"""可配置 Agent 辩论面板"""
from typing import Optional
from concurrent.futures import as_completed, TimeoutError as FuturesTimeoutError
from core.shared.infrastructure.thread_pool import thread_pool
import logging
from core.shared.infrastructure.llm import LLMClient
from core.shared.infrastructure.cache import CacheManager
from agents.prompts import (
    ANALYST_TECH_PROMPT, ANALYST_FUNDAMENTAL_PROMPT,
    ANALYST_CAPITAL_PROMPT, ANALYST_MACRO_PROMPT,
    ANALYST_SENTIMENT_PROMPT,
    ANALYST_RISK_PROMPT, ANALYST_OPPORTUNITY_PROMPT,
    BULL_RESEARCHER_PROMPT, BEAR_RESEARCHER_PROMPT,
    MODERATOR_PROMPT, MODERATOR_WITH_RISK_PROMPT,
)

logger = logging.getLogger(__name__)
AGENT_TIMEOUT = 60

# 角色优先级（主持人永远存在，不在此列表）
ROLE_PRIORITY = ["tech", "fund", "cap", "opp", "bull", "bear", "risk", "macro", "sentiment"]

# 角色 → (名称, 提示词) 映射
ROLE_MAP = {
    "tech":      ("技术分析师", ANALYST_TECH_PROMPT),
    "fund":      ("基本面分析师", ANALYST_FUNDAMENTAL_PROMPT),
    "cap":       ("资金面分析师", ANALYST_CAPITAL_PROMPT),
    "opp":       ("机会分析师", ANALYST_OPPORTUNITY_PROMPT),
    "risk":      ("风控官", ANALYST_RISK_PROMPT),
    "macro":     ("宏观策略师", ANALYST_MACRO_PROMPT),
    "sentiment": ("情绪分析师", ANALYST_SENTIMENT_PROMPT),
    "bull":      ("多方研究员", BULL_RESEARCHER_PROMPT),
    "bear":      ("空方研究员", BEAR_RESEARCHER_PROMPT),
}


def _get_role_distribution(agent_count: int):
    """根据 agent 总数分配分析师角色和风控模式。
    返回 (analyst_keys, risk_merged)
    - risk_merged=True: 风控职责合并到主持人，不创建独立风控 Agent
    """
    agent_count = max(2, min(8, agent_count))

    if agent_count < 5:
        risk_merged = True
        pickable = [r for r in ROLE_PRIORITY if r != "risk"]
        analysts = pickable[:agent_count - 1]  # -1 留给主持人
    else:
        risk_merged = False
        analysts = ROLE_PRIORITY[:agent_count - 1]

    if "risk" in analysts and risk_merged:
        analysts = [r for r in analysts if r != "risk"]

    return analysts, risk_merged


class _Agent:
    def __init__(self, name, role_prompt, llm, cache=None, max_tokens=2048):
        self.name = name
        self.role = role_prompt
        self.llm = llm
        self.cache = cache
        self.max_tokens = max_tokens

    def analyze(self, context, temperature=0.7):
        messages = [
            {"role": "system", "content": self.role},
            {"role": "user", "content": context},
        ]
        return self.llm.chat_with_cache(messages, temperature=temperature, max_tokens=self.max_tokens, cache_manager=self.cache)


class DebatePanel:
    def __init__(self, llm=None, cache=None, agent_count=7):
        llm = llm or LLMClient()
        cache = cache or CacheManager()
        self.agent_count = max(2, min(8, agent_count))
        self.analysts, self.risk_merged = _get_role_distribution(self.agent_count)

        self.a = {}
        for key in self.analysts:
            if key in ROLE_MAP:
                name, prompt = ROLE_MAP[key]
                self.a[key] = _Agent(name, prompt, llm, cache)

        mod_prompt = MODERATOR_WITH_RISK_PROMPT if self.risk_merged else MODERATOR_PROMPT
        self.a["mod"] = _Agent("主持人", mod_prompt, llm, cache, max_tokens=4096)

        logger.info(f"辩论面板: {self.agent_count} agents, "
                    f"分析师=[{', '.join(self.a[k].name for k in self.analysts)}], "
                    f"风控={'合并主持人' if self.risk_merged else '独立'}")

    def debate(self, stock_name, stock_code, data_context):
        header = f"股票名称: {stock_name}\n股票代码: {stock_code}\n\n=== 数据 ===\n{data_context}\n"

        # 并行运行分析师（不含 risk/bull/bear，它们需要看到其他分析结果后执行）
        parallel_keys = [k for k in self.analysts if k not in ("risk", "bull", "bear")]
        results = {}
        if parallel_keys:
            with thread_pool.default.borrow() as pool:
                futures = {pool.submit(self.a[k].analyze, header): k for k in parallel_keys}
                try:
                    for f in as_completed(futures, timeout=AGENT_TIMEOUT + 10):
                        k = futures[f]
                        try:
                            results[k] = f.result(timeout=AGENT_TIMEOUT)
                        except (FuturesTimeoutError, TimeoutError):
                            logger.warning(f"Agent [{self.a[k].name}] 超时，使用降级输出")
                            results[k] = f"[{self.a[k].name}] 分析超时，请稍后重试"
                        except Exception as e:
                            logger.error(f"Agent [{self.a[k].name}] 异常: {e}")
                            results[k] = f"[{self.a[k].name}] 分析失败: {e}"
                except FuturesTimeoutError:
                    logger.warning("并行分析整体超时，部分Agent未返回结果")
                    for k in parallel_keys:
                        if k not in results:
                            results[k] = f"[{self.a[k].name}] 分析超时，请稍后重试"

        # 构建上下文
        ctx = header
        for k in parallel_keys:
            if k in results:
                ctx += f"\n\n[{self.a[k].name}]\n{results[k]}"

        # 独立风控（如果存在）
        if not self.risk_merged and "risk" in self.analysts and "risk" in self.a:
            try:
                risk_output = self.a["risk"].analyze(ctx)
                results["risk"] = risk_output
                ctx += f"\n\n[风控官]\n{risk_output}"
            except Exception as e:
                logger.error(f"风控官异常: {e}")
                risk_output = f"[风控官] 分析失败: {e}"
                results["risk"] = risk_output
                ctx += f"\n\n[风控官]\n{risk_output}"

        # 牛熊辩论：多方 → 空方（空方能看到多方的论证）
        if "bull" in self.analysts and "bull" in self.a:
            try:
                bull_output = self.a["bull"].analyze(ctx)
                results["bull"] = bull_output
                ctx += f"\n\n[多方研究员]\n{bull_output}"
            except Exception as e:
                logger.error(f"多方研究员异常: {e}")
                bull_fail = f"[多方研究员] 分析失败: {e}"
                results["bull"] = bull_fail
                ctx += f"\n\n[多方研究员]\n{bull_fail}"

        if "bear" in self.analysts and "bear" in self.a:
            try:
                bear_output = self.a["bear"].analyze(ctx)
                results["bear"] = bear_output
                ctx += f"\n\n[空方研究员]\n{bear_output}"
            except Exception as e:
                logger.error(f"空方研究员异常: {e}")
                bear_fail = f"[空方研究员] 分析失败: {e}"
                results["bear"] = bear_fail
                ctx += f"\n\n[空方研究员]\n{bear_fail}"

        # 主持人（或兼任风控的主持人）
        try:
            final = self.a["mod"].analyze(ctx)
        except Exception as e:
            logger.error(f"主持人异常: {e}")
            final = f"[主持人] 分析失败: {e}"

        # 构建返回结果
        result = {}
        for k in self.analysts:
            if k == "risk":
                result["风险审查"] = results.get("risk", "")
            elif k == "bull":
                result["多方论证"] = results.get("bull", "")
            elif k == "bear":
                result["空方论证"] = results.get("bear", "")
            elif k in results:
                result[f"{self.a[k].name}"] = results[k]

        if not self.risk_merged and "risk" not in self.analysts:
            pass

        result["最终决议"] = final
        return result