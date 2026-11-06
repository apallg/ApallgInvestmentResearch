"""个股 API"""
import anyio
import asyncio
from fastapi import APIRouter, Depends, HTTPException
from core.shared.infrastructure.thread_pool import thread_pool
from fastapi.responses import StreamingResponse
from backend.api.instrument_identity import require_current_stock_code
from core.data.fetcher import DataFetcher
from core.domain.screening.indicators import Analyzer
from backend.service.market_data import MarketDataService

router = APIRouter()
fetcher = DataFetcher()
analyzer = Analyzer()


@router.get("/search")
async def search_stocks(q: str = "", limit: int = 10):
    from backend.service.instrument_lookup import search_instruments
    return {"data": await anyio.to_thread.run_sync(search_instruments, q, limit)}


@router.get("/{code}")
async def get_stock_info(code: str = Depends(require_current_stock_code)):
    """获取股票基本信息"""
    name = await anyio.to_thread.run_sync(fetcher.get_stock_name, code)
    return {"code": code, "name": name}


@router.get("/{code}/daily")
async def get_stock_daily(code: str = Depends(require_current_stock_code)):
    """获取日K线"""
    try:
        df = await anyio.to_thread.run_sync(fetcher.get_stock_daily, code)
        if df is None or df.empty:
            return {"data": []}
        sources = await anyio.to_thread.run_sync(fetcher.get_sources)
        return {"data": df.to_dict(orient="records"), "source": sources.get("stock_daily", "")}
    except Exception as e:
        raise HTTPException(503, f"获取日K线失败: {e}")


@router.get("/{code}/indicators")
async def get_stock_indicators(code: str = Depends(require_current_stock_code)):
    """获取技术指标"""
    def _compute():
        df = fetcher.get_stock_daily(code)
        if df is None or df.empty:
            return {"indicators": {}, "signals": [], "patterns": []}
        df = analyzer.add_indicators(df)
        ind = analyzer.latest_indicators(df)
        sig = analyzer.detect_cross_signals(df)
        pat = analyzer.detect_kline_patterns(df)
        return {"indicators": ind, "signals": sig or [], "patterns": pat or []}

    try:
        result = await anyio.to_thread.run_sync(_compute)
        return {"data": result}
    except Exception as e:
        raise HTTPException(503, str(e))


@router.get("/{code}/realtime")
async def get_stock_realtime(code: str = Depends(require_current_stock_code)):
    """获取实时行情"""
    try:
        return await anyio.to_thread.run_sync(MarketDataService.get_live_quotes, [code], "live_display")
    except Exception as e:
        return {"data": {}, "meta": {"freshness": "unavailable", "error": str(e)}}


@router.get("/{code}/financial")
async def get_stock_financial(code: str = Depends(require_current_stock_code)):
    """获取财务数据"""
    data = await anyio.to_thread.run_sync(fetcher.get_stock_financial, code)
    return {"data": data or {}}


@router.get("/{code}/capital-flow")
async def get_capital_flow(code: str = Depends(require_current_stock_code)):
    """获取资金流向"""
    try:
        flow = await anyio.to_thread.run_sync(fetcher.get_capital_flow, code)
        if flow is None or (hasattr(flow, 'empty') and flow.empty):
            return {"data": {}}
        if hasattr(flow, 'to_dict'):
            return {"data": flow.to_dict(orient="records")}
        return {"data": flow}
    except Exception as e:
        return {"data": {}, "error": str(e)}


@router.post("/{code}/analyze")
async def analyze_stock(code: str = Depends(require_current_stock_code), mode: str = "single"):
    """AI 分析 (single/panel) —— 非流式版本"""
    from core.shared import config as cfg
    if not cfg.DEEPSEEK_API_KEY:
        raise HTTPException(400, "DEEPSEEK_API_KEY 未配置")

    def _analyze():
        name = fetcher.get_stock_name(code)
        df = fetcher.get_stock_daily(code)
        financial = fetcher.get_stock_financial(code)
        if df is None or df.empty:
            raise HTTPException(400, "日K线数据获取失败")
        context = analyzer.get_analysis_context(df, financial)
        context += f"\n\n股票名称: {name}\n股票代码: {code}"

        if mode == "panel":
            from agents.panel import DebatePanel
            panel = DebatePanel()
            result = panel.debate(name, code, context)
            return {"data": result}
        else:
            from agents.base_agent import BaseAgent
            from agents.prompts import SINGLE_ANALYST_PROMPT
            from core.shared.infrastructure.cache import CacheManager
            agent = BaseAgent("首席分析师", SINGLE_ANALYST_PROMPT, CacheManager())
            analysis = agent.analyze(context)
            return {"data": {"分析报告": analysis}}

    try:
        return await anyio.to_thread.run_sync(_analyze)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"分析失败: {e}")


def _extract_summary(text: str, max_chars: int = 500) -> str:
    """从 Agent 输出中提取摘要：优先取尾部（结论），若太短则取全部"""
    if len(text) <= max_chars:
        return text
    # 取后 max_chars 字，从第一个换行处开始以保证完整性
    tail = text[-max_chars:]
    first_nl = tail.find("\n")
    if first_nl > 0 and first_nl < 100:
        tail = tail[first_nl + 1:]
    return f"...(节选)\n{tail}"


def _collect_stream(llm, messages) -> str:
    """收集流式响应为完整字符串"""
    buf = ""
    for chunk in llm.chat_stream(messages):
        buf += chunk
    return buf


async def _emit_chunks(text: str, chunk_size: int = 80):
    """将文本分块流式发送"""
    for i in range(0, len(text), chunk_size):
        yield f"data: {text[i:i+chunk_size].replace(chr(10), chr(13))}\n"
        await asyncio.sleep(0.02)


@router.post("/{code}/analyze/stream")
async def analyze_stock_stream(code: str = Depends(require_current_stock_code), mode: str = "single"):
    """AI 分析 SSE 流式输出"""
    from core.shared import config as cfg
    if not cfg.DEEPSEEK_API_KEY:
        raise HTTPException(400, "DEEPSEEK_API_KEY 未配置")
    try:
        name = fetcher.get_stock_name(code)
        df = fetcher.get_stock_daily(code)
        if df is None or df.empty:
            raise HTTPException(400, "日K线数据获取失败")
        financial = fetcher.get_stock_financial(code)
        context = analyzer.get_analysis_context(df, financial)
        context += f"\n\n股票名称: {name}\n股票代码: {code}"
        from agents.prompts import SINGLE_ANALYST_PROMPT
        from core.shared.infrastructure.llm import LLMClient
        llm = LLMClient()

        async def generate():
            if mode == "panel":
                from agents.prompts import (
                    ANALYST_TECH_PROMPT, ANALYST_FUNDAMENTAL_PROMPT,
                    ANALYST_CAPITAL_PROMPT, ANALYST_MACRO_PROMPT,
                    ANALYST_RISK_PROMPT, MODERATOR_PROMPT,
                )
                # 截断数据上下文
                max_data_lines = 80
                data_lines = context.split("\n")
                if len(data_lines) > max_data_lines:
                    context_short = "\n".join(data_lines[:max_data_lines]) + "\n...(数据已截断，保留最近80行)"
                else:
                    context_short = context
                header = f"股票名称: {name}\n股票代码: {code}\n\n=== 数据 ===\n{context_short}\n"

                # ─── P0: 4 个独立 Agent 并行执行 ───
                agent_configs = [
                    ("技术分析师", ANALYST_TECH_PROMPT),
                    ("基本面分析师", ANALYST_FUNDAMENTAL_PROMPT),
                    ("资金面分析师", ANALYST_CAPITAL_PROMPT),
                    ("宏观策略师", ANALYST_MACRO_PROMPT),
                ]

                yield "data: [STATUS] 4位分析师并行分析中...\n\n"

                async def run_one(title: str, sys_prompt: str):
                    """在线程池中运行单个 Agent，返回 (title, result) 或 (title, None, error)"""
                    try:
                        result = await anyio.to_thread.run_sync(
                            _collect_stream, llm,
                            [{"role": "system", "content": sys_prompt},
                             {"role": "user", "content": header}],
                        )
                        return (title, result, None)
                    except Exception as e:
                        return (title, None, str(e))

                tasks = [run_one(t, p) for t, p in agent_configs]
                raw_results = await asyncio.gather(*tasks)

                # 按原始顺序排列结果，逐个推送
                reports = []
                for title, result, error in raw_results:
                    if error:
                        yield f"data: [ERROR] {title}: {error}\n\n"
                    else:
                        reports.append((title, result))
                        yield f"data: [AGENT:{title}]\n"
                        async for event in _emit_chunks(result):
                            yield event
                        yield f"data: [END:{title}]\n\n"

                if len(reports) < 2:
                    yield "data: [ERROR] 多数分析师未能完成分析，请稍后重试\n\n"
                    yield "data: [DONE]\n\n"
                    return

                # ─── 风控官审查（依赖前4个结果，取尾部摘要）───
                risk_ctx = f"股票: {name}({code})\n\n"
                for title, text in reports:
                    risk_ctx += f"[{title}]\n{_extract_summary(text)}\n\n"

                yield f"data: [STATUS] 风控官正在审查各分析师报告...\n\n"
                try:
                    risk_buf = await anyio.to_thread.run_sync(
                        _collect_stream, llm,
                        [{"role": "system", "content": ANALYST_RISK_PROMPT},
                         {"role": "user", "content": risk_ctx}],
                    )
                    yield f"data: [AGENT:风控官]\n"
                    async for event in _emit_chunks(risk_buf):
                        yield event
                    yield f"data: [END:风控官]\n\n"
                except Exception as e:
                    risk_buf = f"风控分析失败: {e}"
                    yield f"data: [ERROR] 风控官分析失败: {e}\n\n"

                # ─── 主持人总结 ───
                mod_ctx = f"股票: {name}({code})\n\n"
                for title, text in reports:
                    mod_ctx += f"[{title}]\n{_extract_summary(text)}\n\n"
                mod_ctx += f"[风控官]\n{_extract_summary(risk_buf)}\n\n"
                mod_ctx += "请基于上述各分析师观点，输出最终投研会议纪要。"

                yield f"data: [STATUS] 主持人正在综合总结...\n\n"
                try:
                    yield f"data: [AGENT:主持人]\n"
                    for chunk in llm.chat_stream([
                        {"role": "system", "content": MODERATOR_PROMPT},
                        {"role": "user", "content": mod_ctx},
                    ]):
                        safe = chunk.replace('\n', '\r')
                        yield f"data: {safe}\n"
                    yield f"data: [END:主持人]\n\n"
                except Exception as e:
                    yield f"data: [ERROR] 主持人总结失败: {e}\n\n"

            else:
                messages = [{"role": "system", "content": SINGLE_ANALYST_PROMPT},
                           {"role": "user", "content": context}]
                yield "data: [AGENT:首席分析师]\n"
                result = await anyio.to_thread.run_sync(_collect_stream, llm, messages)
                async for event in _emit_chunks(result):
                    yield event
                yield "data: [END:首席分析师]\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"分析失败: {e}")
