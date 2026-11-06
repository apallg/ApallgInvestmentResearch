"""
周频调仓建议引擎

基于《因子投资方法与实践》第7-8章框架：
- 调仓频率匹配信号半衰期 (20日预测→周频)
- Qlib 预测分数加权分配
- 换手率约束 (≤50%) + 预测相关性动态收紧
- 行业分散约束 (单行业≤30%, 因子协方差代理)
- 容差带 (±12%)
- 止损强制卖出 (-8%)
"""

__all__ = ["WeeklyRebalanceAdvisor", "DEFAULT_PARAMS"]

import json as _json
import logging
import datetime
import math as _math
from pathlib import Path

from core.shared import config
from core.shared.persistence import today_date
from core.domain.portfolio.allocator import (
    PositionAllocator,
    AllocationConfig,
    build_allocation_config,
    compute_one_way_turnover,
)

logger = logging.getLogger(__name__)

_PRED_CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "rebalance_pred"
_PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PARAMS = {
    "top_n": 5,
    "top_k": 10,
    "pool": "a500",
    "sectors": None,
    "code_prefixes": None,
    "sizing": getattr(config, "WEEKLY_REBALANCE_SIZING", "score_weighted"),
    "tolerance_band": 0.12,
    "stop_loss_warn": -0.05,
    "stop_loss_full": -0.10,
    "take_profit_half": 0.15,
    "take_profit_full": 0.30,
    "max_position_pct": 0.20,
    "max_sector_pct": 0.30,
    "max_turnover": 0.30,
    "freq": "week",
    "min_cash_reserve": 0.05,
    "commission_rate": 0.0003,
    "slippage_rate": 0.0001,
    "stamp_duty": 0.0005,
    "cooldown_stop_weeks": 2,
    "cooldown_profit_weeks": 1,
}


def _get_screener():
    from core.domain.screening._screener import BuySideScreener
    return BuySideScreener()


def _get_broker():
    from core.adapters.broker import create_broker
    from core.shared import config
    broker_type = getattr(config, "BROKER_TYPE", "fake")
    return create_broker(broker_type)


def compute_turnover(target_weights, current_weights):
    """计算包含现金腿的单边换手率。纯 L3 计算——不依赖 DB/文件/时间函数。"""
    return compute_one_way_turnover(target_weights, current_weights)


def format_text_report(report):
    """将调仓报告格式化为文本。纯 L3 函数——不依赖 DB/文件/时间函数。"""
    lines = []
    lines.append("═══════════════════════════════════════════")
    lines.append("  Apallg投研 周频调仓建议")
    lines.append(f"  {report['date']} (周一)  |  模型: LGB+XGB 双模型")
    lines.append(f"  股票池: {report['pool']}  |  板块: {report.get('sectors') or '不限'}  |  持仓数: {report['top_n']}")
    lines.append("═══════════════════════════════════════════")
    lines.append("")

    acc = report.get("account", {})
    lines.append(f"💰 账户: 总资产 ¥{acc.get('total_assets', 0):,.0f}  |  总现金 ¥{acc.get('total_cash', acc.get('cash', 0)):,.0f}  |  可用 ¥{acc.get('available_cash', acc.get('cash', 0)):,.0f}  |  冻结 ¥{acc.get('frozen_cash', 0):,.0f}  |  持仓市值 ¥{acc.get('market_value', 0):,.0f}")
    lines.append("")

    lines.append(f"📊 目标组合 (Top-{report['top_n']})")
    lines.append(f"{'排名':<4} {'代码':<8} {'名称':<10} {'预测分':<8} {'操作':<8} {'目标仓位':<10} {'当前':<10}")
    lines.append("-" * 60)
    for r in report.get("target_combo", []):
        cur_str = f"{r['current_weight']:.0%}" if r['current_weight'] > 0 else "-"
        lines.append(
            f"{r['rank']:<4} {r['code']:<8} {r['name']:<10} {r['qlib_score']:<8.4f} "
            f"{r['action']:<8} {r['target_weight']:.0%}{'':<7} {cur_str:<10}"
        )
    lines.append("")

    for ck in report.get("constraints", []):
        if ck["name"] == "行业分布":
            lines.append(f"📊 {ck['value']}")
    lines.append("")

    buys = report.get("to_buy", [])
    if buys:
        lines.append(f"📈 建议买入 ({len(buys)}只)")
        for o in buys:
            lines.append(f"  {o['code']} {o['name']:<8} → ¥{o['amount']:,.0f} ({o['target_weight']:.0%}), {o['reason']}")
        lines.append("")

    sells = report.get("to_sell", [])
    if sells:
        lines.append(f"📉 建议卖出 ({len(sells)}只)")
        for o in sells:
            lines.append(f"  {o['code']} {o['name']:<8} → 现持 ¥{o['amount']:,.0f}, {o['reason']}")
        lines.append("")

    stops = report.get("stop_loss", [])
    if stops:
        lines.append(f"⚠️ 止损提醒 ({len(stops)}只)")
        for o in stops:
            lines.append(f"  {o['code']} {o['name']:<8} → {o['reason']}")
        lines.append("")
    else:
        lines.append("⚠️ 止损提醒: (无触发)")
        lines.append("")

    adjusts = report.get("to_adjust", [])
    if adjusts:
        lines.append(f"🔄 建议调整 ({len(adjusts)}只)")
        for o in adjusts:
            lines.append(f"  {o['code']} {o['name']:<8} → {o['current_weight']:.0%}→{o['target_weight']:.0%}, {o['reason']}")
        lines.append("")

    lines.append("📋 约束检查")
    for ck in report.get("constraints", []):
        if ck["name"] == "行业分布":
            continue
        status = "✓" if ck["pass"] else "⚠"
        note = f" ({ck['note']})" if ck["note"] else ""
        lines.append(f"  {ck['name']}: {ck['value']} (上限 {ck['limit']}) {status}{note}")
    lines.append("")

    aux = report.get("auxiliary", {})
    if aux:
        lines.append("📋 辅助参考 (仅展示, 不影响排序)")
        lines.append(f"{'代码':<8} {'基本面':<12} {'资金流':<8} {'北向':<8} {'情绪':<8}")
        lines.append("-" * 48)
        for code, info in aux.items():
            lines.append(f"{code:<8} {info['fundamental']:<12} {info['flow']:<8} {info.get('north', '-'):<8} {info.get('sentiment', '-'):<8}")
        lines.append("")

    risk = report.get("risk", {})
    if risk:
        lines.append("⚠️ 风控一览")
        lines.append(f"  宏观风险: {risk.get('macro', '-')}  |  情绪风险: {risk.get('sentiment', '-')}")
    lines.append("")
    lines.append("═══════════════════════════════════════════")
    return "\n".join(lines)


def build_combo_table(top_candidates, target_weights, current_weights, current_holdings, tolerance_band):
    """构建调仓组合表。纯 L3 函数——不依赖 DB/文件/时间函数。"""
    table = []
    for i, c in enumerate(top_candidates):
        code = c.get("code", "")
        tgt_w = target_weights.get(code, 0)
        cur_w = current_weights.get(code, 0)
        delta = tgt_w - cur_w

        if code in current_holdings:
            if abs(delta) <= tolerance_band:
                action = "持有"
            elif delta > 0:
                action = "加仓"
            else:
                action = "减仓"
        else:
            action = "★买入"

        table.append({
            "rank": i + 1,
            "code": code,
            "name": c.get("name", ""),
            "qlib_score": round(c.get("qlib_score", 0), 4),
            "action": action,
            "target_weight": round(tgt_w, 4),
            "current_weight": round(cur_w, 4),
            "delta": round(delta, 4),
        })
    return table


def compute_prediction_correlation(current_scores, prev_scores):
    """预测排名相关度。纯 L3 函数——不依赖 DB/文件/时间函数。
    返回 spearman 相关系数 [-1, 1]，或 None (公共代码不足 20 只)。
    """
    if not prev_scores:
        return None
    common_codes = sorted(set(current_scores.keys()) & set(prev_scores.keys()))
    if len(common_codes) < 20:
        return None
    try:
        import math as _m
        from scipy.stats import spearmanr
        cur = [current_scores[c] for c in common_codes]
        prev = [prev_scores[c] for c in common_codes]
        corr, _ = spearmanr(cur, prev, nan_policy='omit')
        if corr is None or _m.isnan(corr):
            return None
        return max(-1.0, min(1.0, corr))
    except Exception:
        return None


def classify_limit_codes(rows):
    """从查询结果判断涨跌停代码。纯 L3 函数——不依赖 DB/文件/时间函数。
    rows: [{code, pct_chg}, ...]
    返回 (limit_up_set, limit_down_set)。
    按板块区分涨跌幅: 主板10% / 创业板·科创板20% / 北交所30%, 不套用统一 ±9.5%。
    """
    from core.domain.backtest.common import limit_pct_for_code
    limit_up = set()
    limit_down = set()
    for r in rows:
        code = r["code"]
        pct = float(r.get("pct_chg", 0) or 0)
        name = str(r.get("name", "") or "").upper()
        is_st = "ST" in name
        threshold = limit_pct_for_code(code, is_st=is_st) * 100 - 0.5
        if pct >= threshold:
            limit_up.add(code)
        elif pct <= -threshold:
            limit_down.add(code)
    return limit_up, limit_down


def _get_industry_map():
    from core.data.fetcher import DataFetcher
    return DataFetcher().get_industry_map()


class WeeklyRebalanceAdvisor:
    def __init__(self, **kwargs):
        for k, v in DEFAULT_PARAMS.items():
            setattr(self, k, kwargs.get(k, v))
        self.screener = _get_screener()
        self.broker = None
        self._cooldown = self._load_persisted_cooldown()
        alloc_cfg = build_allocation_config(
            getattr(self, "freq", "week") or "week",
            sizing=self.sizing,
            max_single_pct=self.max_position_pct,
            max_sector_pct=self.max_sector_pct,
            single_floor_pct=0.10,
            target_sum=1.0 - self.min_cash_reserve,
            max_turnover=self.max_turnover,
            stop_loss_warn=self.stop_loss_warn,
            stop_loss_full=self.stop_loss_full,
            take_profit_half=self.take_profit_half,
            take_profit_full=self.take_profit_full,
            impact_cost_lambda=getattr(config, "PORTFOLIO_IMPACT_LAMBDA", 0.3)
                if getattr(config, "PORTFOLIO_IMPACT_ENABLED", True) else 0.0,
            commission_rate=self.commission_rate,
            slippage_rate=self.slippage_rate,
            stamp_duty=self.stamp_duty,
            cooldown_stop_weeks=self.cooldown_stop_weeks,
            cooldown_profit_weeks=self.cooldown_profit_weeks,
        )
        self._allocator = PositionAllocator(alloc_cfg)

        # 加载 regime 止盈止损配置 (对齐回测引擎)
        self._regime_stops = {}
        try:
            from core.shared.persistence import load_json
            cfg_path = __import__('pathlib').Path('config/regime_ratio.json')
            cfg = load_json(str(cfg_path))
            if cfg:
                stops = cfg.get('stops', {})
                for k, v in stops.items():
                    idx = int(k)
                    self._regime_stops[idx] = {
                        'warn': -abs(float(v.get('warn', 5))) / 100,
                        'full': -abs(float(v.get('full', 10))) / 100,
                        'profitHalf': float(v.get('profitHalf', 15)) / 100,
                        'profitFull': float(v.get('profitFull', 30)) / 100,
                    }
        except Exception:
            pass

    def _load_persisted_cooldown(self):
        """从 settings 表加载持久化的冷冻期映射 {code: 剩余期数}。
        惰性 DB 访问 (与 _detect_limit_codes 一致), 失败回退空映射。
        """
        try:
            from core.data.repos.settings import SettingsRepo
            import json as _json
            _raw = SettingsRepo.get("rebalance_cooldown", None)
            if not _raw:
                return {}
            _parsed = _json.loads(_raw) if isinstance(_raw, str) else _raw
            if "periods" in _parsed:
                self._cooldown_as_of = _parsed.get("as_of")
                _parsed = _parsed.get("periods", {})
            return {k: int(v) for k, v in _parsed.items() if int(v) > 0}
        except Exception as e:
            logger.debug(f"[冷冻期] 加载持久化失败: {e}")
            return {}

    def _persist_cooldown(self):
        """持久化冷冻期到 settings 表, 使止损 2 周/止盈 1 周约束跨 advisor 实例生效。"""
        try:
            from core.data.repos.settings import SettingsRepo
            import json as _json
            payload = {"as_of": getattr(self, "_cooldown_as_of", None),
                       "periods": self._cooldown}
            SettingsRepo.set("rebalance_cooldown", _json.dumps(payload))
        except Exception as e:
            logger.debug(f"[冷冻期] 持久化失败: {e}")

    def _ensure_broker(self):
        if self.broker is None:
            self.broker = _get_broker()
            if not self.broker.connect():
                raise RuntimeError("券商连接失败")

    def generate_advice(self, pool=None, sectors=None, code_prefixes=None,
                        sizing=None, eval_date=None, account_snapshot=None):
        pool = pool or self.pool
        sectors = sectors or self.sectors
        code_prefixes = code_prefixes or self.code_prefixes
        sizing = sizing or self.sizing

        today = eval_date or today_date()

        self._cooldown_cleanup(today)
        self._allocator.config.max_turnover = self.max_turnover  # 重置，防止跨方法状态泄露
        # ── 1. 获取当前持仓 ──
        if account_snapshot is None:
            try:
                self._ensure_broker()
                positions = self.broker.get_positions()
                account = self.broker.get_account()
            except Exception as e:
                logger.warning(f"读取持仓失败: {e}")
                return {"error": "券商持仓或账户读取失败，无法生成调仓建议"}
        else:
            positions = account_snapshot["holdings"]
            account = account_snapshot["account"]
        if account.get("total_assets") is None or (account_snapshot is None and account.get("status") != "fresh"):
            return {"error": "账户资产不可用，无法生成调仓建议"}
        total_assets = account.get("total_assets", account.get("cash", 100000))
        current_holdings = {p["symbol"]: p for p in positions}

        # ── 2. 获取最新 Qlib 预测 ──
        qlib_model, qlib_scores = self.screener.get_qlib_predictions(as_of=today, pool=pool)
        if qlib_model is None or not qlib_scores:
            return {"error": "未找到可用的 Qlib 模型或预测数据，请先训练模型"}

        # ── 换手率惩罚: 预测排名相关度越高 → 收窄换手上限 ──
        _effective_max_turnover = self.max_turnover
        _rank_corr = self._compute_prediction_correlation(qlib_scores, eval_date=today)
        if _rank_corr is not None:
            _turnover_scale = max(0.4, 1.0 - _rank_corr * 0.6)
            _effective_max_turnover = self.max_turnover * _turnover_scale
            logger.info(f"[换手惩罚] 预测排名相关度={_rank_corr:.3f}, "
                        f"换手缩放={_turnover_scale:.2f}, "
                        f"换手上限 {self.max_turnover:.0%}→{_effective_max_turnover:.0%}")
        self._save_predictions(qlib_scores, eval_date=today)

        # ── 3. 跑选股 (Qlib-only) ──
        screen_pool = pool
        screen_custom_sectors = None
        screen_concept_sectors = None
        if sectors and pool not in ("custom", "concept"):
            screen_pool = "custom"
            screen_custom_sectors = sectors

        screen_result = self.screener.screen(
            pool=screen_pool,
            top_n=self.top_k,
            skip_debate=True, skip_l1=True, skip_l2=True,
            skip_tech=True, skip_sent=True,
            skip_fundamental=True, skip_flow=True,
            skip_north_flow=True, skip_size_turnover=True,
            skip_momentum=True, skip_volatility=True,
            skip_illiq=True, skip_overnight_gap=True,
            skip_ht=True,
            skip_wb_industry=True, skip_wb_divergence=True, skip_wb_moneyflow=True,
            source_weights={"qlib": 1.0},
            qlib_model=qlib_model,
            code_prefixes=code_prefixes,
            custom_sectors=screen_custom_sectors,
            concept_sectors=screen_concept_sectors,
            eval_date=today,
        )

        candidates = screen_result.get("candidates", [])
        if not candidates:
            return {"error": "选股结果为空"}

        # 构建代码→候选信息映射
        candidate_map = {}
        for i, c in enumerate(candidates):
            code = c.get("code", "")
            candidate_map[code] = {
                "rank": i + 1,
                "code": code,
                "name": c.get("name", ""),
                "qlib_score": c.get("qlib_score", 0),
                "l0_score": c.get("l0_score", 0),
            }

        # ── 4. 计算目标权重 (统一分配引擎) ──
        top_n_codes = [c.get("code") for c in candidates[:self.top_n]]
        timing_ratio = 1.0
        try:
            from core.domain.timing.model import get_timing_ratio
            timing_ratio = get_timing_ratio(as_of=today)
            if timing_ratio is None:
                timing_ratio = 1.0
        except Exception:
            pass
        sector_map = _get_industry_map() or {}
        self._allocator.config.score_key = "qlib_score"
        self._allocator.config.max_turnover = _effective_max_turnover
        target_weights = self._allocator.allocate_weights_only(
            candidates[:self.top_n], {}, sector_map, self._cooldown)
        target_weights = self._allocator._normalize_with_timing(
            target_weights, timing_ratio)

        # ── 5. 持仓分类 ──
        classification = self._classify_holdings(
            current_holdings, candidate_map, top_n_codes, total_assets
        )
        # 补全 sale/stop_loss 中的股票名称
        name_needed = [it["code"] for k in ("sell", "stop_loss", "reduce") for it in classification.get(k, []) if not it.get("name")]
        if name_needed:
            names = self._lookup_names(name_needed)
            for k in ("sell", "stop_loss", "reduce"):
                for it in classification.get(k, []):
                    if not it.get("name") and it["code"] in names:
                        it["name"] = names[it["code"]]

        # ── 6. 计算当前权重 ──
        prices = {}
        for p in positions:
            prices[p["symbol"]] = p.get("current_price", 0)
        for c in candidates[:self.top_n]:
            code = c.get("code")
            if code not in prices:
                prices[code] = 0

        current_weights = {}
        for sym, p in current_holdings.items():
            current_weights[sym] = p.get("market_value", 0) / total_assets if total_assets > 0 else 0
        initial_build = not current_weights

        # ── 7. 换手率约束 (统一分配引擎) ──
        target_weights = self._allocator._apply_turnover_limit(
            target_weights, current_weights)

        # ── 8. 生成调仓订单 ──
        all_codes = set(list(target_weights.keys()) + list(current_weights.keys()))
        limit_names = {code: item.get("name", "") for code, item in candidate_map.items()}
        limit_names.update({code: item.get("name", "") for code, item in current_holdings.items()})
        limit_up, limit_down = self._detect_limit_codes(all_codes, limit_names)
        orders = self._generate_orders(
            classification, target_weights, current_weights,
            total_assets, account, prices, candidate_map,
            effective_max_turnover=_effective_max_turnover,
            limit_up_codes=limit_up, limit_down_codes=limit_down,
        )

        # ── 9. 约束检查 ──
        turnover = self._compute_turnover(target_weights, current_weights)
        constraint_results = self._check_constraints(
            turnover, target_weights, candidates[:self.top_n],
            effective_max_turnover=_effective_max_turnover,
            rank_corr=_rank_corr,
            initial_build=initial_build,
        )

        # ── 9. 辅助数据 ──
        auxiliary = self._gather_auxiliary(candidates[:self.top_n])

        # ── 10. 构建报告 ──
        report = {
            "date": today.isoformat(),
            "pool": pool,
            "sectors": sectors,
            "model": {"exp_id": qlib_model[0], "run_id": qlib_model[1]},
            "top_n": self.top_n,
            "top_k": self.top_k,
            "sizing": sizing,
            "account": {
                "total_assets": round(total_assets, 2),
                "cash": round(account.get("cash", 0), 2),
                "available_cash": round(account.get("available_cash", account.get("cash", 0)), 2),
                "total_cash": round(account.get("total_cash", account.get("cash", 0)), 2),
                "frozen_cash": round(account.get("frozen_cash", 0), 2),
                "market_value": round(account.get("market_value", 0), 2),
            },
            "target_combo": self._build_combo_table(
                candidates[:self.top_n], target_weights, current_weights, current_holdings
            ),
            "to_buy": [o for o in orders if o["side"] == "buy"],
            "to_sell": [o for o in orders if o["side"] == "sell" and o.get("urgency") != "stop_loss"],
            "to_adjust": [o for o in orders if o["side"] in ("adjust", "reduce", "add")],
            "stop_loss": [o for o in orders if o.get("urgency") == "stop_loss"],
            "turnover": round(turnover, 4),
            "constraints": constraint_results,
            "auxiliary": auxiliary,
            "classification": classification,
            "next_rebalance": (today + datetime.timedelta(days=7)).isoformat(),
        }

        return report

    def generate_advice_from_screening(self, screening_result, holdings_data,
                                        account_data, timing_ratio=1.0, panic=False):
        today = today_date()

        self._cooldown_cleanup(today)

        candidates = screening_result.get("candidates", [])
        if not candidates:
            return {"error": "选股结果为空"}

        total_assets = account_data.get("total_assets", account_data.get("cash", 100000))
        current_holdings = {}
        for h in holdings_data:
            code = h["code"]
            current_holdings[code] = {
                "symbol": code,
                "market_value": float(h.get("market_value", 0) or 0),
                "unrealized_pnl": float(h.get("unrealized_pnl", 0) or 0),
                "current_price": float(h.get("current_price", 0) or 0),
                "shares": int(h.get("shares", 0)),
                "cost_price": float(h.get("cost_price", 0) or 0),
            }

        candidate_map = {}
        for i, c in enumerate(candidates):
            code = c.get("code", "")
            candidate_map[code] = {
                "rank": i + 1,
                "code": code,
                "name": c.get("name", ""),
                "qlib_score": c.get("qlib_score", 0),
                "l0_score": c.get("l0_score", 0),
            }

        # Top-N 只作持仓上限: 全部勾选候选参与权重分配, 截断到前 top_n 进组合, 其余进观察区
        top_n_candidates = candidates[:self.top_n]
        observe_candidates = candidates[self.top_n:]

        if panic:
            timing_ratio = 0.0

        sector_map = _get_industry_map() or {}
        self._allocator.config.max_turnover = self.max_turnover  # 重置，防止跨方法状态泄露
        self._allocator.config.score_key = "l0_score"

        # 按市场状态覆盖止盈止损 (对齐回测引擎)
        regime_info = screening_result.get("regime") or {}
        regime_name = regime_info.get("regime", "")
        if regime_name:
            _STATE_MAP = {n: i for i, n in enumerate(
                ["volatile_up","volatile_down","volatile_flat","steady_up","steady_down","steady_flat"])}
            ri = _STATE_MAP.get(regime_name)
            if ri is not None and hasattr(self, '_regime_stops'):
                s = self._regime_stops.get(ri)
                if s is not None:
                    self._allocator.config.stop_loss_warn = s['warn']
                    self._allocator.config.stop_loss_full = s['full']
                    self._allocator.config.take_profit_half = s['profitHalf']
                    self._allocator.config.take_profit_full = s['profitFull']

        top_n_codes = [c.get("code") for c in top_n_candidates]
        classification = self._classify_holdings(
            current_holdings, candidate_map, top_n_codes, total_assets
        )

        current_weights = {}
        for sym, p in current_holdings.items():
            current_weights[sym] = p.get("market_value", 0) / total_assets if total_assets > 0 else 0
        initial_build = not current_weights

        observe_weights = {}
        for _ob in classification.get("observe", []):
            _obc = _ob["code"]
            if _obc in current_weights:
                observe_weights[_obc] = current_weights[_obc]

        target_weights = self._allocator.allocate_weights_only(
            top_n_candidates, {}, sector_map, self._cooldown)
        top_budget = max(
            0.0,
            self._allocator.config.target_sum * timing_ratio - sum(observe_weights.values()),
        )
        _tw_sum = sum(target_weights.values())
        if _tw_sum > top_budget and _tw_sum > 0:
            target_weights = {k: v / _tw_sum * top_budget for k, v in target_weights.items()}
        elif _tw_sum <= 0:
            target_weights = {}
        target_weights.update(observe_weights)

        name_needed = [it["code"] for k in ("sell", "stop_loss", "reduce") for it in classification.get(k, []) if not it.get("name")]
        if name_needed:
            names = self._lookup_names(name_needed)
            for k in ("sell", "stop_loss", "reduce"):
                for it in classification.get(k, []):
                    if not it.get("name") and it["code"] in names:
                        it["name"] = names[it["code"]]

        prices = {}
        for sym, p in current_holdings.items():
            prices[sym] = p.get("current_price", 0)
        for c in top_n_candidates:
            code = c.get("code")
            if code not in prices:
                prices[code] = c.get("close", 0)

        target_weights = self._allocator._apply_turnover_limit(
            target_weights, current_weights)

        all_codes = set(list(target_weights.keys()) + list(current_weights.keys()))
        limit_names = {code: item.get("name", "") for code, item in candidate_map.items()}
        limit_names.update({code: item.get("name", "") for code, item in current_holdings.items()})
        limit_up, limit_down = self._detect_limit_codes(all_codes, limit_names)
        _skipped = []
        orders = self._generate_orders(
            classification, target_weights, current_weights,
            total_assets, account_data, prices, candidate_map,
            effective_max_turnover=self.max_turnover,
            limit_up_codes=limit_up, limit_down_codes=limit_down,
            keep_observe=True, skipped_out=_skipped,
        )

        turnover = self._compute_turnover(target_weights, current_weights)
        constraint_results = self._check_constraints(
            turnover, target_weights, top_n_candidates,
            initial_build=initial_build,
        )

        return {
            "target_weights": target_weights,
            "classification": classification,
            "orders": orders,
            "constraints": constraint_results,
            "turnover": turnover,
            "observe_candidates": observe_candidates,
            "top_n_codes": top_n_codes,
            "skipped": _skipped,
        }

    def _cooldown_cleanup(self, as_of):
        from core.shared.constants import get_freq_meta

        as_of = as_of if isinstance(as_of, datetime.date) else datetime.date.fromisoformat(str(as_of)[:10])
        previous = getattr(self, "_cooldown_as_of", None)
        if previous:
            previous = datetime.date.fromisoformat(str(previous)[:10])
            elapsed = max(0, (as_of - previous).days // get_freq_meta(getattr(self, "freq", "week")).days)
            if elapsed:
                self._cooldown = {
                    code: periods - elapsed
                    for code, periods in self._cooldown.items()
                    if periods - elapsed > 0
                }
        self._cooldown_as_of = as_of.isoformat()
        self._persist_cooldown()


    def _pred_cache_path(self):
        import re
        safe = re.sub(r'[^a-zA-Z0-9_-]', '_', self.pool or "default")
        return _PRED_CACHE_DIR / f"{safe}.json"

    def _load_prev_predictions(self, as_of_date=None):
        fpath = self._pred_cache_path()
        if not fpath.exists():
            return None
        try:
            from core.shared.persistence import load_json
            data = load_json(str(fpath))
            if as_of_date is not None:
                cached_date = data.get("date", "")
                if cached_date and cached_date >= as_of_date.isoformat():
                    return None
            return data.get("scores", {})
        except Exception as e:
            logger.warning(f"[换手惩罚] 预测缓存读取失败: {e}")
            return None

    def _save_predictions(self, scores, eval_date=None):
        fpath = self._pred_cache_path()
        from core.shared.persistence import save_json
        save_date = eval_date if eval_date is not None else today_date()
        try:
            tmp = fpath.with_suffix(".tmp")
            save_json({
                "date": save_date.isoformat() if hasattr(save_date, 'isoformat') else str(save_date),
                "pool": self.pool,
                "scores": {k: float(v) for k, v in scores.items()},
            }, str(tmp))
            tmp.replace(fpath)
        except Exception as e:
            logger.warning(f"[换手惩罚] 预测缓存写入失败: {e}")

    def _compute_prediction_correlation(self, current_scores, eval_date=None):
        prev_scores = self._load_prev_predictions(as_of_date=eval_date)
        return compute_prediction_correlation(current_scores, prev_scores)

    # ─── 内部方法 ───

    def _allocate_weights(self, top_candidates, sizing, score_key="qlib_score"):
        n = len(top_candidates)
        if n == 0:
            return {}
        target_sum = 1.0 - self.min_cash_reserve

        if sizing == "risk_parity" and n >= 3:
            rp_weights = self._risk_parity_weights(top_candidates, target_sum)
            if rp_weights:
                return rp_weights
            logger.info("[调仓] 风险平价失败, 回退到分数加权")

        if sizing == "equal":
            w = target_sum / n
            return {c.get("code"): w for c in top_candidates}

        # score_squared 和 score_weighted 都默认用 score² (对齐回测)
        use_squared = sizing != "proportional"
        if use_squared:
            total = sum(max(c.get(score_key, 0), 0.001) ** 2 for c in top_candidates)
            if total <= 0:
                w = target_sum / n
                return {c.get("code"): w for c in top_candidates}
            desired = {c.get("code"): max(c.get(score_key, 0), 0.001) ** 2 / total * target_sum
                       for c in top_candidates}
        else:
            scores = [max(0, c.get(score_key, 0)) for c in top_candidates]
            total_score = sum(scores)
            if total_score <= 0:
                w = target_sum / n
                return {c.get("code"): w for c in top_candidates}
            desired = {c.get("code"): s / total_score * target_sum
                       for c, s in zip(top_candidates, scores)}

        # 冲击成本惩罚: 对 desired 权重中的高分股预扣
        if getattr(config, "BUY_SIDE_IMPACT_ENABLED", True):
            impact_lambda = getattr(config, "BUY_SIDE_IMPACT_LAMBDA", 0.3)
            if impact_lambda > 0:
                for c in top_candidates:
                    code = c.get("code")
                    impact = c.get("impact_cost", 0)
                    if impact > 0 and code in desired:
                        penalty = impact_lambda * _math.tanh(impact * 20)
                        desired[code] = max(0, desired[code] - penalty * 0.05)

        des_sum = sum(desired.values())
        if des_sum > 0:
            desired = {k: v / des_sum * target_sum for k, v in desired.items()}

        lo = 0.10
        hi = self.max_position_pct

        weights = {}
        for code, w in desired.items():
            weights[code] = max(lo, min(hi, w))

        cur_sum = sum(weights.values())
        if abs(cur_sum - target_sum) < 1e-6:
            return weights

        for _ in range(20):
            free_codes = [c for c in weights if lo < weights[c] < hi]
            if not free_codes:
                break
            gap = target_sum - sum(weights.values())
            if abs(gap) < 1e-6:
                break
            adj = gap / len(free_codes)
            for c in free_codes:
                weights[c] = max(lo, min(hi, weights[c] + adj))
            if abs(target_sum - sum(weights.values())) < 1e-6:
                break

        return weights

    def _risk_parity_weights(self, top_candidates, target_sum, as_of=None):
        """风险平价 (Equal Risk Contribution) 权重

        从 daily_quotes 取 60 日收益 → Ledoit-Wolf 协方差 → 等风险贡献优化
        返回 {code: weight} 或空 dict (失败时回退)
        """
        try:
            from core.data import Database
            if not Database.is_available():
                return {}

            codes = [c.get("code", "").replace(".SZ", "").replace(".SH", "").replace(".BJ", "")
                    for c in top_candidates]
            bare_to_orig = {b: c.get("code") for b, c in zip(codes, top_candidates)}
            codes = [c for c in codes if c]

            if len(codes) < 3:
                return {}

            ref_date = as_of if as_of is not None else today_date()
            ref_str = ref_date.isoformat() if hasattr(ref_date, 'isoformat') else str(ref_date)

            placeholders = ",".join(["%s"] * len(codes))
            rows = Database.fetchall(
                f"SELECT code, trade_date, close FROM daily_quotes "
                f"WHERE code IN ({placeholders}) "
                f"AND trade_date >= DATE_SUB(%s, INTERVAL 90 DAY) "
                f"AND trade_date <= %s "
                f"ORDER BY code, trade_date",
                codes + [ref_str, ref_str],
            )
            if not rows:
                return {}

            import pandas as pd
            df = pd.DataFrame(rows)
            df["close"] = df["close"].astype(float)
            piv = df.pivot(index="trade_date", columns="code", values="close")
            piv = piv.ffill().dropna(axis=1)
            returns = piv.pct_change(fill_method=None).dropna()

            valid_codes = [c for c in returns.columns if c in bare_to_orig]
            if len(valid_codes) < 3:
                return {}
            returns = returns[valid_codes].iloc[-60:]

            from pypfopt import risk_models
            cov = risk_models.CovarianceShrinkage(returns).ledoit_wolf()

            n = len(valid_codes)
            lo = 0.08
            hi = self.max_position_pct

            # scipy 等风险贡献优化
            from scipy.optimize import minimize
            import numpy as np

            cov_np = cov.values if hasattr(cov, 'values') else np.array(cov)

            def _erc_error(w):
                w = np.abs(w)
                w = w / w.sum()
                pvol = np.sqrt(w @ cov_np @ w)
                if pvol < 1e-10:
                    return 1e10
                mc = cov_np @ w
                rc = w * mc / pvol
                target = np.ones(n) / n
                return np.sum((rc - target) ** 2) * 1000

            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
            bounds = [(lo, hi) for _ in range(n)]
            init = np.ones(n) / n

            result = minimize(_erc_error, init, method="SLSQP",
                            bounds=bounds, constraints=constraints,
                            options={"maxiter": 200, "ftol": 1e-12})

            if not result.success:
                logger.warning(f"[风险平价] 优化未收敛: {result.message}")
                return {}

            raw_w = np.abs(result.x)
            raw_w = raw_w / raw_w.sum() * target_sum
            weights = {}
            for i, bare_code in enumerate(valid_codes):
                orig = bare_to_orig.get(bare_code, bare_code)
                weights[orig] = round(float(raw_w[i]), 4)

            logger.info(f"[风险平价] {n} 只持仓, 权重: "
                        f"{', '.join(f'{k}:{v:.1%}' for k, v in weights.items())}")
            return weights

        except Exception as e:
            logger.warning(f"[风险平价] 异常 (回退分数加权): {e}")
            return {}

    def _apply_sector_diversify(self, raw_weights, top_candidates):
        industry_map = _get_industry_map()
        if not industry_map or len(raw_weights) <= 1:
            return raw_weights

        code_to_industry = {}
        for c in top_candidates:
            code = c.get("code", "")
            ind_info = industry_map.get(code, {})
            code_to_industry[code] = ind_info.get("industry", "未知")

        industry_groups = {}
        for code in raw_weights:
            ind = code_to_industry.get(code, "未知")
            if ind not in industry_groups:
                industry_groups[ind] = []
            industry_groups[ind].append(code)

        weights = dict(raw_weights)
        lo = 0.10
        for _ in range(20):
            # 行业约束
            for ind, codes in industry_groups.items():
                sector_w = sum(weights[c] for c in codes)
                if sector_w > self.max_sector_pct:
                    scale = self.max_sector_pct / sector_w
                    for c in codes:
                        weights[c] *= scale
            # 单只约束
            for c in weights:
                if weights[c] > self.max_position_pct:
                    weights[c] = self.max_position_pct
                if weights[c] < lo:
                    weights[c] = lo
            # 归一化
            total = sum(weights.values())
            if total > 0:
                target_sum = 1.0 - self.min_cash_reserve
                for c in weights:
                    weights[c] = weights[c] / total * target_sum
            # 收敛检查
            worst_sector = max(sum(weights[c] for c in codes) for codes in industry_groups.values())
            if worst_sector <= self.max_sector_pct + 0.005:
                break

        return weights

    def _classify_holdings(self, current_holdings, candidate_map, top_n_codes, total_assets):
        result = {"hold": [], "observe": [], "sell": [], "stop_loss": [], "reduce": []}

        risk_info = []
        for sym, p in current_holdings.items():
            mv = p.get("market_value", 0)
            cost = mv - p.get("unrealized_pnl", 0)
            denom = cost if cost > 0 else (mv if mv > 0 else 1)
            pnl_pct = p.get("unrealized_pnl", 0) / denom
            risk_info.append({"code": sym, "pnl_pct": pnl_pct})

        risk_actions = self._allocator.classify_risk(
            risk_info, use_intraday_low_high=False) if risk_info else {}

        for sym, p in current_holdings.items():
            mv = p.get("market_value", 0)
            cost = mv - p.get("unrealized_pnl", 0)
            denom = cost if cost > 0 else (mv if mv > 0 else 1)
            pnl_pct = p.get("unrealized_pnl", 0) / denom

            ra = risk_actions.get(sym)
            if ra and ra.action == "full":
                result["stop_loss"].append({
                    "code": sym, "name": "", "pnl_pct": round(pnl_pct, 4),
                    "current_weight": mv / total_assets if total_assets > 0 else 0,
                    "reason": ra.reason,
                })
                continue
            if ra and ra.action == "half":
                result["reduce"].append({
                    "code": sym, "name": "", "pnl_pct": round(pnl_pct, 4),
                    "current_weight": mv / total_assets if total_assets > 0 else 0,
                    "reason": ra.reason,
                })
                continue

            if sym not in candidate_map:
                result["sell"].append({
                    "code": sym, "name": "", "rank": None,
                    "current_weight": p.get("market_value", 0) / total_assets if total_assets > 0 else 0,
                    "reason": "不在本周 Top-10 推荐中",
                })
                continue

            rank = candidate_map[sym]["rank"]
            if rank <= self.top_n:
                result["hold"].append({
                    "code": sym, "name": candidate_map[sym]["name"],
                    "rank": rank, "qlib_score": candidate_map[sym]["qlib_score"],
                    "current_weight": p.get("market_value", 0) / total_assets if total_assets > 0 else 0,
                    "reason": f"排名 #{rank}, 仍在 Top-{self.top_n}",
                })
            else:
                result["observe"].append({
                    "code": sym, "name": candidate_map[sym]["name"],
                    "rank": rank, "qlib_score": candidate_map[sym]["qlib_score"],
                    "current_weight": p.get("market_value", 0) / total_assets if total_assets > 0 else 0,
                    "reason": f"排名 #{rank}, 在 Top-{self.top_n+1}~{self.top_k} 观察区",
                })

        return result

    def _detect_limit_codes(self, codes, names=None):
        if not codes:
            return set(), set()
        try:
            from core.data import Database
            if not Database.is_available():
                return set(), set()
            ph = ",".join(["%s"] * len(codes))
            rows = Database.fetchall(
                f"SELECT code, pct_chg FROM daily_quotes "
                f"WHERE code IN ({ph}) AND trade_date = (SELECT MAX(trade_date) FROM daily_quotes)",
                list(codes))
            names = names or {}
            for row in rows:
                row["name"] = names.get(row["code"], "")
            return classify_limit_codes(rows)
        except Exception:
            pass
        return set(), set()

    def _generate_orders(self, classification, target_weights, current_weights,
                         total_assets, account, prices, candidate_map,
                         effective_max_turnover=None,
                         limit_up_codes=None, limit_down_codes=None,
                         keep_observe=False, skipped_out=None):
        orders = []

        # 止损单 — 最高优先级
        for item in classification.get("stop_loss", []):
            code = item["code"]
            price = prices.get(code, 0)
            cur_w = current_weights.get(code, 0)
            if limit_down_codes and code in limit_down_codes:
                logger.info(f"[涨跌停] {code} 跌停封板, 止损单无法成交, 跳过")
                if skipped_out is not None:
                    skipped_out.append({"code": code, "name": item.get("name", ""),
                                        "reason": "跌停封板, 无法卖出"})
                continue
            orders.append({
                "code": code, "name": item.get("name", ""),
                "side": "sell", "action": "止损卖出",
                "current_weight": round(cur_w, 4),
                "target_weight": 0,
                "delta_weight": round(-cur_w, 4),
                "amount": round(cur_w * total_assets, 2),
                "reason": item["reason"],
                "urgency": "stop_loss",
            })

        # 卖出不在观察区的持仓
        for item in classification.get("sell", []):
            code = item["code"]
            cur_w = current_weights.get(code, 0)
            if cur_w <= 0.001:
                continue
            if limit_down_codes and code in limit_down_codes:
                logger.info(f"[涨跌停] {code} 跌停封板, 跳过卖出")
                if skipped_out is not None:
                    skipped_out.append({"code": code, "name": item.get("name", ""),
                                        "reason": "跌停封板, 无法卖出"})
                continue
            orders.append({
                "code": code, "name": item.get("name", ""),
                "side": "sell", "action": "建议卖出",
                "current_weight": round(cur_w, 4),
                "target_weight": 0,
                "delta_weight": round(-cur_w, 4),
                "amount": round(cur_w * total_assets, 2),
                "reason": item["reason"],
                "urgency": "normal",
            })

        for item in classification.get("reduce", []):
            code = item["code"]
            cur_w = current_weights.get(code, 0)
            if cur_w <= 0.001:
                continue
            if limit_down_codes and code in limit_down_codes:
                logger.info(f"[涨跌停] {code} 跌停封板, 减半单无法成交, 跳过")
                if skipped_out is not None:
                    skipped_out.append({"code": code, "name": item.get("name", ""),
                                        "reason": "跌停封板, 无法减半"})
                continue
            orders.append({
                "code": code, "name": item.get("name", ""),
                "side": "reduce", "action": "止盈/损减半",
                "current_weight": round(cur_w, 4),
                "target_weight": round(cur_w / 2, 4),
                "delta_weight": round(-cur_w / 2, 4),
                "amount": round(cur_w / 2 * total_assets, 2),
                "reason": item["reason"],
                "urgency": "stop_loss",
            })

        # 观察区持仓 → 降权或清仓 (目标权重为0时全额卖出); keep_observe=True 时保留不清仓
        for item in classification.get("observe", []):
            if keep_observe:
                continue
            code = item["code"]
            cur_w = current_weights.get(code, 0)
            tgt_w = target_weights.get(code, 0)
            half_target = tgt_w * 0.5
            if abs(cur_w - half_target) < self.tolerance_band:
                continue
            if tgt_w <= 0:
                # 不在目标组合中 → 全额清仓
                if limit_down_codes and code in limit_down_codes:
                    logger.info(f"[涨跌停] {code} 跌停封板, 观察区清仓无法成交, 跳过")
                    continue
                orders.append({
                    "code": code, "name": item.get("name", ""),
                    "side": "sell", "action": "观察区清仓",
                    "current_weight": round(cur_w, 4),
                    "target_weight": 0,
                    "delta_weight": round(-cur_w, 4),
                    "amount": round(cur_w * total_assets, 2),
                    "reason": f"排名 #{item['rank']}, 在观察区, 未进入Top-{self.top_n}, 建议清仓",
                    "urgency": "normal",
                })
            elif cur_w > half_target:
                orders.append({
                    "code": code, "name": item.get("name", ""),
                    "side": "reduce", "action": "降权观察",
                    "current_weight": round(cur_w, 4),
                    "target_weight": round(half_target, 4),
                    "delta_weight": round(half_target - cur_w, 4),
                    "amount": round((cur_w - half_target) * total_assets, 2),
                    "reason": f"排名 #{item['rank']}, 在观察区, 降权至目标一半",
                    "urgency": "normal",
                })

        # Top-N 持仓 → 调整到目标权重
        for item in classification.get("hold", []):
            code = item["code"]
            cur_w = current_weights.get(code, 0)
            tgt_w = target_weights.get(code, 0)
            if abs(cur_w - tgt_w) <= self.tolerance_band:
                continue
            if cur_w > tgt_w:
                orders.append({
                    "code": code, "name": item.get("name", ""),
                    "side": "reduce", "action": "减仓调整",
                    "current_weight": round(cur_w, 4),
                    "target_weight": round(tgt_w, 4),
                    "delta_weight": round(tgt_w - cur_w, 4),
                    "amount": round((cur_w - tgt_w) * total_assets, 2),
                    "reason": f"超配 (当前 {cur_w:.1%} > 目标 {tgt_w:.1%})",
                    "urgency": "normal",
                })
            else:
                orders.append({
                    "code": code, "name": item.get("name", ""),
                    "side": "add", "action": "加仓调整",
                    "current_weight": round(cur_w, 4),
                    "target_weight": round(tgt_w, 4),
                    "delta_weight": round(tgt_w - cur_w, 4),
                    "amount": round((tgt_w - cur_w) * total_assets, 2),
                    "reason": f"低配 (当前 {cur_w:.1%} < 目标 {tgt_w:.1%})",
                    "urgency": "normal",
                })

        # Top-N 中未持仓的 → 建议买入 (冷冻期内跳过)
        held_codes = {o["code"] for o in orders}
        for code, tgt_w in target_weights.items():
            if code in held_codes:
                continue
            if code in current_weights and current_weights[code] > 0.001:
                if skipped_out is not None:
                    skipped_out.append({
                        "code": code, "name": candidate_map.get(code, {}).get("name", ""),
                        "reason": f"已持有, 权重达标, 无需加仓",
                    })
                continue
            if code in self._cooldown and self._cooldown[code] > 0:
                if skipped_out is not None:
                    skipped_out.append({
                        "code": code, "name": candidate_map.get(code, {}).get("name", ""),
                        "reason": "止盈止损冷冻期, 暂不买入",
                    })
                continue
            if limit_up_codes and code in limit_up_codes:
                logger.info(f"[涨跌停] {code} 涨停封板, 跳过买入")
                if skipped_out is not None:
                    skipped_out.append({
                        "code": code, "name": candidate_map.get(code, {}).get("name", ""),
                        "reason": "涨停封板, 无法买入",
                    })
                continue
            _rank = candidate_map.get(code, {}).get("rank", "")
            orders.append({
                "code": code, "name": candidate_map.get(code, {}).get("name", ""),
                "side": "buy", "action": "建议买入",
                "current_weight": 0,
                "target_weight": round(tgt_w, 4),
                "delta_weight": round(tgt_w, 4),
                "amount": round(tgt_w * total_assets, 2),
                "reason": f"排名 #{_rank}, 新进入 Top-{self.top_n}, Qlib分 {candidate_map.get(code, {}).get('qlib_score', 0):.4f}",
                "urgency": "normal",
            })

        # 卖出股票进入冷冻期 (止损/止盈分开, 对齐回测)
        for o in orders:
            if o["side"] == "sell":
                if "止损" in o.get("reason", ""):
                    self._cooldown[o["code"]] = self._allocator.config.cooldown_stop_periods
                else:
                    self._cooldown[o["code"]] = self._allocator.config.cooldown_profit_periods
        # 持久化冷冻期 (跨 advisor 实例生效, 否则每次新建实例后止损/止盈的股票下期立即被重新买入)
        self._persist_cooldown()

        # 排序: 止损→卖→减→加→买
        priority = {"stop_loss": 0, "sell": 1, "reduce": 2, "add": 3, "buy": 4}
        orders.sort(key=lambda o: (priority.get(o.get("urgency", "normal"), 5), o.get("side", "")))

        # ── 换手率上限裁剪: 空仓首次建仓豁免，其余超限时裁剪低优先级调整单 ──
        if effective_max_turnover and orders and current_weights:
            _buy_turnover = sum(
                max(0.0, float(o["delta_weight"])) for o in orders
            )
            _sell_turnover = sum(
                max(0.0, -float(o["delta_weight"])) for o in orders
            )
            _cur_turnover = max(_buy_turnover, _sell_turnover)
            _max_to = effective_max_turnover
            if _cur_turnover > _max_to:
                _must = [o for o in orders if o.get("urgency") == "stop_loss"]
                _optional = [o for o in orders if o.get("urgency") != "stop_loss"]
                _optional.sort(key=lambda o: abs(o["delta_weight"]), reverse=True)
                _must_buy = sum(max(0.0, float(o["delta_weight"])) for o in _must)
                _must_sell = sum(max(0.0, -float(o["delta_weight"])) for o in _must)
                _buy_budget = max(0.0, _max_to - _must_buy)
                _sell_budget = max(0.0, _max_to - _must_sell)
                _kept = []
                _deferred = []
                _used_buy = 0.0
                _used_sell = 0.0
                for o in _optional:
                    _signed_delta = float(o["delta_weight"])
                    _delta = abs(_signed_delta)
                    if _signed_delta >= 0:
                        _fits = _used_buy + _delta <= _buy_budget + 1e-12
                    else:
                        _fits = _used_sell + _delta <= _sell_budget + 1e-12
                    if _fits:
                        _kept.append(o)
                        if _signed_delta >= 0:
                            _used_buy += _delta
                        else:
                            _used_sell += _delta
                    else:
                        _deferred.append(o)
                if skipped_out is not None:
                    for o in _deferred:
                        skipped_out.append({
                            "code": o["code"],
                            "name": o.get("name", ""),
                            "reason": f"换手预算不足（单边上限{_max_to:.0%}），本轮延后",
                        })
                _dropped = len(_deferred)
                if _dropped > 0:
                    _kept_turnover = max(
                        _must_buy + _used_buy,
                        _must_sell + _used_sell,
                    )
                    logger.info(f"[换手惩罚] 裁剪 {_dropped} 条调整单 "
                                f"(换手 {_cur_turnover:.1%}→{_kept_turnover:.1%}, 上限 {_max_to:.0%})")
                orders = _must + _kept
                orders.sort(key=lambda o: (priority.get(o.get("urgency", "normal"), 5), o.get("side", "")))

        return orders

    def _compute_turnover(self, target_weights, current_weights):
        return compute_turnover(target_weights, current_weights)

    def _check_constraints(self, turnover, target_weights, top_candidates,
                           effective_max_turnover=None, rank_corr=None,
                           initial_build=False):
        results = []

        _max_to = effective_max_turnover or self.max_turnover

        # 换手率
        ok = initial_build or turnover <= _max_to
        if initial_build:
            _note = "空仓首次建仓，不适用调仓换手上限"
        else:
            _note = "" if ok else f"超出上限, 可考虑减少调仓或分周执行"
        if rank_corr is not None:
            _note += f" (预测相关度={rank_corr:.3f})"
        results.append({
            "name": "换手率",
            "value": f"{turnover:.1%}",
            "limit": f"{_max_to:.0%}",
            "pass": ok,
            "note": _note,
        })

        # 行业集中度
        industry_map = _get_industry_map()
        if industry_map:
            sector_weights = {}
            for c in top_candidates:
                code = c.get("code", "")
                ind_info = industry_map.get(code, {})
                ind = ind_info.get("industry", "未知")
                w = target_weights.get(code, 0)
                sector_weights[ind] = sector_weights.get(ind, 0) + w
            max_sector = max(sector_weights.items(), key=lambda x: x[1]) if sector_weights else ("未知", 0)
            sector_ok = max_sector[1] <= self.max_sector_pct
            results.append({
                "name": "最大行业集中",
                "value": f"{max_sector[0]} {max_sector[1]:.1%}",
                "limit": f"{self.max_sector_pct:.0%}",
                "pass": sector_ok,
                "note": "" if sector_ok else f"⚠ 行业过于集中，考虑分散",
            })
            results.append({
                "name": "行业分布",
                "value": ", ".join(f"{k} {v:.0%}" for k, v in
                           sorted(sector_weights.items(), key=lambda x: -x[1])),
                "limit": "-",
                "pass": True,
                "note": "",
            })

        # 单只最大仓位
        max_pos = max(target_weights.values()) if target_weights else 0
        pos_ok = max_pos <= self.max_position_pct
        results.append({
            "name": "单只最大仓位",
            "value": f"{max_pos:.1%}",
            "limit": f"{self.max_position_pct:.0%}",
            "pass": pos_ok,
            "note": "" if pos_ok else "单只仓位超限",
        })

        # 现金保留
        allocated = sum(target_weights.values())
        cash = 1.0 - allocated
        cash_ok = cash >= self.min_cash_reserve - 0.02
        results.append({
            "name": "现金保留",
            "value": f"{cash:.1%}",
            "limit": f"{self.min_cash_reserve:.0%}",
            "pass": cash_ok,
            "note": "" if cash_ok else "现金不足，预留流动性",
        })

        return results

    def _gather_auxiliary(self, top_candidates):
        from concurrent.futures import as_completed
        from core.shared.infrastructure.thread_pool import thread_pool
        from core.data.fetcher import DataFetcher

        def _fetch_one(code):
            fetcher = DataFetcher()
            info = {"code": code, "fundamental": "-", "flow": "-", "north": "-", "sentiment": "-"}
            try:
                fin = fetcher.get_stock_financial(code)
                if fin is not None and not fin.empty:
                    pe = fin.iloc[-1].get("市盈率-动态", 0)
                    if pe and pe > 0:
                        info["fundamental"] = f"PE {pe:.0f}"
            except Exception:
                pass
            try:
                cap = fetcher.get_capital_flow(code)
                if cap is not None and not cap.empty:
                    latest = cap.iloc[-1]
                    net = latest.get("主力净流入", 0)
                    info["flow"] = "流入" if net > 0 else "流出" if net < 0 else "持平"
            except Exception:
                pass
            return code, info

        auxiliary = {}
        with thread_pool.default.borrow() as pool:
            futures = {pool.submit(_fetch_one, c.get("code", "")): c for c in top_candidates}
            for f in as_completed(futures):
                try:
                    code, info = f.result()
                    auxiliary[code] = info
                except Exception:
                    pass
        return auxiliary

    def _lookup_names(self, codes):
        """批量补全股票名称"""
        if not codes:
            return {}
        from core.data.fetcher import DataFetcher
        fetcher = DataFetcher()
        names = {}
        for code in codes:
            try:
                name = fetcher.get_stock_name(code)
                if name:
                    names[code] = name
            except Exception:
                pass
        return names

    def _build_combo_table(self, top_candidates, target_weights, current_weights, current_holdings):
        return build_combo_table(top_candidates, target_weights, current_weights, current_holdings, self.tolerance_band)

    def format_text_report(self, report):
        lines = []
        lines.append("═══════════════════════════════════════════")
        lines.append("  Apallg投研 周频调仓建议")
        lines.append(f"  {report['date']} (周一)  |  模型: LGB+XGB 双模型")
        lines.append(f"  股票池: {report['pool']}  |  板块: {report.get('sectors') or '不限'}  |  持仓数: {report['top_n']}")
        lines.append("═══════════════════════════════════════════")
        lines.append("")

        # 账户摘要
        acc = report.get("account", {})
        lines.append(f"💰 账户: 总资产 ¥{acc.get('total_assets', 0):,.0f}  |  总现金 ¥{acc.get('total_cash', acc.get('cash', 0)):,.0f}  |  可用 ¥{acc.get('available_cash', acc.get('cash', 0)):,.0f}  |  冻结 ¥{acc.get('frozen_cash', 0):,.0f}  |  持仓市值 ¥{acc.get('market_value', 0):,.0f}")
        lines.append("")

        # 目标组合
        lines.append(f"📊 目标组合 (Top-{report['top_n']})")
        lines.append(f"{'排名':<4} {'代码':<8} {'名称':<10} {'预测分':<8} {'操作':<8} {'目标仓位':<10} {'当前':<10}")
        lines.append("-" * 60)
        for r in report.get("target_combo", []):
            cur_str = f"{r['current_weight']:.0%}" if r['current_weight'] > 0 else "-"
            lines.append(
                f"{r['rank']:<4} {r['code']:<8} {r['name']:<10} {r['qlib_score']:<8.4f} "
                f"{r['action']:<8} {r['target_weight']:.0%}{'':<7} {cur_str:<10}"
            )
        lines.append("")

        # 行业分布
        for ck in report.get("constraints", []):
            if ck["name"] == "行业分布":
                lines.append(f"📊 {ck['value']}")
        lines.append("")

        # 买入
        buys = report.get("to_buy", [])
        if buys:
            lines.append(f"📈 建议买入 ({len(buys)}只)")
            for o in buys:
                lines.append(f"  {o['code']} {o['name']:<8} → ¥{o['amount']:,.0f} ({o['target_weight']:.0%}), {o['reason']}")
            lines.append("")

        # 卖出
        sells = report.get("to_sell", [])
        if sells:
            lines.append(f"📉 建议卖出 ({len(sells)}只)")
            for o in sells:
                lines.append(f"  {o['code']} {o['name']:<8} → 现持 ¥{o['amount']:,.0f}, {o['reason']}")
            lines.append("")

        # 止损
        stops = report.get("stop_loss", [])
        if stops:
            lines.append(f"⚠️ 止损提醒 ({len(stops)}只)")
            for o in stops:
                lines.append(f"  {o['code']} {o['name']:<8} → {o['reason']}")
            lines.append("")
        else:
            lines.append("⚠️ 止损提醒: (无触发)")
            lines.append("")

        # 调整
        adjusts = report.get("to_adjust", [])
        if adjusts:
            lines.append(f"🔄 建议调整 ({len(adjusts)}只)")
            for o in adjusts:
                lines.append(f"  {o['code']} {o['name']:<8} → {o['current_weight']:.0%}→{o['target_weight']:.0%}, {o['reason']}")
            lines.append("")

        # 约束检查
        lines.append("📋 约束检查")
        for ck in report.get("constraints", []):
            if ck["name"] == "行业分布":
                continue
            status = "✓" if ck["pass"] else "⚠"
            note = f" ({ck['note']})" if ck["note"] else ""
            lines.append(f"  {ck['name']}: {ck['value']} (上限 {ck['limit']}) {status}{note}")
        lines.append("")

        # 辅助参考
        aux = report.get("auxiliary", {})
        if aux:
            lines.append("📋 辅助参考 (仅展示, 不影响排序)")
            lines.append(f"{'代码':<8} {'基本面':<12} {'资金流':<8} {'北向':<8} {'情绪':<8}")
            lines.append("-" * 48)
            for code, info in aux.items():
                lines.append(f"{code:<8} {info['fundamental']:<12} {info['flow']:<8} {info.get('north', '-'):<8} {info.get('sentiment', '-'):<8}")
            lines.append("")

        lines.append("═══════════════════════════════════════════")
        lines.append(f"  调仓依据: Qlib 双模型融合预测未来 20 日收益排名")
        lines.append(f"  下次调仓: {report.get('next_rebalance', '')}")
        lines.append("═══════════════════════════════════════════")

        return "\n".join(lines)
