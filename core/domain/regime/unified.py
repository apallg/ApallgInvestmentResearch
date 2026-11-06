"""
统一市场状态入口 — 所有页面/回测/实盘共用同一套判定代码。

主模型: RegimePredictor (监督学习 6 状态)
回退:   RuleBasedRegimeClassifier (确定性规则 6 状态)
宏观:   MacroAnalyzer.adjust_regime (可选, 前端可调)
平滑:   实时路径 CRF 因果平滑 (可选, 只影响实时, 回测不平滑以免前视)
"""
import logging
import threading

import numpy as np
import pandas as pd

from core.shared.persistence import today_date
from core.domain.regime.types import Regime
from core.domain.regime.classifier import RegimeResult

logger = logging.getLogger(__name__)

# 6 状态顺序 (与 predictor.STATE_NAMES 对齐)
STATE_NAMES = ["volatile_up", "volatile_down", "volatile_flat",
               "steady_up", "steady_down", "steady_flat"]

_CRF_WINDOW = 20      # 观测窗口 (交易日)
_CRF_SELF_PROB = 0.9  # 自转移概率 (对应 ~10 日持续)

# 转移矩阵 (对角占优先验): 学得矩阵不可用时回退
_PRIOR_TRANS = np.full((len(STATE_NAMES), len(STATE_NAMES)),
                       (1.0 - _CRF_SELF_PROB) / (len(STATE_NAMES) - 1))
np.fill_diagonal(_PRIOR_TRANS, _CRF_SELF_PROB)

# 实时路径 CRF 平滑状态 (进程内, 与 get_current_regime 缓存一致)
_prior_probs = []  # 最近 N 个交易日的原始概率序列 (6 维)
_last_date = None  # 上次 append 的日期, 用于同日去重
_crf_lock = threading.Lock()

# 学得转移矩阵缓存 (从 RegimePredictor 读, 读不到回退先验)
_trans = None
_trans_loaded = False
_trans_lock = threading.Lock()


def _get_transition_matrix():
    """优先用学得的转移矩阵 (RegimePredictor 训练时统计), 回退先验对角占优"""
    global _trans, _trans_loaded
    with _trans_lock:
        if _trans_loaded:
            return _trans
        _trans = _PRIOR_TRANS
        try:
            from core.domain.regime.predictor import get_regime_predictor
            rp = get_regime_predictor()
            m = getattr(rp, "transition_matrix", None) if rp is not None else None
            if m is not None and np.asarray(m).shape == (len(STATE_NAMES), len(STATE_NAMES)):
                _trans = np.asarray(m, dtype=float)
        except Exception as e:
            logger.warning(f"[Regime] 学得转移矩阵加载失败, 回退先验: {e}")
        _trans_loaded = True
        return _trans


def _predictor_probs(as_of, index_df, breadth_df):
    """RegimePredictor 主模型 → 6 状态概率 dict, 失败返回 None"""
    from core.domain.regime.predictor import get_regime_predictor, STATE_NAMES
    try:
        rp = get_regime_predictor()
        if rp is None or not rp._is_loaded():
            return None

        if as_of is None:
            pred = rp.predict_regime()
            probs = pred.get("probs") if pred else None
            return probs or None

        # 历史: point-in-time, 需要 index_df (缺省时拉数据)
        df = _ensure_index_df(index_df)
        if df is None or df.empty:
            return None
        df = df.copy()
        if "trade_date" in df.columns and "date" not in df.columns:
            df = df.rename(columns={"trade_date": "date"})
        df = df.sort_values("date").reset_index(drop=True)
        df = df[df["date"] <= pd.Timestamp(as_of)]
        if len(df) < 60:
            return None
        arr = rp.predict_from_df(df, breadth_df)
        if arr is None or len(arr) < 6:
            return None
        return {STATE_NAMES[i]: round(float(arr[i]), 4) for i in range(6)}
    except Exception as e:
        logger.warning(f"[Regime] 主模型预测失败: {e}")
        return None


def _ensure_index_df(index_df):
    """index_df 为空时拉取沪深300历史数据 (实时回退用)"""
    if index_df is not None and not index_df.empty:
        return index_df
    from core.data.fetcher import DataFetcher
    from core.shared.config import REGIME_DEFAULT_INDEX
    return DataFetcher().get_index_daily(REGIME_DEFAULT_INDEX, days=365 * 2)


def _rule_based_regime(as_of, index_df):
    """确定性规则回退 → RegimeResult (6 状态)"""
    from core.domain.regime.classifier import RuleBasedRegimeClassifier
    try:
        df = _ensure_index_df(index_df)
        rc = RuleBasedRegimeClassifier(df)
        return rc.classify(as_of=as_of)
    except Exception as e:
        logger.warning(f"[Regime] 规则回退失败: {e}")
        return RegimeResult(
            regime=Regime.STEADY_FLAT, confidence=0.0,
            sub_scores={s: 0.0 for s in STATE_NAMES},
            details={"error": f"规则回退失败: {e}"},
            as_of_date=str(as_of or today_date()),
        )


def _forward_filter(obs, trans):
    """HMM forward filtering — 因果 CRF 平滑, 输出每步滤波后验 alpha (T×6)"""
    T = obs.shape[0]
    alpha = np.zeros_like(obs)
    alpha[0] = obs[0] / (obs[0].sum() or 1.0)
    for t in range(1, T):
        alpha[t] = obs[t] * (trans.T @ alpha[t - 1])
        s = alpha[t].sum()
        if s > 0:
            alpha[t] /= s
    return alpha


def _crf_smooth(probs):
    """CRF 因果平滑: 对概率序列做 HMM forward filtering, 抑制 argmax 日间跳变

    按交易日累积观测 (同日去重), 窗口 _CRF_WINDOW 个交易日。
    """
    global _prior_probs, _last_date
    arr = np.array([probs.get(s, 0.0) for s in STATE_NAMES], dtype=float)
    today = today_date()
    with _crf_lock:
        if _last_date is not None and _last_date == today and _prior_probs:
            # 同日: 替换最后观测 (模型输出随数据刷新, 保留最新)
            _prior_probs[-1] = arr
        else:
            _prior_probs.append(arr)
            if len(_prior_probs) > _CRF_WINDOW:
                _prior_probs = _prior_probs[-_CRF_WINDOW:]
        _last_date = today
        seq = np.array(_prior_probs)
    alpha = _forward_filter(seq, _get_transition_matrix())
    smoothed = alpha[-1]
    return {STATE_NAMES[i]: round(float(smoothed[i]), 4) for i in range(6)}


def crf_smooth_sequence(obs_list, trans):
    """对概率序列 (按时间顺序) 做因果 forward filter, 返回平滑后的最后一步概率 dict。

    供回测复用: 回测用实例级窗口 + 调仓间隔调整后的转移矩阵, 与实盘语义一致。
    obs_list: list[array], 每个元素是 6 维概率
    """
    seq = np.array(obs_list)
    if seq.ndim != 2 or seq.shape[1] != len(STATE_NAMES):
        return None
    alpha = _forward_filter(seq, trans)
    smoothed = alpha[-1]
    return {STATE_NAMES[i]: round(float(smoothed[i]), 4) for i in range(6)}


def apply_macro_regime_correction(rr, as_of):
    """状态层面宏观修正: 调置信度 + 写 details["macro"] (不改状态 label)"""
    try:
        from core.domain.macro.analyzer import MacroAnalyzer
        from core.domain.macro.data import MacroDataAccess
        snap = MacroDataAccess.get_macro_snapshot(as_of=as_of)
        if snap is not None:
            rr = MacroAnalyzer().adjust_regime(rr, macro_snapshot=snap)
    except Exception as e:
        logger.warning(f"[Regime] 宏观修正失败: {e}")
    return rr


def _vol_switch_direction(as_of, index_df, vol_label_horizon=10):
    """波动率切换 → 方向模型 (高波 oversold_rebound, 低波 trend_momentum)。

    用 VolRegimeClassifier 预测未来波动率 (93% 可靠), 替代 6 状态方向映射。
    返回 (vol_direction, p_vol), 失败返回 (None, None)。
    """
    from core.domain.regime.vol_predictor import get_vol_classifier
    from core.domain.regime.weights import get_vol_direction
    df = _ensure_index_df(index_df)
    if df is None or df.empty:
        return None, None
    df = df.copy()
    if "trade_date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"trade_date": "date"})
    df = df.sort_values("date").reset_index(drop=True)
    if as_of is not None:
        df = df[df["date"] <= pd.Timestamp(as_of)]
    if df.empty:
        return None, None
    p_vol = get_vol_classifier(vol_label_horizon).predict_proba(df)
    if p_vol is None:
        return None, None
    return get_vol_direction(p_vol), p_vol


def get_unified_regime(as_of=None, *, index_df=None, breadth_df=None,
                       apply_macro=True, smooth=True, use_vol_switch=False,
                       vol_label_horizon=10):
    """统一市场状态入口。

    Args:
        as_of: None=实时(最新), 传 date/str=历史 point-in-time (回测/选股历史)
        index_df: 预加载的指数日线 (回测复用, 避免每调仓日重拉)
        breadth_df: 可选市场宽度
        apply_macro: 状态层面宏观修正开关 (前端可调)
        smooth: 实时路径 CRF 序列平滑 (仅 as_of=None 生效)
    """
    probs = _predictor_probs(as_of, index_df, breadth_df)
    method = "predictor"

    if probs is not None:
        if as_of is None and smooth:
            probs = _crf_smooth(probs)
        top_state = max(probs, key=probs.get)
        rr = RegimeResult(
            regime=Regime(top_state),
            confidence=round(float(probs[top_state]), 4),
            sub_scores=probs,
            details={"method": method},
            as_of_date=str(as_of or today_date()),
        )
    else:
        rr = _rule_based_regime(as_of, index_df)
        method = "rule_based"
        if not isinstance(rr.details, dict):
            rr.details = {}
        rr.details["method"] = method

    if apply_macro:
        rr = apply_macro_regime_correction(rr, as_of)

    if use_vol_switch:
        vol_dir, p_vol = _vol_switch_direction(as_of, index_df, vol_label_horizon)
        if not isinstance(rr.details, dict):
            rr.details = {}
        rr.details["vol_direction"] = vol_dir
        rr.details["p_vol"] = round(float(p_vol), 4) if p_vol is not None else None

    return rr
