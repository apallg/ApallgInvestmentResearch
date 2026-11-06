"""Qlib 集成 API — Pydantic 请求模型"""
from typing import Literal, Optional
from pydantic import BaseModel

class SyncRequest(BaseModel):
    full_resync: bool = False
class TrainRequest(BaseModel):
    model_name: str = "lightgbm"
    factor_set: str = "Alpha158"
    custom_factor_keys: Optional[list] = None
    use_factor_pool: bool = False
    instruments: str = "csi300"
    train_start: str = "2019-01-01"
    train_end: str = "2024-12-31"
    valid_start: str = "2025-01-01"
    valid_end: str = "2025-06-30"
    test_start: Optional[str] = None
    test_end: Optional[str] = None
    model_params: dict = {}
    label_horizon: int = 10
    label_mode: Literal["tradable_open_to_open"] = "tradable_open_to_open"
    freq: str = "day"
    auto_backtest: bool = True
    topk: int = 50

    @staticmethod
    def _fix_date(v, default):
        """空字符串返回 None (允许不设测试期), 非法日期用默认值"""
        import re
        if not v or not v.strip():
            return None
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', v.strip()):
            return default
        return v.strip()

    def __init__(self, **data):
        import re
        for field, default in [("train_start", "2019-01-01"), ("train_end", "2024-12-31"),
                                ("valid_start", "2025-01-01"), ("valid_end", "2025-06-30"),
                                ("test_start", "2025-07-01"), ("test_end", "2026-06-30")]:
            if field in data:
                v = self._fix_date(data[field], default)
                if v is not None:
                    data[field] = v
                else:
                    data[field] = None if field.startswith("test_") else default
        super().__init__(**data)
    n_drop: int = 5
    code_prefixes: Optional[list] = None  # 股票代码前缀过滤，如 ['600', '000', '300']
    train_dual: bool = False  # 双模型训练 (LightGBM + XGBoost 按IC加权融合)
    train_lgb_lstm: bool = False  # LGB + LSTM 分数层融合 (0.5/0.5 秩融合)
    training_mode: Literal["full", "state_subset"] = "full"
    expert_role: Optional[Literal["trend", "rebound"]] = None
    use_curated: Optional[str] = None   # 精选因子集名称, "latest"=最新版本, None=不使用
    enable_crossover: bool = False  # 开启因子交叉特征 (两两乘积)
    enable_cross_sectional: bool = False  # 开启截面百分位特征增强 (CSFeatureAugment, 会增加噪声)
    enable_winsorize: bool = True  # 逐日截面 winsorize 5%/95% 抗极值 (价格/量价因子)
    label_norm: str = "csrank"  # label 截面标准化: csrank(截面排名,抗厚尾) | cszscore(z-score) | none(保留幅度)
    early_stop_metric: str = "mse"  # 早停指标: mse(推荐,回归损失) | rank_ic(排序相关)
    param_profile: str = "aggressive"  # 超参 profile: default | conservative(稳定池强正则) | aggressive
    use_reweighter: bool = True  # 训练样本流动性加权 + 次新剔除
    with_linear: bool = False  # 追加 ridge 线性基线为第三模型 (份额上限20%)
    with_catboost: bool = False  # 追加 CatBoost 为第三模型
    catboost_params: dict = {}   # 覆盖 CatBoost 默认参数
    lgb_params: dict = {}        # 覆盖 LightGBM 参数 (Optuna 搜索结果等)
    xgb_params: dict = {}        # 覆盖 XGBoost 参数
class PredictRequest(BaseModel):
    profile_id: Optional[str] = None
    profile_version: Optional[int] = None
    experiment_id: str = ""
    run_id: str = ""
    instruments: str = "csi300"
    start_time: str = "2024-01-01"
    end_time: str = "2025-01-01"
    topk: int = 30
    skip_features: bool = False  # 跳过特征重要性计算，加速预测
    code_prefixes: Optional[list] = None  # 股票代码前缀过滤，如 ['600', '000', '300']
class BacktestRequest(BaseModel):
    predictions_file: str = ""
    topk: int = 50
    n_drop: int = 5
    start_time: str = "2023-01-01"
    end_time: str = "2024-12-31"
class CollectRequest(BaseModel):
    pool: str = "all"       # csi300, csi500, csi_all, all
    retry_missing: bool = False  # 补采数据库中 < 100 条记录的股票（仅 Tencent）
    force_tencent: bool = False  # 跳过 BaoStock，直接 Tencent


# ── Helpers ──
class OptunaSearchRequest(BaseModel):
    model_name: str = "lightgbm"   # lightgbm | xgboost
    search_dual: bool = False     # True=依次搜LGB+XGB后双模型融合重训
    factor_set: str = "Alpha158"
    custom_factor_keys: Optional[list] = None
    use_curated: Optional[str] = None
    instruments: str = "csi300"
    train_start: str = "2019-01-01"
    train_end: str = "2024-12-31"
    valid_start: str = "2025-01-01"
    valid_end: str = "2025-06-30"
    test_start: Optional[str] = None
    test_end: Optional[str] = None
    label_horizon: int = 5
    label_mode: Literal["tradable_open_to_open"] = "tradable_open_to_open"
    label_norm: str = "csrank"
    n_trials: int = 50
    timeout: int = 0   # 超时秒数, 0=不限
    param_profile: str = "default"  # 超参方案: default | conservative | aggressive
class RollingTrainRequest(BaseModel):
    pool: str = "csi300"
    start_anchor: str = "2023-01-01"
    end_anchor: str = ""            # 空=今天
    step_months: int = 3
    train_years: int = 5
    valid_months: int = 6           # 固定验证集长度, 保证早停阈值稳定
    model_name: str = "lightgbm"  # 单模型时使用 (train_dual=False)
    train_dual: bool = False        # 双模型 LGB+XGB; False=仅 model_name 指定模型
    label_horizon: int = 10
    label_mode: Literal["tradable_open_to_open"] = "tradable_open_to_open"
    use_factor_pool: bool = False
    use_curated: Optional[str] = None
    code_prefixes: Optional[list] = None  # 股票代码前缀过滤
    label_norm: str = "csrank"
    early_stop_metric: str = "mse"
    param_profile: str = "aggressive"
    enable_cross_sectional: bool = False
    enable_winsorize: bool = True
    enable_crossover: bool = False
    use_reweighter: bool = True
    with_linear: bool = False
    with_catboost: bool = False
    filter_mode: bool = False       # 嵌入式二次筛选: 追踪特征重要性 → 淘汰低效因子 → 写回因子池
class ModelToggleRequest(BaseModel):
    experiment_id: str
class EnsembleTrainRequest(BaseModel):
    model_names: list = ["lightgbm", "xgboost"]
    ensemble_strategy: str = "ic"       # equal | ic
    factor_set: str = "Alpha158"
    custom_factor_keys: Optional[list] = None
    use_factor_pool: bool = False
    instruments: str = "csi300"
    train_start: str = "2019-01-01"
    train_end: str = "2024-12-31"
    valid_start: str = "2025-01-01"
    valid_end: str = "2025-06-30"
    test_start: Optional[str] = None
    test_end: Optional[str] = None
    model_params: dict = {}
    label_horizon: int = 10
    label_mode: Literal["tradable_open_to_open"] = "tradable_open_to_open"
    freq: str = "day"
    auto_backtest: bool = True
    topk: int = 50
    n_drop: int = 5
    code_prefixes: Optional[list] = None
    label_norm: str = "none"
    early_stop_metric: str = "mse"
    param_profile: str = "aggressive"
    enable_cross_sectional: bool = False
    enable_winsorize: bool = True
    enable_crossover: bool = False
    use_reweighter: bool = True
    with_linear: bool = False
    with_catboost: bool = False
    use_curated: Optional[str] = None
