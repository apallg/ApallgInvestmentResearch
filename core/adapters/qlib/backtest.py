"""
qlib 回测引擎封装 — SimulatorExecutor + TopkDropoutStrategy
"""
import logging
import pandas as pd

from core.adapters.qlib.config import init_qlib

logger = logging.getLogger(__name__)

# A 股默认交易参数
DEFAULT_EXCHANGE_KWARGS = {
    "freq": "day",
    "limit_threshold": 0.095,
    "deal_price": "open",  # 次日开盘成交，避免收盘价乐观偏差
    "open_cost": 0.0005,
    "close_cost": 0.0015,
    "min_cost": 5,
}

DEFAULT_BACKTEST_CONFIG = {
    "start_time": None,
    "end_time": None,
    "account": 100000000,
    "benchmark": "sh000300",  # 沪深300指数基准（全量同步时自动生成）
}


def build_port_analysis_config(signal, topk=50, n_drop=5, start_time=None, end_time=None, freq="day", **kwargs):
    """
    构造 PortAnaRecord 所需配置

    Parameters
    ----------
    signal : tuple — (model, dataset) 或 pd.Series/DataFrame
    topk : int — 每日持仓股票数
    n_drop : int — 每日卖出最差 N 只
    start_time, end_time : str — 回测区间
    """
    backtest_cfg = {**DEFAULT_BACKTEST_CONFIG, "start_time": start_time, "end_time": end_time}
    backtest_cfg.update({k: v for k, v in kwargs.items() if k in DEFAULT_BACKTEST_CONFIG})

    return {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": freq, "generate_portfolio_metrics": True},
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": signal,
                "topk": topk,
                "n_drop": n_drop,
            },
        },
        "backtest": {
            **backtest_cfg,
            "exchange_kwargs": DEFAULT_EXCHANGE_KWARGS,
        },
    }


class QlibBacktestRunner:
    def __init__(self):
        init_qlib()

    def run(self, model, dataset, topk=50, n_drop=5, start_time=None, end_time=None, freq="day", experiment_name="backtest"):
        """
        运行一次完整的回测（预测 → 信号分析 → 投资组合回测）

        Parameters
        ----------
        model : 已训练的 qlib 模型
        dataset : qlib DatasetH 实例
        topk, n_drop : 策略参数
        start_time, end_time : 回测区间

        Returns
        -------
        dict with: port_metrics, report, indicator_values
        """
        from qlib.workflow import R
        from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord

        with R.start(experiment_name=experiment_name):
            recorder = R.get_recorder()

            # 生成预测信号
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()

            # 信号分析 (IC/IR)
            sar = SigAnaRecord(recorder)
            sar.generate()

            # 投资组合回测
            config = build_port_analysis_config(
                signal=(model, dataset),
                topk=topk,
                n_drop=n_drop,
                start_time=start_time,
                end_time=end_time,
                freq=freq,
            )

            par = PortAnaRecord(recorder, config, freq)
            par.generate()

            suffix = "1day" if freq == "day" else f"1{freq}"
            report_normal = par.load(f"report_normal_{suffix}.pkl")
            port_analysis = par.load(f"port_analysis_{suffix}.pkl")
            indicator = par.load(f"indicator_analysis_{suffix}.pkl")

        metrics = {}
        if isinstance(port_analysis, pd.DataFrame):
            for col in port_analysis.columns:
                row = port_analysis[col].to_dict()
                for k, v in row.items():
                    key = f"{col}.{k}"
                    metrics[key] = float(v) if not isinstance(v, (dict, str)) else str(v)

        indicator_vals = {}
        if isinstance(indicator, pd.DataFrame):
            indicator_vals = indicator["value"].to_dict()

        return {
            "metrics": metrics,
            "indicator": indicator_vals,
            "report_normal": report_normal,
        }

    def run_from_predictions(self, predictions, label, topk=50, n_drop=5,
                             start_time=None, end_time=None, freq="day",
                             experiment_name="backtest"):
        """
        从已有预测信号运行回测（不需要 model + dataset）

        predictions : pd.Series 或 pd.DataFrame — 预测分数
        label : pd.Series 或 pd.DataFrame — 对应标签（必需，SigAnaRecord 计算 IC 需要）
        """
        from qlib.workflow import R
        from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord

        with R.start(experiment_name=experiment_name):
            recorder = R.get_recorder()

            recorder.save_objects(**{"pred.pkl": predictions, "label.pkl": label})

            sar = SigAnaRecord(recorder)
            sar.generate()

            config = build_port_analysis_config(
                signal=predictions,
                topk=topk,
                n_drop=n_drop,
                start_time=start_time,
                end_time=end_time,
                freq=freq,
            )

            par = PortAnaRecord(recorder, config, freq)
            par.generate()

            suffix = "1day" if freq == "day" else f"1{freq}"
            report_normal = par.load(f"report_normal_{suffix}.pkl")
            port_analysis = par.load(f"port_analysis_{suffix}.pkl")
            indicator = par.load(f"indicator_analysis_{suffix}.pkl")

        metrics = {}
        if isinstance(port_analysis, pd.DataFrame):
            for col in port_analysis.columns:
                row = port_analysis[col].to_dict()
                for k, v in row.items():
                    key = f"{col}.{k}"
                    metrics[key] = float(v) if not isinstance(v, (dict, str)) else str(v)

        indicator_vals = {}
        if isinstance(indicator, pd.DataFrame):
            indicator_vals = indicator["value"].to_dict()

        return {"metrics": metrics, "indicator": indicator_vals, "report_normal": report_normal}
