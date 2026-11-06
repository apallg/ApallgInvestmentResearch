import json
import pandas as pd

from scripts.h20_confirmation import sorted_stocks
from scripts.h20_confirmation_forward import evaluate


def test_tied_top5_uses_stock_code_order():
    index = pd.MultiIndex.from_arrays(
        [[pd.Timestamp("2026-07-01")] * 6, ["sz000006", "sz000005", "sz000004", "sz000003", "sz000002", "sz000001"]],
        names=["datetime", "instrument"])
    frame = pd.DataFrame({"score": [0.9] * 6}, index=index)
    result = sorted_stocks(frame).index.get_level_values("instrument").tolist()
    assert result[:5] == ["sz000001", "sz000002", "sz000003", "sz000004", "sz000005"]


def test_forward_evaluation_stays_pending_without_observation(tmp_path, monkeypatch, capsys):
    import scripts.h20_confirmation_forward as forward

    monkeypatch.setattr(forward, "OUTPUT", tmp_path)
    args = type("Args", (), {"calendar": None})()
    evaluate(args)
    assert json.loads(capsys.readouterr().out)["reason"] == "observation_not_started"
