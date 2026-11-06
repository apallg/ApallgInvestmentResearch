import datetime
from unittest.mock import Mock, patch

import pytest


@pytest.mark.unit
@pytest.mark.p1
@pytest.mark.parametrize(("matured_at", "expected_timeline"), [
    ("2025-01-01", {"factor_a": ["2025-01-01"]}),
    ("2025-02-01", {}),
    (None, {}),
])
def test_rolling_pool_uses_only_matured_anchor_results(matured_at, expected_timeline, tmp_path):
    from core.domain.factors.rolling import rolling_factor_screen

    pool = Mock()
    anchor = datetime.date(2025, 1, 1)

    def worker(args):
        return {
            "akey": args["anchor_date"],
            "passed": ["factor_a"],
            "detail": {},
            "directions": {"factor_a": "long"},
            "window": {"label_matured_at": matured_at},
        }

    with patch("core.domain.factors.registry.get_pool", return_value=pool), \
         patch("core.domain.screening._screener.get_screener"), \
         patch("core.shared.anchor_cache.clear_mem_cache"):
        result = rolling_factor_screen(
            [{"variant_key": "factor_a", "label": "Factor A", "expr": "$close"}],
            anchor,
            anchor,
            write_pool=True,
            min_periods_frac=1.0,
            precomputed={"cache_dir": str(tmp_path), "pool_codes": ["000001"]},
            worker_fn=worker,
            pool_codes=["000001"],
            cleanup_precomputed=False,
        )

    assert result["factor_timeline"] == expected_timeline
    if expected_timeline:
        pool.record_history.assert_called_once_with(
            ["factor_a"], anchor, category="white_box", source_anchor=anchor,
            factor_specs={"factor_a": {"label": "Factor A", "expr": "$close"}},
        )
        pool.sync_active_set.assert_called_once()
        assert pool.sync_active_set.call_args.kwargs["anchor_date"] == anchor
    else:
        pool.record_history.assert_not_called()
        pool.sync_active_set.assert_not_called()
