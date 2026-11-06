import pytest

from core.adapters.qlib.rolling_train import make_rolling_rounds


pytestmark = [pytest.mark.unit, pytest.mark.p0]


def test_rolling_round_uses_full_five_year_train_before_six_month_validation():
    rounds = make_rolling_rounds("2024-01-01", "2024-01-01", 3, 5, 6)
    _, train, valid, test = rounds[0]
    assert train == ("2018-07-01", "2023-06-30")
    assert valid == ("2023-07-01", "2023-12-31")
    assert test == ("2024-01-01", "2024-03-31")
