"""因子表达式前视(未来函数)防护 单元测试 — 零 I/O。

回归重点:
  - detect_future_shift 识别 Ref/Delta 负位移(未来 N 天)并放过正位移(过去 N 天)
  - WorldQuant 负位移约定经 _fix_qlib_compat 翻转为 Qlib 因果约定后, 不再有前视
"""
import pytest

from core.adapters.qlib.factor_resolver import detect_future_shift


@pytest.mark.parametrize("expr,expect_future", [
    ("Ref($close, -1)", True),
    ("Ref($close, 1)", False),
    ("Delta($close, -5)", True),
    ("Delta($close, 5)", False),
    ("$close / Ref($close, -8) - 1", True),
    ("$close / Ref($close, 8) - 1", False),
    ("Ref(Std($close/Ref($close,-1)-1,5),-5)", True),
    ("Sum($amount * ($close/Ref($close,1)-1), 90)", False),
    ("Ref($high, -1)", True),
    ("Ref(Max($high,5),-1)", True),
    ("Ref(Max($high,5),1)", False),
    ("Mean($close, 20)", False),
    ("Corr($close, $volume, 10)", False),
])
def test_detect_future_shift(expr, expect_future):
    assert bool(detect_future_shift(expr)) is expect_future


@pytest.mark.parametrize("raw,expect_causal", [
    ("Ref($close, -1)", "Ref($close, 1)"),
    ("$close/Ref($close, -20) - 1", "$close/Ref($close, 20) - 1"),
    ("Ref($close,-1)/Ref($close,-60)-1", "Ref($close, 1)/Ref($close, 60)-1"),
    ("Sign($close-Ref($close,-1))", "If(Gt($close-Ref($close, 1),0), 1, If(Lt($close-Ref($close, 1),0), -1, 0))"),
    ("Ref(Max($high,5),-1)", "Ref(Max($high,5), 1)"),
    ("Ref(Std($close/Ref($close,-1)-1,5),-5)", "Ref(Std($close/Ref($close, 1)-1,5), 5)"),
])
def test_worldquant_flip_is_causal(raw, expect_causal):
    from core.adapters.qlib.custom_handler import _fix_qlib_compat
    fixed = _fix_qlib_compat(raw)
    assert not detect_future_shift(fixed)
    assert fixed == expect_causal
