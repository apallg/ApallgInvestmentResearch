from scripts.package_horizon_architecture import choose_structure


def _analyses(horizon, stable):
    basic = ("N-G", "F50-G", "F50-N", "FC-G", "FC-N", "FC-F50", "FR-G", "FR-N", "FR-FC")
    improved = ("AX-A0", "R-AX", "R-F", "R-E")
    return ({"comparisons": [{"horizon": horizon, "comparison": name,
                               "stable_improvement": name in stable} for name in basic]},
            {"comparisons": [{"horizon": horizon, "comparison": name,
                               "stable_improvement": name in stable} for name in improved]})


def test_no_evidence_defaults_to_single_lgb():
    base, improvement = _analyses("H20", set())
    assert choose_structure("H20", base, improvement) == "G"


def test_lstm_and_residual_require_all_comparisons():
    base, improvement = _analyses("H20", {"N-G", "AX-A0", "R-AX", "R-F"})
    assert choose_structure("H20", base, improvement) == "AX"
    improvement["comparisons"][-1]["stable_improvement"] = True
    assert choose_structure("H20", base, improvement) == "R"


def test_dynamic_weight_requires_beating_fixed_weight():
    stable = {"FC-G", "FC-N", "FR-G", "FR-N"}
    base, improvement = _analyses("H10", stable)
    assert choose_structure("H10", base, improvement) == "FC"
    base["comparisons"][-1]["stable_improvement"] = True
    assert choose_structure("H10", base, improvement) == "FR"
