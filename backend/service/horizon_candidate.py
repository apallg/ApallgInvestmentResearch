from core.adapters.qlib.horizon_candidate_runtime import load_candidate, predict_candidate


def predict_horizon_candidate(package_path, horizon, feature_frame, expected_index, calendar):
    candidate = load_candidate(package_path, horizon)
    scores = predict_candidate(candidate, feature_frame, expected_index, calendar)
    profile = candidate["profile"]
    return scores, {"horizon": horizon, "structure": profile["structure"],
                    "qualification": {"live_prediction": profile["live_prediction_qualified"],
                                      "strict_pit": profile["strict_pit_verified"],
                                      "trade_execution": profile["trade_execution_verified"],
                                      "production_switch": profile["production_switch_allowed"]}}
