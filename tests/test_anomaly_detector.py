import json
from unittest.mock import patch

from core.shared.anomaly_detector import AnomalyDetector, SIGNAL_RESULT_KEY, SIGNAL_WINDOW_KEY, compute_signals_and_save


class FakeRedis:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.writes = []

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, ttl, value):
        self.writes.append((key, ttl, value))
        self.values[key] = value

    def eval(self, script, numkeys, key, incoming, max_items, ttl):
        sample = json.loads(incoming)
        raw = self.values.get(key)
        window = json.loads(raw) if raw else []
        if any(not isinstance(item, dict) or not item.get("event_id") for item in window):
            window = []
        if not any(item["event_id"] == sample["event_id"] for item in window):
            window.append(sample)
            window.sort(key=lambda item: item["t"])
            window = window[-max_items:]
            self.setex(key, ttl, json.dumps(window))
        return json.dumps(window)


def test_anomaly_window_adds_each_observed_collection_event_once():
    code = "000001"
    current = {
        "_time": "2026-10-30 10:00:00",
        "7": "10.5",
        "13": "1000",
        "19": "10500",
        "199112": "1.2",
    }
    redis = FakeRedis({f"ths:realtime:{code}": json.dumps(current)})
    detector = AnomalyDetector()

    with patch("core.shared.anomaly_detector.get_redis", return_value=redis):
        first = detector.detect(code)
        repeated = detector.detect(code)

    window = json.loads(redis.values[SIGNAL_WINDOW_KEY.format(code=code)])
    assert first["snapshots"] == repeated["snapshots"] == 1
    assert len(window) == 1
    assert window[0]["observed_at"] == current["_time"]
    assert window[0]["event_id"]
    assert len(redis.writes) == 1

    current["_time"] = "2026-10-30 10:00:10"
    redis.values[f"ths:realtime:{code}"] = json.dumps(current)
    with patch("core.shared.anomaly_detector.get_redis", return_value=redis):
        next_event = detector.detect(code)

    window = json.loads(redis.values[SIGNAL_WINDOW_KEY.format(code=code)])
    assert next_event["snapshots"] == 2
    assert len(window) == 2

    current["_time"] = "2026-10-30 09:59:50"
    redis.values[f"ths:realtime:{code}"] = json.dumps(current)
    with patch("core.shared.anomaly_detector.get_redis", return_value=redis):
        earlier_event = detector.detect(code)

    window = json.loads(redis.values[SIGNAL_WINDOW_KEY.format(code=code)])
    assert earlier_event["snapshots"] == 3
    assert [item["observed_at"] for item in window] == [
        "2026-10-30 09:59:50",
        "2026-10-30 10:00:00",
        "2026-10-30 10:00:10",
    ]


def test_anomaly_sampling_rejects_snapshot_without_collection_time():
    redis = FakeRedis({
        "ths:realtime:000001": json.dumps({"7": "10.5", "13": "1000"})
    })
    with patch("core.shared.anomaly_detector.get_redis", return_value=redis):
        assert AnomalyDetector().detect("000001") is None

    assert not redis.writes


def test_collection_computes_and_caches_signals_for_read_only_api():
    code = "000001"
    current = {
        "_time": "2026-10-30 10:00:00",
        "7": "10.5",
        "13": "1000",
        "19": "10500",
        "199112": "1.2",
    }
    redis = FakeRedis({f"ths:realtime:{code}": json.dumps(current)})
    with patch("core.shared.anomaly_detector.get_redis", return_value=redis):
        compute_signals_and_save([code])

    signals = json.loads(redis.values[SIGNAL_RESULT_KEY.format(code=code)])
    assert signals["snapshots"] == 1
    assert len(json.loads(redis.values[SIGNAL_WINDOW_KEY.format(code=code)])) == 1


def test_signal_get_is_read_only_across_repeated_page_polls():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.ths import router

    payload = {
        "code": "000001",
        "snapshots": 5,
        "ofi": 0.1,
        "big_order_z": 0.5,
        "vwap_deviation": 0.2,
        "vol_ratio_acc": 0.1,
    }
    redis = FakeRedis({"ths:signals:000001": json.dumps(payload)})
    app = FastAPI()
    app.include_router(router)

    with patch("backend.api.ths.get_redis", return_value=redis), \
            patch("backend.api.ths._warmup_sync", side_effect=AssertionError("GET must not collect")), \
            patch("core.shared.anomaly_detector.AnomalyDetector.detect", side_effect=AssertionError("GET must not sample")):
        client = TestClient(app)
        responses = [client.get("/api/ths/signals/000001") for _ in range(100)]

    assert all(response.status_code == 200 for response in responses)
    assert all(response.json()["data"] == payload for response in responses)
    assert not redis.writes
