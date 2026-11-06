import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest


@pytest.mark.unit
@pytest.mark.p0
def test_existing_historical_news_is_skipped_before_analysis():
    from core.adapters.nlp.collector import SentimentCollector

    items = [
        {"hash": "old", "date": "2026-09-14"},
        {"hash": "today", "date": "2026-09-15"},
        {"hash": "new", "date": "2026-09-13"},
    ]
    with patch(
        "core.data.Database.fetchall",
        return_value=[{"hash": "old"}, {"hash": "today"}],
    ):
        selected = SentimentCollector().select_for_analysis(items, "2026-09-15")

    assert [item["hash"] for item in selected] == ["today", "new"]


@pytest.mark.unit
@pytest.mark.p0
def test_local_sentiment_uses_one_pipeline_batch():
    from core.adapters.nlp import analyzer

    calls = []

    def fake_pipeline(titles, **kwargs):
        calls.append((titles, kwargs))
        return [
            {"label": "positive", "score": 0.9},
            {"label": "negative", "score": 0.8},
        ]

    items = [
        {"title": "公司业绩增长", "source_confidence": 0.5},
        {"title": "公司业绩下滑", "source_confidence": 0.5},
    ]
    with patch.object(analyzer, "_get_pipeline", return_value=fake_pipeline):
        result = analyzer.SentimentAnalyzer().analyze(items)

    assert len(calls) == 1
    assert calls[0][0] == ["公司业绩增长", "公司业绩下滑"]
    assert calls[0][1]["batch_size"] == 2
    assert [item["sentiment"] for item in result] == ["positive", "negative"]


@pytest.mark.unit
@pytest.mark.p0
def test_cailian_refresh_is_single_flight():
    from core.adapters.nlp import collector

    class FakeFrame:
        empty = False

        def iterrows(self):
            yield 0, {
                "title": "000001 测试电报",
                "url": "https://example.com/2026-09-15/article.html",
                "date": "2026-09-15",
            }

    calls = []

    def fake_call_ak(*args, **kwargs):
        calls.append((args, kwargs))
        time.sleep(0.05)
        return FakeFrame()

    with patch.object(collector, "_cailian_cache", None), \
         patch.object(collector, "_cailian_cache_time", None), \
         patch.object(collector, "_load_name_map", return_value={}), \
         patch("core.shared.infrastructure.redis_client._get", return_value=None), \
         patch("core.shared.infrastructure.redis_client._set"), \
         patch("core.data.sources.akshare.call_ak", side_effect=fake_call_ak):
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(
                lambda _: collector.SentimentCollector._fetch_cailian_all(),
                range(4),
            ))

    assert len(calls) == 1
    assert all("000001" in result for result in results)


@pytest.mark.unit
@pytest.mark.p0
def test_cailian_snapshot_stays_fixed_for_the_run():
    from core.adapters.nlp.collector import SentimentCollector

    snapshot = {"000001": [{"title": "测试"}]}
    instance = SentimentCollector(cailian_cache=snapshot)
    with patch.object(instance, "_fetch_cailian_all") as fetch:
        result = instance._fetch_cailian("000001")

    assert result == [{"title": "测试"}]
    fetch.assert_not_called()


@pytest.mark.unit
@pytest.mark.p0
def test_sentiment_scheduler_rejects_overlapping_run():
    from core.adapters.nlp.scheduler import SentimentScheduler

    scheduler = SentimentScheduler()
    scheduler._run_lock.acquire()
    try:
        with patch.object(scheduler, "_get_tracked_stocks") as get_stocks:
            scheduler._run_once()
    finally:
        scheduler._run_lock.release()

    get_stocks.assert_not_called()
    assert scheduler.status["status"] == "stopped"
