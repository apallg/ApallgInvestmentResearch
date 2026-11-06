import pytest

from scripts.run_strategy_acceptance import verify_browser_evidence


pytestmark = [pytest.mark.unit, pytest.mark.p0]


def test_browser_evidence_cannot_be_self_attested(tmp_path):
    result = verify_browser_evidence(tmp_path, True, None, None)
    assert result["verified"] is False
    assert any("截图" in reason for reason in result["failures"])
    assert any("控制台" in reason for reason in result["failures"])
    assert any("网络" in reason for reason in result["failures"])


def test_browser_evidence_requires_nonempty_artifacts(tmp_path):
    shots = tmp_path / "shots"
    shots.mkdir()
    for name in ("data-1280x720.png", "factors.png", "qlib.png", "screening.png", "paper.png", "live-1440x900.png"):
        (shots / name).write_bytes(b"png")
    console = tmp_path / "console.log"
    network = tmp_path / "network.har"
    console.write_text("", encoding="utf-8")
    network.write_text("{}", encoding="utf-8")
    assert verify_browser_evidence(shots, True, console, network)["verified"] is False
    console.write_text('{"unhandled_errors":[]}', encoding="utf-8")
    network.write_text('{"log":{"entries":[]}}', encoding="utf-8")
    assert verify_browser_evidence(shots, True, console, network)["verified"] is True
