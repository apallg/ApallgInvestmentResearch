import json
import threading
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_FILE = _DATA_DIR / "disabled_models.json"
_LOCK = threading.Lock()


def _read():
    if not _FILE.exists():
        return {}
    try:
        return json.loads(_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write(data):
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def is_disabled(experiment_id, run_id):
    key = f"{experiment_id}/{run_id}"
    with _LOCK:
        return key in _read()


def set_disabled(experiment_id, run_id, disabled=True):
    from datetime import datetime
    key = f"{experiment_id}/{run_id}"
    with _LOCK:
        data = _read()
        if disabled:
            if key not in data:
                data[key] = {"disabled_at": datetime.now().isoformat()}
        else:
            data.pop(key, None)
        _write(data)


def get_all_disabled():
    with _LOCK:
        return dict(_read())
