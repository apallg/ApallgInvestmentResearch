import json

from core.shared.persistence import load_bytes, load_text


TRADABLE_LABEL_MODE = "tradable_open_to_open"


def get_model_label_mode(run_dir):
    declared = []
    metadata_path = run_dir / "artifacts" / "factor_keys.json"
    if metadata_path.exists():
        try:
            raw = load_bytes(str(metadata_path))
            start = raw.index(b"{")
            end = raw.rindex(b"}")
            metadata = json.loads(raw[start:end + 1].decode("utf-8", errors="ignore"))
            if metadata.get("label_mode"):
                declared.append(str(metadata["label_mode"]))
        except Exception:
            pass
    mode_path = run_dir / "params" / "label_mode"
    if mode_path.exists():
        try:
            mode = load_text(str(mode_path)).strip()
            if mode:
                declared.append(mode)
        except Exception:
            pass
    expression_mode = None
    label_path = run_dir / "params" / "dataset.kwargs.handler.kwargs.label"
    if label_path.exists():
        try:
            label = load_text(str(label_path))
            compact = "".join(label.split())
            if "Ref($open" in label and "Ref($open,-1)" in compact:
                expression_mode = TRADABLE_LABEL_MODE
            elif "Ref($close" in label:
                expression_mode = "close_to_close"
        except Exception:
            pass
    if expression_mode:
        return expression_mode if all(mode == expression_mode for mode in declared) else None
    unique_declared = set(declared)
    return unique_declared.pop() if len(unique_declared) == 1 else None
