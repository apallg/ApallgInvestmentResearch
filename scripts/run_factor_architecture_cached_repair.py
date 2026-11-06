import argparse
import json

from backend.service.factor_architecture import factor_architecture_service
from core.shared.logging import setup_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_run_id")
    parser.add_argument("cache_dir")
    parser.add_argument("--run-id")
    parser.add_argument("--expr-overlay-dir")
    args = parser.parse_args()
    setup_logging()
    result = factor_architecture_service.run_cached_repair(
        args.parent_run_id, args.cache_dir, run_id=args.run_id,
        expr_overlay_dir=args.expr_overlay_dir,
    )
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
