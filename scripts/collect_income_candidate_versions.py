import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.service.income_version_repair import collect_income_candidate_versions


if __name__ == "__main__":
    print(json.dumps(collect_income_candidate_versions(), ensure_ascii=False, indent=2))
