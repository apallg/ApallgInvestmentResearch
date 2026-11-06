import json

from backend.service.income_version_repair import repair_confirmed_income_version_conflicts


if __name__ == "__main__":
    print(json.dumps(repair_confirmed_income_version_conflicts(), ensure_ascii=False, indent=2))
