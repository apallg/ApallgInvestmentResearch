"""
工具函数 — 股票代码格式转换
"""


def mysql_to_qlib_code(code: str) -> str:
    """
    MySQL 代码 → qlib 代码
    600519 → sh600519
    000001 → sz000001
    300750 → sz300750
    688981 → sh688981
    """
    code = str(code).strip()
    if code.startswith("sh") or code.startswith("sz"):
        return code.lower()
    if code.startswith(("600", "601", "603", "605", "688", "689")):
        return "sh" + code
    return "sz" + code
