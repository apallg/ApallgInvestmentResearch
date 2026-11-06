def limit_pct_for_code(code_str, is_st=False):
    code_str = str(code_str)
    if code_str.startswith(("8", "4")):
        return 0.30
    if code_str.startswith(("300", "301", "688")):
        return 0.20
    if is_st:
        return 0.05
    return 0.10
