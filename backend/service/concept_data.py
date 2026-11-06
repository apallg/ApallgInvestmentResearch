from core.data.collection_gate import GateCheckService, GateStatus


def require_fresh_concept_snapshot():
    result = GateCheckService().check_table("stock_sectors", force=True)
    if result is None or result.status != GateStatus.FRESH:
        reason = result.reason if result else "无法检查概念成分数据状态"
        raise ValueError(f"概念成分快照不可用：{reason}")
