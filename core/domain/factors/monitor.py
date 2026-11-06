"""
增强版因子池监测 — 月度IR + 动态淘汰 + 因子归档 + 月度报告

与 factor_pool_manager.py 配合:
- factor_pool_manager: 基础CRUD + IR记录 + 自动淘汰
- factor_monitor: 月度批量检测 + 归档 + 报告生成
"""
import os
import json
import datetime
import logging

logger = logging.getLogger(__name__)

ARCHIVE_PATH = os.path.join(os.path.dirname(__file__), "factor_archive.json")
REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")


class FactorMonitor:

    def __init__(self):
        from core.domain.factors.registry import get_pool
        self.pool = get_pool()
        os.makedirs(REPORT_DIR, exist_ok=True)

    # ══════════════════════════════════════════════
    # 月度检查
    # ══════════════════════════════════════════════

    def monthly_check(self, as_of=None):
        """月末批量检测所有活跃因子。"""
        if as_of is None:
            raise ValueError("as_of 参数为必填项, L3 禁止 date.today_date()")
        active = self.pool.get_active_factors()
        today = as_of
        month_str = today.strftime("%Y-%m")

        warnings = []
        archived = []

        for key, info in active.items():
            ir = self._estimate_monthly_ir(key)
            if ir is None:
                continue

            # 记录IR
            self.pool.update_monitoring(key, ir)

            # 分级判定
            if ir < 0:
                warnings.append((key, info.get("label", key), ir, "negative"))
            elif ir < 0.015:
                warnings.append((key, info.get("label", key), ir, "watch"))

        # 自动淘汰 (IR<0.015连续2月)
        pruned = self.pool.auto_prune(ir_threshold=0.015, consecutive_months=2)
        for key in pruned:
            archived.append({"key": key, "date": today.isoformat(), "reason": "IR连续2月<0.015"})
            self._archive(key, "IR连续2月<0.015", archive_date=today)

        self._save_archive()

        return {
            "month": month_str,
            "active_count": len(active),
            "warnings": [{"key": w[0], "label": w[1], "ir": w[2], "level": w[3]} for w in warnings],
            "pruned": pruned,
            "archived": len(archived),
        }

    def _estimate_monthly_ir(self, key):
        """估算因子近月IR（简化: 从monitoring记录中读取最新IR值）"""
        mon = self.pool.state.get("monitoring", {}).get(key, {})
        ir_history = mon.get("ir_history", [])
        if ir_history:
            return ir_history[-1][1]
        return None

    # ══════════════════════════════════════════════
    # 归档
    # ══════════════════════════════════════════════

    def _archive(self, key, reason, archive_date=None):
        # 收集因子元信息
        all_factors = self.pool.get_all_factors()
        label = None
        category = None
        if key in all_factors:
            label = all_factors[key].get("label", key)
            category = all_factors[key].get("category", "unknown")
        if archive_date is None:
            raise ValueError("archive_date 参数为必填项, L3 禁止 date.today_date()")

        # 1. 优先写 MySQL
        mysql_ok = False
        try:
            from core.data import FactorArchiveRepo
            FactorArchiveRepo.add(key, category=category, label=label, reason=reason)
            mysql_ok = True
        except Exception as e:
            logger.debug(f"[FactorMonitor] MySQL 归档失败: {e}")

        # 2. JSON 文件降级
        try:
            archive = self._load_archive()
            entry = {
                "key": key,
                "archived_date": archive_date.isoformat(),
                "reason": reason,
            }
            if label:
                entry["label"] = label
            if category:
                entry["category"] = category
            archive.append(entry)
            from core.shared.persistence import save_json
            save_json(archive, ARCHIVE_PATH)
        except Exception as e:
            logger.warning(f"[FactorMonitor] JSON 归档保存失败: {e}")

    def _load_archive(self):
        # 1. 优先从 MySQL 读取
        try:
            from core.data import FactorArchiveRepo
            rows = FactorArchiveRepo.get_all()
            if rows:
                return [{
                    "key": r["factor_key"],
                    "label": r.get("label", r["factor_key"]),
                    "category": r.get("category", "unknown"),
                    "archived_date": r["archived_at"].strftime("%Y-%m-%d") if r.get("archived_at") else "",
                    "reason": r.get("reason", ""),
                } for r in rows]
        except Exception as e:
            logger.debug(f"[FactorMonitor] MySQL 读取归档失败: {e}")

        # 2. JSON 文件降级
        try:
            from core.shared.persistence import load_json
            data = load_json(ARCHIVE_PATH)
            if data is not None:
                return data
        except Exception:
            pass
        return []

    def _save_archive(self):
        # 批量操作后写 JSON 降级文件
        try:
            archive = self._load_archive()
            from core.shared.persistence import save_json
            save_json(archive, ARCHIVE_PATH)
        except Exception as e:
            logger.warning(f"[FactorMonitor] 保存归档失败: {e}")

    def get_archive(self):
        return self._load_archive()

    # ══════════════════════════════════════════════
    # 月度报告
    # ══════════════════════════════════════════════

    def generate_monthly_report(self, as_of=None):
        """生成因子池月度健康报告 (Markdown)"""
        if as_of is None:
            raise ValueError("as_of 参数为必填项, L3 禁止 date.today_date()")
        today = as_of
        active = self.pool.get_active_factors()
        all_factors = self.pool.get_all_factors()
        monitoring = self.pool.state.get("monitoring", {})
        archive = self._load_archive()

        lines = [
            f"# 因子池月报 — {today.strftime('%Y年%m月')}",
            "",
            "## 一、概况",
            f"- 活跃因子: {len(active)}",
            f"- 总因子数: {len(all_factors)}",
            f"- 已归档: {len(archive)}",
            "",
            "## 二、活跃因子IR趋势",
            "",
            "| 因子 | 标签 | 近3月IR | 趋势 |",
            "|------|------|---------|------|",
        ]

        for key, info in sorted(active.items()):
            mon = monitoring.get(key, {})
            ir_hist = mon.get("ir_history", [])
            recent = ir_hist[-3:] if len(ir_hist) >= 3 else ir_hist
            ir_values = [f"{r[1]:.3f}" for r in recent]
            ir_str = " → ".join(ir_values) if ir_values else "无数据"
            trend = "↑" if len(recent) >= 2 and recent[-1][1] > recent[0][1] else "↓"
            lines.append(f"| {key} | {info.get('label', key)} | {ir_str} | {trend} |")

        lines.append("")
        lines.append("## 三、淘汰预警")
        lines.append("")

        warnings = []
        for key, info in active.items():
            mon = monitoring.get(key, {})
            ir_hist = mon.get("ir_history", [])
            if ir_hist and ir_hist[-1][1] < 0.015:
                warnings.append(f"- ⚠️ **{key}** ({info.get('label', key)}): IR={ir_hist[-1][1]:.3f}，低于淘汰阈值0.015")

        if warnings:
            lines.extend(warnings)
        else:
            lines.append("无预警因子。")

        lines.append("")
        lines.append("## 四、本月归档")
        lines.append("")
        recent_archive = [a for a in archive if a.get("archived_date", "").startswith(today.strftime("%Y-%m"))]
        if recent_archive:
            for a in recent_archive:
                lines.append(f"- {a['key']} ({a.get('label', '')}): {a.get('reason', '')}")
        else:
            lines.append("本月无归档。")

        lines.append("")
        lines.append(f"*报告由 Apallg投研 FactorMonitor 自动生成*")

        filename = f"factor_report_{today.strftime('%Y%m')}.md"
        path = os.path.join(REPORT_DIR, filename)
        from core.shared.persistence import save_text
        save_text("\n".join(lines), path)

        logger.info(f"[FactorMonitor] 月报已生成: {path}")
        return path
