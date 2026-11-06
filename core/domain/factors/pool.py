"""
因子池管理器 — JSON文件持久化，管理活跃因子、月度监测、自动淘汰
"""

__all__ = ["FactorPoolManager", "get_pool", "reload_pool"]

import os
import json
import copy
import datetime
import logging
import threading

from core.shared.config import FACTOR_POOL_STATE_PATH

logger = logging.getLogger(__name__)

STATE_PATH = FACTOR_POOL_STATE_PATH
LEGACY_STATE_PATH = os.path.join(os.path.dirname(__file__), "factor_pool_state.json")

DEFAULT_STATE = {
    "white_box": {},      # {key: {label, template_key, params, added_date, metrics, active}}
    "traditional": {
        "tech":       {"label": "技术指标", "active": True},
        "momentum":   {"label": "动量因子", "active": True},
        "volatility": {"label": "波动率", "active": True},
        "size_turnover": {"label": "小市值换手", "active": True},
        "fundamental": {"label": "基本面", "active": False},
        "sentiment":  {"label": "情绪分析", "active": False},
        "flow":       {"label": "资金流向", "active": False},
        "qlib":       {"label": "Qlib模型", "active": False},
        "north_flow": {"label": "北向资金", "active": False},
        "size":       {"label": "规模因子(SMB)", "active": False},
        "value":      {"label": "价值因子(HML)", "active": False},
        "quality":    {"label": "质量因子(QMJ)", "active": False},
    },
    "history": {          # {category: [(anchor_date, [factor_keys])]}, 按时间排序
        "white_box": [],
        "traditional": [],
    },
    "monitoring": {},     # {key: {ir_history: [(month, ir)], last_check: str}}
    "updated_at": "",
}

_lock = threading.RLock()  # 可重入, 允许写方法内部 _save 再次获取


class FactorPoolManager:
    def __init__(self):
        self.state = self._load()
        self._persisted_state = copy.deepcopy(self.state)

    @staticmethod
    def _deep_merge(base, defaults):
        """将 defaults 中缺失的 key 合并到 base (原地修改)"""
        for k, v in defaults.items():
            if k not in base:
                base[k] = v
            elif isinstance(v, dict) and isinstance(base.get(k), dict):
                FactorPoolManager._deep_merge(base[k], v)
        return base

    def _load(self):
        from core.data import FactorPoolStateRepo
        try:
            state = FactorPoolStateRepo.load()
            if state is not None:
                logger.info("[FactorPool] 从 MySQL 加载状态")
        except Exception as e:
            raise RuntimeError(f"因子池权威数据库不可用，拒绝读取 JSON 旧快照: {e}") from e

        if state is None:
            if os.path.exists(STATE_PATH) or (
                not os.getenv("APALLG_FACTOR_POOL_STATE_PATH") and os.path.exists(LEGACY_STATE_PATH)
            ):
                raise RuntimeError("因子池数据库无权威状态但存在 JSON 快照；请先显式核验并恢复，拒绝覆盖旧状态")
            state = json.loads(json.dumps(DEFAULT_STATE))
        else:
            state = self._deep_merge(state, json.loads(json.dumps(DEFAULT_STATE)))
        return state

    def _save(self):
        """内部方法: 调用方必须持有 _lock"""
        from core.data import FactorPoolStateRepo
        previous = copy.deepcopy(self._persisted_state)
        self.state["updated_at"] = FactorPoolStateRepo.now_iso()
        try:
            FactorPoolStateRepo.save(self.state)
        except Exception as e:
            self.state = previous
            raise RuntimeError(f"因子池权威数据库写入失败，变更已回滚: {e}") from e

        self._persisted_state = copy.deepcopy(self.state)
        try:
            FactorPoolStateRepo.save_json(self.state, STATE_PATH)
        except Exception as e:
            logger.warning(f"[FactorPool] 数据库已提交，但 JSON 审计快照导出失败: {e}")

    def add_factor(self, key, label, template_key=None, params=None, metrics=None,
                   category="white_box", expr=None, curated_set=None, data_safety="unknown"):
        """新因子入库。如果已存在则更新 metrics。
        expr: 可选 qlib 表达式, 存进 entry 供 resolve_factor_expression 识别 (遗传因子等动态表达式)。"""
        from core.data import FactorPoolStateRepo
        with _lock:
            actual_curated = None
            if curated_set:
                curated = self.state.get("curated", {})
                if curated_set == "current_curated":
                    if not curated:
                        raise ValueError("当前没有可晋级的精选因子集")
                    actual_curated = max(curated, key=lambda name: curated[name].get("created", ""))
                else:
                    actual_curated = curated_set
                if actual_curated not in curated:
                    raise ValueError(f"目标精选因子集不存在: {actual_curated}")
            if category not in self.state:
                self.state[category] = {}
            entry = {
                "label": label,
                "template_key": template_key,
                "params": params or {},
                "added_date": FactorPoolStateRepo.today_iso(),
                "metrics": metrics or {},
                "active": True,
                "data_safety": data_safety or "unknown",
            }
            if expr:
                entry["expr"] = expr
            self.state[category][key] = entry
            if actual_curated:
                curated_entry = self.state["curated"][actual_curated]
                flat_list = curated_entry.setdefault("flat_list", [])
                if key not in flat_list:
                    flat_list.append(key)
                factors = curated_entry.setdefault("factors", {})
                mined = factors.setdefault("挖掘因子", [])
                if key not in mined:
                    mined.append(key)
                curated_entry.setdefault("stats", {})["total"] = len(flat_list)
            logger.info(f"[FactorPool] 入库: {key} ({label})")
            self._save()
            return actual_curated

    def remove_factor(self, key):
        """手动移除因子"""
        with _lock:
            for cat in ("white_box", "traditional", "hybrid"):
                if key in self.state.get(cat, {}):
                    self.state[cat][key]["active"] = False
                    logger.info(f"[FactorPool] 停用: {key}")
                    self._save()
                    return True
        return False

    def delete_factor(self, key):
        """硬删除因子 (从状态文件中彻底移除)"""
        with _lock:
            deleted = False
            for cat in ("white_box", "traditional", "hybrid"):
                if key in self.state.get(cat, {}):
                    del self.state[cat][key]
                    deleted = True
                    logger.info(f"[FactorPool] 删除: {key} (from {cat})")
            if key in self.state.get("monitoring", {}):
                del self.state["monitoring"][key]
                deleted = True
            if deleted:
                self._save()
            return deleted

    def toggle_factor(self, key, active=True):
        """切换因子活跃状态"""
        with _lock:
            for cat in ("white_box", "traditional", "hybrid"):
                if key in self.state.get(cat, {}):
                    self.state[cat][key]["active"] = active
                    self._save()
                    return True
        return False

    def get_active_factors(self, category=None, as_of=None):
        """获取活跃因子列表。
        category=None 返回全部。
        as_of: 历史日期(str或date) — 返回该日期时的活跃因子集，而非当前快照。
               仅使用带可信生效日期的历史快照；缺少时点证据时返回空集。
        """
        with _lock:
            if as_of is not None:
                snapshot = self.get_active_factor_snapshot(as_of, category)
                return snapshot["factors"] if snapshot else {}

            result = {}
            cats = [category] if category else ["white_box", "traditional", "hybrid"]
            for cat in cats:
                for key, info in self.state.get(cat, {}).items():
                    if info.get("active", True):
                        if key in result:
                            continue  # 同名键冲突, 保留首次出现的 (跨分类重复是设计行为)
                        result[key] = {**info, "pool_category": cat}
            return result

    def get_active_factor_snapshot(self, as_of, category="white_box"):
        if hasattr(as_of, "isoformat"):
            as_of = as_of.isoformat()
        as_of = str(as_of)[:10]
        with _lock:
            result = {}
            selected = {}
            cats = [category] if category else ["white_box", "traditional", "hybrid"]
            for cat in cats:
                hist = self.state.get("history", {}).get(cat, [])
                eligible = [entry for entry in hist
                            if isinstance(entry, dict) and
                            entry.get("provenance") in ("explicit_snapshot", "matured_rolling_test") and
                            isinstance(entry.get("effective_date"), str) and
                            entry["effective_date"][:10] <= as_of and
                            isinstance(entry.get("keys"), list) and
                            isinstance(entry.get("factor_specs"), dict) and
                            set(entry["keys"]) == set(entry["factor_specs"])]
                if not eligible:
                    continue
                latest = max(eligible, key=lambda entry: (
                    entry["effective_date"][:10], str(entry.get("source_anchor") or ""),
                ))
                selected[cat] = {
                    "effective_date": latest["effective_date"][:10],
                    "source_anchor": latest.get("source_anchor"),
                    "provenance": latest["provenance"],
                    "keys": list(latest["keys"]),
                    "factor_specs": copy.deepcopy(latest["factor_specs"]),
                }
                for key in latest["keys"]:
                    if key in result:
                        logger.warning(f"[FactorPool] 同名键冲突(历史): {key} 已在 {result[key].get('pool_category', '?')} 中, "
                                       f"{cat} 中的同名条目被忽略")
                        continue
                    info = latest["factor_specs"].get(key, {})
                    result[key] = {**copy.deepcopy(info), "pool_category": cat, "active": True}
            if not selected:
                return None
            return {"as_of": as_of, "categories": selected, "factors": result}

    def get_all_factors(self):
        """返回全部因子(含非活跃)的完整状态"""
        with _lock:
            result = {}
            for cat in ("white_box", "traditional", "hybrid"):
                for key, info in self.state.get(cat, {}).items():
                    result[key] = {**info, "pool_category": cat, "key": key}
            return result

    def update_monitoring(self, key, ir_value):
        """记录月度 IR 值"""
        from core.data import FactorPoolStateRepo
        with _lock:
            month = FactorPoolStateRepo.today_iso()[:7]
            if key not in self.state["monitoring"]:
                self.state["monitoring"][key] = {"ir_history": [], "last_check": month}
            mon = self.state["monitoring"][key]
            mon["ir_history"].append((month, round(ir_value, 4)))
            mon["last_check"] = month
            # 只保留最近 12 个月
            if len(mon["ir_history"]) > 12:
                mon["ir_history"] = mon["ir_history"][-12:]
            self._save()

    def get_icir_weights(self, as_of=None, halflife=12, min_obs=10):
        """ICIR 自适应权重: w_i = max(0, ICIR_i) / sum(max(0, ICIR_j))

        从 factor_validation_scores 表读取滚动 IC 序列，计算 ICIR (mean/std)，
        用指数衰减加权 (halflife 期) 给近期更高权重。
        返回 {source_key: weight}，可直接传入 screen(source_weights=...)。
        """
        try:
            from core.data import Database
            if not Database.is_available():
                return {}
        except Exception:
            return {}

        active = self.get_active_factors(as_of=as_of)
        if not active:
            return {}

        icirs = {}
        from core.data import FactorPoolStateRepo
        now = FactorPoolStateRepo.today_iso()
        for key in active:
            try:
                rows = Database.fetchall(
                    "SELECT daily_ics, icir, ic_mean FROM factor_validation_scores "
                    "WHERE factor_key=%s AND anchor_date <= %s "
                    "ORDER BY anchor_date DESC LIMIT %s",
                    (key, str(now), halflife * 3),
                )
                if not rows or len(rows) < min_obs:
                    continue
                import numpy as np
                ics = []
                stored_icir = None
                for r in rows:
                    if r.get("daily_ics"):
                        try:
                            pairs = json.loads(r["daily_ics"]) if isinstance(r["daily_ics"], str) else r["daily_ics"]
                            ics.extend([float(p[1]) for p in pairs if len(p) >= 2])
                        except Exception:
                            pass
                    if stored_icir is None and r.get("icir"):
                        stored_icir = float(r["icir"])
                if ics and len(ics) >= min_obs:
                    weights = np.exp(-np.arange(len(ics)) / halflife)[::-1]
                    w_mean = np.average(ics, weights=weights)
                    w_std = np.sqrt(np.average((np.array(ics) - w_mean) ** 2, weights=weights))
                    icir = w_mean / w_std if w_std > 0 else 0
                elif stored_icir is not None:
                    icir = stored_icir
                else:
                    continue
                if icir > 0:
                    icirs[key] = icir
            except Exception:
                pass

        if not icirs:
            return {}
        total = sum(icirs.values())
        # 映射到 source_weights key
        KEY_MAP = {
            "tech": "tech", "momentum": "momentum", "volatility": "volatility",
            "size_turnover": "size_turnover", "fundamental": "fundamental",
            "sentiment": "sentiment", "flow": "flow", "qlib": "qlib",
            "north_flow": "north_flow", "illiq": "illiq", "overnight_gap": "overnight_gap",
            "wb_industry": "wb_industry", "wb_divergence": "wb_divergence", "wb_moneyflow": "wb_moneyflow",
        }
        weights = {}
        for key, icir in icirs.items():
            w = icir / total
            for prefix, source in KEY_MAP.items():
                if key.startswith(prefix):
                    weights[source] = weights.get(source, 0) + w
                    break
            else:
                weights[key] = weights.get(key, 0) + w
        return weights

    def get_top_by_regime(self, regime=None, top_n=30, min_dates=5):
        """按市场状态聚合因子 IC，返回该状态下表现最好的因子集。

        已融入 curate_direction (regime_aware=True)，本方法保留作为只读查询接口。
        """
        from core.domain.factors.selector import _compute_regime_icir

        if regime is None:
            try:
                from core.data import Database
                rows = Database.fetchall(
                    "SELECT regime FROM market_regime_history ORDER BY trade_date DESC LIMIT 1"
                )
                regime = rows[0]["regime"] if rows else "trending_up"
            except Exception:
                regime = "trending_up"

        try:
            from core.data import Database
            regime_rows = Database.fetchall(
                "SELECT trade_date FROM market_regime_history WHERE regime=%s ORDER BY trade_date",
                (regime,),
            )
            regime_dates = {str(r["trade_date"]) for r in regime_rows}
        except Exception:
            return {"regime": regime, "factors": [], "error": "无法读取市场状态数据"}

        if len(regime_dates) < min_dates:
            return {"regime": regime, "factors": [], "error": f"该状态仅有 {len(regime_dates)} 天数据"}

        active_factors = self.get_active_factors()
        all_keys = list(active_factors)
        if not all_keys:
            return {"regime": regime, "factors": [], "error": "因子池为空"}

        factor_regime_ics = _compute_regime_icir(all_keys, regime_dates, min_dates)

        if not factor_regime_ics:
            return {"regime": regime, "factors": [], "error": "该状态下无因子有足够IC观测"}

        long_factors = [(k, v) for k, v in factor_regime_ics.items() if v["ic_mean"] > 0]
        short_factors = [(k, v) for k, v in factor_regime_ics.items() if v["ic_mean"] < 0]
        long_factors.sort(key=lambda x: x[1]["icir"], reverse=True)
        short_factors.sort(key=lambda x: x[1]["icir"], reverse=True)

        n_long = min(top_n * 3 // 4, len(long_factors))
        n_short = min(top_n - n_long, len(short_factors))
        selected = long_factors[:n_long] + short_factors[:n_short]

        factors = []
        for k, v in selected:
            info = active_factors.get(k, {})
            factors.append({
                "key": k,
                "label": info.get("label", k),
                "category": info.get("pool_category", ""),
                "ic_mean": v["ic_mean"],
                "icir": v["icir"],
                "n_dates": v["n_dates"],
                "direction": "long" if v["ic_mean"] > 0 else "short",
            })

        return {
            "regime": regime,
            "total_checked": len(factor_regime_ics),
            "selected": len(factors),
            "factors": factors,
        }

    def auto_prune(self, ir_threshold=0.015, consecutive_months=2):
        """IR 连续N月 < 阈值 → 自动停用"""
        with _lock:
            removed = []
            for key, mon in self.state.get("monitoring", {}).items():
                recent = mon.get("ir_history", [])[-consecutive_months:]
                if len(recent) >= consecutive_months and all(r[1] < ir_threshold for r in recent):
                    if self.remove_factor(key):
                        removed.append(key)
            if removed:
                logger.info(f"[FactorPool] 自动淘汰: {removed}")
            return removed

    def get_source_weights(self):
        """根据活跃因子生成选股用的 source_weights dict"""
        active = self.get_active_factors()
        # 映射表: factor_key → source_weights key
        KEY_MAP = {
            "tech": "tech", "qlib": "qlib", "sentiment": "sentiment",
            "fundamental": "fundamental", "flow": "flow", "north_flow": "north_flow",
            "size_turnover": "size_turnover", "momentum": "momentum", "volatility": "volatility",
            "wb_industry": "wb_industry", "wb_divergence": "wb_divergence", "wb_moneyflow": "wb_moneyflow",
            "wb_industry_n5": "wb_industry", "wb_industry_n10": "wb_industry",
            "wb_industry_n20": "wb_industry", "wb_industry_n30": "wb_industry",
        }
        weights = {}
        for key in active:
            wk = KEY_MAP.get(key, key)
            weights[wk] = weights.get(wk, 0) + 1.0
        total = max(sum(weights.values()), 1)
        return {k: round(v/total, 3) for k, v in weights.items()}

    def batch_add_from_phase1(self, phase1_signals):
        """从 Phase 1 结果批量入库有效信号"""
        count = 0
        for sig in phase1_signals:
            if sig.get("status") == "有效" and sig.get("is_sub"):
                tk = sig.get("template_key", "")
                cat = "white_box" if tk else "traditional"
                self.add_factor(
                    key=sig["key"],
                    label=sig["label"],
                    template_key=tk,
                    params={"param_value": sig.get("param_value")},
                    metrics=sig.get("metrics", {}),
                    category=cat,
                )
                count += 1
        logger.info(f"[FactorPool] Phase1 批量入库: {count} 个")
        return count

    def _backup_before_sync(self, category):
        """sync_active_set 前自动备份当前状态到 backups/ 目录"""
        from core.data import FactorPoolStateRepo
        backup_root = os.path.join(os.path.dirname(__file__), "..", "backups")
        timestamp = FactorPoolStateRepo.now_strftime("%Y%m%d_%H%M%S")
        try:
            backup_path = FactorPoolStateRepo.backup_state(
                self.state, backup_root, timestamp)
            logger.info(f"[FactorPool] 同步前备份: {backup_path}")
        except Exception as e:
            logger.warning(f"[FactorPool] 备份失败: {e}")
        try:
            FactorPoolStateRepo.cleanup_old_backups(backup_root, keep=10)
        except Exception:
            pass

    def list_backups(self):
        """列出所有可用备份"""
        from core.data import FactorPoolStateRepo
        backup_root = os.path.join(os.path.dirname(__file__), "..", "backups")
        return FactorPoolStateRepo.list_backups(backup_root)

    def restore_from_backup(self, backup_name=None):
        """从备份还原因子池状态。
        backup_name=None 时还原最近一次备份。
        返回还原后的状态摘要。"""
        from core.data import FactorPoolStateRepo
        backups = self.list_backups()
        if not backups:
            logger.warning("[FactorPool] 无可用备份")
            return None
        if backup_name is None:
            target = backups[0]
        else:
            matched = [b for b in backups if b["name"] == backup_name]
            if not matched:
                logger.warning(f"[FactorPool] 未找到备份: {backup_name}")
                return None
            target = matched[0]
        try:
            old_state = FactorPoolStateRepo.read_backup(target["path"])
        except Exception as e:
            logger.warning(f"[FactorPool] 读取备份失败: {e}")
            return None
        with _lock:
            self.state = old_state
            self._save()
        logger.info(f"[FactorPool] 从备份还原: {target['name']} "
                    f"({target['total']} 因子, {target['active']} 活跃)")
        return {"name": target["name"], "total": target["total"], "active": target["active"]}

    def sync_active_set(self, keys, factor_meta=None, category="white_box", anchor_date=None):
        """用 keys 覆盖为当期动态因子池: keys 中的入库/激活, 同类中不在 keys 的停用。
        供滚动筛选每期写回, 形成随时间更替的动态池。返回 (added, deactivated)。
        factor_meta: {key: {label, template_key, params, metrics, expr}} 可选元数据。
        anchor_date: 锚点日期 — 记录到 history 中, 供训练时按历史日期查询对应活跃集。"""
        if keys is None:
            logger.warning("[FactorPool] sync_active_set: keys 为 None, 跳过")
            return [], []
        if category is None:
            category = "white_box"
        from core.data import FactorPoolStateRepo
        with _lock:
            self._backup_before_sync(category)
            keys = set(keys)
            meta = factor_meta or {}
            added, deactivated = [], []
            # 停用同类中本期未通过的因子
            for k, info in list(self.state.get(category, {}).items()):
                if info.get("active", True) and k not in keys:
                    info["active"] = False
                    deactivated.append(k)
            # 入库/激活本期通过的因子
            for k in keys:
                m = meta.get(k, {})
                existing = self.state.get(category, {}).get(k)
                if existing:
                    existing["active"] = True
                    for field in ("label", "template_key", "params", "expr", "data_safety"):
                        if field in m:
                            existing[field] = copy.deepcopy(m[field])
                    if m.get("metrics"):
                        existing["metrics"] = copy.deepcopy(m["metrics"])
                else:
                    self.state.setdefault(category, {})[k] = {
                        "label": m.get("label", k),
                        "template_key": m.get("template_key"),
                        "params": m.get("params", {}),
                        "expr": m.get("expr"),
                        "data_safety": m.get("data_safety") or "unknown",
                        "added_date": FactorPoolStateRepo.today_iso(),
                        "metrics": m.get("metrics", {}),
                        "active": True,
                    }
                    added.append(k)
            # 记录历史锚点快照
            if anchor_date is not None:
                akey = anchor_date.isoformat() if hasattr(anchor_date, "isoformat") else str(anchor_date)
                factor_specs = {k: copy.deepcopy(self.state[category][k]) for k in keys}
                hist = self.state.setdefault("history", {}).setdefault(category, [])
                for i, entry in enumerate(hist):
                    if (isinstance(entry, dict) and entry.get("provenance") == "explicit_snapshot" and
                            entry.get("effective_date") == akey):
                        hist[i] = {"effective_date": akey, "keys": list(keys),
                                   "factor_specs": factor_specs,
                                   "provenance": "explicit_snapshot"}
                        break
                else:
                    hist.append({"effective_date": akey, "keys": list(keys),
                                 "factor_specs": factor_specs,
                                 "provenance": "explicit_snapshot"})
                hist.sort(key=lambda entry: (
                    entry.get("effective_date", "") if isinstance(entry, dict) else str(entry[0]),
                    str(entry.get("source_anchor") or "") if isinstance(entry, dict) else "",
                ))
            self._save()
            logger.info(f"[FactorPool] 动态池同步({category}): +{len(added)} 激活, -{len(deactivated)} 停用"
                         + (f" anchor={akey}" if anchor_date else ""))
        return added, deactivated

    def record_history(self, keys, effective_date, category="white_box", source_anchor=None,
                       factor_specs=None):
        """记录滚动测试结果在标签成熟后的历史有效日期。"""
        with _lock:
            akey = effective_date.isoformat() if hasattr(effective_date, "isoformat") else str(effective_date)
            source_key = source_anchor.isoformat() if hasattr(source_anchor, "isoformat") else str(source_anchor or "")
            keys = list(keys)
            if factor_specs is None:
                factor_specs = {k: copy.deepcopy(self.state[category][k])
                                for k in keys if k in self.state.get(category, {})}
            else:
                factor_specs = copy.deepcopy(factor_specs)
            if set(keys) != set(factor_specs):
                raise ValueError("历史因子快照必须包含每个因子的完整定义")
            hist = self.state.setdefault("history", {}).setdefault(category, [])
            for i, entry in enumerate(hist):
                if (isinstance(entry, dict) and entry.get("provenance") == "matured_rolling_test" and
                        entry.get("source_anchor") == source_key):
                    hist[i] = {"effective_date": akey, "source_anchor": source_key,
                               "keys": keys, "factor_specs": factor_specs,
                               "provenance": "matured_rolling_test"}
                    break
            else:
                hist.append({"effective_date": akey, "source_anchor": source_key,
                             "keys": keys, "factor_specs": factor_specs,
                             "provenance": "matured_rolling_test"})
            hist.sort(key=lambda entry: (
                entry.get("effective_date", "") if isinstance(entry, dict) else str(entry[0]),
                str(entry.get("source_anchor") or "") if isinstance(entry, dict) else "",
            ))
            self._save()
            logger.info(f"[FactorPool] 历史记录({category}): {len(keys)} 因子 effective={akey} anchor={source_key}")

    def register_fundamental_factors(self):
        """将适配器中的基本面因子同步到因子池 white_box 中 (启动时调用, 增量)。
        已有因子不覆盖, 仅补注册缺失的。返回新增数量。
        """
        from core.data import FactorPoolStateRepo
        try:
            from core.domain.factors.adapter import PublicFactorAdapter
            adapter = PublicFactorAdapter()
        except Exception as e:
            logger.warning(f"[FactorPool] 无法加载适配器, 跳过基本面注册: {e}")
            return 0

        with _lock:
            wb = self.state.setdefault("white_box", {})
            added = 0
            for key, info in adapter.all_factors.items():
                if info.get("source") != "基本面":
                    continue
                if key in wb:
                    continue
                # 注入 expr 供训练管线使用
                expr = info.get("expr")
                entry = {
                    "label": info.get("label", key),
                    "template_key": key,
                    "params": {"n": info.get("n", 20)},
                    "added_date": FactorPoolStateRepo.today_iso(),
                    "metrics": {},
                    "active": True,
                    "category": info.get("category", ""),
                    "source": "基本面",
                    "data_safety": info.get("data_safety") or "unknown",
                }
                if expr:
                    entry["expr"] = expr
                wb[key] = entry
                added += 1

            if added:
                self._save()
                logger.info(f"[FactorPool] 注册 {added} 个基本面因子到 white_box")
            return added

    # ══════════════════════════════════════════════
    # 精选集管理 (curated factor sets)
    # ══════════════════════════════════════════════

    def save_curated_set(self, name, factors_by_category, config, stats,
                         factor_specs=None, qualification="research_only", validation_evidence=None):
        """保存精选因子集到 factor_pool_state 的 curated 段。

        factors_by_category: {category: [(key, composite, normed, raw), ...]}
        config: 精选配置 dict
        stats: 统计信息 dict
        """
        from core.data import FactorPoolStateRepo
        with _lock:
            if "curated" not in self.state:
                self.state["curated"] = {}

            flat_list = []
            factors_detail = {}
            for cat, items in factors_by_category.items():
                keys = [item[0] for item in items]
                factors_detail[cat] = keys
                flat_list.extend(keys)
            resolved_specs = {}
            for key in flat_list:
                for category in ("white_box", "traditional", "hybrid"):
                    if key in self.state.get(category, {}):
                        resolved_specs[key] = copy.deepcopy(self.state[category][key])
                        break
            resolved_specs.update(copy.deepcopy(factor_specs or {}))

            self.state["curated"][name] = {
                "created": FactorPoolStateRepo.now_iso(),
                "description": config.get("description", ""),
                "config": config,
                "factors": factors_detail,
                "flat_list": flat_list,
                "factor_specs": resolved_specs,
                "qualification": qualification or "research_only",
                "production_eligible": False,
                "validation_evidence": copy.deepcopy(validation_evidence or {}),
                "stats": stats,
            }
            self._save()
            logger.info(f"[FactorPool] 精选集已保存: {name} ({len(flat_list)} 因子)")

    def get_curated_factors(self, name="latest"):
        """返回精选因子集的 flat_list。name="latest" 取最新版本。"""
        with _lock:
            curated = self.state.get("curated", {})
            if not curated:
                return []
            if name == "latest":
                latest_key = max(curated, key=lambda k: curated[k].get("created", ""))
                return curated[latest_key].get("flat_list", [])
            entry = curated.get(name)
            return entry.get("flat_list", []) if entry else []

    def get_curated_set(self, name="latest"):
        """返回完整精选集信息。"""
        with _lock:
            curated = self.state.get("curated", {})
            if not curated:
                return None
            if name == "latest":
                latest_key = max(curated, key=lambda k: curated[k].get("created", ""))
                return {"name": latest_key, **curated[latest_key]}
            entry = curated.get(name)
            return {"name": name, **entry} if entry else None

    def list_curated_sets(self):
        """列出所有精选集版本 (按创建时间降序)。"""
        with _lock:
            curated = self.state.get("curated", {})
            return sorted(
                [{"name": k, "created": v.get("created", ""),
                  "count": len(v.get("flat_list", [])),
                  "description": v.get("description", ""),
                  "qualification": v.get("qualification") or "unknown",
                  "production_eligible": v.get("production_eligible") is True}
                 for k, v in curated.items()],
                key=lambda x: x["created"], reverse=True)

    def delete_curated_set(self, name):
        """删除指定精选集, 同时清理历史版本 (最多保留 5 套)。"""
        with _lock:
            curated = self.state.get("curated", {})
            if name not in curated:
                return False
            del curated[name]

            # 自动清理: 超过 5 套时删除最旧的
            if len(curated) > 5:
                sorted_names = sorted(curated, key=lambda k: curated[k].get("created", ""))
                to_delete = sorted_names[:len(curated) - 5]
                for n in to_delete:
                    del curated[n]
                    logger.info(f"[FactorPool] 自动清理旧精选集: {n}")

            self._save()
            return True


_pool = None


def get_pool() -> "FactorPoolManager":
    from core.domain.factors.registry import get_pool as _reg_get_pool
    return _reg_get_pool()


def reload_pool():
    from core.domain.factors.registry import reload_pool as _reg_reload_pool
    return _reg_reload_pool()
