"""
盘中异动检测 — 4个实时信号

信号:
  1. OFI (订单流失衡)         — 买盘vs卖盘力量对比, 方向突变 = 异动
  2. 大单z-score              — 主力资金异常流入/流出
  3. VWAP乖离                 — 价格偏离日内均价的极端程度
  4. 量比加速                 — 资金涌入/撤退的速度变化

数据源: Redis ths:realtime:{code} (80字段实时快照)
窗口: 最近20个快照 (~4分钟 @12s间隔)

用法:
  from core.shared.anomaly_detector import AnomalyDetector
  ad = AnomalyDetector()
  signals = ad.detect('600519')  # -> {ofi, big_order_z, vwap_dev, vol_ratio_acc}
"""

import hashlib
import json
import logging
import numpy as np
from datetime import datetime
from collections import defaultdict
from core.shared.infrastructure.redis_client import get_redis

logger = logging.getLogger(__name__)

WINDOW_SIZE = 20          # 快照窗口
OFI_THRESHOLD = 0.25      # OFI 突变阈值
BIG_ORDER_Z_THRESHOLD = 2.0   # 大单z-score阈值
VWAP_DEV_THRESHOLD = 2.0  # VWAP偏离σ倍数
VOL_ACC_THRESHOLD = 0.5   # 量比加速阈值

SIGNAL_WINDOW_KEY = "ths:signal_window:{code}"
SIGNAL_RESULT_KEY = "ths:signals:{code}"
APPEND_EVENT_SCRIPT = """
local raw = redis.call('GET', KEYS[1])
local window = {}
if raw then
    local ok, decoded = pcall(cjson.decode, raw)
    if ok and type(decoded) == 'table' then
        local valid = true
        for _, item in ipairs(decoded) do
            if type(item) ~= 'table' or not item.event_id then
                valid = false
                break
            end
        end
        if valid then window = decoded end
    end
end
local incoming = cjson.decode(ARGV[1])
for _, item in ipairs(window) do
    if item.event_id == incoming.event_id then
        return cjson.encode(window)
    end
end
local insert_at = #window + 1
for index, item in ipairs(window) do
    if (tonumber(item.t) or math.huge) > (tonumber(incoming.t) or math.huge) then
        insert_at = index
        break
    end
end
table.insert(window, insert_at, incoming)
while #window > tonumber(ARGV[2]) do table.remove(window, 1) end
redis.call('SETEX', KEYS[1], tonumber(ARGV[3]), cjson.encode(window))
return cjson.encode(window)
"""


class AnomalyDetector:

    def detect(self, code):
        """计算单只股票的4个异动信号"""
        r = get_redis()
        if not r:
            return None

        raw = r.get(f"ths:realtime:{code}")
        if not raw:
            return None

        current = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))

        try:
            observed_at = str(current["_time"])
            observed_time = datetime.fromisoformat(observed_at)
        except (KeyError, TypeError, ValueError):
            return None
        event_payload = json.dumps(current, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        event_id = hashlib.sha256(f"{code}|{observed_at}|{event_payload}".encode("utf-8")).hexdigest()
        sample = self._extract(current, code)
        sample["t"] = observed_time.timestamp()
        sample["event_id"] = event_id
        sample["observed_at"] = observed_at
        raw_window = r.eval(
            APPEND_EVENT_SCRIPT,
            1,
            SIGNAL_WINDOW_KEY.format(code=code),
            json.dumps(sample, ensure_ascii=False),
            WINDOW_SIZE,
            900,
        )
        window = json.loads(raw_window) if isinstance(raw_window, str) else json.loads(raw_window.decode("utf-8"))

        if len(window) < 5:
            return {
                "code": code,
                "ofi": None, "big_order_z": None,
                "vwap_deviation": None, "vol_ratio_acc": None,
                "snapshots": len(window),
            }

        ofi = self._calc_ofi(window)
        big_order_z = self._calc_big_order_z(window)
        vwap_dev = self._calc_vwap_deviation(window, current)
        vol_acc = self._calc_vol_ratio_acc(window)
        composite = self._calc_composite(ofi, big_order_z, vwap_dev, vol_acc)

        return {
            "code": code,
            "ofi": round(float(ofi), 4),
            "ofi_alert": bool(abs(ofi) > OFI_THRESHOLD and self._ofi_divergence(window)),
            "big_order_z": round(float(big_order_z), 4),
            "big_order_alert": bool(abs(big_order_z) > BIG_ORDER_Z_THRESHOLD),
            "vwap_deviation": round(float(vwap_dev), 4) if vwap_dev is not None else None,
            "vwap_alert": bool(abs(vwap_dev) > VWAP_DEV_THRESHOLD) if vwap_dev is not None else False,
            "vol_ratio_acc": round(float(vol_acc), 4) if vol_acc is not None else None,
            "vol_acc_alert": bool(abs(vol_acc) > VOL_ACC_THRESHOLD) if vol_acc is not None else False,
            "composite": composite,
            "snapshots": len(window),
            "time": observed_time.strftime("%H:%M:%S"),
            "observed_at": observed_at,
        }

    def detect_batch(self, codes):
        """批量检测, 返回有异动的股票列表"""
        results = []
        r = get_redis()
        for code in codes:
            s = self.detect(code)
            if s:
                if r:
                    try:
                        r.setex(SIGNAL_RESULT_KEY.format(code=code), 60, json.dumps(s, ensure_ascii=False))
                    except Exception:
                        pass
                alerts = sum([
                    s.get("ofi_alert", False),
                    s.get("big_order_alert", False),
                    s.get("vwap_alert", False),
                    s.get("vol_acc_alert", False),
                ])
                if alerts > 0:
                    s["alert_count"] = alerts
                    results.append(s)
        results.sort(key=lambda x: x["alert_count"], reverse=True)
        return results

    # ── 数据提取 ──────────────────────────────────────────

    def _extract(self, items, code):
        def f(k, d=0.0):
            try:
                return float(items.get(str(k), d))
            except (ValueError, TypeError):
                return d

        bid_vol = sum(f(30 + i * 2 + 1) for i in range(5))
        ask_vol = sum(f(40 + i * 2 + 1) for i in range(5))

        return {
            "t": datetime.now().timestamp(),
            "price": f("7"),
            "volume": f("13"),
            "amount": f("19"),
            "bid_vol": bid_vol,
            "ask_vol": ask_vol,
            "big_in": f("223"),
            "big_out": f("224"),
            "main_net": f("237"),
            "vol_ratio": f("1771976"),
            "pct_chg": f("199112"),
        }

    # ── 信号计算 ──────────────────────────────────────────

    def _calc_ofi(self, window):
        """订单流失衡 — 当前快照的(买-卖)/(买+卖)"""
        cur = window[-1]
        total = cur["bid_vol"] + cur["ask_vol"]
        if total <= 0:
            return 0
        return (cur["bid_vol"] - cur["ask_vol"]) / total

    def _ofi_divergence(self, window):
        """OFI背离检测: OFI方向突变但价格未跟随"""
        if len(window) < 4:
            return False
        recent_ofi = [self._calc_ofi_single(w) for w in window[-4:]]
        recent_px = [w["price"] for w in window[-4:]]
        ofi_prev = np.mean(recent_ofi[:2])
        ofi_cur = np.mean(recent_ofi[2:])
        px_prev = np.mean(recent_px[:2])
        px_cur = np.mean(recent_px[2:])
        ofi_change = ofi_cur - ofi_prev
        px_change = px_cur - px_prev
        if px_prev <= 0 or abs(ofi_change) < OFI_THRESHOLD:
            return False
        if ofi_change > 0 and px_change <= 0:
            return True   # 买盘增加但价格没涨 → 隐蔽吸筹
        if ofi_change < 0 and px_change >= 0:
            return True   # 卖盘增加但价格没跌 → 隐蔽出货
        return False

    def _calc_ofi_single(self, w):
        total = w["bid_vol"] + w["ask_vol"]
        return (w["bid_vol"] - w["ask_vol"]) / total if total > 0 else 0

    def _calc_big_order_z(self, window):
        """大单z-score — 当前主力净流入偏离均值的标准差倍数"""
        values = [w["main_net"] for w in window[:-1]]
        if len(values) < 4:
            return 0
        mean = np.mean(values)
        std = np.std(values)
        if std < 1000:
            return 0
        return (window[-1]["main_net"] - mean) / std

    def _calc_vwap_deviation(self, window, current):
        """VWAP乖离 — (当前价-VWAP)/σ, std地板0.01防涨停板除零"""
        prices = [w["price"] for w in window]
        volumes = [max(w["volume"], 1) for w in window]
        vwap = np.average(prices, weights=volumes)
        std = max(float(np.std(prices)), 0.01)
        cur_price = current.get("7")
        if not cur_price:
            return None
        try:
            return (float(cur_price) - vwap) / std
        except (ValueError, TypeError):
            return None

    def _calc_vol_ratio_acc(self, window):
        """量比加速 — 量比的变化率"""
        if len(window) < 3:
            return None
        recent = [w["vol_ratio"] for w in window[-3:]]
        if any(v is None for v in recent):
            return None
        try:
            return (recent[-1] - recent[-2]) / max(abs(recent[-2]), 0.01)
        except (ValueError, TypeError, ZeroDivisionError):
            return None

    def _calc_composite(self, ofi, big_order_z, vwap_dev, vol_acc):
        """
        综合买卖信号 — 加权评分卡
        OFI(买盘压力) + 大单方向 + VWAP位置 + 量能加速
        返回: strong_buy / buy / hold / sell / strong_sell
        """
        score = 0.0
        if abs(ofi) > 0.05:
            score += ofi * 2.0
        if big_order_z is not None:
            score += big_order_z * 0.5
        if vwap_dev is not None:
            score -= vwap_dev * 0.3
        if vol_acc is not None:
            score += vol_acc * 1.0

        if score >= 1.5:
            return "strong_buy"
        elif score >= 0.5:
            return "buy"
        elif score <= -1.5:
            return "strong_sell"
        elif score <= -0.5:
            return "sell"
        return "hold"

def compute_signals_and_save(codes):
    """采集后调用: 计算异动信号并写入Redis"""
    ad = AnomalyDetector()
    alerts = ad.detect_batch(codes)
    if alerts:
        r = get_redis()
        if r:
            r.setex("ths:alerts", 60, json.dumps(alerts, ensure_ascii=False))
        logger.info(f"异动检测: {len(alerts)} 只触发信号")
    return alerts
