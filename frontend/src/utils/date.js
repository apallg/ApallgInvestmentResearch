// 本地时区日期工具 — 避免 toISOString() 的 UTC 日期在中国时区 (UTC+8) 08:00 前差一天
export function todayStr() {
  const d = new Date()
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

export function dateStr(d) {
  if (!d) return todayStr()
  if (typeof d === 'string') d = new Date(d)
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}
