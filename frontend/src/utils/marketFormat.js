export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

export function fmtPrice(v) {
  if (v == null) return '--'
  return Number(v).toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

export function fmtDelta(v) {
  if (v == null) return '--'
  const n = Number(v)
  if (!Number.isFinite(n)) return '--'
  return (n >= 0 ? '+' : '') + n.toFixed(2) + '%'
}

export function fmtChange(v) {
  if (v == null) return '--'
  const n = Number(v)
  if (!Number.isFinite(n)) return '--'
  return (n >= 0 ? '+' : '') + n.toFixed(2)
}

export function fmtAmt(v) {
  if (v == null) return '--'
  const n = Number(v)
  if (n >= 1e12) return (n / 1e12).toFixed(2) + '万亿'
  if (n >= 1e8) return (n / 1e8).toFixed(1) + '亿'
  return n.toLocaleString()
}

export function fmtVol(v) {
  if (v == null) return '--'
  const n = Number(v)
  if (n >= 1e8) return (n / 1e8).toFixed(2) + '亿'
  if (n >= 1e4) return (n / 1e4).toFixed(1) + '万'
  return String(n)
}

export function fmtM(v) {
  if (v == null) return '--'
  const n = Number(v)
  if (Math.abs(n) < 10) return n.toFixed(3)
  return n.toFixed(2)
}
