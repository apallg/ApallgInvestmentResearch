export function alignBenchmarkToStockDates(stockRows, benchmarkRows) {
  const stockByDate = new Map()
  const benchmarkByDate = new Map()
  for (const row of stockRows || []) {
    const close = Number(row?.close)
    if (row?.trade_date && Number.isFinite(close) && close > 0) {
      stockByDate.set(String(row.trade_date).slice(0, 10), close)
    }
  }
  for (const row of benchmarkRows || []) {
    const close = Number(row?.close)
    if (row?.trade_date && Number.isFinite(close) && close > 0) {
      benchmarkByDate.set(String(row.trade_date).slice(0, 10), close)
    }
  }
  const dates = [...stockByDate.keys()].sort()
  const baseDate = dates.find(date => benchmarkByDate.has(date)) || null
  if (!baseDate) return { line: null, baseDate: null, matchedCount: 0 }
  const baseStock = stockByDate.get(baseDate)
  const baseBenchmark = benchmarkByDate.get(baseDate)
  const line = dates.map(date => {
    const benchmarkClose = benchmarkByDate.get(date)
    if (benchmarkClose == null) return null
    return +(benchmarkClose / baseBenchmark * baseStock).toFixed(2)
  })
  return { line, baseDate, matchedCount: line.filter(value => value != null).length }
}
