import { describe, expect, it } from 'vitest'
import { alignBenchmarkToStockDates } from '../src/utils/benchmark.js'

describe('benchmark alignment', () => {
  it('uses the first common date when the benchmark has older history', () => {
    const aligned = alignBenchmarkToStockDates([
      { trade_date: '2025-01-02', close: 11 },
      { trade_date: '2025-01-03', close: 12 },
    ], [
      { trade_date: '2020-01-02', close: 3000 },
      { trade_date: '2025-01-02', close: 4000 },
      { trade_date: '2025-01-03', close: 4200 },
    ])

    expect(aligned).toEqual({
      line: [11, 11.55],
      baseDate: '2025-01-02',
      matchedCount: 2,
    })
  })

  it('preserves missing trading dates as gaps', () => {
    const aligned = alignBenchmarkToStockDates([
      { trade_date: '2025-01-02', close: 11 },
      { trade_date: '2025-01-03', close: 12 },
      { trade_date: '2025-01-06', close: 13 },
    ], [
      { trade_date: '2025-01-02', close: 4000 },
      { trade_date: '2025-01-06', close: 4200 },
    ])

    expect(aligned.line).toEqual([11, null, 11.55])
    expect(aligned.baseDate).toBe('2025-01-02')
  })

  it('does not build a line when there is no common valid date', () => {
    const aligned = alignBenchmarkToStockDates(
      [{ trade_date: '2025-01-02', close: 11 }],
      [{ trade_date: '2025-01-03', close: 4000 }],
    )

    expect(aligned).toEqual({ line: null, baseDate: null, matchedCount: 0 })
  })
})
