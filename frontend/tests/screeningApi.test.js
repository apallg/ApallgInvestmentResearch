import { beforeEach, describe, expect, it, vi } from 'vitest'

const { post } = vi.hoisted(() => ({ post: vi.fn() }))

vi.mock('../src/api/index.js', () => ({
  default: { post },
}))

import { runScreening } from '../src/api/screening.js'

describe('screening api', () => {
  beforeEach(() => post.mockReset())

  it('forwards volatility state model mappings', async () => {
    const stateModels = {
      high_vol: ['exp', 'rebound-lgb-lstm'],
      low_vol: ['exp', 'trend-lgb-lstm'],
    }

    await runScreening({ use_vol_switch: true, state_models: stateModels })

    expect(post).toHaveBeenCalledOnce()
    expect(post.mock.calls[0][0]).toBe('/screening/run')
    expect(post.mock.calls[0][1]).toMatchObject({
      use_vol_switch: true,
      state_models: stateModels,
    })
  })

  it('uses the explainable diagnostic baseline by default', async () => {
    await runScreening()

    expect(post.mock.calls[0][1]).toMatchObject({
      pool: 'stable_history_500d',
      macro_bias: false,
      macro_regime: false,
      sector_fusion: false,
      neutralize_qlib: false,
      decision_day_filter: true,
    })
  })

  it('forwards an exact profile reference', async () => {
    await runScreening({ profile_id: 'baseline', profile_version: 3 })
    expect(post.mock.calls[0][1]).toMatchObject({ profile_id: 'baseline', profile_version: 3 })
  })
})
