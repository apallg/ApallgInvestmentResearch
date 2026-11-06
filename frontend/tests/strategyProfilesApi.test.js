import { beforeEach, describe, expect, it, vi } from 'vitest'

const { get, post } = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }))

vi.mock('../src/api/index.js', () => ({ default: { get, post } }))

import {
  activateStrategyProfile, createStrategyProfile, evaluateStrategyProfile,
  getChampionProfile, listStrategyProfiles, recordPaperPeriod, validateStrategyProfile,
} from '../src/api/strategyProfiles.js'

describe('strategy profile api', () => {
  beforeEach(() => { get.mockReset(); post.mockReset() })

  it('uses immutable profile endpoints', async () => {
    await listStrategyProfiles()
    await getChampionProfile()
    await createStrategyProfile({ profile_id: 'baseline' })
    await validateStrategyProfile('baseline', 2)
    expect(get).toHaveBeenNthCalledWith(1, '/strategy-profiles')
    expect(get).toHaveBeenNthCalledWith(2, '/strategy-profiles/champion')
    expect(post).toHaveBeenNthCalledWith(1, '/strategy-profiles', { profile_id: 'baseline' })
    expect(post).toHaveBeenNthCalledWith(2, '/strategy-profiles/baseline/2/validate')
  })

  it('submits only evidence references for oos and paper', async () => {
    await evaluateStrategyProfile('baseline', 1, ['a', 'b'])
    await recordPaperPeriod('baseline', 1, { screening_save_id: 7, decision_id: 'decision' })
    await activateStrategyProfile('baseline', 1)
    expect(post.mock.calls[0][1]).toEqual({ backtest_job_ids: ['a', 'b'] })
    expect(post.mock.calls[1][1]).toEqual({ screening_save_id: 7, decision_id: 'decision' })
    expect(post.mock.calls[2][0]).toBe('/strategy-profiles/baseline/1/activate')
  })
})
