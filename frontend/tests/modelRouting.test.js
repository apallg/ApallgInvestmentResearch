import { describe, expect, it } from 'vitest'

import { modelRefPayload, stateModelsPayload } from '../src/utils/modelRouting.js'

describe('model routing payload', () => {
  it('serializes a fixed model reference', () => {
    expect(modelRefPayload('exp/run')).toEqual(['exp', 'run'])
  })

  it('rejects an invalid fixed model reference', () => {
    expect(() => modelRefPayload('exp')).toThrow('请选择一个有效的已训练模型')
  })

  it('keeps the original complete state routing contract', () => {
    expect(stateModelsPayload({ high_vol: 'exp/rebound', low_vol: 'exp/trend' }, true)).toEqual({
      high_vol: ['exp', 'rebound'],
      low_vol: ['exp', 'trend'],
    })
  })
})
