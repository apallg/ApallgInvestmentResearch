import api from './index.js'

export function getPresets() {
  return api.get('/tuner/presets')
}

export function startSweep(configs, { pool = 'all', top_n = 50, code_prefixes = null } = {}) {
  return api.post('/tuner/sweep', { configs, pool, top_n, code_prefixes })
}

export function startValidate(configs, { eval_weeks = 12, top_k = 10, pool = 'all', date_start = null, date_end = null, code_prefixes = null } = {}) {
  return api.post('/tuner/validate', { configs, eval_weeks, top_k, pool, date_start, date_end, code_prefixes })
}

export function startAutoTune({
  weights, skips, tech_presets = ['all'], l0_thresholds = [0.30],
  pool = 'all', top_n = 30, eval_weeks = 12, top_k = 10,
  date_start = null, date_end = null, code_prefixes = null,
}) {
  return api.post('/tuner/auto', {
    weights, skips, tech_presets, l0_thresholds, pool, top_n, eval_weeks, top_k,
    date_start, date_end, code_prefixes,
  })
}

export function startPhase1SignalTest({
  pool = 'all', eval_weeks = 12, date_start = null, date_end = null,
  top_k = 10, code_prefixes = null, top_n = null,
  qlib_model = null, thresholds = null,
  skip_tech_subs = false, date_step = 5,
  regime = null, signal_filter = null,
}) {
  return api.post('/tuner/phase1-signal-test', {
    pool, eval_weeks, date_start, date_end, top_k, code_prefixes, top_n,
    qlib_model, thresholds, skip_tech_subs, date_step,
    regime, signal_filter,
  })
}

export function startPhase2WeightOpt({
  active_signals, pool = 'all', top_n = 30, eval_weeks = 12,
  date_start = null, date_end = null, top_k = 10, code_prefixes = null,
  regime = null,
}) {
  return api.post('/tuner/phase2-weight-opt', {
    active_signals, pool, top_n, eval_weeks, date_start, date_end, top_k, code_prefixes,
    regime,
  })
}

export function getTunerResult(jobId) {
  return api.get(`/tuner/result/${jobId}`)
}

export function generateRegimeWeights(regimeBreakdown) {
  return api.post('/tuner/generate-regime-weights', regimeBreakdown)
}

export function preloadCache(params) {
  return api.post('/tuner/preload-cache', params, { timeout: 3600000 })
}

export function startRegimeReplay({ eval_weeks = 52, date_step = 5, date_start = null, date_end = null, index_name = '沪深300' } = {}) {
  return api.post('/tuner/regime-replay', { eval_weeks, date_step, date_start, date_end, index_name })
}

export function startRegimeWalkForward({ pool = 'all', eval_weeks = 52, top_k = 10, date_start = null, date_end = null, code_prefixes = null, date_step = 5 } = {}) {
  return api.post('/tuner/regime-walk-forward', { pool, eval_weeks, top_k, date_start, date_end, code_prefixes, date_step })
}

export function getFactorPool() {
  return api.get('/tuner/factor-pool')
}

export function factorPoolAdd(signals) {
  return api.post('/tuner/factor-pool/add', signals)
}

export function factorPoolToggle(key, active) {
  return api.post('/tuner/factor-pool/toggle', { key, active })
}

export function factorPoolRemove(key) {
  return api.post('/tuner/factor-pool/remove', { key })
}

export function factorPoolDelete(key) {
  return api.delete('/tuner/factor-pool/' + key)
}

export function getTunerHistory(days = 7) {
  return api.get('/tuner/history', { params: { days } })
}
