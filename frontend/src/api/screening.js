import api from './index.js'

export function runScreening({
  pool = 'stable_history_500d', top_n = 50, debate_top = 5, debate_agents = 6,
  skip_debate = false, skip_l0 = false, skip_l1 = false, skip_l2 = false,
  skip_tech = false, skip_qlib = false, skip_sent = false,
  skip_fundamental = false, skip_flow = false,
  skip_north_flow = false, skip_size_turnover = false,
  skip_momentum = false, skip_volatility = false,
  skip_illiq = false, skip_overnight_gap = false,
  skip_wb_industry = false, skip_wb_divergence = false, skip_wb_moneyflow = false,
  wb_industry_n = 20, wb_divergence_n = 10, wb_moneyflow_n = 5,
  illiq_n = 20, overnight_gap_n = 20,
  source_weights = null, tech_preset = 'all', qlib_model = null,
  async_mode = true, hot_sector_count = 10, custom_sectors = null,
  code_prefixes = null,
  concept_sectors = null,
  regime = null,
  eval_date = null,
  macro_bias = false,
  sector_fusion = false,
  macro_regime = false,
  smooth_regime = false,
  use_vol_switch = false,
  state_models = null,
  neutralize_qlib = false,
  decision_day_filter = true,
  profile_id = null,
  profile_version = null,
} = {}) {
  return api.post('/screening/run', {
    pool, top_n, debate_top, debate_agents,
    skip_debate, skip_l0, skip_l1, skip_l2,
    skip_tech, skip_qlib, skip_sent,
    skip_fundamental, skip_flow, skip_north_flow, skip_size_turnover,
    skip_momentum, skip_volatility,
    skip_illiq, skip_overnight_gap,
    skip_wb_industry, skip_wb_divergence, skip_wb_moneyflow,
    wb_industry_n, wb_divergence_n, wb_moneyflow_n,
    illiq_n, overnight_gap_n,
    source_weights, tech_preset, qlib_model,
    async_mode, hot_sector_count, custom_sectors, code_prefixes,
    concept_sectors,
    regime,
    eval_date,
    macro_bias,
    sector_fusion,
    macro_regime,
    smooth_regime,
    use_vol_switch,
    state_models,
    neutralize_qlib,
    decision_day_filter,
    profile_id,
    profile_version,
  })
}

export function listTrainedModels() {
  return api.get('/qlib/trained-models')
}

export function getScreeningResult(jobId) {
  return api.get(`/screening/result/${jobId}`)
}

export function getLatestScreening() {
  return api.get('/screening/latest')
}

export function getScreeningHistory(days = 7) {
  return api.get('/screening/history', { params: { days } })
}

export function getHotSectors(topN = 5) {
  return api.get('/screening/hot-sectors', { params: { top_n: topN } })
}

export function getAvailableSectors() {
  return api.get('/screening/sectors')
}

export function getConceptSectors() {
  return api.get('/screening/concept-sectors')
}

export function setScreeningPredictions(predictions) {
  return api.post('/screening/set-predictions', predictions)
}
