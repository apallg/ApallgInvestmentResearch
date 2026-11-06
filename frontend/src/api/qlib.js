import api from './index'

export default {
  syncData(full)           { return api.post('/qlib/data/sync', { full_resync: full }) },
  syncBenchmarks()         { return api.post('/qlib/data/sync-benchmarks') },
  collectData(pool, retryMissing, forceTencent) {
    const body = { pool: pool || 'all' }
    if (retryMissing) body.retry_missing = true
    if (forceTencent) body.force_tencent = true
    return api.post('/qlib/data/collect', body)
  },
  collectIncremental(pool, lookbackDays, threads) {
    return api.post('/qlib/data/collect-incremental', { pool, lookback_days: lookbackDays, threads })
  },
  retryFailedIncremental(jobId, threads) {
    return api.post(`/qlib/data/collect-incremental/${jobId}/retry-failed`, { threads })
  },
  collectFull(pool, threads, delay) {
    return api.post('/qlib/data/collect-full', { pool, threads, delay })
  },
  getSyncStatus(id)        { return api.get(`/qlib/data/sync/${id}`) },
  getDataStatus()          { return api.get('/qlib/data/status') },
  getDataQuality()         { return api.get('/qlib/data/quality') },
  getDataCheck()           { return api.get('/qlib/data/check') },
  fixUnitIssues()          { return api.post('/qlib/data/fix-units') },
  validateQlibData()       { return api.get('/qlib/data/validate-qlib') },
  getModels()              { return api.get('/qlib/models') },
  trainModel(cfg)          { return api.post('/qlib/models/train', cfg) },
  getTrainStatus(id)       { return api.get(`/qlib/models/train/${id}`) },
  predict(cfg)             { return api.post('/qlib/models/predict', cfg, { timeout: 600000 }) },
  predictDual(cfg)         { return api.post('/qlib/models/predict-dual', cfg, { timeout: 600000 }) },
  getPredictStatus(jobId)  { return api.get(`/qlib/models/predict/${jobId}`) },
  runBacktest(cfg)         { return api.post('/qlib/backtest/run', cfg) },
  
  
  getTrainedModels()       { return api.get('/qlib/trained-models') },
  deleteTrainedModel(expId, runId) { return api.delete(`/qlib/trained-models/${expId}/${runId}`) },
  disableModel(expId, runId)  { return api.put(`/qlib/trained-models/${runId}/disable`, { experiment_id: expId }) },
  enableModel(expId, runId)   { return api.put(`/qlib/trained-models/${runId}/enable`, { experiment_id: expId }) },
  rollingTrain(cfg)         { return api.post('/qlib/models/rolling-train', cfg) },
  optunaSearch(cfg)         { return api.post('/qlib/models/optuna-search', cfg) },

  // 调度器状态
  getSchedulerStatus(type)  { const p = type ? (type.endsWith('/') ? type : type + '/') : ''; return api.get(`/scheduler/${p}status`) },
  startScheduler(type)      { const p = type ? (type.endsWith('/') ? type : type + '/') : ''; return api.post(`/scheduler/${p}start`) },
  stopScheduler(type)       { const p = type ? (type.endsWith('/') ? type : type + '/') : ''; return api.post(`/scheduler/${p}stop`) },
  triggerScheduler(type)    { const p = type ? (type.endsWith('/') ? type : type + '/') : ''; return api.post(`/scheduler/${p}trigger`) },
  collectFundamentals(pool) { return api.post('/qlib/data/collect-fundamentals', { pool: pool || 'all', max_workers: 4 }) },
}
