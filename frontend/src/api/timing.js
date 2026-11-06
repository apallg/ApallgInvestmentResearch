import api from './index.js'

export function getBreadth() {
  return api.get('/timing/breadth')
}

export function getBreadthHistory(start, end) {
  return api.get('/timing/breadth/history', { params: { start, end } })
}

export function trainTiming(params = {}) {
  return api.post('/timing/train', params)
}

export function getTrainStatus(jobId) {
  return api.get(`/timing/train/${jobId}`)
}

export function predictTiming(asOfDate = null) {
  return api.post('/timing/predict', { as_of_date: asOfDate })
}

export function backtestTiming(params = {}) {
  return api.post('/timing/backtest', params)
}

export function getBacktestStatus(jobId) {
  return api.get(`/timing/backtest/${jobId}`)
}

export function deleteTimingModel(modelName) {
  return api.delete(`/timing/model/${modelName}`)
}

export function predictRegime(asOfDate = null) {
  return api.post('/timing/regime/predict', { as_of_date: asOfDate })
}

export function getCurrentRegime() {
  return api.get('/timing/regime/current')
}

export function getRegimeModelInfo() {
  return api.get('/timing/regime/model-info')
}
