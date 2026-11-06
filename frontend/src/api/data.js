import api from './index'

export default {
  getOverview(force = false) {
    return api.get('/data/overview', { params: { force } })
  },
  getHistoryQuality() {
    return api.get('/data/history-quality')
  },
  getQuoteAudit(params = {}) {
    return api.get('/data/history/quote-audit', { params })
  },
  historyPreflight() {
    return api.post('/data/history/preflight')
  },
  startIndustryHistorySnapshots(payload) {
    return api.post('/data/history/industry-snapshots', payload)
  },
  getIndustryHistorySnapshots(params = {}) {
    return api.get('/data/history/industry-snapshots', { params })
  },
  cancelIndustryHistorySnapshots(jobId) {
    return api.post(`/data/history/industry-snapshots/${jobId}/cancel`)
  },
  createHistoryBaseline(start = '2019-01-01', end) {
    return api.post('/data/history/baseline', null, { params: { start, ...(end ? { end } : {}) } })
  },
  startHistoryDenominatorShadow(payload) {
    return api.post('/data/history/denominator-shadow', payload)
  },
  getHistoryDenominatorShadow(batchId) {
    return api.get(`/data/history/denominator-shadow/${batchId}`)
  },
  cancelHistoryDenominatorShadow(batchId) {
    return api.post(`/data/history/denominator-shadow/${batchId}/cancel`)
  },
  previewHistoryShadowBatch(payload) {
    return api.post('/data/history/shadow-batches/preview', payload)
  },
  startHistoryShadowBatch(batchId) {
    return api.post(`/data/history/shadow-batches/${batchId}/start`)
  },
  getHistoryShadowBatch(batchId) {
    return api.get(`/data/history/shadow-batches/${batchId}`)
  },
  getHistoryShadowConflicts(batchId, code, year, limit = 500, denominatorBatchId) {
    return api.get(`/data/history/shadow-batches/${batchId}/${code}/${year}/conflicts`, {
      params: { limit, ...(denominatorBatchId ? { denominator_batch_id: denominatorBatchId } : {}) },
    })
  },
  cancelHistoryShadowBatch(batchId) {
    return api.post(`/data/history/shadow-batches/${batchId}/cancel`)
  },
  prepareHistoryRelease(batchId, code, year, denominatorBatchId, replaceConflicts = false) {
    return api.post(`/data/history/shadow-batches/${batchId}/${code}/${year}/release-preview`, {
      denominator_batch_id: denominatorBatchId,
      replace_conflicts: replaceConflicts,
    })
  },
  getHistoryRelease(releaseId) {
    return api.get(`/data/history/releases/${releaseId}`)
  },
  publishHistoryRelease(releaseId) {
    return api.post(`/data/history/releases/${releaseId}/publish`)
  },
  rollbackHistoryRelease(releaseId) {
    return api.post(`/data/history/releases/${releaseId}/rollback`)
  },
  repairTable(table) {
    return api.post(`/data/repair/${table}`)
  },
  repairAll(previewId) {
    return api.post('/data/repair-all', previewId ? { preview_id: previewId } : {})
  },
  repairPreview() {
    return api.post('/data/repair-preview')
  },
  getJobs(limit = 50) {
    return api.get('/data/jobs', { params: { limit } })
  },
  getRepairStatus(jobId) {
    return api.get(`/system/tasks/${jobId}`)
  },
  collectContext(kind) {
    return api.post(`/data/context/collect/${kind}`)
  },
}
