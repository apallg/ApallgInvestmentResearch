import api from './index.js'

export function saveScreening(resultId, saveName, note = '') {
  return api.post(`/screening/${resultId}/save`, { save_name: saveName, note })
}

export function listSavedScreenings(days = 30, pool = null) {
  return api.get('/screening/saves', { params: { days, pool } })
}

export function getSavedScreening(saveId) {
  return api.get(`/screening/saves/${saveId}`)
}

export function deleteSavedScreening(saveId) {
  return api.delete(`/screening/saves/${saveId}`)
}

export function runDailyCheck(params = {}) {
  return api.post('/rebalance/daily-check', params)
}

export function runRebalance(screeningSaveId, params = {}) {
  return api.post('/rebalance/run', { screening_save_id: screeningSaveId, ...params, preview_only: true }, { timeout: 600000 })
}

export function commitRebalance(screeningSaveId, params = {}) {
  return api.post('/rebalance/run', { screening_save_id: screeningSaveId, ...params, preview_only: false }, { timeout: 600000 })
}

export function getLocalAccount() {
  return api.get('/rebalance/account/local')
}

export function activateLocalAccount(openingCash, holdingsFingerprint, reason) {
  return api.post('/rebalance/account/local/activate', {
    opening_cash: openingCash, holdings_fingerprint: holdingsFingerprint, reason,
  })
}

export function recordHistoricalFill(payload) {
  return api.post('/rebalance/account/local/historical-fills', payload)
}

export function getLocalReconciliation() {
  return api.get('/rebalance/account/local/reconciliation')
}

export function getUnresolvedBrokerTrades(limit = 500) {
  return api.get('/rebalance/account/local/broker-trades/unresolved', { params: { limit } })
}

export function getUnresolvedBrokerOrders(limit = 500) {
  return api.get('/rebalance/account/local/broker-reports/unresolved', { params: { limit } })
}

export function getLocalQuarantine(limit = 50, offset = 0) {
  return api.get('/rebalance/account/local/quarantine', { params: { limit, offset } })
}

export function bindLocalBroker(expectedAccountVersion) {
  return api.post('/rebalance/account/local/bind-broker', {
    expected_account_version: expectedAccountVersion,
  })
}

export function listLocalIntents(limit = 100) {
  return api.get('/rebalance/account/local/intents', { params: { limit } })
}

export function prepareLocalIntent(payload) {
  return api.post('/rebalance/account/local/intents', payload)
}

export function cancelLocalIntent(intentId) {
  return api.post(`/rebalance/account/local/intents/${intentId}/cancel`)
}

export function attachLocalBrokerOrder(intentId, brokerOrderId) {
  return api.post(`/rebalance/account/local/intents/${intentId}/attach-order`, { broker_order_id: brokerOrderId })
}

export function getLocalBrokerEvidence(intentId) {
  return api.get(`/rebalance/account/local/intents/${intentId}/broker-evidence`)
}

export function recordLocalBrokerFill(intentId, payload) {
  return api.post(`/rebalance/account/local/intents/${intentId}/broker-fill`, payload)
}

export function finalizeLocalBrokerOrder(intentId) {
  return api.post(`/rebalance/account/local/intents/${intentId}/finalize`)
}

export function previewDecision(screeningSaveId, params = {}) {
  return api.post('/rebalance/decision-preview', { screening_save_id: screeningSaveId, ...params })
}

export function getDecision(decisionId) {
  return api.get(`/rebalance/decisions/${decisionId}`)
}

export function getPendingSignals(checkType = null, limit = 200, offset = 0) {
  return api.get('/rebalance/signals/pending', { params: { check_type: checkType, limit, offset } })
}

export function getSignal(signalId) {
  return api.get(`/rebalance/signals/${signalId}`)
}

export function getBatchSignals(batchId, executionStatus = null) {
  return api.get(`/rebalance/batch/${batchId}`, { params: { execution_status: executionStatus } })
}

export function executeSignal(signalId, detail = {}) {
  return api.put(`/rebalance/signals/${signalId}/execute`, detail)
}

export function closeSignalRemainder(signalId, expectedAccountVersion, reason) {
  return api.post(`/rebalance/signals/${signalId}/close-remainder`, {
    expected_account_version: expectedAccountVersion, reason,
  })
}

export function skipSignal(signalId, reason = '') {
  return api.put(`/rebalance/signals/${signalId}/skip`, { reason })
}

export function batchExecuteSignals(signalDetails) {
  return api.put('/rebalance/signals/batch-execute', { signal_ids: signalDetails })
}

export function getHistory(days = 30, limit = 50) {
  return api.get('/rebalance/history', { params: { days, limit } })
}

export function listSessions(days = 90, limit = 30) {
  return api.get('/rebalance/sessions', { params: { days, limit } })
}
export function getSession(batchId) {
  return api.get(`/rebalance/sessions/${batchId}`)
}

export function getCash() {
  return api.get('/rebalance/cash')
}

export function setCash(cash) {
  return api.put('/rebalance/cash', cash)
}

export function getCooldown() {
  return api.get('/rebalance/cooldown')
}

export function checkIntraday(codes, maData = null, sellSignals = null) {
  return api.post('/rebalance/intraday-check', { codes, ma_data: maData, sell_signals: sellSignals })
}

export function previewStocks(codes, stopMode = 'fixed') {
  return api.post('/rebalance/preview', { codes, stop_mode: stopMode })
}

export function checkT0(codes) {
  return api.post('/rebalance/t0-check', { codes })
}
