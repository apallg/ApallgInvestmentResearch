<template>
  <div>
    <h1 class="page-title">持仓管理</h1>
    <div v-if="error" class="error">{{ error }}</div>

    <!-- 账户总览 -->
    <div class="grid-4 mb-2">
      <MetricCard :value="formatMoney(account.available)" label="券商可用资金" />
      <MetricCard :value="formatMoney(account.market_value)" label="券商持仓市值" />
      <MetricCard :value="formatMoney(account.total_assets)" label="券商总资产" />
      <MetricCard :value="formatMoney(account.unrealized_pnl)" label="券商未实现盈亏" :color="account.unrealized_pnl == null ? '' : account.unrealized_pnl >= 0 ? 'up' : 'down'" />
    </div>
    <div class="snapshot-state">账户：{{ accountStatusLabel }} · {{ account.account_id || brokerLabel || '来源未知' }} · 观测时间 {{ account.as_of || '--' }}<span v-if="accountFetchError"> · {{ accountFetchError }}</span></div>
    <div class="snapshot-state">本地账本：{{ localAccount?.status === 'active' ? '已激活，持仓按成交事务更新' : localAccount?.status === 'uninitialized' ? '期初待核准，可维护旧持仓快照' : '状态不可用，持仓编辑已锁定' }}</div>

    <!-- 资金管理 -->
    <div class="card" style="margin-bottom:16px;">
      <div class="card-title">
        资金管理
        <span v-if="!cashEnabled" style="font-size:11px;color:var(--danger);margin-left:8px;">{{ brokerLabel }} · 不可操作</span>
      </div>
      <div v-if="cashEnabled" style="display:flex;gap:8px;align-items:flex-end;flex-wrap:wrap;">
        <div class="flex" style="flex-direction:column;">
          <label style="font-size:11px;color:var(--text-muted);margin-bottom:4px;">金额</label>
          <input v-model.number="cashAmount" type="number" min="0" step="1000" style="width:150px;" placeholder="金额" />
        </div>
        <button class="btn btn-success" @click="addCash">充值</button>
        <button class="btn btn-warning" @click="setCash">设为</button>
        <button class="btn btn-danger" @click="withdrawCash">出金</button>
      </div>
      <div style="font-size:12px;color:var(--text-muted);margin-top:8px;">
        现金: {{ formatMoney(account.cash) }} &nbsp; 冻结: {{ formatMoney(account.frozen) }}
        <span v-if="!cashEnabled" style="margin-left:8px;color:var(--danger);">(来自 {{ brokerLabel }}，不可修改)</span>
      </div>
    </div>

    <!-- 持仓切换 -->
    <div class="tab-bar flex gap-4 mb-2">
      <button :style="tabStyle('list')" @click="tab='list'">持仓列表</button>
      <button v-if="localAccount?.status === 'uninitialized'" :style="tabStyle('add')" @click="tab='add'">添加持仓</button>
    </div>

    <template v-if="tab==='list'">
      <div v-if="holdings.length" class="snapshot-state">
        本地持仓估值：{{ pricedCount }}/{{ holdings.length }} 只有效报价（{{ coveragePct }}） · 缺价 {{ missingCodes.length ? missingCodes.join('、') : '无' }} · 来源 {{ priceMeta.source || '--' }} · 最新报价观测 {{ priceMeta.observed_at || '--' }}
        <span v-if="priceFetchError"> · {{ priceFetchError }} · 当前估值不可用</span>
      </div>
      <div v-if="holdings.length" class="card">
        <table><thead>
            <tr><th>代码</th><th>名称</th><th>持股数</th><th>成本价</th><th>当前价</th><th>持仓市值</th><th>买入日期</th><th>操作</th></tr>
          </thead>
        <tbody>
            <tr v-for="h in holdings" :key="h.code">
            <template v-if="editingCode !== h.code">
              <td>{{ h.code }}</td>
              <td>{{ h.name }}</td>
              <td>{{ h.shares }}</td>
              <td>{{ Number(h.cost_price).toFixed(2) }}</td>
              <td :class="priceColor(h.code)" :title="quoteDetail(h.code)">{{ formatPrice(h.code) }}</td>
              <td>{{ formatMarketValue(h) }}</td>
              <td>{{ h.buy_date || '--' }}</td>
              <td style="display:flex;gap:6px;">
                <button class="btn btn-sm" @click="startEdit(h)">编辑</button>
                <button v-if="localAccount?.status === 'uninitialized'" class="btn btn-sm btn-danger" @click="del(h.code)">删除</button>
              </td>
            </template>
            <template v-else>
              <td>{{ h.code }}</td>
              <td><input v-model="editForm.name" style="width:80px" /></td>
              <td><input v-model.number="editForm.shares" type="number" min="1" style="width:80px" :disabled="localAccount?.status === 'active'" /></td>
              <td><input v-model="editForm.cost_price" type="text" style="width:80px" :disabled="localAccount?.status === 'active'" /></td>
              <td>{{ formatPrice(h.code) }}</td>
              <td>{{ formatEditMarketValue() }}</td>
              <td><input v-model="editForm.buy_date" type="date" style="width:110px" :disabled="localAccount?.status === 'active'" /></td>
              <td style="display:flex;gap:6px;">
                <button class="btn btn-sm btn-success" @click="saveEdit">保存</button>
                <button class="btn btn-sm" style="background:var(--text-muted)" @click="cancelEdit">取消</button>
              </td>
            </template>
          </tr>
            </tbody>
          <tfoot>
            <tr style="font-weight:bold;border-top:2px solid var(--border);">
              <td colspan="5" style="text-align:right;">{{ completeValuation ? '本地持仓总市值' : `未确认估值 (${pricedCount}/${holdings.length})` }}</td>
              <td>{{ completeValuation ? formatMoney(totalMarketValue) : '--' }}</td>
              <td colspan="2"></td>
            </tr>
          </tfoot>
        </table>
      </div>
      <div v-else-if="holdingsLoadError" class="card empty" role="alert">
        <p style="margin-bottom:10px">持仓快照不可用：{{ holdingsLoadError }}。这不代表账户没有持仓。</p>
      </div>
      <div v-else class="card empty">
        <p style="margin-bottom:10px">还没有持仓</p>
        <router-link to="/screening" class="btn btn-sm">去选股筛选候选股 →</router-link>
      </div>
    </template>

    <div v-if="tab==='add' && localAccount?.status === 'uninitialized'" class="card" style="max-width:400px">
      <div class="card-title">新增持仓</div>
      <div class="flex mb-2">
        <label style="width:80px">股票代码</label>
        <input v-model="form.code" placeholder="6位数字 如 600519" style="flex:1" maxlength="6" @blur="fetchStockName" />
      </div>
      <div class="flex mb-2"><label style="width:80px">名称</label><input v-model="form.name" placeholder="输入代码后自动识别" style="flex:1" readonly /></div>
      <div class="flex mb-2"><label style="width:80px">数量</label><input v-model.number="form.shares" type="number" min="1" style="flex:1" /></div>
      <div class="flex mb-2"><label style="width:80px">成本价</label><input v-model="form.cost_price" type="text" style="flex:1" placeholder="如 100.50" /></div>
      <div class="flex mb-2"><label style="width:80px">买入日期</label><input v-model="form.buy_date" type="date" style="flex:1" /></div>
      <button class="btn w-full mt-2" @click="addHolding">添加</button>
    </div>
  </div>
</template>
<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import api from '../api/index.js'
import MetricCard from '../components/MetricCard.vue'
import { useBroker } from '../composables/useBroker.js'
import { useHoldingsStore } from '../stores/holdings.js'
import { todayStr } from '../utils/date.js'

const { cashEnabled, brokerLabel, loadBroker } = useBroker()
const holdingsStore = useHoldingsStore()
const holdingsLoadError = computed(() => holdingsStore.loadError.value)

const tab = ref('list')
const holdings = holdingsStore.holdings
const account = ref({})
const localAccount = ref(null)
const accountFetchError = ref('')
const cashAmount = ref(10000)
const error = ref('')
const form = ref({ code: '', name: '', shares: 100, cost_price: 100, buy_date: todayStr() })

const editingCode = ref(null)
const editForm = ref({})
const prices = ref({})
const priceMeta = ref({})
const priceFetchError = ref('')
const nowTick = ref(Date.now())
let refreshTimer
let priceRequestGeneration = 0

const accountStatusLabel = computed(() => ({ fresh: '可用', stale: '已过期', degraded: '部分可用', unavailable: '不可用' })[account.value.status] || '不可用')
const pricedCount = computed(() => holdings.value.filter(h => Number(prices.value[h.code]?.price) > 0).length)
const missingCodes = computed(() => holdings.value.filter(h => !(Number(prices.value[h.code]?.price) > 0)).map(h => h.code))
const coveragePct = computed(() => holdings.value.length ? `${(pricedCount.value / holdings.value.length * 100).toFixed(1)}%` : '--')
const completeValuation = computed(() => {
  const expected = Number(priceMeta.value.expected_count)
  const accepted = Number(priceMeta.value.priced_count)
  return !priceFetchError.value &&
    priceMeta.value.freshness === 'fresh' &&
    priceMeta.value.data_envelope?.status === 'available' &&
    Number.isInteger(expected) && expected === holdings.value.length &&
    Number.isInteger(accepted) && accepted === expected &&
    pricedCount.value === expected
})

function formatMoney(value) {
  return value == null || !Number.isFinite(Number(value)) ? '--' : `¥${Number(value).toFixed(2)}`
}

function quoteDetail(code) {
  const quote = prices.value[code]
  if (!quote) return '无有效报价'
  const observed = quote.observed_at ? new Date(quote.observed_at.replace(' ', 'T')).getTime() : NaN
  const age = Number.isFinite(observed) ? `${Math.max(0, Math.floor((nowTick.value - observed) / 60_000))} 分钟前` : '时间未知'
  return `${quote.source || '来源未知'} · ${quote.observed_at || '观测时间未知'} · ${age}`
}

const totalMarketValue = computed(() => {
  return holdings.value.reduce((sum, h) => {
    const p = prices.value[h.code]
    if (p && p.price) return sum + p.price * h.shares
    return sum
  }, 0)
})

function formatPrice(code) {
  const p = prices.value[code]
  if (Number(p?.price) > 0) return Number(p.price).toFixed(2)
  return '--'
}

function priceColor(code) {
  const p = prices.value[code]
  if (p && p.pct_chg > 0) return 'color-up'
  if (p && p.pct_chg < 0) return 'color-down'
  return ''
}

function formatMarketValue(h) {
  const p = prices.value[h.code]
  if (p && p.price) return '¥' + (p.price * h.shares).toFixed(2)
  return '--'
}

function formatEditMarketValue() {
  const shares = Number(editForm.value.shares) || 0
  const p = prices.value[editForm.value.code]
  if (p && p.price) return '¥' + (p.price * shares).toFixed(2)
  return '--'
}

const tabStyle = (name) => tab.value === name
  ? 'background:var(--primary);color:var(--on-primary);border:none;border-radius:6px;padding:8px 20px;cursor:pointer;font-size:14px;font-weight:600;'
  : 'background:var(--card-bg);border:1px solid var(--border);color:var(--text-secondary);border-radius:6px;padding:8px 20px;cursor:pointer;font-size:14px;'

async function loadAccount() {
  try {
    const r = await api.get('/trading/account')
    account.value = r.data || {}
    accountFetchError.value = ''
  } catch (e) {
    account.value = {}
    accountFetchError.value = '读取失败'
    error.value = e.message
  }
}
async function loadLocalAccount() {
  try {
    const r = await api.get('/rebalance/account/local')
    localAccount.value = r?.data ?? r
    if (localAccount.value?.status !== 'uninitialized') tab.value = 'list'
  } catch (e) {
    localAccount.value = null
    tab.value = 'list'
  }
}
async function load() {
  try { await holdingsStore.load(true) }
  catch (e) { error.value = e.message }
}
async function loadPrices() {
  const requestGeneration = ++priceRequestGeneration
  if (!holdings.value.length) {
    prices.value = {}
    priceMeta.value = {}
    priceFetchError.value = ''
    return
  }
  const codes = holdings.value.map(h => h.code)
  try {
    const r = await api.post('/portfolio/holdings/prices', { codes })
    if (requestGeneration !== priceRequestGeneration) return
    prices.value = r.data || {}
    priceMeta.value = r.meta || {}
    priceFetchError.value = ''
  } catch (e) {
    if (requestGeneration !== priceRequestGeneration) return
    prices.value = {}
    priceMeta.value = {}
    priceFetchError.value = e.message || '读取失败'
  }
}
function isValidCode(code) { return /^\d{6}$/.test(code) }

async function fetchStockName() {
  const code = form.value.code.trim()
  if (!isValidCode(code)) { form.value.name = ''; return }
  try {
    const r = await api.get(`/stock/${code}`)
    form.value.name = r.name || ''
  } catch { form.value.name = '' }
}

async function addHolding() {
  if (localAccount.value?.status !== 'uninitialized') { error.value = '本地账本状态不允许新增持仓'; return }
  form.value.code = form.value.code.trim()
  if (!isValidCode(form.value.code)) { error.value = '股票代码必须为6位数字'; return }
  if (!form.value.name) { error.value = '无法识别该代码，请确认代码正确'; return }
  try {
    await holdingsStore.add(form.value)
    form.value = { code: '', name: '', shares: 100, cost_price: 100, buy_date: todayStr() }
    await loadPrices(); tab.value = 'list'
  } catch (e) { error.value = e.message }
}
async function del(code) {
  if (localAccount.value?.status !== 'uninitialized') { error.value = '本地账本状态不允许删除持仓'; return }
  if (!confirm('确认删除本地持仓记录？此操作不会向券商下单，也不会记录卖出成交。')) return
  const holding = holdings.value.find(h => h.code === code)
  if (!holding) { error.value = '持仓已变化，请刷新后重试'; return }
  const reason = prompt('请填写删除本地持仓记录的原因（至少8字）')?.trim()
  if (!reason) return
  if (reason.length < 8) { error.value = '原因至少需要8字'; return }
  try {
    await holdingsStore.remove(code, {
      reason, expected_shares: holding.shares,
      expected_cost_price: holding.cost_price,
      expected_buy_date: holding.buy_date || '',
      expected_id: holding.id,
      expected_version: holding.version,
    })
    await loadPrices()
  }
  catch (e) { error.value = e.message }
}
function startEdit(h) {
  editingCode.value = h.code
  editForm.value = {
    code: h.code,
    name: h.name,
    shares: h.shares,
    cost_price: Number(h.cost_price),
    buy_date: h.buy_date || '',
  }
}
function cancelEdit() {
  editingCode.value = null
  editForm.value = {}
}
async function saveEdit() {
  if (!editForm.value.name) { error.value = '名称不能为空'; return }
  if (!localAccount.value) { error.value = '本地账本状态不可用，请刷新后重试'; return }
  try {
    const { code, name, shares, cost_price, buy_date } = editForm.value
    const original = holdings.value.find(h => h.code === code)
    if (!original) { error.value = '持仓已变化，请刷新后重试'; return }
    const positionChanged = Number(shares) !== Number(original.shares) ||
      Number(cost_price) !== Number(original.cost_price) || buy_date !== (original.buy_date || '')
    if (localAccount.value.status === 'active' && positionChanged) {
      error.value = '实盘账户已激活，持仓数量、成本和日期只能通过成交事务变更'
      return
    }
    if (localAccount.value.status !== 'active' && localAccount.value.status !== 'uninitialized') {
      error.value = '本地账本状态不可用，请刷新后重试'
      return
    }
    const reason = positionChanged ? prompt('请填写手工调整持仓数量、成本或日期的原因（至少8字）')?.trim() : undefined
    if (positionChanged && !reason) return
    if (positionChanged && reason.length < 8) { error.value = '原因至少需要8字'; return }
    const data = { name, expected_id: original.id, expected_version: original.version }
    if (localAccount.value.status === 'uninitialized') {
      Object.assign(data, {
        shares: Number(shares), cost_price: String(cost_price), buy_date,
        expected_shares: original.shares, expected_cost_price: original.cost_price,
        expected_buy_date: original.buy_date || '', reason,
      })
    }
    await holdingsStore.update(code, data)
    editingCode.value = null
    editForm.value = {}
  } catch (e) {
    if (Number(e?.status ?? e?.response?.status) === 409) {
      cancelEdit()
      error.value = `${e.message || '持仓版本已变化'}；已重新读取当前持仓，请重新打开编辑后再提交`
      return
    }
    error.value = e.message
  }
}

async function addCash() {
  if (!cashAmount.value || cashAmount.value <= 0) { error.value = '请输入有效金额'; return }
  try { await api.post('/trading/account/cash', { amount: cashAmount.value }); await loadAccount(); error.value = '' }
  catch (e) { error.value = e.message }
}
async function setCash() {
  if (cashAmount.value < 0) { error.value = '金额不能为负'; return }
  try { await api.put('/trading/account/cash', { amount: cashAmount.value }); await loadAccount(); error.value = '' }
  catch (e) { error.value = e.message }
}
async function withdrawCash() {
  if (!cashAmount.value || cashAmount.value <= 0) { error.value = '请输入有效金额'; return }
  try { await api.post('/trading/account/withdraw', { amount: cashAmount.value }); await loadAccount(); error.value = '' }
  catch (e) { error.value = e.message }
}

onMounted(async () => {
  await loadBroker()
  await loadAccount()
  await loadLocalAccount()
  await load()
  await loadPrices()
  refreshTimer = setInterval(() => {
    nowTick.value = Date.now()
    loadAccount()
    loadLocalAccount()
    loadPrices()
  }, 60_000)
})
onUnmounted(() => {
  priceRequestGeneration += 1
  clearInterval(refreshTimer)
})
</script>

<style scoped>
.card:not(.metric) { margin: 16px 32px; }
.page-title { padding: 0 32px; }
.grid-4 { margin-left: 32px; margin-right: 32px; }
.tab-bar { margin-left: 32px; margin-right: 32px; }
.error { margin-left: 32px; margin-right: 32px; }
.snapshot-state { margin: 0 32px 12px; color: var(--text-muted); font-size: 12px; }
.color-up { color: var(--up); }
.color-down { color: var(--down); }
tfoot td { padding: 8px 10px; background: var(--card-bg); }
</style>
