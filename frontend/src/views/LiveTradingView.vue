<template>
  <div class="ltv-root">
    <StrategyProfileBanner :profile="selectedSnapshotProfile" title="实盘策略版本" />
    <div v-if="!strategyControlEnabled" class="ltv-unmanaged-warning" role="alert">
      实盘策略版本管控已关闭：当前页面会话可使用非 Champion 快照生成并记录新增买入，其他行情、风控和成交校验仍然生效。
    </div>
    <div v-if="executionBlockReason" class="error">新增买入已阻断：{{ executionBlockReason }}</div>
    <div v-if="error" class="error">{{ error }}</div>
    <div v-if="holdingsLoadError" class="ltv-unmanaged-warning" role="alert">
      本地持仓快照不可用：{{ holdingsLoadError }}。当前持仓状态未知，不能按空仓继续操作。
    </div>

    <section class="ltv-config" aria-labelledby="rebalance-config-title">
      <div class="ltv-stage__header ltv-config__header">
        <div class="ltv-stage__identity">
          <span class="ltv-stage__number">1</span>
          <div>
            <h3 id="rebalance-config-title">组合输入与约束</h3>
            <p>先确定选股快照，再设置组合规模、权重与风控口径。</p>
          </div>
        </div>
        <span class="ltv-plan-status" :class="{ 'is-stale': planDirty, 'is-ready': planCalculated && !planDirty }">
          {{ planStatusText }}
        </span>
      </div>

      <div class="ltv-config__grid">
        <div class="ltv-config__group ltv-config__group--source">
          <strong>数据来源</strong>
          <div class="ltv-field ltv-field--wide">
            <label for="rebalance-screening">保存的选股结果</label>
            <select id="rebalance-screening" v-model="selectedSaveId">
              <option :value="null">选择一份保存的选股</option>
              <option v-for="s in savedScreenings" :key="s.id" :value="s.id">{{ s.save_name }}（{{ s.eval_date }} · {{ s.pool === 'a500' ? '稳定历史池500D' : s.pool }} · {{ s.profile_id ? `${s.profile_id}/第${s.profile_version}版` : '旧版' }}）</option>
            </select>
            <span class="ltv-source-meta">{{ modelRoutingLabel }}</span>
          </div>
          <div class="ltv-strategy-control" :class="{ 'is-unmanaged': !strategyControlEnabled }">
            <label for="rebalance-strategy-control">
              <input
                id="rebalance-strategy-control"
                type="checkbox"
                :checked="strategyControlEnabled"
                aria-describedby="rebalance-strategy-control-help"
                @change="onStrategyControlChange"
              />
              <span>实盘策略版本管控</span>
            </label>
            <small id="rebalance-strategy-control-help">
              {{ strategyControlEnabled ? '开启：仅当前 Champion 可生成新增买入' : '已关闭：刷新页面后自动恢复开启' }}
            </small>
            <label v-if="!strategyControlEnabled" class="ltv-takeover-reason" for="rebalance-takeover-reason">
              <span>接管理由</span>
              <input id="rebalance-takeover-reason" v-model.trim="manualTakeoverReason" type="text" minlength="8" maxlength="200" placeholder="说明本次绕过 Champion 准入的原因" />
            </label>
          </div>
        </div>

        <div class="ltv-config__group">
          <strong>组合构建</strong>
          <div class="ltv-config__fields">
            <div class="ltv-field ltv-field--sm">
              <label for="rebalance-top-n">持仓上限</label>
              <input id="rebalance-top-n" v-model.number="topN" type="number" min="3" max="30" />
            </div>
            <div class="ltv-field">
              <label for="rebalance-sizing">目标权重</label>
              <span id="rebalance-sizing" class="ltv-field__static">等权</span>
            </div>
            <div class="ltv-field ltv-field--decision">
              <label>候选排序</label>
              <span class="ltv-field__static">模型原始排序</span>
            </div>
          </div>
        </div>

        <div class="ltv-config__group">
          <strong>卖出与冷冻</strong>
          <div class="ltv-config__fields">
            <div class="ltv-field ltv-field--sm">
              <label for="rebalance-stop-cooldown" title="止损清仓后 N 周内不再买入该股票">止损冷冻</label>
              <div class="ltv-input-unit"><input id="rebalance-stop-cooldown" v-model.number="cooldownStopWeeks" type="number" min="1" max="12" placeholder="2" /><span>周</span></div>
            </div>
            <div class="ltv-field ltv-field--sm">
              <label for="rebalance-profit-cooldown" title="止盈清仓后 N 周内不再买入该股票">止盈冷冻</label>
              <div class="ltv-input-unit"><input id="rebalance-profit-cooldown" v-model.number="cooldownProfitWeeks" type="number" min="1" max="12" placeholder="1" /><span>周</span></div>
            </div>
          </div>
          <div class="ltv-mode-switch" role="group" aria-label="止盈止损模式">
            <button v-for="m in STOP_MODES" :key="m" type="button"
                    :class="{ 'ltv-mode--active': stopMode === m }"
                    :aria-pressed="stopMode === m"
                    @click="stopMode = m; refreshPreview()">{{ STOP_LABELS[m] }}</button>
          </div>
        </div>
      </div>

      <div class="ltv-config__actions">
        <div class="ltv-config__feedback" aria-live="polite">
          <span v-if="loadingCandidates">正在加载候选股票…</span>
          <span v-else-if="loadingPreview">正在更新个股行情…</span>
          <span v-else>当前勾选 {{ checkedCount }} 只，最终组合最多 {{ topN }} 只</span>
        </div>
        <button class="btn-sm btn-outline" type="button" @click="loadSavedScreenings" :disabled="loadingCandidates || loadingPreview">刷新快照列表</button>
        <button class="btn-sm ltv-primary-action" type="button" @click="runRebalanceAction" :disabled="!selectedSaveId || !checkedCodes.length || runningRebalance || pendingPlanCommit || pendingPlanCommitError || uncertainFillRequests.length || uncertainFillRequestsError || (planCalculated && !planDirty && planCommitBlocked && !planCommitted)">
          {{ planActionLabel }}
        </button>
      </div>
      <div v-if="pendingPlanCommit" class="ltv-opening ltv-intent-form" role="alert">
        <strong>上次调仓提交结果尚未确认</strong>
        <span class="ltv-source-meta">选股快照 {{ pendingPlanCommit.screening_save_id }} · 请求 {{ pendingPlanCommit.payload.commit_request_id }}。请用原请求ID核对/重试；完成前不能创建新方案。</span>
        <button class="btn-sm" type="button" :disabled="runningRebalance" @click="retryPendingPlanCommit">用原请求ID核对并重试</button>
      </div>
      <div v-else-if="pendingPlanCommitError" class="error" role="alert">{{ pendingPlanCommitError }} 新提交已停止，请先人工核对调仓信号历史。</div>
    </section>

    <section v-if="uncertainFillRequests.length || uncertainFillRequestsError" class="ltv-opening ltv-intent-form" role="alert">
      <strong>存在尚未确认的成交请求，已暂停新调仓方案</strong>
      <span v-if="uncertainFillRequestsError" class="error">{{ uncertainFillRequestsError }} 请核对账本和信号历史后处理。</span>
      <div v-for="request in uncertainFillRequests" :key="request.signalId" class="ltv-intent-actions">
        <span>信号 #{{ request.signalId }} · 请求 {{ request.payload.request_id }} · {{ request.payload.shares }} 股 @ {{ request.payload.price }}</span>
        <button class="btn-sm" type="button" :disabled="recoveringFillSignalId !== null || showExecutionModal" @click="retryUncertainFill(request)">{{ recoveringFillSignalId === request.signalId ? '正在核对…' : '读取信号并用原请求ID核对' }}</button>
      </div>
    </section>


    <section class="ltv-ledger" aria-label="本地账本与券商对账">
      <div><strong>本地权威可用余额：</strong>{{ localLedgerAvailableCash }}</div>
<div><strong>人工预算设置：</strong>{{ manualBudget == null ? '读取失败' : `¥${Number(manualBudget).toFixed(2)}` }} <span class="ltv-source-meta">独立预算，不作为本地账本余额或实盘下单资金</span></div>
      <div><strong>券商对账：</strong>{{ reconciliation?.can_execute ? (reconciliation.sellable_missing_codes?.length ? '已绑定账户快照一致；对应卖出委托因可卖股数缺失会阻断，真实委托仍关闭' : '已绑定账户快照一致；真实委托仍关闭') : reconciliation?.can_bind ? '对账一致；请核对券商账号后完成绑定，真实委托仍关闭' : (reconciliation?.reason || '尚未读取') }}</div>
      <div v-if="invalidatedSignalDetails.length" class="ltv-reconciliation-detail" role="status">
        最近一次账本变更后因账户版本变化失效的计划：
        <span v-for="item in invalidatedSignalDetails" :key="item.signal_id" class="ltv-reconciliation-item">
          {{ item.code || '未知证券' }} #{{ item.signal_id }} · 原计划 {{ item.planned_shares }} 股 / 已成交 {{ item.filled_shares }} 股 / 失效余量 {{ item.invalidated_remaining_shares }} 股
        </span>
      </div>
      <div v-if="reconciliation?.broker_account_id || reconciliation?.broker_as_of" class="ltv-source-meta">
        券商账户 {{ reconciliation.broker_account_id || '未知' }}<span v-if="reconciliation.broker_as_of"> · 快照 {{ reconciliation.broker_as_of }}</span>
      </div>
      <div v-if="reconciliation?.broker_cash != null" class="ltv-source-meta">
        本地总现金 {{ reconciliation.local_cash == null ? '待核准' : `¥${Number(reconciliation.local_cash).toFixed(2)}` }} / 冻结 {{ reconciliation.local_frozen == null ? '待核准' : `¥${Number(reconciliation.local_frozen).toFixed(2)}` }}；
        券商总现金 ¥{{ Number(reconciliation.broker_cash).toFixed(2) }} / 冻结 ¥{{ Number(reconciliation.broker_frozen).toFixed(2) }} / 可用 ¥{{ Number(reconciliation.broker_available).toFixed(2) }}
      </div>
      <div v-if="reconciliation?.cash_difference != null" class="ltv-reconciliation-detail">
        资金差额（券商－本地）：总现金 ¥{{ Number(reconciliation.cash_balance_difference).toFixed(2) }} / 冻结 ¥{{ Number(reconciliation.frozen_difference).toFixed(2) }} / 可用 ¥{{ Number(reconciliation.cash_difference).toFixed(2) }}
      </div>
      <div class="ltv-source-meta">本地绑定账号：{{ !localAccount ? '账本不可用' : (localAccount.broker_account_id || '尚未绑定') }} · 冻结资金 {{ localAccount?.frozen == null ? '未知' : `¥${Number(localAccount.frozen).toFixed(2)}` }}</div>
      <div v-if="reconciliation?.pending_local_intent" class="ltv-source-meta">存在待处理的本地委托意图，已暂停新增计划及人工成交。</div>
      <div v-if="localAccount?.broker_account_id" class="ltv-source-meta">已绑定券商账户的成交须经本地委托意图与券商成交回报核验后入账；普通人工成交入口已关闭。</div>
      <div v-if="reconciliation?.position_differences?.length" class="ltv-reconciliation-detail">
        持仓差异：
        <span v-for="item in reconciliation.position_differences" :key="item.code" class="ltv-reconciliation-item">
          {{ item.code }}（本地 {{ item.local_shares == null ? '不可核验' : item.local_shares }} 股 / 券商 {{ item.broker_shares }} 股）
        </span>
      </div>
      <div v-if="reconciliation?.sellable_missing_codes?.length" class="ltv-reconciliation-detail">
        可卖股数缺项：{{ reconciliation.sellable_missing_codes.join('、') }}（这些证券的卖出委托不可准备）
      </div>
      <div v-if="reconciliationMissingItems.length" class="ltv-reconciliation-detail">
        对账缺项：{{ reconciliationMissingItems.join('、') }}
      </div>
      <div v-if="reconciliation?.broker_trade_report_health" class="ltv-reconciliation-detail">
        成交回报：未解决冲突 {{ reconciliation.broker_trade_report_health.conflicts }} 条 / 费用待核 {{ reconciliation.broker_trade_report_health.unpriced }} 笔 / 关联委托未入账 {{ reconciliation.broker_trade_report_health.unbooked }} 笔 / 期初后无本地委托归属（阻断） {{ reconciliation.broker_trade_report_health.unassigned }} 笔 / 期初前隔离 {{ reconciliation.broker_trade_report_health.legacy_quarantined || 0 }} 笔
      </div>
      <div v-if="showBrokerReportArchive && reconciliation?.broker_trade_report_health" class="ltv-broker-trades" role="region" aria-label="待核对券商成交回报">
        <strong>本地保存的待核对成交与期初前隔离记录</strong>
        <span class="ltv-source-meta">期初核准后的未归属成交会阻断新实盘计划；期初前记录仅作隔离展示。不会猜测关联或自动入账。</span>
        <span v-if="unresolvedBrokerTradesError" class="ltv-source-meta">明细读取失败：{{ unresolvedBrokerTradesError }}</span>
        <span v-else-if="unresolvedBrokerTrades === null" class="ltv-source-meta">正在读取…</span>
        <span v-else-if="!unresolvedBrokerTrades.length" class="ltv-source-meta">当前没有待核对成交回报。</span>
        <div v-else class="ltv-broker-trades__list">
          <div v-for="trade in unresolvedBrokerTrades" :key="`${trade.issue_type}:${trade.trade_id}`" class="ltv-broker-trade">
            <span><strong>{{ trade.issue_type === 'conflict' ? '回报冲突' : trade.issue_type === 'manual_mismatch' ? '人工成交与券商回报不一致' : trade.issue_type === 'broker_mismatch' ? '账本成交与券商回报不一致' : trade.issue_type === 'ledger_missing' ? '关联委托待入账' : trade.issue_type === 'unassigned' ? '期初后无本地委托归属，阻断' : trade.issue_type === 'legacy_quarantined' ? '期初前旧成交，隔离' : '费用待核' }}</strong> · {{ trade.trade_id }} · {{ trade.symbol }} · {{ sideLabel({ side: trade.side }) }} {{ trade.quantity }} 股</span>
            <span class="ltv-source-meta">成交价 {{ trade.price }} · 佣金 {{ trade.commission ?? '缺失' }} · 印花税 {{ trade.stamp_duty ?? '缺失' }} · {{ trade.observed_at }}</span>
          </div>
        </div>
      </div>
      <div v-if="showBrokerReportArchive" class="ltv-broker-trades" role="region" aria-label="待核对券商委托回报" aria-live="polite">
        <strong>本地保存的待核对委托</strong>
        <span class="ltv-source-meta">这是本地保存的异常委托回报，不代表当前券商在线状态；不会自动改变本地账本。</span>
        <span v-if="unresolvedBrokerOrdersError" class="ltv-source-meta">明细读取失败：{{ unresolvedBrokerOrdersError }}</span>
        <span v-else-if="unresolvedBrokerOrders === null" class="ltv-source-meta">正在读取…</span>
        <span v-else-if="!unresolvedBrokerOrders.length" class="ltv-source-meta">当前没有未核对委托回报。</span>
        <div v-else class="ltv-broker-trades__list">
          <div v-for="order in unresolvedBrokerOrders" :key="order.report_key" class="ltv-broker-trade">
            <span><strong>{{ order.disposition === 'conflict' ? '委托回报冲突' : '委托状态未知' }}</strong> · {{ order.order_id }} · {{ order.symbol }} · {{ sideLabel({ side: order.side }) }} {{ order.filled_qty }}/{{ order.quantity }} 股</span>
            <span class="ltv-source-meta">状态 {{ order.status }} · 来源 {{ order.source }} · {{ order.observed_at }}</span>
          </div>
        </div>
      </div>
      <div v-if="legacyQuarantineCount > 0" class="ltv-broker-trades" role="region" aria-label="旧记录隔离清单">
        <strong>旧记录隔离（只读）</strong>
        <span class="ltv-source-meta">身份不完整订单 {{ localAccount.legacy_unknown.orders }} 条 · 成交 {{ localAccount.legacy_unknown.trades }} 条 · 缺账户/版本/计划股数的旧信号 {{ localAccount.legacy_unknown.signals }} 条 · 缺账户/版本的历史调仓会话 {{ localAccount.legacy_unknown.sessions }} 条。记录不会自动归入本地账本。</span>
        <span v-if="legacyQuarantineError" class="ltv-source-meta">隔离明细读取失败：{{ legacyQuarantineError }}</span>
        <template v-else-if="legacyQuarantine">
          <div v-for="kind in ['orders', 'trades', 'signals', 'sessions']" :key="kind" class="ltv-quarantine-group">
            <strong v-if="legacyQuarantine[kind]?.length">{{ quarantineGroupLabels[kind] }}（{{ legacyQuarantine.counts[kind] }} 条）</strong>
            <div v-for="row in legacyQuarantine[kind] || []" :key="`${kind}:${row.id}`" class="ltv-broker-trade ltv-quarantine-row">
              <span>{{ formatQuarantineEntry(kind, row) }}</span>
              <span class="ltv-source-meta">缺少：{{ missingQuarantineIdentity(kind, row).join('、') }}</span>
            </div>
          </div>
          <div class="ltv-quarantine-pagination">
            <span class="ltv-source-meta">第 {{ Math.floor(quarantineOffset / 50) + 1 }} 页</span>
            <button class="btn-sm btn-outline" type="button" :disabled="quarantineLoading || quarantineOffset === 0" @click="changeQuarantinePage(-50)">上一页</button>
            <button class="btn-sm btn-outline" type="button" :disabled="quarantineLoading || !quarantineHasNext" @click="changeQuarantinePage(50)">下一页</button>
          </div>
        </template>
        <span v-else class="ltv-source-meta">正在读取隔离明细…</span>
      </div>
      <button class="btn-sm btn-outline" type="button" @click="loadLedgerState">刷新账本与对账</button>
      <button v-if="localAccount?.status === 'active' && !localAccount?.broker_account_id && reconciliation?.can_bind" class="btn-sm btn-outline" type="button" :disabled="bindingSaving" @click="bindLedgerBroker">{{ bindingSaving ? '绑定中…' : '核对并绑定券商账号' }}</button>
      <button v-if="localAccount?.status === 'active' && !localAccount?.broker_account_id" class="btn-sm btn-outline" type="button" @click="toggleHistoricalFillForm">{{ historicalFillOpen ? '收起历史成交补录' : '补录历史成交' }}</button>
      <div v-if="historicalFillOpen" class="ltv-opening ltv-intent-form" role="region" aria-label="历史成交补录">
        <strong>历史成交独立入账</strong>
        <span class="ltv-source-meta">仅供未绑定券商的本地账户补录早于今日的真实成交。必须填写券商成交编号和理由；旧信号只作来源审计，不会改变状态或恢复计划。补录将更新当前现金/持仓，之后请重新生成计划。</span>
        <label>成交日期<input v-model="historicalFillForm.tradeDate" type="date" :disabled="Boolean(historicalFillPending)" /></label>
        <label>证券代码<input v-model.trim="historicalFillForm.code" inputmode="numeric" maxlength="6" :disabled="Boolean(historicalFillPending)" /></label>
        <label>方向<select v-model="historicalFillForm.side" :disabled="Boolean(historicalFillPending)"><option value="buy">买入</option><option value="sell">卖出</option></select></label>
        <label>成交股数<input v-model="historicalFillForm.shares" type="number" min="1" step="1" :disabled="Boolean(historicalFillPending)" /></label>
        <label>成交价格<input v-model.trim="historicalFillForm.price" type="text" inputmode="decimal" :disabled="Boolean(historicalFillPending)" /></label>
        <label>实际佣金<input v-model.trim="historicalFillForm.commission" type="text" inputmode="decimal" :disabled="Boolean(historicalFillPending)" /></label>
        <label>实际印花税<input v-model.trim="historicalFillForm.stampDuty" type="text" inputmode="decimal" :disabled="Boolean(historicalFillPending || historicalFillForm.side === 'buy')" /></label>
        <label>券商成交编号<input v-model.trim="historicalFillForm.brokerTradeId" maxlength="64" :disabled="Boolean(historicalFillPending)" /></label>
        <label>来源信号编号（可选）<input v-model.trim="historicalFillForm.sourceSignalId" type="number" min="1" step="1" :disabled="Boolean(historicalFillPending)" /></label>
        <label>补录理由<input v-model.trim="historicalFillForm.reason" minlength="8" maxlength="500" :disabled="Boolean(historicalFillPending)" /></label>
        <label v-if="historicalFillForm.side === 'sell' && Number(historicalFillForm.shares) % 100 !== 0" class="ltv-form-hint">
          <input v-model="historicalFillForm.allowOddLotSell" type="checkbox" :disabled="Boolean(historicalFillPending)" />确认这是券商实际卖出的零股数量
        </label>
        <span class="ltv-source-meta">提交时账户版本：{{ localAccount?.version }} · 请求ID：{{ historicalFillPending?.request_id || historicalFillForm.requestId }}</span>
        <div v-if="historicalFillError" class="error" role="alert">{{ historicalFillError }}</div>
        <div v-if="historicalFillMessage" class="ltv-source-meta" role="status">{{ historicalFillMessage }}</div>
        <div class="ltv-config__actions">
          <button class="btn-sm" type="button" :disabled="historicalFillSaving || (!historicalFillPending && !historicalFillReady)" @click="submitHistoricalFill">{{ historicalFillSaving ? '提交中…' : historicalFillPending ? '用原请求ID重试' : '核对并补录' }}</button>
          <button v-if="historicalFillPending" class="btn-sm btn-outline" type="button" :disabled="historicalFillSaving" @click="discardHistoricalFillRequest">放弃未确认请求并重新填写</button>
        </div>
      </div>
      <div v-if="localAccount?.status === 'uninitialized'" class="ltv-opening">
        <input v-model="openingCash" type="number" min="0" step="0.01" placeholder="核准期初现金" aria-label="核准期初现金" />
        <input v-model="openingReason" type="text" placeholder="核准原因（至少8字）" aria-label="核准原因" />
        <button class="btn-sm" type="button" :disabled="openingSaving" @click="activateLedger">核准期初</button>
        <span class="ltv-source-meta">持仓指纹：{{ localAccount.holdings_fingerprint }}</span>
        <div v-if="localAccount.pending_holdings?.length" class="ltv-opening-holdings" role="status">
          <strong>待核准旧持仓（核准后才归入本地账户）</strong>
          <span v-for="holding in localAccount.pending_holdings" :key="holding.code" class="ltv-source-meta">
            {{ holding.code }} · {{ holding.shares }} 股 · 成本 {{ holding.cost_price }} 元
          </span>
        </div>
      </div>
      <div v-if="localAccount?.status === 'active' && localAccount.pending_holdings?.length" class="ltv-opening-holdings ltv-opening-holdings--blocked" role="alert">
        <strong>发现未归属的旧持仓，账本校验失败，已暂停实盘计划</strong>
        <span v-for="holding in localAccount.pending_holdings" :key="holding.code" class="ltv-source-meta">
          {{ holding.code }} · {{ holding.shares }} 股 · 成本 {{ holding.cost_price }} 元
        </span>
      </div>
    </section>

    <section v-if="localAccount?.broker_account_id" class="ltv-stage" aria-labelledby="local-intents-title">
      <div class="ltv-stage__header">
        <div class="ltv-stage__identity">
          <span class="ltv-stage__number ltv-stage__number--muted">委托</span>
          <div>
            <h3 id="local-intents-title">券商委托与成交回报</h3>
            <p>应用只预留资金并核验回报，不会向券商提交或撤销委托。请先在券商客户端按意图参数下单，再关联委托编号。</p>
          </div>
        </div>
        <button class="btn-sm btn-outline" type="button" :disabled="intentActionId !== ''" @click="loadLedgerState">刷新委托与回报</button>
      </div>
      <div v-if="localIntentsError" class="error" role="alert">本地委托列表读取失败：{{ localIntentsError }}</div>
      <div v-if="intentFormSignal" class="ltv-opening ltv-intent-form">
        <strong>准备 {{ intentFormSignal.code }} {{ sideLabel(intentFormSignal) }} 意图</strong>
        <span class="ltv-source-meta">预留后请按显示的证券、方向、数量和限价在券商客户端手工下单；费用上限用于冻结买入资金。</span>
        <label>委托股数<input v-model="intentForm.quantity" type="number" min="1" step="1" /></label>
        <label>限价<input v-model="intentForm.limitPrice" type="text" inputmode="decimal" /></label>
        <label>费用上限<input v-model="intentForm.feeCap" type="text" inputmode="decimal" /></label>
        <div class="ltv-source-meta">请求 ID {{ intentForm.intentId }}</div>
        <div v-if="intentFormError" class="error" role="alert">{{ intentFormError }}</div>
        <div class="ltv-config__actions">
          <button class="btn-sm" type="button" :disabled="intentFormSaving || reconciliation?.can_execute !== true" @click="submitIntentPreparation">{{ intentFormSaving ? '预留中…' : '核验并预留资金' }}</button>
          <button class="btn-sm btn-outline" type="button" :disabled="intentFormSaving" @click="cancelIntentPreparation">关闭</button>
        </div>
      </div>
      <div v-if="!localIntentsError && !localIntents.length" class="ltv-plan-empty">当前没有本地券商委托意图。</div>
      <div v-for="intent in localIntents" :key="intent.intent_id" class="ltv-intent-card">
        <div class="ltv-result__row-top">
          <strong>{{ intent.code }} · {{ sideLabel({ side: intent.side }) }} · {{ intent.quantity }} 股 @ ¥{{ Number(intent.limit_price).toFixed(3) }}</strong>
          <span class="ltv-result__label">{{ intent.state }}</span>
          <span class="ltv-result__detail">预留 ¥{{ Number(intent.reserved_cash || 0).toFixed(2) }} · 已成 {{ intent.filled_quantity || 0 }} 股 · 账号版本 {{ intent.account_version }}</span>
        </div>
        <div class="ltv-source-meta">意图 {{ intent.intent_id }}<span v-if="intent.broker_order_id"> · 券商委托 {{ intent.broker_order_id }}</span> · {{ intent.created_at }}</div>
        <div v-if="intent.state === 'prepared'" class="ltv-intent-actions">
          <span class="ltv-source-meta">请在券商客户端按上述参数手工下单后输入券商委托编号；若未下单，可取消意图并释放冻结资金。</span>
          <input v-model.trim="intentOrderInputs[intent.intent_id]" maxlength="64" autocomplete="off" placeholder="券商委托编号" :aria-label="`${intent.code} 券商委托编号`" />
          <button class="btn-sm" type="button" :disabled="intentActionId !== ''" @click="runIntentAction(intent, 'attach')">核验并关联</button>
          <button class="btn-sm btn-outline" type="button" :disabled="intentActionId !== ''" @click="runIntentAction(intent, 'cancel')">未下单，取消预留</button>
        </div>
        <div v-else-if="intent.broker_order_id" class="ltv-intent-actions">
          <button class="btn-sm btn-outline" type="button" :disabled="intentActionId !== ''" @click="loadIntentEvidence(intent.intent_id)">读取券商回报</button>
          <button v-if="['cancelled','rejected'].includes(intentEvidence[intent.intent_id]?.order?.status)" class="btn-sm btn-outline" type="button" :disabled="intentActionId !== ''" @click="runIntentAction(intent, 'finalize')">核验终态并释放剩余预留</button>
        </div>
        <div v-if="intentEvidence[intent.intent_id]" class="ltv-intent-evidence">
          <span v-if="intentEvidence[intent.intent_id].order" class="ltv-source-meta">券商状态 {{ intentEvidence[intent.intent_id].order.status }} · 已报成交 {{ intentEvidence[intent.intent_id].order.filled_qty }}/{{ intentEvidence[intent.intent_id].order.quantity }} 股</span>
          <span v-if="intentEvidence[intent.intent_id].order_conflicts" class="error" role="alert">委托状态或累计成交数有 {{ intentEvidence[intent.intent_id].order_conflicts }} 条未解决冲突，停止该委托全部成交入账。</span>
          <span v-if="intentEvidence[intent.intent_id].trade_conflicts" class="error" role="alert">逐笔成交有 {{ intentEvidence[intent.intent_id].trade_conflicts }} 条未解决冲突，停止该委托全部成交入账。</span>
          <span v-if="!intentEvidence[intent.intent_id].trades?.length" class="ltv-source-meta">尚无该委托的逐笔成交回报。</span>
          <div v-for="trade in intentEvidence[intent.intent_id].trades || []" :key="trade.trade_id" class="ltv-intent-trade">
            <span><strong>{{ trade.trade_id }}</strong> · {{ trade.symbol }} · {{ sideLabel({ side: trade.side }) }} {{ trade.quantity }} 股 @ ¥{{ trade.price }} · ¥{{ trade.amount }}</span>
            <span class="ltv-source-meta">回报费用：佣金 {{ trade.commission ?? '缺失' }} / 印花税 {{ trade.stamp_duty ?? '缺失' }} · {{ trade.observed_at }}<span v-if="trade.booked"> · 已入账</span><span v-else-if="trade.booked_intent_id"> · 已由意图 {{ trade.booked_intent_id }} 入账</span></span>
            <div v-if="!trade.booked && !trade.booked_intent_id" class="ltv-intent-actions">
              <label>实际佣金<input v-model="intentFees[`${intent.intent_id}:${trade.trade_id}`].commission" type="text" inputmode="decimal" /></label>
              <label>实际印花税<input v-model="intentFees[`${intent.intent_id}:${trade.trade_id}`].stampDuty" type="text" inputmode="decimal" /></label>
              <button class="btn-sm" type="button" :disabled="intentActionId !== '' || intentEvidence[intent.intent_id].conflicts > 0" @click="bookIntentTrade(intent, trade)">核对回报并入账</button>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section v-if="restoredSignals.length || pendingSignalsError" class="ltv-stage" aria-labelledby="rebalance-unfinished-title">
      <div class="ltv-stage__header">
        <div class="ltv-stage__identity">
          <span class="ltv-stage__number ltv-stage__number--muted">待办</span>
          <div>
            <h3 id="rebalance-unfinished-title">未完成成交确认</h3>
            <p>恢复当日待办及历史部分成交；过期信号只能核对后关闭剩余计划。</p>
          </div>
        </div>
        <button class="btn-sm btn-outline" type="button" @click="loadPersistedSignals">刷新</button>
      </div>
      <div v-if="pendingSignalsError" class="error" role="alert">{{ pendingSignalsError }}</div>
      <div v-for="s in restoredSignals" :key="s.id" class="ltv-result__row">
        <div class="ltv-result__row-top">
          <span class="ltv-result__code">{{ s.code }} {{ s.name }}</span>
          <span class="ltv-result__label" :class="sideClass(s.side)">{{ sideLabel(s) }}</span>
          <span class="ltv-result__detail">计划 {{ s.planned_shares }} 股 · 已成交 {{ s.filled_shares }} 股 · 剩余 {{ s.planned_shares - s.filled_shares }} 股</span>
          <span v-if="isExpiredSignal(s)" class="ltv-result__detail">{{ String(s.created_at).slice(0, 10) }} 已过期</span>
          <button v-if="!isExpiredSignal(s) && !localAccount?.broker_account_id" class="btn-sm" type="button"
            :disabled="localAccount?.status !== 'active' || Number(localAccount?.version) !== Number(s.account_version)"
            @click="executeSignal(s)">记录成交</button>
          <span v-else-if="!isExpiredSignal(s) && localAccount?.broker_account_id" class="ltv-result__reason">请从已核验的券商委托回报入账</span>
          <button v-if="canPrepareIntent(s)" class="btn-sm btn-outline" type="button" @click="startIntentPreparation(s)">准备券商委托</button>
          <button v-if="s.execution_status === 'partial'" class="btn-sm btn-outline" type="button"
            :disabled="localAccount?.status !== 'active' || closingRemainder"
            @click="closeRemainderId = closeRemainderId === s.id ? null : s.id; closeRemainderError = ''">关闭剩余计划</button>
        </div>
        <div v-if="!isExpiredSignal(s) && localAccount?.status === 'active' && Number(localAccount?.version) !== Number(s.account_version)" class="ltv-result__reason">账户版本已变化，请核对并重新生成计划。</div>
        <div v-if="closeRemainderId === s.id" class="ltv-opening">
          <input v-model.trim="closeRemainderReason" type="text" minlength="8" maxlength="255" :aria-label="`关闭 ${s.code} 剩余计划的理由`" placeholder="填写关闭理由（8至255字）" />
          <button class="btn-sm" type="button" :disabled="closingRemainder || closeRemainderReason.length < 8" @click="closeSignalRemainder(s)">{{ closingRemainder ? '处理中…' : '核对并确认关闭' }}</button>
          <span class="ltv-source-meta">仅关闭剩余 {{ s.planned_shares - s.filled_shares }} 股，不撤销已成交记录或券商委托。</span>
        </div>
        <div v-if="closeRemainderId === s.id && closeRemainderError" class="error" role="alert">{{ closeRemainderError }}</div>
      </div>
    </section>

    <!-- ═══ 加载中 ═══ -->
    <div v-if="loadingCandidates" class="ltv-empty-state">
      <div class="ltv-empty-state__index">01</div>
      <div class="ltv-empty-state__title">正在加载选股结果...</div>
      <div class="ltv-empty-state__desc">{{ selectedSaveId ? '获取候选股列表' : '获取选股列表' }}</div>
    </div>

    <!-- ═══ 空状态 ═══ -->
    <div v-else-if="!holdingsLoadError && !candidates.length && !holdings.length" class="ltv-empty-state">
      <div class="ltv-empty-state__index">01</div>
      <div class="ltv-empty-state__title">选择一份选股结果开始</div>
      <div class="ltv-empty-state__desc">在“选股”页面保存候选结果，再回到这里构建组合方案。</div>
    </div>

    <main v-else class="ltv-main">
      <section class="ltv-stage" aria-labelledby="rebalance-candidates-title">
        <div class="ltv-stage__header">
          <div class="ltv-stage__identity">
            <span class="ltv-stage__number">2</span>
            <div>
              <h3 id="rebalance-candidates-title">候选确认与风险审计</h3>
              <p>模型排名不做二次复筛；这里只核对硬风险排除和最终入选结果。</p>
            </div>
          </div>
          <span class="ltv-stage__summary">候选 {{ displayCandidates.length }} · 已选 {{ checkedCount }}</span>
        </div>
        <div class="ltv-decision-grid">
          <div class="ltv-candidate-panel">
            <div class="ltv-left__section">
              <span class="ltv-left__title">候选股票</span>
          <div class="ltv-left__actions">
            <span class="ltv-left__count">{{ checkedCount }}/{{ displayCandidates.length }} 已选</span>
            <button
              class="ltv-left__collapse"
              type="button"
              :aria-expanded="!candidatesCollapsed"
              aria-controls="rebalance-candidate-list"
              @click="candidatesCollapsed = !candidatesCollapsed"
            >{{ candidatesCollapsed ? '展开' : '收起' }}</button>
          </div>
            </div>
          <div id="rebalance-candidate-list" v-show="!candidatesCollapsed" class="ltv-left__list">
          <div v-for="c in displayCandidates" :key="c.code"
               class="ltv-row" :class="{ 'ltv-row--active': activeCode === c.code }"
               role="button" tabindex="0"
               @click="activeCode = c.code"
               @keydown.enter.prevent="activeCode = c.code"
               @keydown.space.prevent="activeCode = c.code">
            <input type="checkbox" :checked="isChecked(c.code)" :disabled="c.is_holding" :aria-label="`${c.code} ${c.name}参与调仓计算`" @click.stop @change="toggleCheck(c.code)" />
            <span class="ltv-row__code ticker">{{ c.code }}</span>
            <span class="ltv-row__name">{{ c.name }}</span>
            <span class="ltv-row__score" v-if="c.score != null">{{ c.score.toFixed(1) }}</span>
            <span v-if="c.final_rank" class="ltv-row__rank" :class="`ltv-row__rank--${c.decision_status}`">#{{ c.final_rank }}</span>
            <span class="ltv-row__label" v-if="c.is_holding">持仓 {{ c.shares }}股</span>
            <button v-if="c.is_holding" class="ltv-row__edit" @click.stop="editHolding(c)" :aria-label="`编辑${c.code}持仓`" title="编辑">✎</button>
          </div>
        </div>

          <div class="ltv-candidate-panel__footer">
            <button v-if="localAccount?.status === 'uninitialized'" class="btn-sm btn-outline" type="button" @click="showAddHolding = true">添加持仓</button>
            <span v-else>账户激活后，持仓数量通过成交确认更新</span>
            <span>持仓股票固定参与计算，不能取消勾选</span>
          </div>
        </div>

        <section class="ltv-decision" aria-labelledby="decision-audit-title">
          <div class="ltv-decision__header">
            <div>
              <strong id="decision-audit-title">候选决策审计</strong>
              <span v-if="decisionData">{{ decisionData.data_cutoff }} · {{ decisionData.summary?.selected_count || decisionSelected.length }}只入选</span>
            </div>
            <button type="button" :aria-expanded="!decisionCollapsed" @click="decisionCollapsed = !decisionCollapsed">
              {{ decisionCollapsed ? '展开' : '收起' }}
            </button>
          </div>
          <div v-show="!decisionCollapsed" class="ltv-decision__body">
            <div v-if="loadingDecision" class="ltv-decision__hint" role="status">正在生成可审计快照…</div>
            <div v-else-if="decisionError" class="ltv-decision__error" role="alert">{{ decisionError }}</div>
            <template v-else-if="decisionData">
              <div v-if="decisionSelected.length" class="ltv-decision__list">
                <button
                  v-for="row in decisionSelected"
                  :key="`decision-${row.code}`"
                  type="button"
                  :class="{ 'is-active': decisionActiveRow?.code === row.code }"
                  :aria-pressed="decisionActiveRow?.code === row.code"
                  @click="activeCode = row.code"
                >
                  <span class="ticker">#{{ row.final_rank }} {{ row.code }}</span>
                  <span>{{ row.name }}</span>
                  <span class="ltv-decision__score">{{ formatDecisionScore(row.utility_score) }}</span>
                  <small>{{ formatDecisionRankChange(row) }}</small>
                </button>
              </div>
              <div v-else class="ltv-decision__empty">快照已生成，但未返回入选明细，请刷新后重试。</div>
              <div class="ltv-decision__notice">候选顺序完全沿用选股模型；市场状态只控制总仓位，行业、资金和风格信息不改变排名。</div>
              <details v-if="decisionData.excluded?.length" class="ltv-decision__excluded">
                <summary>查看 {{ decisionData.excluded.length }} 只风险排除项</summary>
                <p>{{ decisionData.excluded.map(row => `${row.code} ${row.exclusion_reason}`).join('；') }}</p>
              </details>
              <div class="ltv-decision__fingerprint" :title="decisionData.fingerprint">
                快照 {{ decisionData.fingerprint?.slice(0, 12) }}
              </div>
            </template>
            <div v-else class="ltv-decision__empty">生成调仓方案后，这里会展示入选排名、硬风险排除和决策快照。</div>
          </div>
        </section>
        </div>

        <div v-if="cooldownStocks.length" class="ltv-cooldown">
          <div class="ltv-cooldown__header">
            <span>冷冻期股票 ({{ cooldownStocks.length }})</span>
            <span v-if="cooldownAsOf">更新 {{ cooldownAsOf }}</span>
          </div>
          <button v-for="item in cooldownStocks" :key="item.code" class="ltv-cooldown__item" @click="viewCooldownStock(item)">
            <span class="ticker">{{ item.code }}</span>
            <span>{{ item.name || '点击查看K线' }}</span>
            <strong>剩余 {{ item.remaining_periods }} 期</strong>
          </button>
        </div>

      </section>

      <section class="ltv-stage" aria-labelledby="rebalance-plan-title">
        <div class="ltv-stage__header">
          <div class="ltv-stage__identity">
            <span class="ltv-stage__number">3</span>
            <div>
              <h3 id="rebalance-plan-title">调仓建议与执行</h3>
              <p>先处理卖出与减仓，再确认买入；观察和剔除项保留为解释信息。</p>
            </div>
          </div>
          <button class="btn-sm ltv-primary-action" type="button" @click="runRebalanceAction" :disabled="!selectedSaveId || !checkedCodes.length || runningRebalance || pendingPlanCommit || pendingPlanCommitError || uncertainFillRequests.length || uncertainFillRequestsError || (planCalculated && !planDirty && planCommitBlocked && !planCommitted)">
            {{ planActionLabel }}
          </button>
        </div>
        <div class="ltv-plan-overview" aria-label="调仓方案摘要">
          <span><strong>{{ sellSignals.length }}</strong> 卖出/减仓</span>
          <span><strong>{{ buySignals.length }}</strong> 买入/加仓</span>
          <span><strong>{{ classificationData?.hold?.length || 0 }}</strong> 继续持有</span>
          <span><strong>{{ observeCandidates.length + (classificationData?.observe?.length || 0) }}</strong> 观察</span>
          <span><strong>{{ filteredOut.length + skippedCandidates.length }}</strong> 剔除/跳过</span>
        </div>
        <div v-if="planDirty && planCalculated" class="ltv-plan-stale-note" role="status">以下仍是上次计算结果，当前参数尚未应用；请重新计算后再执行。</div>
        <div v-if="planFingerprint && !planDirty" class="ltv-plan-stale-note" role="status">
          {{ planCommitted ? '方案已显式提交并保存人工成交确认信号。' : '当前为只读预览，尚未保存调仓信号；检查结果后再次点击“确认并提交方案”。' }}
        </div>

        <div class="ltv-result" :class="{ 'is-stale': planDirty }" v-if="hasPlanSummary">
          <div class="ltv-result__meta" v-if="priceSource">
            <span :class="priceSource === 'realtime' ? 'ltv-meta--live' : 'ltv-meta--close'">
              {{ priceSource === 'realtime' ? '● 按盘中实时价计算' : '○ 按最近收盘价计算' }}
            </span>
            换手约束基于该价格 · 前{{ appliedTopN }}名为本方案持仓上限
            <span v-if="positionRatio != null">· 市场状态仓位 {{ formatDecisionCoverage(positionRatio) }}（{{ positionSourceLabel }}）</span>
          </div>
          <div v-if="sellSignals.length" class="ltv-result__group">
            <div class="ltv-result__group-title ltv-result__group-title--sell">优先处理 · 卖出/减仓 ({{ sellSignals.length }})</div>
            <div v-for="s in sellSignals" :key="s.id" class="ltv-result__row" :class="{ 'ltv-result__row--done': s.executed }">
              <div class="ltv-result__row-top">
                <span class="ltv-result__code">{{ s.code }} {{ s.name }}</span>
                <span class="ltv-result__label" :class="sideClass(s.side)">{{ sideLabel(s) }}</span>
                <span class="ltv-result__detail">{{ (s.current_weight * 100).toFixed(1) }}% → {{ (s.target_weight * 100).toFixed(1) }}%</span>
                <span class="ltv-result__price" v-if="executionDisplay(s)">{{ executionDisplay(s) }}</span>
                <button v-if="!s.executed && s.execution_status !== 'cancelled' && !localAccount?.broker_account_id" class="btn-sm" :disabled="planDirty || !s.id || localAccount?.status !== 'active'" @click="executeSignal(s)">记录已卖</button>
                <span v-else-if="!s.executed && s.execution_status !== 'cancelled' && localAccount?.broker_account_id" class="ltv-result__reason">等待券商成交回报入账</span>
                <button v-if="canPrepareIntent(s)" class="btn-sm btn-outline" type="button" @click="startIntentPreparation(s)">准备委托</button>
                <span v-else-if="s.execution_status === 'cancelled'" class="ltv-result__cancelled">计划已失效</span>
                <span v-else class="ltv-result__done">✓</span>
              </div>
              <div class="ltv-result__reason" v-if="s.unfilled_reason || s.reason">{{ s.unfilled_reason || s.reason }}</div>
            </div>
          </div>
          <div v-if="buySignals.length" class="ltv-result__group">
            <div class="ltv-result__group-title ltv-result__group-title--buy">后续确认 · 买入/加仓 ({{ buySignals.length }})</div>
            <div v-for="s in buySignals" :key="s.id" class="ltv-result__row" :class="{ 'ltv-result__row--done': s.executed }">
              <div class="ltv-result__row-top">
                <span class="ltv-result__code">{{ s.code }} {{ s.name }}</span>
                <span class="ltv-result__label ltv-up">{{ s.side === 'add' ? '加仓' : '买入' }}</span>
                <span class="ltv-result__detail">
                  {{ (s.target_weight * 100).toFixed(1) }}%
                  <template v-if="s.amount > 0"> · ¥{{ s.amount.toFixed(0) }}</template>
                </span>
                <span class="ltv-result__price" v-if="executionDisplay(s)">{{ executionDisplay(s) }}</span>
                <button v-if="!s.executed && s.execution_status !== 'cancelled' && !localAccount?.broker_account_id" class="btn-sm" :disabled="planDirty || !s.id || localAccount?.status !== 'active'" @click="executeSignal(s)">记录已买</button>
                <span v-else-if="!s.executed && s.execution_status !== 'cancelled' && localAccount?.broker_account_id" class="ltv-result__reason">等待券商成交回报入账</span>
                <button v-if="canPrepareIntent(s)" class="btn-sm btn-outline" type="button" @click="startIntentPreparation(s)">准备委托</button>
                <span v-else-if="s.execution_status === 'cancelled'" class="ltv-result__cancelled">计划已失效</span>
                <span v-else class="ltv-result__done">✓</span>
              </div>
              <div class="ltv-result__row-sub">
                <span class="ltv-result__reason" v-if="s.unfilled_reason || s.reason">{{ s.unfilled_reason || s.reason }}</span>
                <span class="ltv-result__stops" v-if="stopInfo(s.code)">
                  止损 {{ stopInfo(s.code).loss?.toFixed(2) }} | 止盈 {{ stopInfo(s.code).profit?.toFixed(2) }}
                </span>
                <span
                  v-if="s.liquidity_capacity?.participation_rate != null"
                  class="ltv-result__capacity"
                  :class="{ 'is-warning': s.liquidity_capacity.status === 'warning' }"
                >
                  20日成交额中位数 {{ formatDecisionAmount(s.liquidity_capacity.median_amount_20d) }}
                  · 下单占比 {{ formatDecisionCoverage(s.liquidity_capacity.participation_rate) }}
                </span>
              </div>
            </div>
          </div>
          <div v-if="classificationData?.hold?.length" class="ltv-result__group">
            <div class="ltv-result__group-title ltv-result__group-title--hold">继续持有 · 权重达标 ({{ classificationData.hold.length }})</div>
            <div v-for="h in classificationData.hold" :key="'h' + h.code" class="ltv-result__row">
              <div class="ltv-result__row-top">
                <span class="ltv-result__code">{{ h.code }} {{ h.name }}</span>
                <span class="ltv-result__label ltv-hold">持有</span>
                <span class="ltv-result__detail" v-if="h.current_weight">{{ (h.current_weight * 100).toFixed(1) }}%</span>
              </div>
              <div class="ltv-result__reason" v-if="h.reason">{{ h.reason }}</div>
            </div>
          </div>
          <details v-if="classificationData?.observe?.length" class="ltv-result__group ltv-result__group--secondary">
            <summary class="ltv-result__group-title ltv-result__group-title--observe">继续持有 · 观察 ({{ classificationData.observe.length }})</summary>
            <div v-for="h in classificationData.observe" :key="'ob-h' + h.code" class="ltv-result__row">
              <div class="ltv-result__row-top">
                <span class="ltv-result__code">{{ h.code }} {{ h.name }}</span>
                <span class="ltv-result__label ltv-observe">继续持有</span>
                <span class="ltv-result__detail" v-if="h.current_weight">当前 {{ (h.current_weight * 100).toFixed(1) }}%</span>
              </div>
              <div class="ltv-result__reason">{{ h.reason }} · 本轮不主动调仓</div>
            </div>
          </details>
          <details v-if="observeCandidates.length" class="ltv-result__group ltv-result__group--secondary">
            <summary class="ltv-result__group-title ltv-result__group-title--observe">候选观察 · 本轮未进组合 ({{ observeCandidates.length }})</summary>
            <div v-for="c in observeCandidates" :key="'o' + c.code" class="ltv-result__row">
              <div class="ltv-result__row-top">
                <span class="ltv-result__code">{{ c.code }} {{ c.name }}</span>
                <span class="ltv-result__label ltv-observe">观察</span>
                <span class="ltv-result__detail" v-if="c.qlib_score != null">qlib {{ c.qlib_score.toFixed(4) }}</span>
              </div>
              <div class="ltv-result__reason">排名在前{{ appliedTopN }}名之外，本轮不建仓，保留观察</div>
            </div>
          </details>
          <details v-if="skippedCandidates.length" class="ltv-result__group ltv-result__group--secondary">
            <summary class="ltv-result__group-title ltv-result__group-title--skip">组合内跳过 ({{ skippedCandidates.length }})</summary>
            <div v-for="s in skippedCandidates" :key="'sk' + s.code" class="ltv-result__row">
              <div class="ltv-result__row-top">
                <span class="ltv-result__code">{{ s.code }} {{ s.name }}</span>
                <span class="ltv-result__label ltv-skip">跳过</span>
              </div>
              <div class="ltv-result__reason">{{ s.reason }}</div>
            </div>
          </details>
          <details v-if="filteredOut.length" class="ltv-result__group ltv-result__group--secondary">
            <summary class="ltv-result__group-title ltv-result__group-title--filter">风险筛选剔除 ({{ filteredOut.length }})</summary>
            <div v-for="f in filteredOut" :key="'f' + f.code" class="ltv-result__row">
              <div class="ltv-result__row-top">
                <span class="ltv-result__code">{{ f.code }} {{ f.name }}</span>
                <span class="ltv-result__label ltv-filter">剔除</span>
              </div>
              <div class="ltv-result__reason">{{ f.reason }}</div>
            </div>
          </details>
        </div>
        <div v-else class="ltv-plan-empty">
            <strong>{{ planCalculated && !planDirty ? '当前组合无需主动调仓' : '尚未预览当前参数下的调仓方案' }}</strong>
            <span>{{ planDirty ? '候选或参数已经变化，请重新预览。' : '系统会将持仓、候选排名、冷冻期和换手约束合并计算。预览后需再次确认才会保存调仓信号。' }}</span>
        </div>
      </section>

      <section class="ltv-stage ltv-stage--evidence" aria-labelledby="rebalance-evidence-title">
        <div class="ltv-stage__header">
          <div class="ltv-stage__identity">
            <span class="ltv-stage__number ltv-stage__number--muted">辅助</span>
            <div>
              <h3 id="rebalance-evidence-title">个股价位与盘中确认</h3>
              <p>用于校验入场价、支撑压力和盘中可执行性，不改变上方组合决策。</p>
            </div>
          </div>
          <span v-if="activeCode" class="ltv-stage__summary ticker">当前 {{ activeCode }}</span>
        </div>
        <div class="ltv-right">
        <div v-if="!activeCode" class="ltv-right__empty">
          从候选或调仓建议中选择一只股票查看执行辅助信息
        </div>
        <template v-else>
          <div class="ltv-chart-box">
            <div class="ltv-chart-tabs">
              <button class="ltv-chart-tab" :class="{ 'ltv-chart-tab--active': chartTab === 'kline' }" @click="switchChart('kline')">日K（前复权）</button>
              <button class="ltv-chart-tab" :class="{ 'ltv-chart-tab--active': chartTab === 'intraday' }" @click="switchChart('intraday')">分时</button>
              <button v-if="checkedCodes.length > 1" class="ltv-chart-tab" :class="{ 'ltv-chart-tab--active': chartTab === 'compare' }" @click="switchChart('compare')">对比</button>
            </div>
            <div v-if="chartTab === 'intraday' && intradayBars?.length && intradayMeta?.quantity_status !== 'verified'" class="ltv-intraday-note" role="status">
              分时量额口径未核验；当前数据仅作价格参考，不用于量价或VWAP执行判断。
            </div>
            <div class="ltv-chart-wrap ltv-chart--main">
              <div ref="mainChartRef" class="ltv-chart"></div>
              <div class="ltv-chart__overlay" v-if="chartOverlayText">{{ chartOverlayText }}</div>
            </div>
          </div>

          <!-- 建议价位面板 -->
          <div class="ltv-suggest" v-if="activePreview">
            <div class="ltv-suggest__header">
              <span class="ltv-suggest__code">{{ activePreview.code }} {{ activePreview.name }}</span>
              <span class="data-value data-value--lg">{{ activeLivePrice?.toFixed(2) || activePreview.realtime?.price?.toFixed(2) || '--' }}</span>
              <span class="ltv-suggest__quality" :class="qualityClass">
                {{ activePreview.entry_quality?.label || '--' }} {{ activePreview.entry_quality?.score || '' }}
              </span>
            </div>
            <div v-if="!activePreview.execution_allowed" class="text-muted">
              {{ activePreview.price_status === 'historical_preview' ? '历史预览仅展示前复权日线，不提供实时执行价位。' : '实时价与前复权日线缺少可核验锚点，支撑压力和建议价位不可用。' }}
            </div>
            <div class="ltv-suggest__grid">
              <div class="ltv-suggest__col">
                <strong>支撑位</strong>
                <div v-for="s in activePreview.supports?.slice(0,4)" :key="s.name" class="ltv-suggest__line">
                  <span>{{ s.name }}</span>
                  <span class="ltv-suggest__val">{{ s.price.toFixed(2) }}</span>
                </div>
              </div>
              <div class="ltv-suggest__col">
                <strong>压力位</strong>
                <div v-for="r in activePreview.resistances?.slice(0,4)" :key="r.name" class="ltv-suggest__line">
                  <span>{{ r.name }}</span>
                  <span class="ltv-suggest__val">{{ r.price.toFixed(2) }}</span>
                </div>
              </div>
              <div class="ltv-suggest__col">
                <strong>参考价位</strong>
                <div class="ltv-suggest__line">
                  <span>{{ activeExecution ? '盘中执行' : '买入参考' }}</span>
                  <span class="ltv-suggest__val text-red" v-if="activeExecution">
                    {{ executionPrice(activeExecution) }}
                  </span>
                  <span class="ltv-suggest__val text-red" v-else-if="activePreview.suggested_buy">
                    {{ activePreview.suggested_buy.low.toFixed(2) }}~{{ activePreview.suggested_buy.high.toFixed(2) }}
                  </span>
                </div>
                <div class="ltv-suggest__line">
                  <span>止损参考</span>
                  <span class="ltv-suggest__val text-green">{{ activePreview.stop_loss_ref?.toFixed(2) || '--' }}</span>
                </div>
                <div class="ltv-suggest__line">
                  <span>止盈参考</span>
                  <span class="ltv-suggest__val text-red">{{ activePreview.take_profit_ref?.toFixed(2) || '--' }}</span>
                </div>
              </div>
            </div>
            <div v-if="activePreview.technical_analysis?.items?.length" class="ltv-technical">
              <div class="ltv-technical__header">
                <div>
                  <strong>技术指标判读</strong>
                  <span>日线截至 {{ activePreview.technical_analysis.as_of }}；“出货/洗盘”均为条件推断，需后续确认</span>
                </div>
                <span class="ltv-technical__overall" :class="technicalStatusClass(activePreview.technical_analysis.overall?.bias)">
                  {{ technicalStatusIcon(activePreview.technical_analysis.overall?.bias) }}
                  {{ activePreview.technical_analysis.overall?.label }}
                  · 利好 {{ activePreview.technical_analysis.overall?.bullish || 0 }}
                  / 利空 {{ activePreview.technical_analysis.overall?.bearish || 0 }}
                  / 风险 {{ activePreview.technical_analysis.overall?.risk || 0 }}
                </span>
              </div>
              <div class="ltv-technical__grid">
                <div v-for="indicator in activePreview.technical_analysis.items" :key="indicator.key" class="ltv-technical__item">
                  <div class="ltv-technical__item-head">
                    <strong>{{ indicator.label }}</strong>
                    <span class="ltv-technical__status" :class="technicalStatusClass(indicator.status)">
                      {{ technicalStatusIcon(indicator.status) }} {{ technicalStatusLabel(indicator.status) }}
                    </span>
                  </div>
                  <span class="ltv-technical__value">{{ indicator.value }}</span>
                  <span class="ltv-technical__detail">{{ indicator.detail }}</span>
                </div>
              </div>
            </div>
          </div>
        </template>
      </div>
      </section>
    </main>

    <section class="ltv-stage" aria-labelledby="rebalance-execution-history-title">
      <div class="ltv-stage__header">
        <div class="ltv-stage__identity">
          <span class="ltv-stage__number ltv-stage__number--muted">历史</span>
          <div>
            <h3 id="rebalance-execution-history-title">调仓执行历史</h3>
            <p>按本地实盘账户汇总最近30天；笔数与成交/剩余股数分开统计。</p>
          </div>
        </div>
        <button class="btn-sm btn-outline" type="button" @click="toggleExecutionHistory">
          {{ executionHistoryOpen ? '收起' : '查看历史' }}
        </button>
      </div>
      <div v-if="executionHistoryOpen">
        <div v-if="executionHistoryError" class="error" role="alert">{{ executionHistoryError }}</div>
        <div v-else-if="executionHistoryLoading" class="ltv-source-meta" role="status">正在读取执行历史…</div>
        <div v-else-if="!executionHistory.length" class="ltv-source-meta">最近30天没有调仓信号记录。</div>
        <div v-else class="ltv-execution-history-wrap">
          <table class="ltv-execution-history">
            <thead><tr><th>生成时间</th><th>计划数</th><th>已执行</th><th>待处理</th><th>部分成交</th><th>已跳过</th><th>已取消</th><th>失败/未知</th><th>计划/成交/剩余股数</th></tr></thead>
            <tbody>
              <tr v-for="row in executionHistory" :key="`${row.batch_id}:${row.check_type}:${row.screening_save_id}`">
                <td>{{ String(row.created_at || '').slice(0, 19) || '--' }}</td>
                <td>{{ row.total ?? 0 }}</td>
                <td>{{ row.executed ?? 0 }}</td>
                <td>{{ row.pending_only ?? 0 }}</td>
                <td>{{ row.partial ?? 0 }}</td>
                <td>{{ row.skipped ?? 0 }}</td>
                <td>{{ row.cancelled ?? 0 }}</td>
                <td>{{ row.failed ?? 0 }} / {{ row.unknown ?? 0 }}<span v-if="Number(row.unplanned) > 0"> · 缺计划股数 {{ row.unplanned }}</span></td>
                <td>{{ row.planned_shares ?? 0 }} / {{ row.filled_shares ?? 0 }} / {{ row.remaining_shares ?? 0 }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <span class="ltv-source-meta">计划数和状态数按信号行计；股数列为计划股数、已成交股数、当前待处理计划余量。旧记录缺少计划股数时单独标出，不将其推算为0股有效计划。</span>
        <button class="btn-sm btn-outline" type="button" :disabled="executionHistoryLoading" @click="loadExecutionHistory">刷新历史</button>
      </div>
    </section>

    <!-- ═══ 底部: 日内执行决策区 ═══ -->
    <IntradayCheckPanel
      v-if="buyCodes.length || sellSignals.length"
      :buy-codes="buyCodes"
      :ma-data="maData"
      :sell-signals="sellSignals"
      :refresh-interval="5000"
      @update="onIntradayUpdate"
    />

    <!-- ═══ 底部: T+0 做T建议 (I激进 / N保守) ═══ -->
    <T0Panel
      v-if="heldCodes.length"
      :codes="heldCodes"
      :refresh-interval="10000"
    />

    <!-- ═══ 持仓编辑弹窗 ═══ -->
    <div v-if="showEditModal" class="ltv-modal-overlay" @click.self="closeEditModal">
      <div class="ltv-modal">
        <h4>{{ editingHolding ? '编辑持仓' : '添加持仓' }}</h4>
        <div v-if="localAccount?.status === 'active'" class="ltv-form-hint">实盘账户已激活，仅可修改股票名称；数量、成本和日期由成交事务更新。</div>
        <div class="ltv-modal__form">
          <input v-model="holdForm.code" placeholder="代码" :disabled="!!editingHolding" />
          <input v-model="holdForm.name" placeholder="名称" />
          <input v-model.number="holdForm.shares" placeholder="股数" type="number" :disabled="localAccount?.status === 'active'" />
          <input v-model.number="holdForm.cost_price" placeholder="成本价" type="number" step="0.01" :disabled="localAccount?.status === 'active'" />
          <input v-model="holdForm.buy_date" type="date" :disabled="localAccount?.status === 'active'" />
        </div>
        <div class="ltv-modal__btns">
          <button v-if="editingHolding && localAccount?.status === 'uninitialized'" class="btn-sm btn-danger" @click="deleteHolding">删除</button>
          <button class="btn-sm btn-outline" @click="closeEditModal">取消</button>
          <button class="btn-sm" @click="saveHolding">保存</button>
        </div>
      </div>
    </div>

    <!-- ═══ 成交记录弹窗 ═══ -->
    <div v-if="showExecutionModal" class="ltv-modal-overlay" @click.self="closeExecutionModal" @keydown.esc="closeExecutionModal">
      <div class="ltv-modal ltv-modal--execution" role="dialog" aria-modal="true" aria-labelledby="execution-modal-title">
        <h4 id="execution-modal-title">记录实际成交</h4>
        <div class="ltv-execution-summary">
          <strong>{{ executingSignal?.code }} {{ executingSignal?.name }}</strong>
          <span :class="sideClass(executingSignal?.side)">{{ sideLabel(executingSignal || {}) }}</span>
        </div>
        <div class="ltv-modal__form">
          <label class="ltv-form-field" for="execution-shares">
            <span>实际成交股数</span>
            <input
              id="execution-shares"
              ref="executionSharesInput"
              v-model.number="executionForm.shares"
              type="number"
              min="1"
              :step="isExecutionBuy ? 100 : 1"
              inputmode="numeric"
              :disabled="!!uncertainExecution"
            />
          </label>
          <label class="ltv-form-field" for="execution-price">
            <span>实际成交价格</span>
            <input
              id="execution-price"
              v-model="executionForm.price"
              type="number"
              min="0.001"
              step="0.001"
              inputmode="decimal"
              :disabled="!!uncertainExecution"
            />
          </label>
          <label class="ltv-form-field" for="execution-commission">
            <span>实际佣金及其他费用</span>
            <input id="execution-commission" v-model="executionForm.commission" type="number" min="0" step="0.01" :disabled="!!uncertainExecution" />
          </label>
          <label class="ltv-form-field" for="execution-stamp-duty">
            <span>实际印花税</span>
            <input id="execution-stamp-duty" v-model="executionForm.stampDuty" type="number" min="0" step="0.01" :disabled="isExecutionBuy || !!uncertainExecution" />
          </label>
          <label v-if="localAccount?.broker_account_id" class="ltv-form-field" for="execution-broker-trade-id">
            <span>券商逐笔成交编号</span>
            <input id="execution-broker-trade-id" v-model.trim="executionForm.brokerTradeId" maxlength="64" autocomplete="off" :disabled="!!uncertainExecution" />
          </label>
          <div v-if="localAccount?.broker_account_id" class="ltv-form-hint">请按券商回报填写成交编号；同一编号只能入账一次。</div>
          <div class="ltv-form-hint">剩余计划 {{ Math.max(0, Number(executingSignal?.planned_shares ?? executingSignal?.suggested_shares ?? 0) - Number(executingSignal?.filled_shares || 0)) }} 股；本次可部分成交。</div>
          <div class="ltv-form-hint" v-if="isExecutionBuy && !executingSignal?.suggested_shares">
            原方案未生成有效股数，请重新生成计划。
          </div>
          <div class="ltv-form-error" v-if="executionFormError" role="alert">{{ executionFormError }}</div>
        </div>
        <div class="ltv-modal__btns">
          <button class="btn-sm btn-outline" :disabled="recordingExecution" @click="closeExecutionModal">取消</button>
          <button class="btn-sm" :disabled="recordingExecution" @click="submitExecution">
            {{ recordingExecution ? '记录中…' : '确认记录' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'
import api from '../api/index.js'
import { useHoldingsStore } from '../stores/holdings.js'
import * as rebalanceApi from '../api/rebalance.js'
import IntradayCheckPanel from '../components/IntradayCheckPanel.vue'
import T0Panel from '../components/T0Panel.vue'
import StrategyProfileBanner from '../components/StrategyProfileBanner.vue'
import { todayStr } from '../utils/date.js'

const holdingsStore = useHoldingsStore()
const holdingsLoadError = ref('')
const error = ref('')
const pendingPlanCommitStorageKey = 'rebalance-plan-commit:LOCAL-LIVE'
const executionStoragePrefix = 'rebalance-fill-'
const localAccount = ref(null)
const reconciliation = ref(null)
const invalidatedSignalDetails = ref([])
const showBrokerReportArchive = computed(() => Boolean(reconciliation.value) &&
  reconciliation.value.broker_status !== 'simulation')
const localLedgerAvailableCash = computed(() => {
  const account = localAccount.value
  if (!account) return '账本不可用'
  if (account.status !== 'active') return '期初未核准'
  if (account.integrity?.ok !== true || Number(account.integrity.account_version) !== Number(account.version)) {
    return '账本完整性未通过'
  }
  return account.available == null ? '余额不可用' : `¥${Number(account.available).toFixed(2)}`
})
const manualBudget = ref(null)
const unresolvedBrokerTrades = ref(null)
const unresolvedBrokerTradesError = ref('')
const unresolvedBrokerOrders = ref(null)
const unresolvedBrokerOrdersError = ref('')
const localIntents = ref([])
const localIntentsError = ref('')
const intentActionId = ref('')
const intentFormSignal = ref(null)
const intentForm = reactive({ intentId: '', quantity: '', limitPrice: '', feeCap: '' })
const intentFormError = ref('')
const intentFormSaving = ref(false)
const intentOrderInputs = reactive({})
const intentEvidence = reactive({})
const intentFees = reactive({})
const legacyQuarantine = ref(null)
const legacyQuarantineError = ref('')
const quarantineLoading = ref(false)
const quarantineOffset = ref(0)
const quarantineGroupLabels = { orders: '旧委托', trades: '旧成交', signals: '旧信号', sessions: '历史调仓会话' }
const legacyQuarantineCount = computed(() => Object.values(localAccount.value?.legacy_unknown || {})
  .reduce((total, count) => total + (Number.isInteger(Number(count)) ? Number(count) : 0), 0))
const quarantineHasNext = computed(() => ['orders', 'trades', 'signals', 'sessions'].some(kind =>
  quarantineOffset.value + (legacyQuarantine.value?.[kind]?.length || 0) < Number(legacyQuarantine.value?.counts?.[kind] || 0)))
const reconciliationMissingLabels = {
  'local.position.shares': '本地账本持仓股数',
  'account.account_id': '券商资金账号',
  'account.as_of': '账户快照时间',
  'account.cash': '账户总现金',
  'account.frozen': '账户冻结资金',
  'account.available': '账户可用资金',
  'account.fund_values': '账户资金数值',
  'account.fund_consistency': '账户资金分项一致性',
  'position.symbol': '持仓证券代码',
  'position.shares': '持仓股数',
  'position.sellable': '券商持仓可卖股数',
  'position.duplicate_code': '重复持仓代码',
  'order.id': '委托编号',
  'order.status': '委托状态',
  'broker.trade_report_health': '成交回报完整性状态',
  'broker.trade_fees': '成交佣金或印花税',
  'broker.trade_unassigned': '期初核准后无本地委托归属的成交',
  'broker.verified_account_id': '可核验的券商资金账号',
  'broker.configuration.account': 'QMT资金账号配置',
  'broker.configuration.userdata_dir': 'QMT userdata_mini目录',
  'broker.configuration.user': 'Easytrader登录账号配置',
  'broker.configuration.easytrader_account': 'Easytrader资金账号配置',
  broker_snapshot: '券商账户/持仓/委托快照',
}
const reconciliationMissingItems = computed(() => (reconciliation.value?.missing_items || [])
  .map(item => reconciliationMissingLabels[item] || item))
const openingCash = ref('')
const openingReason = ref('')
const openingSaving = ref(false)
const bindingSaving = ref(false)
const historicalFillOpen = ref(false)
const historicalFillSaving = ref(false)
const historicalFillError = ref('')
const historicalFillMessage = ref('')
const historicalFillPending = ref(null)
const historicalFillForm = reactive({
  requestId: crypto.randomUUID(), tradeDate: '', code: '', side: 'buy', shares: '', price: '',
  commission: '', stampDuty: '0', brokerTradeId: '', sourceSignalId: '', reason: '', allowOddLotSell: false,
})
const historicalFillReady = computed(() => {
  const shares = Number(historicalFillForm.shares)
  const price = Number(historicalFillForm.price)
  const commission = Number(historicalFillForm.commission)
  const stampDuty = Number(historicalFillForm.stampDuty)
  const sourceSignalId = historicalFillForm.sourceSignalId === '' ? null : Number(historicalFillForm.sourceSignalId)
  return localAccount.value?.status === 'active' && !localAccount.value?.broker_account_id &&
    Number.isInteger(Number(localAccount.value?.version)) && !historicalFillPending.value &&
    /^\d{4}-\d{2}-\d{2}$/.test(historicalFillForm.tradeDate) && historicalFillForm.tradeDate < todayStr() &&
    /^\d{6}$/.test(historicalFillForm.code) && ['buy', 'sell'].includes(historicalFillForm.side) &&
    Number.isInteger(shares) && shares > 0 && (historicalFillForm.side !== 'buy' || shares % 100 === 0) &&
    Number.isFinite(price) && price > 0 && Number.isFinite(commission) && commission >= 0 &&
    Number.isFinite(stampDuty) && stampDuty >= 0 && (historicalFillForm.side !== 'buy' || stampDuty === 0) &&
    historicalFillForm.brokerTradeId.trim().length > 0 && historicalFillForm.brokerTradeId.trim().length <= 64 &&
    historicalFillForm.reason.trim().length >= 8 &&
    (sourceSignalId === null || (Number.isInteger(sourceSignalId) && sourceSignalId > 0)) &&
    (historicalFillForm.side !== 'sell' || shares % 100 === 0 || historicalFillForm.allowOddLotSell)
})
const savedScreenings = ref([])
const selectedSaveId = ref(null)
const candidates = ref([])
const screeningMeta = ref({})
const loadedSave = ref(null)
const executionBlockReason = ref('')
const selectedSnapshotProfile = computed(() => loadedSave.value?.profile_id ? ({
  profile_id: loadedSave.value.profile_id,
  version: loadedSave.value.profile_version,
  name: loadedSave.value.save_name,
  status: loadedSave.value.profile_status || 'snapshot',
  fingerprint: loadedSave.value.profile_fingerprint || '',
}) : null)
const holdings = ref([])
const activeCode = ref(null)
const previewData = ref({})
const loadingCandidates = ref(false)
const loadingPreview = ref(false)
const runningRebalance = ref(false)
const planCalculated = ref(false)
const planDirty = ref(false)
const planFingerprint = ref('')
const planDecisionId = ref('')
const planCommitRequestId = ref('')
const pendingPlanCommit = ref(null)
const pendingPlanCommitError = ref('')
const planCommitted = ref(false)
const planCommitBlocked = ref(false)
const candidatesCollapsed = ref(false)
const rebalanceSignals = ref([])
const persistedSignals = ref([])
const pendingSignalsError = ref('')
const executionHistoryOpen = ref(false)
const executionHistory = ref([])
const executionHistoryError = ref('')
const executionHistoryLoading = ref(false)
const closeRemainderId = ref(null)
const closeRemainderReason = ref('')
const closeRemainderError = ref('')
const closingRemainder = ref(false)
const observeCandidates = ref([])
const skippedCandidates = ref([])
const filteredOut = ref([])
const classificationData = ref(null)
const decisionData = ref(null)
const decisionError = ref('')
const loadingDecision = ref(false)
const decisionCollapsed = ref(false)
const priceSource = ref('')
const positionRatio = ref(null)
const positionSource = ref('')
const cooldownStocks = ref([])
const cooldownAsOf = ref('')
const intradayData = ref({ results: [], sell_results: [], timestamp: '', source: '' })
const sellSignals = computed(() => rebalanceSignals.value.filter(s => s.side === 'sell' || s.side === 'reduce'))
const buySignals = computed(() => rebalanceSignals.value.filter(s => s.side === 'buy' || s.side === 'add'))
const restoredSignals = computed(() => persistedSignals.value.filter(s =>
  !rebalanceSignals.value.some(current => current.id === s.id && !planDirty.value)
))
const buyCodes = computed(() => buySignals.value.map(s => s.code))
const heldCodes = computed(() => (holdings.value || []).map(h => h.code))
const hasPlanSummary = computed(() => Boolean(
  rebalanceSignals.value.length || observeCandidates.value.length ||
  classificationData.value?.hold?.length || classificationData.value?.observe?.length ||
  classificationData.value?.sell?.length || classificationData.value?.stop_loss?.length ||
  classificationData.value?.reduce?.length || skippedCandidates.value.length || filteredOut.value.length
))
const planStatusText = computed(() => {
  if (runningRebalance.value) return '正在计算组合方案'
  if (loadingCandidates.value) return '正在读取选股快照'
  if (!selectedSaveId.value) return '等待选择选股快照'
  if (planDirty.value) return '参数已变更，方案待更新'
  if (planCalculated.value) return hasPlanSummary.value ? '方案已生成' : '当前无需主动调仓'
  return '等待生成方案'
})
const planActionLabel = computed(() => {
  if (runningRebalance.value) return '正在计算…'
  if (planCalculated.value && !planDirty.value && planFingerprint.value && !planCommitted.value) return '确认并提交方案'
  return planCalculated.value ? '重新预览方案' : '预览调仓方案'
})
const decisionSelected = computed(() => {
  if (Array.isArray(decisionData.value?.selected)) return decisionData.value.selected
  return (decisionData.value?.ranked_candidates || []).filter(row => row.decision_status === 'selected')
})
const decisionActiveRow = computed(() => (
  decisionSelected.value.find(row => row.code === activeCode.value) || decisionSelected.value[0] || null
))
const positionSourceLabel = computed(() => ({
  unified_regime: '统一市场模型',
  explicit: '显式设置',
  timing_fallback: '择时回退',
  panic_override: '极端风险覆盖',
}[positionSource.value] || '统一仓位'))
const modelRoutingLabel = computed(() => {
  const routing = screeningMeta.value?.model_routing || screeningMeta.value?.runtime_model_routing
  if (!routing) return '模型路由未记录，请确认该快照来源'
  const direction = {
    trend_momentum: '趋势动量',
    oversold_rebound: '超跌反转',
    volume_breakout: '放量突破',
  }[routing.direction] || routing.direction || '未记录方向'
  return routing.mode === 'volatility'
    ? `波动率模型切换 · ${direction}`
    : `未启用波动率切换 · ${direction}`
})
const activeExecution = computed(() => {
  const buy = buySignals.value.find(s => s.code === activeCode.value)
  const sell = sellSignals.value.find(s => s.code === activeCode.value)
  const item = buy
    ? intradayData.value.results?.find(r => r.code === activeCode.value)
    : sell
      ? intradayData.value.sell_results?.find(r => r.code === activeCode.value)
      : null
  return item?.execution || null
})
const activeLivePrice = computed(() => {
  const buy = intradayData.value.results?.find(r => r.code === activeCode.value)
  const sell = intradayData.value.sell_results?.find(r => r.code === activeCode.value)
  return buy?.current_price ?? sell?.current_price ?? null
})
const maData = computed(() => {
  const out = {}
  for (const code of buyCodes.value) {
    const p = previewData.value[code]
    if (p?.ma && Object.keys(p.ma).length) out[code] = p.ma
  }
  return out
})
const topN = ref(5)
const appliedTopN = ref(5)
const sizing = ref('equal')
const stopMode = ref('fixed')
const strategyControlEnabled = ref(true)
const manualTakeoverReason = ref('')
const cooldownStopWeeks = ref(null)
const cooldownProfitWeeks = ref(null)
const manualChecked = reactive(new Set())
const mainChartRef = ref(null)
const intradayBars = ref(null)
const intradayMeta = ref(null)
const intradayError = ref('')
const chartTab = ref('kline')
let mainInstance = null
let resizeBound = false
let intradayRequestSeq = 0
let decisionRequestSeq = 0
const showEditModal = ref(false)
const showAddHolding = ref(false)
const editingHolding = ref(null)
const holdForm = reactive({ code: '', name: '', shares: 100, cost_price: '', buy_date: '' })
const showExecutionModal = ref(false)
const executingSignal = ref(null)
const executionSharesInput = ref(null)
const executionForm = reactive({ shares: '', price: '', commission: '', stampDuty: '', brokerTradeId: '', requestId: '' })
const executionAccountVersion = ref(null)
const uncertainExecution = ref(null)
const uncertainFillRequests = ref([])
const uncertainFillRequestsError = ref('')
const recoveringFillSignalId = ref(null)

let ledgerLoadSeq = 0
let pendingLoadSeq = 0
let executionHistorySeq = 0
async function loadLedgerState() {
  const seq = ++ledgerLoadSeq
  const [accountRes, budgetRes, reconciliationRes, tradesRes, ordersRes, intentsRes] = await Promise.allSettled([
    rebalanceApi.getLocalAccount(), rebalanceApi.getCash(), rebalanceApi.getLocalReconciliation(),
    rebalanceApi.getUnresolvedBrokerTrades(500), rebalanceApi.getUnresolvedBrokerOrders(500),
    rebalanceApi.listLocalIntents(100),
  ])
  if (seq !== ledgerLoadSeq) return
  localAccount.value = accountRes.status === 'fulfilled' ? (accountRes.value?.data || accountRes.value) : null
  manualBudget.value = budgetRes.status === 'fulfilled' ? (budgetRes.value?.data || budgetRes.value)?.cash ?? null : null
  reconciliation.value = accountRes.status === 'rejected'
    ? { can_execute: false, reason: `本地账本读取失败：${accountRes.reason?.message || '服务不可用'}` }
    : reconciliationRes.status === 'rejected'
      ? { can_execute: false, reason: `券商对账读取失败：${reconciliationRes.reason?.message || '服务不可用'}` }
      : (reconciliationRes.value?.data || reconciliationRes.value)
  unresolvedBrokerTrades.value = tradesRes.status === 'fulfilled'
    ? (tradesRes.value?.data || tradesRes.value || [])
    : null
  unresolvedBrokerTradesError.value = tradesRes.status === 'rejected'
    ? (tradesRes.reason?.response?.data?.detail || tradesRes.reason?.message || '服务不可用')
    : ''
  unresolvedBrokerOrders.value = ordersRes.status === 'fulfilled'
    ? (ordersRes.value?.data || ordersRes.value || [])
    : null
  unresolvedBrokerOrdersError.value = ordersRes.status === 'rejected'
    ? (ordersRes.reason?.response?.data?.detail || ordersRes.reason?.message || '服务不可用')
    : ''
  localIntents.value = intentsRes.status === 'fulfilled' ? (intentsRes.value?.data || intentsRes.value || []) : []
  localIntentsError.value = intentsRes.status === 'rejected'
    ? (intentsRes.reason?.response?.data?.detail || intentsRes.reason?.message || '服务不可用')
    : ''
  await loadLegacyQuarantine(seq)
  return accountRes.status === 'fulfilled'
}

function startIntentPreparation(signal) {
  if (localAccount.value?.status !== 'active' || !localAccount.value?.broker_account_id || reconciliation.value?.can_execute !== true) {
    error.value = '本地账本、券商绑定或只读对账未通过，不能准备委托'
    return
  }
  const signalId = signal?.id ?? signal?.signal_id
  const plannedShares = Number(signal?.planned_shares ?? signal?.suggested_shares ?? 0)
  const remaining = plannedShares - Number(signal?.filled_shares || 0)
  if (!signalId || !Number.isInteger(remaining) || remaining <= 0 || (!planCommitted.value && !signal?._persisted)) {
    error.value = '信号尚未提交或缺少有效剩余计划股数，不能准备委托'
    return
  }
  intentFormSignal.value = signal
  intentFormError.value = ''
  const key = `rebalance-intent-${signalId}`
  try {
    const saved = JSON.parse(sessionStorage.getItem(key) || 'null')
    if (saved?.intent_id && Number(saved.signal_id) === Number(signalId)) {
      intentForm.intentId = saved.intent_id
      intentForm.quantity = String(saved.quantity)
      intentForm.limitPrice = String(saved.limit_price)
      intentForm.feeCap = String(saved.fee_cap)
    } else {
      intentForm.intentId = crypto.randomUUID()
      intentForm.quantity = String(remaining)
      intentForm.limitPrice = String(signal?.planned_price ?? signal?.current_price ?? '')
      intentForm.feeCap = ''
    }
  } catch (e) {
    intentForm.intentId = crypto.randomUUID()
    intentForm.quantity = String(remaining)
    intentForm.limitPrice = String(signal?.planned_price ?? signal?.current_price ?? '')
    intentForm.feeCap = ''
  }
}

function canPrepareIntent(signal) {
  const signalId = signal?.id ?? signal?.signal_id
  const remaining = Number(signal?.planned_shares ?? signal?.suggested_shares ?? 0) - Number(signal?.filled_shares || 0)
  const committed = signal?._persisted || (planCommitted.value && !planDirty.value)
  return Boolean(localAccount.value?.status === 'active' && localAccount.value?.broker_account_id &&
    reconciliation.value?.can_execute === true && !localIntentsError.value && committed && signalId &&
    Number.isInteger(remaining) && remaining > 0 && !isExpiredSignal(signal))
}

function cancelIntentPreparation() {
  if (intentFormSaving.value) return
  intentFormSignal.value = null
  intentFormError.value = ''
}

async function submitIntentPreparation() {
  const signal = intentFormSignal.value
  const signalId = signal?.id ?? signal?.signal_id
  const quantity = Number(intentForm.quantity)
  const limitPrice = String(intentForm.limitPrice).trim()
  const feeCap = String(intentForm.feeCap).trim()
  if (!signalId || localAccount.value?.status !== 'active' || reconciliation.value?.can_execute !== true) {
    intentFormError.value = '账户状态或券商对账已变化，请刷新后重试'
    return
  }
  const remaining = Number(signal?.planned_shares ?? signal?.suggested_shares ?? 0) - Number(signal?.filled_shares || 0)
  if (!Number.isInteger(quantity) || quantity <= 0 || quantity > remaining) {
    intentFormError.value = '委托股数须为正整数且不能超过剩余计划量'
    return
  }
  if (!/^\d+(?:\.\d{1,3})?$/.test(limitPrice) || Number(limitPrice) <= 0 ||
      !/^\d+(?:\.\d{1,2})?$/.test(feeCap)) {
    intentFormError.value = '请填写有效限价和非负费用上限'
    return
  }
  const payload = {
    intent_id: intentForm.intentId,
    signal_id: Number(signalId),
    expected_account_version: Number(localAccount.value.version),
    quantity,
    limit_price: limitPrice,
    fee_cap: feeCap,
  }
  const storageKey = `rebalance-intent-${signalId}`
  try {
    sessionStorage.setItem(storageKey, JSON.stringify(payload))
  } catch (e) {
    intentFormError.value = '无法保留本次委托请求 ID，已停止提交'
    return
  }
  intentFormSaving.value = true
  intentFormError.value = ''
  try {
    const response = await rebalanceApi.prepareLocalIntent(payload)
    const result = response?.data || response
    if (result?.intent_id !== payload.intent_id || !result?.state) throw new Error('准备委托回执异常，请刷新意图列表核对')
    applyInvalidatedSignalResult(result)
    sessionStorage.removeItem(storageKey)
    intentFormSignal.value = null
    const [accountLoaded, pendingLoaded] = await Promise.all([loadLedgerState(), loadPersistedSignals()])
    if (!accountLoaded || !pendingLoaded) error.value = '委托已预留，但页面刷新不完整，请刷新账本与委托列表核对'
  } catch (e) {
    const status = Number(e.response?.status ?? e.status)
    if (status >= 400 && status < 500) sessionStorage.removeItem(storageKey)
    intentFormError.value = e.response?.data?.detail || e.message || String(e)
  } finally {
    intentFormSaving.value = false
  }
}

async function runIntentAction(intent, action) {
  if (intentActionId.value) return
  intentActionId.value = intent.intent_id
  error.value = ''
  try {
    if (action === 'cancel') {
      const response = await rebalanceApi.cancelLocalIntent(intent.intent_id)
      applyInvalidatedSignalResult(response?.data || response)
    } else if (action === 'attach') {
      const orderId = String(intentOrderInputs[intent.intent_id] || '').trim()
      if (!orderId) throw new Error('请填写券商委托编号')
      const response = await rebalanceApi.attachLocalBrokerOrder(intent.intent_id, orderId)
      applyInvalidatedSignalResult(response?.data || response)
    } else if (action === 'finalize') {
      const response = await rebalanceApi.finalizeLocalBrokerOrder(intent.intent_id)
      applyInvalidatedSignalResult(response?.data || response)
    }
    await loadLedgerState()
    await loadPersistedSignals()
    if (action === 'attach') await loadIntentEvidence(intent.intent_id)
  } catch (e) {
    error.value = e.response?.data?.detail || e.message || String(e)
  } finally {
    intentActionId.value = ''
  }
}

async function loadIntentEvidence(intentId) {
  intentActionId.value = intentId
  error.value = ''
  try {
    const response = await rebalanceApi.getLocalBrokerEvidence(intentId)
    const evidence = response?.data || response
    intentEvidence[intentId] = evidence
    for (const trade of evidence.trades || []) {
      const key = `${intentId}:${trade.trade_id}`
      if (!intentFees[key]) intentFees[key] = {
        commission: trade.commission == null ? '' : String(trade.commission),
        stampDuty: trade.stamp_duty == null ? '' : String(trade.stamp_duty),
      }
    }
  } catch (e) {
    error.value = e.response?.data?.detail || e.message || String(e)
  } finally {
    intentActionId.value = ''
  }
}

async function bookIntentTrade(intent, trade) {
  const key = `${intent.intent_id}:${trade.trade_id}`
  const fees = intentFees[key] || {}
  if (trade.booked || !/^\d+(?:\.\d{1,2})?$/.test(String(fees.commission)) ||
      !/^\d+(?:\.\d{1,2})?$/.test(String(fees.stampDuty))) {
    error.value = '请核对并填写券商回报中的实际佣金和印花税'
    return
  }
  intentActionId.value = intent.intent_id
  error.value = ''
  try {
    const response = await rebalanceApi.recordLocalBrokerFill(intent.intent_id, {
      broker_trade_id: trade.trade_id,
      commission: String(fees.commission),
      stamp_duty: String(fees.stampDuty),
    })
    const result = response?.data || response
    if (result?.broker_trade_id !== trade.trade_id || typeof result?.already_recorded !== 'boolean') {
      throw new Error('券商成交入账回执异常，请刷新账本和回报核对')
    }
    applyInvalidatedSignalResult(result)
    await loadLedgerState()
    await loadPersistedSignals()
    await loadIntentEvidence(intent.intent_id)
  } catch (e) {
    error.value = e.response?.data?.detail || e.message || String(e)
  } finally {
    intentActionId.value = ''
  }
}

function applyInvalidatedSignalResult(result) {
  const invalidatedIds = new Set((result?.invalidated_signal_ids || []).map(Number))
  invalidatedSignalDetails.value = Array.isArray(result?.invalidated_signal_details)
    ? result.invalidated_signal_details.filter(item =>
      Number.isSafeInteger(item?.signal_id) && item.signal_id > 0 &&
      Number.isSafeInteger(item?.planned_shares) && Number.isSafeInteger(item?.filled_shares) &&
      Number.isSafeInteger(item?.invalidated_remaining_shares) && item.invalidated_remaining_shares >= 0)
    : []
  rebalanceSignals.value.forEach(item => {
    if (invalidatedIds.has(Number(item.id ?? item.signal_id))) {
      item.execution_status = 'cancelled'
      item.unfilled_reason = '账户版本已变化，旧计划已失效，请重新生成'
    }
  })
}

async function loadLegacyQuarantine(seq = ledgerLoadSeq) {
  if (legacyQuarantineCount.value <= 0) {
    legacyQuarantine.value = null
    legacyQuarantineError.value = ''
    quarantineOffset.value = 0
    return
  }
  quarantineLoading.value = true
  legacyQuarantineError.value = ''
  try {
    const response = await rebalanceApi.getLocalQuarantine(50, quarantineOffset.value)
    if (seq !== ledgerLoadSeq) return
    legacyQuarantine.value = response?.data || response
  } catch (requestError) {
    if (seq === ledgerLoadSeq) {
      legacyQuarantine.value = null
      legacyQuarantineError.value = requestError?.message || '服务不可用'
    }
  } finally {
    if (seq === ledgerLoadSeq) quarantineLoading.value = false
  }
}

async function changeQuarantinePage(delta) {
  const nextOffset = quarantineOffset.value + delta
  if (nextOffset < 0 || quarantineLoading.value) return
  quarantineOffset.value = nextOffset
  await loadLegacyQuarantine()
}

function missingQuarantineIdentity(kind, row) {
  if (kind === 'sessions') {
    return [
      row.account_id == null || String(row.account_id).trim() === '' ? '账户' : '',
      Number(row.account_version) > 0 ? '' : '账户版本',
    ].filter(Boolean)
  }
  if (kind === 'signals') {
    return [
      row.account_id == null || String(row.account_id).trim() === '' ? '账户' : '',
      Number(row.account_version) > 0 ? '' : '账户版本',
      Number(row.planned_shares) > 0 ? '' : '计划股数',
    ].filter(Boolean)
  }
  return [['account_id', '账户'], ['broker', '券商'], ['environment', '环境'], ['source', '来源']]
    .filter(([key]) => row[key] == null || String(row[key]).trim() === '').map(([, label]) => label)
}

function formatQuarantineEntry(kind, row) {
  if (kind === 'orders') return `委托 #${row.id} · ${row.symbol} · ${sideLabel({ side: row.side })} ${row.quantity} 股 · ${row.status} · ${row.created_at}`
  if (kind === 'trades') return `成交 #${row.id}（委托 #${row.order_id}）· ${row.symbol} · ${sideLabel({ side: row.side })} ${row.quantity} 股 · ${row.trade_time}`
  if (kind === 'sessions') return `调仓会话 #${row.id} · ${row.batch_id} · ${row.eval_date} · ${row.created_at}`
  return `信号 #${row.id} · ${row.batch_id} · ${row.code} · ${sideLabel({ side: row.side })} · 已成交 ${row.filled_shares || 0}/${row.planned_shares ?? '未知'} 股 · ${row.execution_status}`
}

async function loadPersistedSignals() {
  const seq = ++pendingLoadSeq
  try {
    const rows = []
    const ids = new Set()
    let offset = 0
    while (true) {
      const response = await rebalanceApi.getPendingSignals(null, 200, offset)
      const page = response?.data ?? response
      if (!Array.isArray(page)) throw new Error('待确认信号格式无效')
      if (seq !== pendingLoadSeq) return false
      for (const signal of page) {
        if (!Number.isInteger(signal?.id) || ids.has(signal.id)) {
          throw new Error('待确认信号分页重复或缺少编号，请刷新核对')
        }
        ids.add(signal.id)
        rows.push(signal)
      }
      if (page.length < 200) break
      offset += page.length
    }
    persistedSignals.value = rows.map(s => ({
      ...s, _persisted: true, suggested_shares: Number(s.planned_shares),
      filled_shares: Number(s.filled_shares || 0),
    }))
    pendingSignalsError.value = ''
    return true
  } catch (e) {
    if (seq !== pendingLoadSeq) return false
    persistedSignals.value = []
    pendingSignalsError.value = `未完成信号读取失败：${e.message || e}`
    return false
  }
}

async function loadExecutionHistory() {
  const seq = ++executionHistorySeq
  executionHistoryLoading.value = true
  try {
    const response = await rebalanceApi.getHistory(30, 50)
    const rows = response?.data ?? response
    if (seq !== executionHistorySeq) return
    if (!Array.isArray(rows)) throw new Error('执行历史格式无效')
    executionHistory.value = rows
    executionHistoryError.value = ''
  } catch (e) {
    if (seq !== executionHistorySeq) return
    executionHistory.value = []
    executionHistoryError.value = `执行历史读取失败：${e.message || e}`
  } finally {
    if (seq === executionHistorySeq) executionHistoryLoading.value = false
  }
}

function toggleExecutionHistory() {
  executionHistoryOpen.value = !executionHistoryOpen.value
  if (executionHistoryOpen.value) loadExecutionHistory()
}

function isExpiredSignal(signal) {
  return String(signal.created_at || '').slice(0, 10) !== todayStr()
}

async function closeSignalRemainder(signal) {
  if (closingRemainder.value || closeRemainderReason.value.length < 8) return
  if (localAccount.value?.status !== 'active') {
    closeRemainderError.value = '本地账户未核准或账本不可用'
    return
  }
  closingRemainder.value = true
  closeRemainderError.value = ''
  try {
    const response = await rebalanceApi.closeSignalRemainder(
      signal.id, Number(localAccount.value.version), closeRemainderReason.value,
    )
    const result = response?.data ?? response
    if (result?.signal_id !== signal.id || result?.execution_status !== 'cancelled' ||
        result?.closed_shares !== signal.planned_shares - signal.filled_shares) {
      throw new Error('关闭回执格式异常，请刷新账本与未完成信号核对')
    }
    const [accountLoaded, pendingLoaded] = await Promise.all([loadLedgerState(), loadPersistedSignals()])
    if (!accountLoaded || !pendingLoaded) {
      error.value = '剩余计划关闭已返回成功，但账本或待办刷新不完整；请刷新核对后继续'
      return
    }
    closeRemainderId.value = null
    closeRemainderReason.value = ''
    error.value = ''
  } catch (e) {
    closeRemainderError.value = e.response?.data?.detail || e.message || String(e)
  } finally {
    closingRemainder.value = false
  }
}

async function activateLedger() {
  if (!localAccount.value?.holdings_fingerprint) return
  openingSaving.value = true
  try {
    await rebalanceApi.activateLocalAccount(openingCash.value, localAccount.value.holdings_fingerprint, openingReason.value)
    openingCash.value = ''
    openingReason.value = ''
    await loadLedgerState()
    invalidatePlan()
  } catch (e) {
    error.value = '期初核准失败: ' + (e.message || e)
  } finally {
    openingSaving.value = false
  }
}

function historicalFillStorageKey() {
  return `local-historical-fill:${localAccount.value?.account_id || 'LOCAL-LIVE'}`
}

function clearHistoricalFillForm() {
  Object.assign(historicalFillForm, {
    requestId: crypto.randomUUID(), tradeDate: '', code: '', side: 'buy', shares: '', price: '',
    commission: '', stampDuty: '0', brokerTradeId: '', sourceSignalId: '', reason: '', allowOddLotSell: false,
  })
}

function toggleHistoricalFillForm() {
  historicalFillOpen.value = !historicalFillOpen.value
  historicalFillError.value = ''
  if (!historicalFillOpen.value) return
  try {
    const saved = JSON.parse(sessionStorage.getItem(historicalFillStorageKey()) || 'null')
    if (saved?.request_id) {
      historicalFillPending.value = saved
      Object.assign(historicalFillForm, {
        requestId: saved.request_id, tradeDate: saved.trade_date, code: saved.code, side: saved.side,
        shares: String(saved.shares), price: String(saved.price), commission: String(saved.commission),
        stampDuty: String(saved.stamp_duty), brokerTradeId: saved.broker_trade_id,
        sourceSignalId: saved.source_signal_id == null ? '' : String(saved.source_signal_id),
        reason: saved.reason, allowOddLotSell: saved.allow_odd_lot_sell === true,
      })
      historicalFillError.value = '上次请求结果未确认；只能使用原请求ID和原参数重试，或明确放弃后重新填写。'
    } else {
      historicalFillPending.value = null
      clearHistoricalFillForm()
    }
  } catch (error) {
    historicalFillError.value = '无法读取本地待确认请求，请刷新页面后核对账本版本。'
  }
}

function discardHistoricalFillRequest() {
  if (historicalFillSaving.value || !window.confirm('确认放弃该未确认请求？若服务器已成功入账，重新填写前请先刷新账本核对。')) return
  sessionStorage.removeItem(historicalFillStorageKey())
  historicalFillPending.value = null
  historicalFillError.value = ''
  clearHistoricalFillForm()
}

async function submitHistoricalFill() {
  if (historicalFillSaving.value) return
  let payload = historicalFillPending.value
  if (!payload) {
    if (!historicalFillReady.value) return
    const sideText = historicalFillForm.side === 'buy' ? '买入' : '卖出'
    if (!window.confirm(`确认将 ${historicalFillForm.tradeDate} ${historicalFillForm.code} ${sideText} ${historicalFillForm.shares} 股 @ ${historicalFillForm.price} 元，按券商成交编号 ${historicalFillForm.brokerTradeId.trim()} 补入本地账本？此操作更新当前现金和持仓，不会修改旧信号。`)) return
    payload = {
      request_id: historicalFillForm.requestId,
      trade_date: historicalFillForm.tradeDate,
      code: historicalFillForm.code.trim(),
      side: historicalFillForm.side,
      shares: Number(historicalFillForm.shares),
      price: historicalFillForm.price,
      expected_account_version: Number(localAccount.value.version),
      commission: historicalFillForm.commission,
      stamp_duty: historicalFillForm.side === 'buy' ? '0' : historicalFillForm.stampDuty,
      broker_trade_id: historicalFillForm.brokerTradeId.trim(),
      source_signal_id: historicalFillForm.sourceSignalId === '' ? null : Number(historicalFillForm.sourceSignalId),
      reason: historicalFillForm.reason.trim(),
      allow_odd_lot_sell: historicalFillForm.side === 'sell' && historicalFillForm.allowOddLotSell,
    }
    try {
      sessionStorage.setItem(historicalFillStorageKey(), JSON.stringify(payload))
      historicalFillPending.value = payload
    } catch (error) {
      historicalFillError.value = '浏览器无法保存幂等请求，未发送补录。'
      return
    }
  }
  historicalFillSaving.value = true
  historicalFillError.value = ''
  historicalFillMessage.value = ''
  try {
    const response = await rebalanceApi.recordHistoricalFill(payload)
    const result = response?.data ?? response
    if (result?.signal_updated !== false || !Number.isInteger(Number(result?.account_version))) {
      throw new Error('服务器返回结果不完整；保留原请求ID供核对重试。')
    }
    sessionStorage.removeItem(historicalFillStorageKey())
    historicalFillPending.value = null
    historicalFillMessage.value = result.already_recorded
      ? `该请求此前已入账；账本版本 ${result.account_version}。`
      : `历史成交已单独入账；账本版本 ${result.account_version}，旧信号未改变。`
    clearHistoricalFillForm()
    await Promise.all([loadLedgerState(), loadPersistedSignals(), loadExecutionHistory()])
  } catch (error) {
    const status = error?.response?.status
    historicalFillError.value = error?.response?.data?.detail || error.message || '补录结果未知，请用原请求ID重试或刷新账本核对。'
    if (status === 400 || status === 422) {
      sessionStorage.removeItem(historicalFillStorageKey())
      historicalFillPending.value = null
      historicalFillForm.requestId = crypto.randomUUID()
    }
  } finally {
    historicalFillSaving.value = false
  }
}

async function bindLedgerBroker() {
  bindingSaving.value = true
  try {
    const response = await rebalanceApi.bindLocalBroker(Number(localAccount.value?.version))
    applyInvalidatedSignalResult(response?.data || response)
    await loadLedgerState()
    invalidatePlan()
  } catch (e) {
    error.value = '绑定券商账号失败: ' + (e.message || e)
  } finally {
    bindingSaving.value = false
  }
}
const executionFormError = ref('')
const recordingExecution = ref(false)
const isExecutionBuy = computed(() => ['buy', 'add'].includes(executingSignal.value?.side))

const displayCandidates = computed(() => {
  const list = []
  const seen = new Set()
  for (const c of candidates.value) {
    if (!seen.has(c.code)) {
      const h = holdings.value.find(h => h.code === c.code)
      const audit = decisionData.value?.ranked_candidates?.find(row => row.code === c.code) || {}
      list.push({ ...c, ...audit, is_holding: !!h, shares: h?.shares, cost_price: h?.cost_price, realtime_price: null, score: c.score ?? c.l0_score })
      seen.add(c.code)
    }
  }
  for (const h of holdings.value) {
    if (!seen.has(h.code)) {
      list.push({ code: h.code, name: h.name, is_holding: true, shares: h.shares, cost_price: h.cost_price, score: null, realtime_price: null })
      seen.add(h.code)
    }
  }
  return list
})

function isChecked(code) {
  const h = holdings.value.find(h => h.code === code)
  if (h) return true
  return manualChecked.has(code)
}

function toggleCheck(code) {
  const h = holdings.value.find(h => h.code === code)
  if (h) return
  if (manualChecked.has(code)) manualChecked.delete(code)
  else manualChecked.add(code)
}

const checkedCount = computed(() => {
  const codes = new Set(holdings.value.map(h => h.code))
  for (const c of manualChecked) codes.add(c)
  return codes.size
})

const checkedCodes = computed(() => {
  const codes = [...new Set([...holdings.value.map(h => h.code), ...manualChecked])]
  return codes
})

const activePreview = computed(() => {
  if (!activeCode.value) return null
  return previewData.value[activeCode.value] || null
})

const qualityClass = computed(() => {
  const label = activePreview.value?.entry_quality?.label
  return { '优秀': 'ltv-q--great', '良好': 'ltv-q--good', '一般': 'ltv-q--ok', '观望': 'ltv-q--poor' }[label] || 'ltv-q--poor'
})

function technicalStatusClass(status) {
  return `ltv-tech--${['bullish', 'bearish', 'neutral', 'risk'].includes(status) ? status : 'neutral'}`
}

function technicalStatusIcon(status) {
  return { bullish: '↑', bearish: '↓', neutral: '—', risk: '!' }[status] || '—'
}

function technicalStatusLabel(status) {
  return { bullish: '利好', bearish: '利空', neutral: '中性', risk: '风险' }[status] || '中性'
}

// ─── 数据加载 ───

async function loadSavedScreenings() {
  try {
    const res = await rebalanceApi.listSavedScreenings(30)
    savedScreenings.value = res?.data || res || []
  } catch (e) { /* ignore */ }
}

function restorePendingPlanCommit() {
  pendingPlanCommit.value = null
  pendingPlanCommitError.value = ''
  try {
    const saved = localStorage.getItem(pendingPlanCommitStorageKey)
    const legacy = sessionStorage.getItem(pendingPlanCommitStorageKey)
    const raw = saved ?? legacy
    const pending = JSON.parse(raw || 'null')
    if (!pending) return
    const payload = pending?.payload || {}
    if (pending.version !== 1 || !Number.isInteger(Number(pending.screening_save_id)) || Number(pending.screening_save_id) <= 0 ||
        !/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(payload.commit_request_id || '') ||
        !/^[0-9a-f]{64}$/i.test(payload.expected_plan_fingerprint || '') ||
        !payload.decision_id) {
      pendingPlanCommitError.value = '浏览器中存在格式无效的未确认调仓请求。'
      return
    }
    if (!saved && legacy) {
      localStorage.setItem(pendingPlanCommitStorageKey, legacy)
      sessionStorage.removeItem(pendingPlanCommitStorageKey)
    }
    pendingPlanCommit.value = pending
  } catch {
    pendingPlanCommitError.value = '无法读取浏览器中的未确认调仓请求。'
  }
}

function loadUnconfirmedExecutionRequests() {
  const requests = new Map()
  const keys = new Set()
  try {
    for (const storage of [localStorage, sessionStorage]) {
      for (let index = 0; index < storage.length; index += 1) {
        const key = storage.key(index)
        if (key?.startsWith(executionStoragePrefix)) keys.add(key)
      }
    }
    for (const key of keys) {
      const signalId = Number(key.slice(executionStoragePrefix.length))
      if (!Number.isSafeInteger(signalId) || signalId <= 0) {
        throw new Error('浏览器中的待确认成交信号编号无效')
      }
      const raw = localStorage.getItem(key) ?? sessionStorage.getItem(key)
      const payload = JSON.parse(raw || 'null')
      if (!isValidStoredExecutionPayload(payload)) {
        throw new Error(`信号 #${signalId} 的待确认成交请求内容不完整`)
      }
      if (!localStorage.getItem(key)) localStorage.setItem(key, raw)
      sessionStorage.removeItem(key)
      requests.set(signalId, { signalId, payload })
    }
    uncertainFillRequests.value = [...requests.values()].sort((left, right) => left.signalId - right.signalId)
    uncertainFillRequestsError.value = ''
  } catch (error) {
    uncertainFillRequests.value = [...requests.values()].sort((left, right) => left.signalId - right.signalId)
    uncertainFillRequestsError.value = error.message || '无法读取浏览器中的待确认成交请求'
  }
}

function isValidStoredExecutionPayload(payload) {
  return Boolean(payload &&
    /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(payload.request_id || '') &&
    Number.isSafeInteger(payload.shares) && payload.shares > 0 &&
    Number.isSafeInteger(payload.expected_account_version) && payload.expected_account_version > 0 &&
    /^\d+(?:\.\d{1,3})?$/.test(String(payload.price || '')) &&
    /^\d+(?:\.\d{1,2})?$/.test(String(payload.commission ?? '')) &&
    /^\d+(?:\.\d{1,2})?$/.test(String(payload.stamp_duty ?? '')))
}

function saveUnconfirmedExecutionRequest(signalId, payload) {
  const key = `${executionStoragePrefix}${signalId}`
  localStorage.setItem(key, JSON.stringify(payload))
  try { sessionStorage.removeItem(key) } catch {}
  loadUnconfirmedExecutionRequests()
}

function clearUnconfirmedExecutionRequest(signalId) {
  const key = `${executionStoragePrefix}${signalId}`
  localStorage.removeItem(key)
  sessionStorage.removeItem(key)
  loadUnconfirmedExecutionRequests()
}

function syncPendingRequestsAcrossTabs(event) {
  if (event.storageArea !== localStorage) return
  if (event.key === null || event.key === pendingPlanCommitStorageKey) restorePendingPlanCommit()
  if (event.key === null || event.key.startsWith(executionStoragePrefix)) loadUnconfirmedExecutionRequests()
}

async function retryUncertainFill(request) {
  if (recoveringFillSignalId.value !== null || showExecutionModal.value) return
  recoveringFillSignalId.value = request.signalId
  try {
    const response = await rebalanceApi.getSignal(request.signalId)
    const signal = response?.data ?? response
    if (!signal || Number(signal.id) !== request.signalId || signal.account_id !== localAccount.value?.account_id) {
      throw new Error('信号账户或编号与待确认请求不一致')
    }
    executeSignal({ ...signal, _persisted: true })
  } catch (error) {
    uncertainFillRequestsError.value = `信号 #${request.signalId} 暂时无法核对：${error.message || error}`
  } finally {
    recoveringFillSignalId.value = null
  }
}

async function loadHoldings() {
  try {
    await holdingsStore.load(true)
    holdings.value = holdingsStore.holdings?.value ?? []
    holdingsLoadError.value = ''
    return true
  } catch (e) {
    console.error('loadHoldings error:', e)
    holdings.value = []
    holdingsLoadError.value = e.message || '读取失败'
    return false
  }
}

async function loadCooldown() {
  try {
    const res = await rebalanceApi.getCooldown()
    const data = res?.data || res || {}
    cooldownStocks.value = data.items || []
    cooldownAsOf.value = data.as_of || ''
  } catch (e) {
    cooldownStocks.value = []
    cooldownAsOf.value = ''
  }
}

async function viewCooldownStock(item) {
  activeCode.value = item.code
  await loadStockPreview(item.code)
}

async function loadStockPreview(code) {
  if (!code || previewData.value[code]) return
  loadingPreview.value = true
  try {
    const res = await rebalanceApi.previewStocks([code], stopMode.value)
    const data = res?.data || res
    previewData.value = { ...previewData.value, ...(data.results || {}) }
    await nextTick()
    renderCharts()
  } catch (e) {
    error.value = '加载冷冻期股票K线失败: ' + (e.message || e)
  } finally {
    loadingPreview.value = false
  }
}

async function onSaveSelected() {
  if (!selectedSaveId.value) {
    candidates.value = []
    screeningMeta.value = {}
    loadedSave.value = null
    executionBlockReason.value = ''
    manualChecked.clear()
    previewData.value = {}
    rebalanceSignals.value = []
    observeCandidates.value = []
    skippedCandidates.value = []
    filteredOut.value = []
    classificationData.value = null
    decisionData.value = null
    decisionError.value = ''
    positionRatio.value = null
    positionSource.value = ''
    planCalculated.value = false
    planDirty.value = false
    planFingerprint.value = ''
    planDecisionId.value = ''
    planCommitRequestId.value = ''
    planCommitted.value = false
    planCommitBlocked.value = false
    return
  }
  loadingCandidates.value = true
  error.value = ''
  candidates.value = []
  screeningMeta.value = {}
  previewData.value = {}
  decisionData.value = null
  decisionError.value = ''
  positionRatio.value = null
  positionSource.value = ''
  planCalculated.value = false
  planDirty.value = false
  planFingerprint.value = ''
  planDecisionId.value = ''
  planCommitRequestId.value = ''
  planCommitted.value = false
  planCommitBlocked.value = false
  try {
    const res = await rebalanceApi.getSavedScreening(selectedSaveId.value)
    const data = res?.data || res
    loadedSave.value = data
    const cands = data?.candidates || []
    screeningMeta.value = data?.params_json || {}
    if (!cands.length) {
      error.value = '该选股结果没有候选股数据'
      loadingCandidates.value = false
      return
    }
    candidates.value = cands.map((c, i) => ({
      ...c,
      code: c.code,
      name: c.name || c.code,
      score: c.l0_score ?? c.score ?? (100 - i),
      close: c.close,
    }))
    suppressAutoRebalance = true
    manualChecked.clear()
    for (const c of candidates.value) manualChecked.add(c.code)
    if (!activeCode.value) activeCode.value = cands[0].code
    await loadPreview()
    planDirty.value = true
  } catch (e) {
    error.value = '加载选股结果失败: ' + (e.message || e)
  } finally {
    loadingCandidates.value = false
    suppressAutoRebalance = false
  }
}

async function loadPreview(silent = false) {
  const codes = [...new Set([...checkedCodes.value, activeCode.value].filter(Boolean))]
  if (!codes.length) return
  if (!silent) loadingPreview.value = true
  try {
    const res = await rebalanceApi.previewStocks(codes, stopMode.value)
    const data = res?.data || res
    previewData.value = { ...previewData.value, ...(data.results || {}) }
    if (!silent) {
      await nextTick()
      renderCharts()
    }
  } catch (e) {
    if (!silent) error.value = 'K线数据加载失败: ' + (e.message || e)
  } finally {
    if (!silent) loadingPreview.value = false
  }
}

const STOP_MODES = ['fixed', 'dynamic', 'combined']
const STOP_LABELS = { fixed: '固定百分比', dynamic: '动态支撑压力', combined: '固定+动态' }

function refreshPreview() {
  if (checkedCodes.value.length) loadPreview()
}

// ─── 持仓编辑 ───

function editHolding(h) {
  const holding = holdings.value.find(row => row.code === h.code)
  if (!holding) { error.value = '持仓已变化，请刷新后重试'; return }
  editingHolding.value = holding
  holdForm.code = holding.code
  holdForm.name = holding.name
  holdForm.shares = holding.shares || 100
  holdForm.cost_price = holding.cost_price || ''
  holdForm.buy_date = holding.buy_date || ''
  showEditModal.value = true
}

function closeEditModal() {
  showEditModal.value = false
  editingHolding.value = null
  holdForm.code = ''; holdForm.name = ''; holdForm.shares = 100; holdForm.cost_price = ''; holdForm.buy_date = ''
}

async function recoverHoldingConflict(error) {
  if (Number(error?.status ?? error?.response?.status) !== 409) return false
  closeEditModal()
  invalidatePlan()
  const [holdingsReady, ledgerReady] = await Promise.all([
    loadHoldings(), loadLedgerState(), loadCooldown(),
  ])
  return Boolean(holdingsReady && ledgerReady)
}

async function saveHolding() {
  if (!holdForm.code || !holdForm.name) return
  if (!localAccount.value) { error.value = '本地账本状态不可用，请刷新后重试'; return }
  try {
    if (editingHolding.value) {
      const original = editingHolding.value
      const positionChanged = Number(holdForm.shares) !== Number(original.shares) ||
        Number(holdForm.cost_price) !== Number(original.cost_price) ||
        (holdForm.buy_date || '') !== (original.buy_date || '')
      if (localAccount.value.status === 'active' && positionChanged) {
        error.value = '实盘账户已激活，持仓数量、成本和日期只能通过成交事务变更'
        return
      }
      const data = { name: holdForm.name, expected_id: original.id, expected_version: original.version }
      if (localAccount.value.status === 'uninitialized') {
        data.shares = holdForm.shares
        data.cost_price = String(holdForm.cost_price)
        data.buy_date = holdForm.buy_date || ''
        data.expected_shares = original.shares
        data.expected_cost_price = original.cost_price
        data.expected_buy_date = original.buy_date || ''
      }
      if (localAccount.value.status !== 'active' && localAccount.value.status !== 'uninitialized') {
        error.value = '本地账本状态不可用，请刷新后重试'
        return
      }
      const reason = positionChanged ? prompt('请填写手工调整持仓数量、成本或日期的原因（至少8字）')?.trim() : undefined
      if (positionChanged && !reason) return
      if (positionChanged && reason.length < 8) { error.value = '原因至少需要8字'; return }
      if (reason) data.reason = reason
      await api.put('/portfolio/holdings/' + editingHolding.value.code, data)
    } else {
      if (localAccount.value.status !== 'uninitialized') { error.value = '实盘账户已激活，新增持仓须通过成交事务'; return }
      await api.post('/portfolio/holdings', {
        code: holdForm.code, name: holdForm.name, shares: holdForm.shares,
        cost_price: holdForm.cost_price, buy_date: holdForm.buy_date || undefined
      })
    }
    closeEditModal()
    await Promise.all([loadHoldings(), loadCooldown()])
    await loadPreview()
    invalidatePlan()
  } catch (e) {
    const recovered = await recoverHoldingConflict(e)
    error.value = Number(e?.status ?? e?.response?.status) === 409
      ? `${e.message || '持仓版本已变化'}；${recovered ? '已刷新持仓与账本，请重新打开编辑后再提交' : '刷新状态失败，请手动刷新核对'}`
      : '保存持仓失败: ' + (e.message || e)
  }
}

async function deleteHolding() {
  if (localAccount.value?.status !== 'uninitialized') { error.value = '实盘账户已激活，删除持仓须通过成交事务'; return }
  if (!editingHolding.value || !confirm(`确认删除 ${editingHolding.value.code} ${editingHolding.value.name} 的本地持仓记录？此操作不会向券商下单，也不会记录卖出成交。`)) return
  const reason = prompt('请填写删除本地持仓记录的原因（至少8字）')?.trim()
  if (!reason) return
  if (reason.length < 8) { error.value = '原因至少需要8字'; return }
  try {
    await api.delete('/portfolio/holdings/' + editingHolding.value.code, {
      data: { reason, expected_shares: editingHolding.value.shares,
        expected_cost_price: editingHolding.value.cost_price,
        expected_buy_date: editingHolding.value.buy_date || '',
        expected_id: editingHolding.value.id,
        expected_version: editingHolding.value.version }
    })
    closeEditModal()
    await Promise.all([loadHoldings(), loadCooldown()])
    await loadPreview()
    invalidatePlan()
  } catch (e) {
    const recovered = await recoverHoldingConflict(e)
    error.value = Number(e?.status ?? e?.response?.status) === 409
      ? `${e.message || '持仓版本已变化'}；${recovered ? '已刷新持仓与账本，请重新确认后操作' : '刷新状态失败，请手动刷新核对'}`
      : '删除失败: ' + (e.message || e)
  }
}

watch(showAddHolding, (v) => { if (v) { showEditModal.value = true; showAddHolding.value = false } })

// ─── 图表渲染 ───

const UP = '#ef4444'
const DOWN = '#22c55e'

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

async function renderCharts() {
  await nextTick()
  renderMainChart()
  fetchIntraday()
}

const chartOverlayText = computed(() => {
  if (chartTab.value === 'intraday') return intradayBars.value?.length ? '' : (intradayError.value || '加载分时...')
  if (chartTab.value === 'compare') return checkedCodes.value.length > 1 ? '' : '勾选2只以上显示走势对比'
  return activePreview.value?.kline?.length ? '' : (loadingPreview.value ? '加载K线数据...' : '暂无K线数据')
})

function switchChart(tab) {
  chartTab.value = tab
  nextTick(() => renderMainChart())
}

async function renderMainChart() {
  const el = mainChartRef.value
  if (!el) return
  const echarts = await import('echarts')
  if (!mainInstance || mainInstance.isDisposed()) mainInstance = echarts.init(el)
  else mainInstance.resize()
  if (chartTab.value === 'intraday') buildIntradayOption(mainInstance)
  else if (chartTab.value === 'compare') buildCompareOption(mainInstance)
  else buildKlineOption(mainInstance)
}

async function fetchIntraday() {
  const requestSeq = ++intradayRequestSeq
  const code = activeCode.value
  if (!code) {
    intradayBars.value = []
    intradayMeta.value = null
    intradayError.value = ''
    return
  }
  intradayBars.value = []
  intradayMeta.value = null
  intradayError.value = ''
  if (chartTab.value === 'intraday') await renderMainChart()
  try {
    const res = await api.get(`/rebalance/intraday-bars/${code}`)
    if (requestSeq !== intradayRequestSeq) return
    const payload = res?.data || res
    intradayBars.value = payload?.bars || []
    intradayMeta.value = payload
    if (!intradayBars.value.length) intradayError.value = payload?.message || '暂无可用分时数据'
    if (chartTab.value === 'intraday') await renderMainChart()
  } catch (e) {
    if (requestSeq === intradayRequestSeq) {
      intradayBars.value = []
      intradayMeta.value = null
      intradayError.value = e?.response?.data?.detail || e?.message || '分时数据读取失败'
      if (chartTab.value === 'intraday') await renderMainChart()
    }
  }
}

function buildIntradayOption(inst) {
  const bars = intradayBars.value
  if (!bars?.length) {
    inst.clear()
    return
  }
  const times = bars.map(b => b.time)
  const prices = bars.map(b => b.price)
  const avgPrices = bars.map(b => b.avg_price)
  const borderC = cssVar('--border')

  inst.setOption({
    title: { text: '日内分时', left: 8, top: 4, textStyle: { fontSize: 11, fontWeight: 'normal', color: cssVar('--text-secondary') } },
    tooltip: { trigger: 'axis' },
    grid: { left: 8, right: 8, top: 28, bottom: 20 },
    xAxis: { type: 'category', data: times, axisLabel: { fontSize: 8, interval: 29, color: cssVar('--text-muted') } },
    yAxis: { type: 'value', scale: true, axisLabel: { fontSize: 9, color: cssVar('--text-muted') }, splitLine: { lineStyle: { color: borderC, type: 'dashed' } } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 12, bottom: 4 }],
    series: [
      { type: 'line', name: '价格', data: prices, smooth: true, symbol: 'none', lineStyle: { color: UP, width: 1 } },
      { type: 'line', name: '均价', data: avgPrices, smooth: true, symbol: 'none', lineStyle: { color: '#f59e0b', width: 1, type: 'dashed' } },
    ],
  }, { notMerge: true })
}

function getChartColors() {
  return ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6',
    '#ec4899', '#06b6d4', '#f97316', '#14b8a6', '#6366f1',
    '#e11d48', '#0891b2', '#84cc16', '#d946ef', '#64748b']
}

function buildCompareOption(inst) {
  const codes = checkedCodes.value
  if (codes.length < 2) {
    inst.setOption({ series: [] }, { notMerge: true })
    return
  }

  const colors = getChartColors()
  const series = []
  let allDates = []
  const borderC = cssVar('--border')
  const mutedC = cssVar('--text-muted')

  codes.forEach((code, idx) => {
    const item = previewData.value[code]
    if (!item?.kline?.length) return
    const closes = item.kline.map(b => b.close)
    const dates = item.kline.map(b => b.date.slice(5))
    if (dates.length > allDates.length) allDates = dates
    const base = closes[closes.length - 1] || 1
    const pctData = closes.map(c => ((c - base) / base * 100).toFixed(2))
    series.push({
      name: `${code} ${item.name || ''}`,
      type: 'line', data: pctData, smooth: true, symbol: 'none',
      lineStyle: { color: colors[idx % colors.length], width: 1.5 },
    })
  })

  inst.setOption({
    title: { text: '多股走势对比 (% 偏离)', left: 10, top: 4, textStyle: { fontSize: 12, fontWeight: 'normal', color: cssVar('--text-secondary') } },
    tooltip: { trigger: 'axis' },
    legend: { bottom: 0, textStyle: { fontSize: 10, color: cssVar('--text-secondary') } },
    grid: { left: 8, right: 8, top: 32, bottom: 28 },
    xAxis: { type: 'category', data: allDates, axisLabel: { fontSize: 9, color: mutedC } },
    yAxis: { type: 'value', axisLabel: { fontSize: 9, color: mutedC, formatter: '{value}%' }, splitLine: { lineStyle: { color: borderC, type: 'dashed' } } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 14, bottom: 14 }],
    series,
  }, { notMerge: true })
}

function buildKlineOption(inst) {
  const item = activePreview.value
  if (!item?.kline?.length) return

  const data = item.kline
  const dates = data.map(d => d.date.slice(5))
  const ohlc = data.map(d => [d.open, d.close, d.low, d.high])
  const closes = data.map(d => d.close)

  function ma(arr, n) {
    if (arr.length < n) return Array(arr.length).fill(null)
    const result = Array(n - 1).fill(null)
    for (let i = n - 1; i < arr.length; i++) {
      let sum = 0
      for (let j = 0; j < n; j++) sum += arr[i - j]
      result.push(sum / n)
    }
    return result
  }

  const ma5 = ma(closes, 5)
  const ma20 = ma(closes, 20)
  const borderC = cssVar('--border')
  const mutedC = cssVar('--text-muted')

  const series = [
    { type: 'candlestick', name: 'K线', data: ohlc,
      itemStyle: { color: UP, color0: DOWN, borderColor: UP, borderColor0: DOWN } },
    { type: 'line', name: 'MA5', data: ma5, smooth: true, symbol: 'none', lineStyle: { color: '#f59e0b', width: 1 } },
    { type: 'line', name: 'MA20', data: ma20, smooth: true, symbol: 'none', lineStyle: { color: '#8b5cf6', width: 1 } },
  ]

  const markLines = []
  if (item.supports) {
    item.supports.slice(0, 3).forEach(s => {
      markLines.push({ name: s.name, yAxis: s.price, lineStyle: { color: DOWN, type: 'dashed', width: 1 }, label: { formatter: s.name, fontSize: 9 } })
    })
  }
  if (item.resistances) {
    item.resistances.slice(0, 2).forEach(r => {
      markLines.push({ name: r.name, yAxis: r.price, lineStyle: { color: UP, type: 'dashed', width: 1 }, label: { formatter: r.name, fontSize: 9 } })
    })
  }
  if (item.stop_loss_ref) {
    markLines.push({ name: '止损', yAxis: item.stop_loss_ref, lineStyle: { color: UP, type: 'solid', width: 1.5 }, label: { formatter: '止损', fontSize: 9 } })
  }

  series[0].markLine = { silent: true, symbol: 'none', data: markLines }

  inst.setOption({
    title: { text: `${activeCode.value} ${item.name || ''}`, left: 10, top: 4, textStyle: { fontSize: 12, fontWeight: 'bold', color: cssVar('--text') } },
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    grid: { left: 8, right: 8, top: 32, bottom: 24 },
    xAxis: { type: 'category', data: dates, axisLabel: { fontSize: 9, color: mutedC } },
    yAxis: { type: 'value', scale: true, axisLabel: { fontSize: 9, color: mutedC }, splitLine: { lineStyle: { color: borderC, type: 'dashed' } } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 14, bottom: 6 }],
    series,
  }, { notMerge: true })
}

// ─── 调仓 ───

async function runRebalanceAction() {
  if (runningRebalance.value) return
  if (pendingPlanCommit.value || pendingPlanCommitError.value) {
    error.value = '请先核对浏览器中尚未确认的调仓提交，不能创建新方案'
    return
  }
  if (uncertainFillRequests.value.length || uncertainFillRequestsError.value) {
    error.value = '请先核对浏览器中尚未确认的成交请求，不能创建新方案'
    return
  }
  if (!selectedSaveId.value) { error.value = '请先选择选股结果'; return }
  const codes = checkedCodes.value
  if (!codes.length) { error.value = '请至少勾选一只股票'; return }
  if (!strategyControlEnabled.value && manualTakeoverReason.value.trim().length < 8) {
    error.value = '人工接管理由至少需要8个字符'
    return
  }
  if (planCalculated.value && !planDirty.value && planFingerprint.value && !planCommitted.value) {
    return commitRebalancePlan()
  }
  return previewRebalancePlan(codes)
}

function rebalanceRequestParams(decisionId) {
  return {
    top_n: topN.value, sizing: sizing.value, frequency: 'weekly', code_prefixes: checkedCodes.value,
    cooldown_stop_weeks: cooldownStopWeeks.value || null,
    cooldown_profit_weeks: cooldownProfitWeeks.value || null,
    decision_id: decisionId,
    enforce_strategy_profile: strategyControlEnabled.value,
    unmanaged_reason: strategyControlEnabled.value ? null : manualTakeoverReason.value.trim(),
  }
}

async function previewRebalancePlan(codes) {
  runningRebalance.value = true; error.value = ''
  intradayData.value = { results: [], sell_results: [], timestamp: '', source: '' }
  try {
    if (!await loadHoldings()) throw new Error('本地持仓快照不可用，已停止生成调仓方案')
    const decision = await refreshDecisionAudit(codes, true)
    if (!decision?.decision_id) throw new Error('未生成有效的决策快照')
    const res = await rebalanceApi.runRebalance(selectedSaveId.value, rebalanceRequestParams(decision.decision_id))
    const data = res?.data || res
    if (data.error) { error.value = data.error; return }
    rebalanceSignals.value = (data.signals || []).map(s => ({ ...s, id: s.id ?? s.signal_id, executed: false, filled_shares: 0 }))
    observeCandidates.value = data.observe_candidates || []
    skippedCandidates.value = data.skipped || []
    filteredOut.value = data.filtered_out || []
    classificationData.value = data.classification || null
    priceSource.value = data.price_source || ''
    positionRatio.value = data.timing_ratio ?? null
    positionSource.value = data.position_source || ''
    executionBlockReason.value = data.execution_block_reason || ''
    appliedTopN.value = topN.value
    planCalculated.value = true
    planDirty.value = false
    planFingerprint.value = data.plan_fingerprint || ''
    planDecisionId.value = decision.decision_id
    planCommitRequestId.value = ''
    planCommitted.value = false
    planCommitBlocked.value = Boolean(data.commit_blocked)
    if (!planFingerprint.value) throw new Error('预览未返回方案指纹，不能提交')
    const [holdingsReady] = await Promise.all([loadHoldings(), loadCooldown(), loadLedgerState(), loadPersistedSignals()])
    if (!holdingsReady) throw new Error('生成方案后无法确认当前持仓快照，请刷新后重试')
  } catch (e) {
    error.value = '调仓预览失败: ' + (e.message || e)
    planDirty.value = true
  } finally {
    runningRebalance.value = false
  }
}

function clearPendingPlanCommit() {
  try {
    localStorage.removeItem(pendingPlanCommitStorageKey)
    sessionStorage.removeItem(pendingPlanCommitStorageKey)
    pendingPlanCommit.value = null
    pendingPlanCommitError.value = ''
    return true
  } catch {
    pendingPlanCommitError.value = '服务端已确认提交，但浏览器无法清除待确认记录；请用同一请求ID再次核对。'
    return false
  }
}

async function sendPendingPlanCommit(pending) {
  runningRebalance.value = true
  error.value = ''
  try {
    const res = await rebalanceApi.commitRebalance(pending.screening_save_id, pending.payload)
    const data = res?.data || res
    if (data.error) {
      error.value = data.error
      if (data.error.includes('账户、行情或方案内容已变化')) {
        clearPendingPlanCommit()
        planDirty.value = true
      }
      return
    }
    if (data.preview_only || data.committed !== true) {
      if (data.committed === false) {
        clearPendingPlanCommit()
        planDirty.value = true
        error.value = data.execution_block_reason || '服务未提交方案；请检查账户状态并重新预览'
        return
      }
      throw new Error('服务未确认方案已写入账本；已保留原请求ID供安全重试')
    }
    const visiblePlan = planFingerprint.value === pending.payload.expected_plan_fingerprint
    if (visiblePlan) {
      const committedSignals = data.signals || []
      const idsBySignal = new Map(committedSignals.map(signal => [`${signal.code}:${signal.side}`, signal.id ?? signal.signal_id]))
      rebalanceSignals.value = data.already_committed
        ? rebalanceSignals.value.map(signal => ({
          ...signal,
          id: idsBySignal.get(`${signal.code}:${signal.side}`) ?? signal.id,
          signal_id: idsBySignal.get(`${signal.code}:${signal.side}`) ?? signal.signal_id,
        }))
        : committedSignals.map(signal => ({ ...signal, id: signal.id ?? signal.signal_id, executed: false, filled_shares: 0 }))
      planCommitRequestId.value = pending.payload.commit_request_id
      planCommitted.value = true
    }
    const requestCleared = clearPendingPlanCommit()
    await Promise.all([loadHoldings(), loadCooldown(), loadLedgerState(), loadPersistedSignals()])
    if (!requestCleared) error.value = '调仓方案已提交，刷新后可继续使用原请求ID核对，不会重复创建'
  } catch (e) {
    if (e?.commitOutcome === 'not_committed' || Number(e?.status) === 422) {
      const requestCleared = clearPendingPlanCommit()
      planDirty.value = true
      error.value = `服务确认本次调仓未写入（HTTP ${e.status || 422}）：${e.message || e}`
      if (!requestCleared) error.value += '；浏览器未能清除待确认记录，请刷新后检查原请求状态'
      return
    }
    error.value = '调仓方案提交结果尚未确认；请使用同一请求ID安全重试: ' + (e.message || e)
  } finally {
    runningRebalance.value = false
  }
}

async function commitRebalancePlan() {
  if (runningRebalance.value) return
  if (!planFingerprint.value || !planDecisionId.value || planDirty.value) {
    error.value = '方案已过期，请重新预览后提交'
    return
  }
  if (pendingPlanCommit.value || pendingPlanCommitError.value) {
    error.value = '请先核对上次调仓提交结果'
    return
  }
  runningRebalance.value = true
  const holdingsReady = await loadHoldings()
  runningRebalance.value = false
  if (!holdingsReady) {
    invalidatePlan()
    error.value = '本地持仓快照不可用，已阻止提交调仓方案'
    return
  }
  const pending = {
    version: 1,
    screening_save_id: Number(selectedSaveId.value),
    payload: {
      ...rebalanceRequestParams(planDecisionId.value),
      commit_request_id: planCommitRequestId.value || crypto.randomUUID(),
      expected_plan_fingerprint: planFingerprint.value,
    },
  }
  try {
    localStorage.setItem(pendingPlanCommitStorageKey, JSON.stringify(pending))
    sessionStorage.removeItem(pendingPlanCommitStorageKey)
    pendingPlanCommit.value = pending
    planCommitRequestId.value = pending.payload.commit_request_id
  } catch {
    error.value = '浏览器无法保存幂等请求，未发送调仓提交'
    return
  }
  await sendPendingPlanCommit(pending)
}

async function retryPendingPlanCommit() {
  if (!pendingPlanCommit.value || runningRebalance.value) return
  await sendPendingPlanCommit(pendingPlanCommit.value)
}

function onStrategyControlChange(event) {
  const enabled = event.target.checked
  if (!enabled && !confirm('确认关闭实盘策略版本管控？\n\n关闭后，非 Champion 或旧版快照也可生成并记录新增买入。刷新页面后将自动恢复开启。')) {
    event.target.checked = true
    return
  }
  strategyControlEnabled.value = enabled
  if (enabled) manualTakeoverReason.value = ''
  executionBlockReason.value = ''
}

async function refreshDecisionAudit(codes = checkedCodes.value, throwOnError = false) {
  if (!selectedSaveId.value || !codes.length) return
  const requestSeq = ++decisionRequestSeq
  loadingDecision.value = true
  decisionError.value = ''
  try {
    const decisionRes = await rebalanceApi.previewDecision(selectedSaveId.value, {
      top_n: topN.value,
      policy_id: 'legacy',
      selected_codes: codes,
    })
    const nextDecision = decisionRes?.data || decisionRes || null
    if (requestSeq === decisionRequestSeq) decisionData.value = nextDecision
    return nextDecision
  } catch (decisionFailure) {
    if (requestSeq === decisionRequestSeq) {
      decisionData.value = null
      decisionError.value = '决策审计生成失败：' + (decisionFailure.message || decisionFailure)
    }
    if (throwOnError) throw decisionFailure
    return null
  } finally {
    if (requestSeq === decisionRequestSeq) loadingDecision.value = false
  }
}

function formatDecisionScore(value) {
  const score = Number(value)
  return Number.isFinite(score) ? score.toFixed(3) : '--'
}

function formatDecisionCoverage(value) {
  const coverage = Number(value)
  return Number.isFinite(coverage) ? `${Math.round(coverage * 100)}%` : '--'
}

function formatDecisionAmount(value) {
  const amount = Number(value)
  if (!Number.isFinite(amount)) return '--'
  if (amount >= 1e8) return `${(amount / 1e8).toFixed(2)}亿`
  return `${(amount / 1e4).toFixed(0)}万`
}

function formatDecisionRankChange(row) {
  const change = Number(row?.rank_change ?? (row?.original_rank - row?.final_rank))
  if (!Number.isFinite(change) || change === 0) return `原#${row?.original_rank || '--'}`
  return `${change > 0 ? '↑' : '↓'}${Math.abs(change)} · 原#${row.original_rank}`
}

function executeSignal(signal) {
  const signalId = signal.id ?? signal.signal_id
  if (!signalId) { error.value = '该信号缺少编号，无法记录执行'; return }
  let saved = null
  try {
    const key = `${executionStoragePrefix}${signalId}`
    const raw = localStorage.getItem(key) ?? sessionStorage.getItem(key)
    saved = JSON.parse(raw || 'null')
    if (saved && !isValidStoredExecutionPayload(saved)) {
      uncertainFillRequestsError.value = `信号 #${signalId} 的待确认成交请求内容不完整`
      return
    }
    if (saved && !localStorage.getItem(key)) {
      localStorage.setItem(key, raw)
      sessionStorage.removeItem(key)
    }
  } catch {
    error.value = '无法读取待确认成交请求；为避免重复记账，已停止新的成交确认'
    return
  }
  const hasUnconfirmedRequest = Boolean(saved?.request_id)
  if (planDirty.value && !signal._persisted && !hasUnconfirmedRequest) { error.value = '方案已过期，请重新生成后再记录成交'; return }
  if (localAccount.value?.status !== 'active' && !hasUnconfirmedRequest) { error.value = '本地账户未核准或账本不可用，不能记录成交'; return }
  if (localAccount.value?.broker_account_id && !hasUnconfirmedRequest) { error.value = '已绑定券商账户须通过本地委托意图和券商成交回报入账'; return }
  const live = executionForSignal(signal)
  const defaultPrice = live?.current_price ?? signal.current_price ?? 0
  const suggestedShares = Math.max(0, Number(signal.planned_shares ?? signal.suggested_shares ?? 0) - Number(signal.filled_shares || 0))
  const normalizedPrice = Number(defaultPrice)
  executingSignal.value = signal
  executionForm.shares = suggestedShares > 0 ? suggestedShares : ''
  executionForm.price = normalizedPrice > 0 ? normalizedPrice.toFixed(3) : ''
  executionForm.commission = ''
  executionForm.stampDuty = signal.side === 'buy' || signal.side === 'add' ? '0' : ''
  executionForm.brokerTradeId = ''
  executionForm.requestId = crypto.randomUUID()
  executionAccountVersion.value = Number(localAccount.value?.version)
  executionFormError.value = ''
  uncertainExecution.value = null
  if (saved?.request_id && Number(saved.expected_account_version) > 0) {
    uncertainExecution.value = { signalId, payload: saved }
    executionForm.shares = saved.shares
    executionForm.price = saved.price
    executionForm.commission = saved.commission
    executionForm.stampDuty = saved.stamp_duty
    executionForm.brokerTradeId = saved.broker_trade_id || ''
    executionForm.requestId = saved.request_id
    executionAccountVersion.value = saved.expected_account_version
    executionFormError.value = '上次成交结果尚未确认；本次将按原请求 ID 和原成交参数重试'
  }
  showExecutionModal.value = true
  nextTick(() => executionSharesInput.value?.focus())
}

function closeExecutionModal() {
  if (recordingExecution.value) return
  showExecutionModal.value = false
  executingSignal.value = null
  executionForm.shares = ''
  executionForm.price = ''
  executionForm.brokerTradeId = ''
  executionForm.requestId = ''
  executionAccountVersion.value = null
  uncertainExecution.value = null
  executionFormError.value = ''
}

async function submitExecution() {
  if (planDirty.value && !executingSignal.value?._persisted && !uncertainExecution.value) { executionFormError.value = '方案已过期，请重新生成后再记录成交'; return }
  if (localAccount.value?.status !== 'active' && !uncertainExecution.value) { executionFormError.value = '本地账户未核准或账本不可用，不能记录成交'; return }
  const signal = executingSignal.value
  const signalId = signal?.id ?? signal?.signal_id
  const shares = Number(executionForm.shares)
  const price = String(executionForm.price).trim()
  const commission = String(executionForm.commission).trim()
  const stampDuty = String(executionForm.stampDuty).trim()
  if (executionForm.commission === '' || (!isExecutionBuy.value && executionForm.stampDuty === '')) {
    executionFormError.value = '请按券商成交回报填写实际费用，零费用也须填写0'
    return
  }
  if (!Number.isInteger(shares) || shares <= 0) {
    executionFormError.value = '成交股数必须是有效正整数'
    return
  }
  const remaining = Number(signal?.planned_shares ?? signal?.suggested_shares ?? 0) - Number(signal?.filled_shares || 0)
  if (shares > remaining && !uncertainExecution.value) {
    executionFormError.value = '成交股数超过剩余计划量'
    return
  }
  if (signal?._persisted && Number(localAccount.value?.version) !== Number(signal.account_version) && !uncertainExecution.value) {
    executionFormError.value = '账户版本已变化，请刷新并核对计划'
    return
  }
  if (isExecutionBuy.value && shares % 100 !== 0) {
    executionFormError.value = 'A股买入股数必须是100股的整数倍'
    return
  }
  if (!/^\d+(?:\.\d{1,3})?$/.test(price) || Number(price) <= 0) {
    executionFormError.value = '成交价格必须是有效正数且最多三位小数'
    return
  }
  if (![commission, stampDuty].every(value => /^\d+(?:\.\d{1,2})?$/.test(value))) {
    executionFormError.value = '实际费用必须是非负且最多两位小数'
    return
  }
  if (isExecutionBuy.value && Number(stampDuty) !== 0) {
    executionFormError.value = '买入不应填写印花税'
    return
  }
  if (!uncertainExecution.value && localAccount.value?.broker_account_id && (!executionForm.brokerTradeId || executionForm.brokerTradeId.length > 64)) {
    executionFormError.value = '已绑定券商账户的人工成交必须填写券商逐笔成交编号'
    return
  }
  const requestPayload = uncertainExecution.value?.payload || {
    shares, price, commission, stamp_duty: stampDuty,
    request_id: executionForm.requestId,
    expected_account_version: executionAccountVersion.value,
    date: todayStr(),
    ...(localAccount.value?.broker_account_id ? { broker_trade_id: executionForm.brokerTradeId } : {}),
  }
  try {
    saveUnconfirmedExecutionRequest(signalId, requestPayload)
  } catch (e) {
    executionFormError.value = '无法持久保存本次请求 ID，已停止提交成交确认'
    return
  }
  uncertainExecution.value = { signalId, payload: requestPayload }
  recordingExecution.value = true
  executionFormError.value = ''
  try {
    const response = await rebalanceApi.executeSignal(signalId, requestPayload)
    const result = response?.data || response
    const responseRemaining = Number(result?.remaining_plan_shares)
    if (!['partial', 'executed'].includes(result?.execution_status) ||
        typeof result?.already_recorded !== 'boolean' ||
        !Number.isInteger(responseRemaining) || responseRemaining < 0 ||
        responseRemaining > Number(signal.planned_shares ?? signal.suggested_shares ?? 0) ||
        (!result.already_recorded && responseRemaining !== remaining - shares) ||
        (result.execution_status === 'executed' && responseRemaining !== 0) ||
        (result.execution_status === 'partial' && responseRemaining === 0)) {
      executionFormError.value = '成交回执格式异常，结果尚未确认；请保留本次请求 ID，先核对账本和未完成信号'
      return
    }
    clearUnconfirmedExecutionRequest(signalId)
    uncertainExecution.value = null
    if (responseRemaining <= remaining) {
      signal.filled_shares = Number(signal.planned_shares ?? signal.suggested_shares ?? 0) - responseRemaining
      signal.execution_status = result.execution_status
      signal.executed = result.execution_status === 'executed'
    }
    applyInvalidatedSignalResult(result)
    showExecutionModal.value = false
    executingSignal.value = null
    const [holdingsOk, , ledgerOk, pendingOk] = await Promise.all([
      loadHoldings(), loadCooldown(), loadLedgerState(), loadPersistedSignals(),
    ])
    if (!holdingsOk || !ledgerOk || !pendingOk) {
      planDirty.value = true
      error.value = '成交已记入本地账本，但页面状态刷新不完整，请刷新页面核对后再操作'
    }
  } catch (e) {
    if (e.status && e.status < 500) {
      if (String(e.message || '').includes('请求 ID 已用于另一笔成交')) {
        uncertainFillRequestsError.value = `信号 #${signalId} 的请求 ID 与已记账成交冲突；待确认请求已保留并阻止新方案，请先核对账本`
        executionFormError.value = uncertainFillRequestsError.value
      } else {
        clearUnconfirmedExecutionRequest(signalId)
        uncertainExecution.value = null
        executionFormError.value = '执行失败: ' + (e.message || e)
      }
    } else {
      executionFormError.value = '成交结果未确认，请保留本次请求 ID，按原参数重试或先核对账本：' + (e.message || e)
    }
  } finally {
    recordingExecution.value = false
  }
}

function executionForSignal(signal) {
  if (signal.side === 'sell' || signal.side === 'reduce') {
    return intradayData.value.sell_results?.find(r => r.code === signal.code)?.execution || null
  }
  return intradayData.value.results?.find(r => r.code === signal.code)?.execution || null
}
function executionPrice(execution) {
  if (!execution) return ''
  if (execution.limit_price != null) return `¥${Number(execution.limit_price).toFixed(2)}`
  return execution.action === 'market' ? '市价' : '观望'
}
function executionDisplay(signal) {
  const execution = executionForSignal(signal)
  if (execution) {
    if (execution.can_execute) return `可成交 · ${executionPrice(execution)}`
    if (execution.action === 'wait') return '盘中观望'
    return execution.trigger_text || `等待 ${executionPrice(execution)}`
  }
  if (signal.side === 'buy' || signal.side === 'add') {
    const p = previewData.value[signal.code]?.suggested_buy
    return p ? `参考 ¥${p.low.toFixed(2)}~${p.high.toFixed(2)}` : ''
  }
  return ''
}
function onIntradayUpdate(data) {
  intradayData.value = data || { results: [], sell_results: [], timestamp: '', source: '' }
}
function stopInfo(code) {
  const p = previewData.value[code]
  if (!p || (!p.stop_loss_ref && !p.take_profit_ref)) return null
  return { loss: p.stop_loss_ref, profit: p.take_profit_ref }
}
function sideClass(side) {
  return { buy: 'ltv-up', sell: 'ltv-down', reduce: 'ltv-warn', add: 'ltv-up' }[side] || ''
}
function sideLabel(s) {
  const map = { buy: '买入', sell: '卖出', reduce: '减仓', add: '加仓' }
  if (s.side === 'sell' && s.action?.includes('止损')) return '止损'
  if (s.side === 'reduce' && s.action?.includes('止盈')) return '止盈'
  return map[s.side] || s.action || s.side
}

// ─── 监听 ───

let suppressAutoRebalance = false
function invalidatePlan() {
  planDirty.value = true
  planCalculated.value = false
  planFingerprint.value = ''
  planDecisionId.value = ''
  planCommitRequestId.value = ''
  planCommitted.value = false
  planCommitBlocked.value = false
  rebalanceSignals.value = []
  observeCandidates.value = []
  skippedCandidates.value = []
  filteredOut.value = []
  classificationData.value = null
  positionRatio.value = null
  positionSource.value = ''
  priceSource.value = ''
  executionBlockReason.value = ''
}

function markPlanDirty() {
  if (suppressAutoRebalance) return
  if (selectedSaveId.value) planDirty.value = true
}

watch(selectedSaveId, onSaveSelected)
watch(checkedCodes, () => { loadPreview() })
watch(activeCode, async (code) => {
  chartTab.value = 'kline'
  await loadStockPreview(code)
  await nextTick()
  renderMainChart()
  fetchIntraday()
})
watch(manualChecked, () => {
  loadPreview()
  markPlanDirty()
})
watch([topN, sizing, stopMode, strategyControlEnabled, manualTakeoverReason, cooldownStopWeeks, cooldownProfitWeeks], markPlanDirty)
watch([buyCodes, sellSignals], ([buys, sells]) => {
  if (!buys.length && !sells.length) onIntradayUpdate(null)
}, { immediate: true })
watch(() => checkedCodes.value.length, (n) => {
  if (n > 1 && chartTab.value === 'compare') nextTick(() => renderMainChart())
  else if (n <= 1 && chartTab.value === 'compare') { chartTab.value = 'kline'; nextTick(() => renderMainChart()) }
})

function onResize() {
  if (mainInstance && !mainInstance.isDisposed()) mainInstance.resize()
}

let pagePollTimer = null
function scheduleNextPoll() {
  stopPagePoll()
  const jitter = 2500 + Math.random() * 1000
  pagePollTimer = setTimeout(async () => {
    if (!document.hidden && checkedCodes.value.length) {
      await loadPreview(true)
      if (activeCode.value) await fetchIntraday()
    }
    scheduleNextPoll()
  }, jitter)
}
function startPagePoll() { stopPagePoll(); scheduleNextPoll() }
function stopPagePoll() { if (pagePollTimer) { clearTimeout(pagePollTimer); pagePollTimer = null } }
function onVisibilityChange() {
  if (document.hidden) stopPagePoll()
  else {
    startPagePoll()
    loadLedgerState()
    loadPersistedSignals()
  }
}

onMounted(async () => {
  restorePendingPlanCommit()
  loadUnconfirmedExecutionRequests()
  window.addEventListener('storage', syncPendingRequestsAcrossTabs)
  await Promise.all([loadSavedScreenings(), loadHoldings(), loadCooldown(), loadLedgerState(), loadPersistedSignals()])
  window.addEventListener('resize', onResize)
  resizeBound = true
  startPagePoll()
  document.addEventListener('visibilitychange', onVisibilityChange)
})

onUnmounted(() => {
  intradayRequestSeq += 1
  executionHistorySeq += 1
  stopPagePoll()
  window.removeEventListener('storage', syncPendingRequestsAcrossTabs)
  document.removeEventListener('visibilitychange', onVisibilityChange)
  if (mainInstance && !mainInstance.isDisposed()) mainInstance.dispose()
  if (resizeBound) window.removeEventListener('resize', onResize)
})
</script>

<style scoped>
.ltv-root {
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-height: calc(100vh - 100px);
  padding: var(--space-3) var(--content-padding) 32px;
}
.ltv-execution-history-wrap { max-width: 100%; overflow-x: auto; margin: 12px 0; }
.ltv-execution-history { width: 100%; min-width: 900px; border-collapse: collapse; font-size: var(--fs-caption); }
.ltv-execution-history th, .ltv-execution-history td { padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: right; white-space: nowrap; }
.ltv-execution-history th:first-child, .ltv-execution-history td:first-child { text-align: left; }

.ltv-ledger {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  align-items: start;
  gap: 10px 18px;
  padding: 14px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg-secondary);
  font-size: var(--fs-sm);
}
.ltv-ledger > button { justify-self: start; }
.ltv-opening { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; grid-column: 1 / -1; }
.ltv-opening-holdings { display: flex; flex-wrap: wrap; gap: 6px 12px; width: 100%; padding: 8px; border: 1px solid var(--border); border-radius: var(--radius-sm); }
.ltv-opening-holdings strong { width: 100%; }
.ltv-opening-holdings--blocked { grid-column: 1 / -1; border-color: var(--warning); }
.ltv-opening input { padding: 5px 8px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg); color: var(--text); }
.ltv-intent-form { grid-column: 1 / -1; padding: 10px; border: 1px solid var(--border); border-radius: var(--radius-sm); }
.ltv-intent-form label, .ltv-intent-actions label { display: grid; gap: 4px; color: var(--text-secondary); font-size: var(--fs-sm); }
.ltv-intent-card { display: grid; gap: 8px; grid-column: 1 / -1; padding: 10px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--surface); }
.ltv-intent-actions { display: flex; align-items: end; flex-wrap: wrap; gap: 8px; }
.ltv-intent-actions input, .ltv-intent-form input { max-width: 220px; min-height: 34px; padding: 5px 8px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg); color: var(--text); }
.ltv-intent-evidence, .ltv-intent-trade { display: grid; gap: 6px; }
.ltv-intent-trade { padding: 8px; border-left: 3px solid var(--border); background: var(--bg); overflow-wrap: anywhere; }

.ltv-field { display: flex; align-items: center; gap: 4px; }
.ltv-field label { font-size: var(--fs-sm); color: var(--text-muted); white-space: nowrap; }
.ltv-field__static { min-height: 30px; display: inline-flex; align-items: center; color: var(--text); font-size: var(--fs-sm); }
.ltv-source-meta { color: var(--text-muted); font-size: var(--fs-caption); line-height: 1.4; }
.ltv-reconciliation-detail { color: var(--text); font-size: var(--fs-body); line-height: 1.5; overflow-wrap: anywhere; }
.ltv-reconciliation-item { display: inline-block; margin: 0 8px 4px 0; }
.ltv-broker-trades { display: grid; grid-column: 1 / -1; gap: 6px; }
.ltv-broker-trades__list { display: grid; gap: 6px; }
.ltv-broker-trade { display: grid; gap: 2px; padding: 8px 10px; border-left: 3px solid var(--danger); background: var(--bg); overflow-wrap: anywhere; }
.ltv-quarantine-group { display: grid; gap: 6px; }
.ltv-quarantine-row { border-left-color: var(--warning); }
.ltv-quarantine-pagination { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.ltv-field select, .ltv-field input {
  font-size: var(--fs-body); padding: 4px 8px;
  border: 1px solid var(--border); border-radius: var(--radius-sm);
  background: var(--bg); color: var(--text);
}
.ltv-field--sm input { width: 56px; text-align: center; }
.ltv-field--decision select { min-width: 132px; }
.ltv-mode-switch { display: inline-flex; border: 1px solid var(--border); border-radius: var(--radius-sm); overflow: hidden; }
.ltv-mode-switch button {
  font-size: var(--fs-caption); padding: 3px 10px; border: none;
  background: var(--bg-secondary); color: var(--text-muted); cursor: pointer;
  border-right: 1px solid var(--border); font-family: var(--font);
}
.ltv-mode-switch button:last-child { border-right: none; }
.ltv-mode--active { background: var(--primary) !important; color: var(--on-primary) !important; }

/* ── Left panel ── */
.ltv-left__section { order: -2; flex-shrink: 0; display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.ltv-left__title { font-size: var(--fs-body); font-weight: 600; color: var(--text); }
.ltv-left__count { font-size: var(--fs-caption); color: var(--text-muted); }
.ltv-left__actions { display: flex; align-items: center; gap: 6px; }
.ltv-left__collapse {
  min-height: 28px; padding: 2px 8px; border: 1px solid var(--border); border-radius: var(--radius-sm);
  background: var(--bg-secondary); color: var(--text-secondary); font: inherit; font-size: var(--fs-caption); cursor: pointer;
}
.ltv-left__collapse:hover, .ltv-left__collapse:focus-visible { border-color: var(--primary); color: var(--text); outline: none; }
.ltv-left__list { flex: 1; min-height: 160px; overflow-y: auto; overscroll-behavior: contain; }
.ltv-row {
  display: flex; align-items: center; gap: 6px;
  padding: 4px 6px; border-radius: var(--radius-sm); cursor: pointer; font-size: var(--fs-sm);
}
.ltv-row:hover { background: var(--primary-muted); }
.ltv-row--active { background: rgba(107,159,212,.12); border: 1px solid rgba(107,159,212,.3); }
.ltv-row input[type="checkbox"] { margin: 0; }
.ltv-row__code { font-weight: 600; min-width: 54px; }
.ltv-row__name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-secondary); }
.ltv-row__score { color: var(--primary); font-weight: 600; min-width: 36px; text-align: right; font-variant-numeric: tabular-nums; }
.ltv-row__rank { min-width: 28px; text-align: right; font-size: var(--fs-caption); color: var(--text-muted); font-variant-numeric: tabular-nums; }
.ltv-row__rank--selected { color: var(--primary); font-weight: 600; }
.ltv-row__rank--excluded { color: var(--danger); text-decoration: line-through; }
.ltv-row__label { font-size: 10px; background: var(--bg-secondary); padding: 0 4px; border-radius: 3px; color: var(--text-muted); }
.ltv-row__edit { font-size: var(--fs-caption); border: none; background: none; color: var(--primary); cursor: pointer; padding: 0 2px; opacity: .5; }
.ltv-row__edit:hover { opacity: 1; }
.ltv-row__sell {
  font-size: 10px; border: 1px solid rgba(245,85,85,.3); background: rgba(245,85,85,.08);
  color: var(--danger); cursor: pointer; padding: 0 4px; border-radius: 3px; font-family: var(--font);
}
.ltv-row__sell:hover { background: var(--danger); color: #fff; }
.ltv-up { color: var(--up); }
.ltv-down { color: var(--down); }
.ltv-warn { color: var(--warning); }

.ltv-decision { order: -1; margin-bottom: 8px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg-secondary); flex-shrink: 0; }
.ltv-decision__header { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 7px 8px; }
.ltv-decision__header > div { min-width: 0; display: flex; flex-direction: column; gap: 2px; }
.ltv-decision__header strong { color: var(--text); font-size: var(--fs-sm); }
.ltv-decision__header span { color: var(--text-muted); font-size: var(--fs-caption); }
.ltv-decision__header button { min-height: 28px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg); color: var(--text-secondary); cursor: pointer; }
.ltv-decision__header button:hover, .ltv-decision__header button:focus-visible { border-color: var(--primary); color: var(--text); outline: none; }
.ltv-decision__body { max-height: 230px; overflow-y: auto; border-top: 1px solid var(--border); padding: 8px; }
.ltv-decision__hint, .ltv-decision__error, .ltv-decision__notice, .ltv-decision__empty { font-size: var(--fs-caption); line-height: 1.5; }
.ltv-decision__hint { color: var(--text-muted); }
.ltv-decision__error { color: var(--danger); }
.ltv-decision__empty { padding: 8px; border: 1px dashed var(--border); border-radius: var(--radius-sm); color: var(--text-muted); }
.ltv-decision__notice { color: var(--text-secondary); margin: 6px 0; }
.ltv-decision__notice--shadow { color: var(--warning); }
.ltv-decision__quality { display: flex; flex-wrap: wrap; gap: 4px; margin: 6px 0; }
.ltv-decision__quality > span { display: inline-flex; align-items: center; gap: 3px; padding: 3px 5px; border: 1px solid var(--border); border-radius: var(--radius-sm); color: var(--text-secondary); font-size: var(--fs-caption); }
.ltv-decision__quality > span.is-disabled { border-style: dashed; color: var(--text-muted); }
.ltv-decision__quality strong { color: var(--text); font-weight: 600; }
.ltv-decision__quality small { color: var(--primary); }
.ltv-decision__quality .is-disabled small { color: var(--warning); }
.ltv-decision__list { display: flex; flex-direction: column; gap: 2px; }
.ltv-decision__list button { min-height: 44px; display: grid; grid-template-columns: 88px minmax(0, 1fr) 50px auto; align-items: center; gap: 5px; width: 100%; padding: 5px 6px; border: 0; border-radius: var(--radius-sm); background: transparent; color: var(--text-secondary); text-align: left; cursor: pointer; font: inherit; font-size: var(--fs-caption); }
.ltv-decision__list button:hover, .ltv-decision__list button:focus-visible { background: var(--primary-muted); outline: 1px solid var(--primary); }
.ltv-decision__list button.is-active { background: var(--primary-muted); }
.ltv-decision__list button > span:nth-child(2) { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ltv-decision__score { color: var(--primary); text-align: right; font-variant-numeric: tabular-nums; }
.ltv-decision__list small { color: var(--warning); white-space: nowrap; }
.ltv-decision__detail { margin-top: 6px; padding: 7px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg); }
.ltv-decision__detail-head { display: flex; justify-content: space-between; gap: 6px; color: var(--text-secondary); font-size: var(--fs-caption); }
.ltv-decision__detail-head strong { color: var(--text); }
.ltv-decision__components { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 3px 8px; margin-top: 5px; color: var(--text-muted); font-size: var(--fs-caption); }
.ltv-decision__components span { display: flex; justify-content: space-between; gap: 4px; }
.ltv-decision__components strong.is-positive { color: var(--up); }
.ltv-decision__components strong.is-negative { color: var(--down); }
.ltv-decision__detail p { margin: 5px 0 0; color: var(--text-secondary); font-size: var(--fs-caption); line-height: 1.5; }
.ltv-decision__excluded { margin-top: 6px; color: var(--danger); font-size: var(--fs-caption); line-height: 1.5; }
.ltv-decision__excluded summary { min-height: 28px; cursor: pointer; color: var(--warning); }
.ltv-decision__excluded p { margin: 5px 0 0; color: var(--danger); }
.ltv-decision__fingerprint { margin-top: 6px; color: var(--text-muted); font-size: var(--fs-caption); font-family: var(--font-mono); }

.ltv-cooldown {
  flex-shrink: 0; max-height: 126px; overflow-y: auto; margin-top: 8px;
  border: 1px solid rgba(244,167,66,.35); border-radius: var(--radius-sm); background: rgba(244,167,66,.06);
}
.ltv-cooldown__header {
  position: sticky; top: 0; z-index: 1; display: flex; justify-content: space-between; gap: 6px;
  padding: 6px 8px; background: var(--bg-secondary); color: var(--warning); font-size: var(--fs-caption); font-weight: 600;
}
.ltv-cooldown__header span:last-child { color: var(--text-muted); font-weight: 400; }
.ltv-cooldown__item {
  width: 100%; display: grid; grid-template-columns: 58px 1fr auto; gap: 6px; align-items: center;
  padding: 6px 8px; border: 0; border-top: 1px solid var(--border); background: transparent;
  color: var(--text-secondary); font: inherit; font-size: var(--fs-sm); text-align: left; cursor: pointer;
}
.ltv-cooldown__item:hover, .ltv-cooldown__item:focus-visible { background: var(--primary-muted); outline: none; }
.ltv-cooldown__item strong { color: var(--warning); font-size: var(--fs-caption); white-space: nowrap; }

/* ── Right panel ── */
.ltv-right__empty {
  flex: 1; display: flex; align-items: center; justify-content: center;
  color: var(--text-muted); font-size: var(--fs-base);
  border: 1px dashed var(--border); border-radius: var(--radius); min-height: 300px;
}
.ltv-chart-box { border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; flex-shrink: 0; }
.ltv-chart-tabs { display: flex; border-bottom: 1px solid var(--border); background: var(--bg-secondary); }
.ltv-intraday-note { padding: 6px 10px; color: var(--text-muted); font-size: var(--fs-xs); border-bottom: 1px solid var(--border); }
.ltv-chart-tab {
  font-size: var(--fs-sm); padding: 8px 18px; border: none; background: transparent;
  color: var(--text-muted); cursor: pointer; font-family: var(--font); border-right: 1px solid var(--border);
}
.ltv-chart-tab:hover { color: var(--text); }
.ltv-chart-tab--active { background: var(--primary-muted); color: var(--primary); font-weight: 600; }
.ltv-chart-wrap { position: relative; }
.ltv-chart--main .ltv-chart { width: 100%; height: 400px; }
.ltv-chart__overlay {
  position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  color: var(--text-muted); font-size: var(--fs-body);
  background: rgba(8,8,15,.85); pointer-events: none;
}

/* ── Suggest panel ── */
.ltv-suggest {
  border: 1px solid var(--border); border-radius: var(--radius);
  padding: var(--space-2) var(--space-3); flex-shrink: 0;
}
.ltv-suggest__header { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.ltv-suggest__code { font-weight: 600; font-size: var(--fs-body); color: var(--text); }
.ltv-suggest__quality { font-size: var(--fs-caption); padding: 2px 8px; border-radius: var(--radius-sm); font-weight: 600; }
.ltv-q--great { background: rgba(239,68,68,.12); color: var(--up); }
.ltv-q--good { background: rgba(244,167,66,.12); color: var(--warning); }
.ltv-q--ok { background: rgba(34,197,94,.12); color: var(--success); }
.ltv-q--poor { background: var(--bg-secondary); color: var(--text-muted); }
.ltv-suggest__grid { display: flex; gap: 16px; font-size: var(--fs-sm); }
.ltv-suggest__col { flex: 1; }
.ltv-suggest__col strong { display: block; margin-bottom: 4px; font-size: var(--fs-caption); color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.3px; }
.ltv-suggest__line { display: flex; justify-content: space-between; padding: 2px 0; color: var(--text-secondary); }
.ltv-suggest__val { font-family: var(--font-mono); font-weight: 500; }
.ltv-technical { margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border); }
.ltv-technical__header { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
.ltv-technical__header > div { display: flex; flex-direction: column; gap: 2px; }
.ltv-technical__header strong { color: var(--text); font-size: var(--fs-sm); }
.ltv-technical__header span { color: var(--text-secondary); font-size: var(--fs-caption); }
.ltv-technical__overall { flex-shrink: 0; padding: 4px 8px; border: 1px solid currentColor; border-radius: var(--radius-sm); font-weight: 600; }
.ltv-technical__grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
.ltv-technical__item { min-width: 0; padding: 8px 10px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg-secondary); }
.ltv-technical__item-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.ltv-technical__item-head strong { color: var(--text-secondary); font-size: var(--fs-caption); }
.ltv-technical__status { flex-shrink: 0; font-size: var(--fs-caption); font-weight: 600; }
.ltv-technical__value { display: block; margin-top: 4px; color: var(--text); font-family: var(--font-mono); font-size: var(--fs-sm); }
.ltv-technical__detail { display: block; margin-top: 3px; color: var(--text-secondary); font-size: var(--fs-caption); line-height: 1.45; }
.ltv-tech--bullish { color: var(--up) !important; }
.ltv-tech--bearish { color: var(--down) !important; }
.ltv-tech--neutral { color: var(--text-secondary) !important; }
.ltv-tech--risk { color: var(--warning) !important; }

/* ── Rebalance results ── */
.ltv-result { flex: 1; min-height: 160px; margin-top: 10px; border: 1px solid var(--border); border-radius: var(--radius-sm); overflow-y: auto; overscroll-behavior: contain; }
.ltv-result__group { }
.ltv-result__group-title { padding: 10px 12px; font-size: var(--fs-base); font-weight: 600; border-bottom: 1px solid var(--border); }
.ltv-result__group-title--sell { background: rgba(245,85,85,.15); color: var(--danger); }
.ltv-result__group-title--buy { background: rgba(34,197,94,.12); color: var(--success); }
.ltv-result__group-title--observe { background: rgba(244,167,66,.12); color: var(--warning); }
.ltv-result__group-title--hold { background: rgba(78,201,176,.12); color: var(--accent-cool); }
.ltv-hold { color: var(--accent-cool); }
.ltv-result__group-title--skip { background: var(--bg-secondary); color: var(--text-secondary); }
.ltv-skip { color: var(--text-secondary); }
.ltv-result__group-title--filter { background: rgba(245,85,85,.15); color: var(--danger); }
.ltv-filter { color: var(--danger); }
.ltv-result__meta {
  padding: 6px 12px; font-size: var(--fs-caption); color: var(--text-muted);
  border-bottom: 1px solid var(--border); background: var(--bg-secondary);
  display: flex; gap: 8px; align-items: center;
}
.ltv-meta--live { color: var(--success); font-weight: 600; }
.ltv-meta--close { color: var(--text-secondary); font-weight: 600; }
.ltv-observe { color: var(--warning); }
.ltv-result__row { padding: 10px 12px; border-top: 1px solid rgba(42,42,80,.3); font-size: var(--fs-body); }
.ltv-result__row--done { opacity: .45; background: rgba(255,255,255,.01); }
.ltv-result__row-top { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.ltv-result__code { font-weight: 600; font-family: var(--font-mono); flex: 1; color: var(--text); font-size: var(--fs-base); }
.ltv-result__label { font-size: var(--fs-caption); padding: 2px 6px; border-radius: 3px; font-weight: 600; }
.ltv-result__detail { color: var(--text-secondary); font-size: var(--fs-sm); }
.ltv-result__reason { color: var(--text-secondary); font-size: var(--fs-sm); }
.ltv-result__price { color: var(--up); font-size: var(--fs-body); font-family: var(--font-mono); font-weight: 600; }
.ltv-result__stops { color: var(--text-secondary); font-size: var(--fs-sm); margin-left: auto; }
.ltv-result__row-sub { display: flex; gap: 10px; padding-top: 4px; }
.ltv-result__done { color: var(--success); font-weight: bold; }
.ltv-result__cancelled { color: var(--warning); font-size: var(--fs-caption); font-weight: 600; }

/* ── Empty state ── */
.ltv-empty-state { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; color: var(--text-muted); gap: 8px; }
.ltv-empty-state__icon { font-size: 40px; }
.ltv-empty-state__title { font-size: var(--fs-lg); font-weight: 600; color: var(--text); }
.ltv-empty-state__desc { font-size: var(--fs-body); }

/* ── Modal ── */
.ltv-modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,.45); display: flex; align-items: center; justify-content: center; z-index: 100; }
.ltv-modal {
  background: var(--card-bg); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 20px; width: 360px;
  box-shadow: var(--shadow-lg);
}
.ltv-modal h4 { margin: 0 0 12px; font-size: var(--fs-base); color: var(--text); }
.ltv-modal__form { display: flex; flex-direction: column; gap: 8px; }
.ltv-modal__form input {
  font-size: var(--fs-body); padding: 6px 10px;
  border: 1px solid var(--border); border-radius: var(--radius-sm);
  background: var(--bg); color: var(--text); font-family: var(--font); outline: none;
}
.ltv-modal__form input:focus { border-color: var(--primary); }
.ltv-modal__form input:disabled { opacity: .5; }
.ltv-modal__btns { display: flex; gap: 8px; margin-top: 12px; justify-content: flex-end; }
.ltv-modal--execution { width: min(400px, calc(100vw - 32px)); }
.ltv-execution-summary { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 14px; color: var(--text); }
.ltv-form-field { display: flex; flex-direction: column; gap: 6px; font-size: var(--fs-body); color: var(--text-secondary); }
.ltv-form-field input { width: 100%; box-sizing: border-box; }
.ltv-form-hint { font-size: var(--fs-sm); line-height: 1.5; color: var(--text-secondary); }
.ltv-form-error { padding: 8px 10px; border: 1px solid var(--danger); border-radius: var(--radius-sm); color: var(--danger); font-size: var(--fs-sm); line-height: 1.4; }

/* ── Workflow layout ── */
.ltv-config,
.ltv-stage {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--card-bg);
}
.ltv-stage { padding: 16px; }
.ltv-stage__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 14px;
}
.ltv-stage__identity { display: flex; align-items: center; gap: 12px; min-width: 0; }
.ltv-stage__identity h3 { margin: 0; color: var(--text); font-size: var(--fs-base); font-weight: 650; }
.ltv-stage__identity p { margin: 3px 0 0; color: var(--text-muted); font-size: var(--fs-sm); line-height: 1.45; }
.ltv-stage__number {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  flex: 0 0 32px;
  border-radius: 50%;
  background: var(--primary);
  color: white;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}
.ltv-stage__number--muted { width: auto; padding: 0 9px; border-radius: var(--radius-sm); background: var(--bg-secondary); color: var(--text-secondary); font-size: var(--fs-caption); }
.ltv-stage__summary { color: var(--text-secondary); font-size: var(--fs-sm); white-space: nowrap; }

.ltv-config { overflow: hidden; }
.ltv-config__header { margin: 0; padding: 14px 16px; border-bottom: 1px solid var(--border); background: var(--bg-secondary); }
.ltv-config__grid { display: grid; grid-template-columns: minmax(280px, 1.15fr) minmax(300px, 1fr) minmax(300px, 1fr); gap: 12px; padding: 14px 16px; }
.ltv-config__group { display: flex; flex-direction: column; gap: 10px; min-width: 0; padding: 12px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg); }
.ltv-config__group > strong { color: var(--text); font-size: var(--fs-sm); font-weight: 600; }
.ltv-strategy-control { padding: 9px 10px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg-secondary); }
.ltv-strategy-control.is-unmanaged { border-color: var(--warning); }
.ltv-strategy-control label { display: flex; align-items: center; gap: 8px; min-height: 28px; color: var(--text); font-size: var(--fs-sm); font-weight: 600; cursor: pointer; }
.ltv-strategy-control input { width: 18px; height: 18px; margin: 0; accent-color: var(--primary); }
.ltv-strategy-control small { display: block; margin-top: 4px; color: var(--text-muted); font-size: var(--fs-caption); line-height: 1.45; }
.ltv-strategy-control.is-unmanaged small { color: var(--warning); }
.ltv-takeover-reason { display: grid !important; align-items: stretch !important; gap: 5px !important; margin-top: 8px; }
.ltv-takeover-reason input { width: 100%; min-height: 34px; padding: 6px 8px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg); color: var(--text); font: inherit; }
.ltv-unmanaged-warning { margin-bottom: 12px; padding: 10px 12px; border: 1px solid var(--warning); border-radius: var(--radius-sm); background: var(--bg-secondary); color: var(--warning); font-size: var(--fs-sm); line-height: 1.5; }
.ltv-config__fields { display: flex; align-items: end; flex-wrap: wrap; gap: 10px; }
.ltv-field { align-items: stretch; flex-direction: column; gap: 5px; min-width: 0; }
.ltv-field label { color: var(--text-secondary); }
.ltv-field select,
.ltv-field input { box-sizing: border-box; min-height: 36px; }
.ltv-field--wide,
.ltv-field--wide select { width: 100%; }
.ltv-field--sm input { width: 70px; }
.ltv-input-unit { display: flex; align-items: center; gap: 5px; color: var(--text-muted); font-size: var(--fs-sm); }
.ltv-mode-switch { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); width: 100%; }
.ltv-mode-switch button { min-height: 34px; padding: 5px 8px; }
.ltv-config__actions { display: flex; align-items: center; justify-content: flex-end; gap: 8px; padding: 10px 16px; border-top: 1px solid var(--border); }
.ltv-config__feedback { flex: 1; color: var(--text-muted); font-size: var(--fs-sm); }
.ltv-primary-action { min-width: 128px; min-height: 36px; border-color: var(--primary); background: var(--primary); color: var(--on-primary); font-weight: 600; }
.ltv-primary-action:hover:not(:disabled),
.ltv-primary-action:focus-visible { filter: brightness(1.08); outline: 2px solid var(--primary); outline-offset: 2px; }
.ltv-primary-action:disabled { cursor: not-allowed; opacity: .45; }
.ltv-plan-status { padding: 5px 9px; border: 1px solid var(--border); border-radius: 999px; color: var(--text-secondary); background: var(--bg); font-size: var(--fs-caption); white-space: nowrap; }
.ltv-plan-status.is-stale { border-color: var(--warning); color: var(--warning); }
.ltv-plan-status.is-ready { border-color: var(--success); color: var(--success); }

.ltv-main { display: flex; flex-direction: column; gap: 16px; overflow: visible; min-height: 0; }
.ltv-decision-grid { display: grid; grid-template-columns: minmax(340px, .92fr) minmax(420px, 1.08fr); gap: 12px; align-items: stretch; }
.ltv-candidate-panel,
.ltv-decision { min-width: 0; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg-secondary); }
.ltv-candidate-panel { display: flex; flex-direction: column; overflow: hidden; }
.ltv-left__section { order: initial; margin: 0; padding: 9px 10px; border-bottom: 1px solid var(--border); }
.ltv-left__list { flex: none; min-height: 180px; max-height: 360px; padding: 5px; overflow-y: auto; }
.ltv-row { min-height: 36px; }
.ltv-candidate-panel__footer { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-top: auto; padding: 8px 10px; border-top: 1px solid var(--border); color: var(--text-muted); font-size: var(--fs-caption); }
.ltv-decision { order: initial; margin: 0; }
.ltv-decision__header { min-height: 43px; box-sizing: border-box; padding: 7px 10px; border-bottom: 1px solid var(--border); }
.ltv-decision__body { max-height: 360px; border-top: 0; }
.ltv-decision__header button,
.ltv-left__collapse { min-height: 32px; }
.ltv-cooldown { max-height: none; margin-top: 12px; }

.ltv-plan-overview { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; margin-bottom: 12px; }
.ltv-plan-overview > span { display: flex; align-items: baseline; gap: 6px; padding: 9px 10px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg-secondary); color: var(--text-secondary); font-size: var(--fs-caption); }
.ltv-plan-overview strong { color: var(--text); font-size: var(--fs-base); font-variant-numeric: tabular-nums; }
.ltv-plan-stale-note { margin-bottom: 10px; padding: 8px 10px; border: 1px solid var(--warning); border-radius: var(--radius-sm); color: var(--warning); background: var(--bg-secondary); font-size: var(--fs-sm); }
.ltv-result.is-stale { opacity: .72; }
.ltv-result { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); align-items: start; gap: 10px; min-height: 0; margin: 0; border: 0; overflow: visible; }
.ltv-result__meta { grid-column: 1 / -1; border: 1px solid var(--border); border-radius: var(--radius-sm); }
.ltv-result__group { overflow: hidden; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg); }
.ltv-result__group--secondary > summary { cursor: pointer; list-style-position: inside; user-select: none; }
.ltv-result__group--secondary[open] > summary { border-bottom: 1px solid var(--border); }
.ltv-result__capacity { color: var(--text-muted); font-size: var(--fs-caption); }
.ltv-result__capacity.is-warning { color: var(--warning); }
.ltv-result__group-title { padding: 9px 11px; font-size: var(--fs-sm); }
.ltv-result__row { padding: 9px 11px; }
.ltv-plan-empty { display: flex; flex-direction: column; align-items: center; gap: 5px; padding: 28px 16px; border: 1px dashed var(--border); border-radius: var(--radius-sm); background: var(--bg-secondary); color: var(--text-muted); text-align: center; }
.ltv-plan-empty strong { color: var(--text); font-size: var(--fs-body); }
.ltv-plan-empty span { font-size: var(--fs-sm); }

.ltv-right { display: grid; grid-template-columns: minmax(0, 1.65fr) minmax(280px, .75fr); gap: 12px; overflow: visible; padding: 0; }
.ltv-right__empty { grid-column: 1 / -1; min-height: 220px; }
.ltv-chart-box { min-width: 0; }
.ltv-chart--main .ltv-chart { height: 360px; }
.ltv-suggest { padding: 12px; }
.ltv-suggest__header { flex-wrap: wrap; }
.ltv-suggest__grid { flex-direction: column; gap: 12px; }
.ltv-suggest__col + .ltv-suggest__col { padding-top: 10px; border-top: 1px solid var(--border); }

.ltv-empty-state { min-height: 340px; border: 1px dashed var(--border); border-radius: var(--radius); background: var(--card-bg); }
.ltv-empty-state__index { display: inline-flex; align-items: center; justify-content: center; width: 48px; height: 48px; border-radius: 50%; background: var(--primary-muted); color: var(--primary); font-weight: 700; }

@media (max-width: 1180px) {
  .ltv-config__grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .ltv-config__group--source { grid-column: 1 / -1; }
  .ltv-decision-grid { grid-template-columns: 1fr; }
  .ltv-left__list,
  .ltv-decision__body { max-height: 320px; }
  .ltv-right { grid-template-columns: 1fr; }
  .ltv-suggest__grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .ltv-suggest__col + .ltv-suggest__col { padding-top: 0; padding-left: 12px; border-top: 0; border-left: 1px solid var(--border); }
}

@media (max-width: 760px) {
  .ltv-root { gap: 12px; padding-inline: 12px; }
  .ltv-stage { padding: 12px; }
  .ltv-stage__header,
  .ltv-config__header { align-items: flex-start; flex-direction: column; }
  .ltv-stage__summary,
  .ltv-plan-status { align-self: flex-start; }
  .ltv-config__grid { grid-template-columns: 1fr; padding: 12px; }
  .ltv-config__group--source { grid-column: auto; }
  .ltv-config__fields { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .ltv-field--decision { grid-column: 1 / -1; }
  .ltv-field select,
  .ltv-field input { min-height: 44px; width: 100%; }
  .ltv-input-unit { display: grid; grid-template-columns: minmax(0, 1fr) auto; }
  .ltv-mode-switch button { min-height: 44px; }
  .ltv-config__actions { align-items: stretch; flex-wrap: wrap; padding: 12px; }
  .ltv-config__feedback { flex-basis: 100%; }
  .ltv-config__actions button { min-height: 44px; flex: 1; }
  .ltv-decision-grid { grid-template-columns: minmax(0, 1fr); }
  .ltv-row { min-height: 44px; }
  .ltv-decision__header button,
  .ltv-left__collapse,
  .ltv-candidate-panel__footer button { min-height: 44px; }
  .ltv-plan-overview { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .ltv-result { grid-template-columns: 1fr; }
  .ltv-suggest__grid { display: flex; }
  .ltv-technical__header { flex-direction: column; }
  .ltv-technical__grid { grid-template-columns: 1fr; }
  .ltv-suggest__col + .ltv-suggest__col { padding-top: 10px; padding-left: 0; border-top: 1px solid var(--border); border-left: 0; }
  .ltv-chart--main .ltv-chart { height: 300px; }
  .ltv-candidate-panel__footer { align-items: flex-start; flex-direction: column; }
  .ltv-modal { width: min(360px, calc(100vw - 24px)); box-sizing: border-box; }
}
</style>
