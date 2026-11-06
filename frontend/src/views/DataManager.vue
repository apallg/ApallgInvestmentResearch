<template>
  <div class="dm-root">
    <div class="page-header">
      <div>
        <h1 class="page-title">数据中心</h1>
        <p class="dm-subtitle">查看近期更新、历史数据质量、采集作业与实验准入状态</p>
      </div>
      <button class="btn" @click="refreshOverview" :disabled="overviewLoading">
        {{ overviewLoading ? '加载中...' : '刷新概览' }}
      </button>
    </div>

    <nav class="dm-tabs" aria-label="数据中心工作区">
      <button v-for="tab in tabs" :key="tab.id" :class="['dm-tab', { 'dm-tab--active': activeTab === tab.id }]"
        :aria-current="activeTab === tab.id ? 'page' : undefined" @click="setActiveTab(tab.id)">
        {{ tab.label }}
      </button>
    </nav>

    <!-- ═══ 总览仪表条 ═══ -->
    <div v-if="activeTab === 'overview'" class="dm-bar">
      <div class="dm-bar-item">
        <span class="dm-bar-val" :class="healthColor">{{ overallHealth.recent_update_rate ?? overallHealth.health_pct }}%</span>
        <span class="dm-bar-label">近期更新达标率</span>
        <span class="dm-dot" :class="healthDot"></span>
      </div>
      <span class="dm-bar-sep"></span>
      <div class="dm-bar-item">
        <span class="dm-bar-val dm-bar-val--ok">{{ overallHealth.fresh }}</span>
        <span class="dm-bar-label">正常</span>
      </div>
      <div class="dm-bar-item">
        <span class="dm-bar-val dm-bar-val--warn">{{ overallHealth.stale + overallHealth.incomplete }}</span>
        <span class="dm-bar-label">警告</span>
      </div>
      <div class="dm-bar-item" v-if="overallHealth.unavailable">
        <span class="dm-bar-val dm-bar-val--muted">{{ overallHealth.unavailable }}</span>
        <span class="dm-bar-label">来源已停发</span>
      </div>
      <div class="dm-bar-item">
        <span class="dm-bar-val dm-bar-val--bad">{{ overallHealth.missing + overallHealth.error }}</span>
        <span class="dm-bar-label">异常</span>
      </div>
      <span class="dm-bar-sep"></span>
      <div class="dm-bar-item" v-if="overview">
        <span class="dm-bar-val dm-bar-val--muted">{{ overview.qlib?.stocks || '--' }}</span>
        <span class="dm-bar-label">量化数据标的</span>
      </div>
      <div class="dm-bar-item" v-if="overview">
        <span class="dm-bar-val dm-bar-val--muted">{{ overview.qlib?.dates || '--' }}</span>
        <span class="dm-bar-label">交易日</span>
      </div>
      <div class="dm-bar-item" v-if="overview">
        <span class="dm-bar-status" :class="overview.qlib?.fresh ? 'dm-bar-status--on' : 'dm-bar-status--off'">
          {{ overview.qlib?.restart_required ? '需重启后端' : overview.qlib?.fresh ? '末日已同步' : overview.qlib?.synced ? '待同步' : '未同步' }}
        </span>
        <span class="dm-bar-label">截面未验证</span>
      </div>
    </div>

    <div v-if="activeTab === 'overview' && overview" class="dm-state-summary">
      <article><span>历史行情</span><strong>{{ historyQuality?.history_denominator_verified ? '已核验' : '无法判定' }}</strong><small>{{ historyQuality?.history_denominator_verified ? '固定股票池分母可用' : '只掌握现库行数，不代表历史覆盖率' }}</small></article>
      <article><span>Qlib 数据</span><strong>{{ overview.qlib?.restart_required ? '需重启后端' : overview.qlib?.fresh ? '末日已同步' : '待检查' }}</strong><small>{{ overview.qlib?.validation_note || '末日状态不代表逐日截面完整' }}</small></article>
      <article><span>实验准入</span><strong>{{ historyQuality?.training_ready ? '可进入实验' : '阻止' }}</strong><small>{{ historyQuality?.training_ready ? '已通过历史认证门槛' : '历史质量及 PIT 认证未全部通过' }}</small></article>
      <article><span>最近更新</span><strong>{{ overallHealth.fresh }} / {{ overallHealth.monitored_total ?? overallHealth.total }}</strong><small>按仍可采集的数据频率计算</small></article>
    </div>

    <div v-if="overviewLoading && !overview" class="msg">加载数据概览中...</div>
    <div v-if="overviewError" class="error">{{ overviewError }}</div>
    <div v-if="overview?.qlib?.restart_required" class="dm-hint-warn" role="alert">Qlib 当前数据代已变化，请重启后端后刷新并验证数据；重启前暂不可读取，也无需重复同步。</div>

    <section v-if="activeTab === 'history'" class="dm-snapshot">
      <div><strong>稳定历史池500D</strong><span>仅使用决策日以前至少500个有效交易观测</span></div>
      <label>决策日期<input v-model="snapshotDate" type="date" /></label>
      <button class="btn-sm" :disabled="snapshotLoading" @click="checkSnapshot">{{ snapshotLoading ? '校验中…' : '构建并校验时点一致性快照' }}</button>
      <dl v-if="snapshot">
        <dt>股票数量</dt><dd>{{ snapshot.member_count }}</dd><dt>时点一致性状态</dt><dd>{{ snapshot.pit_status }}</dd>
        <dt>证券状态历史</dt><dd>{{ snapshot.security_reference_status }} · {{ ((snapshot.security_reference_coverage || 0) * 100).toFixed(1) }}%</dd><dt>数据截止</dt><dd>{{ snapshot.latest_trade_date }}</dd>
        <dt>快照指纹</dt><dd><code :title="snapshot.snapshot_id" @click="copySnapshot">{{ snapshot.snapshot_id }}</code></dd>
      </dl>
      <span v-if="snapshotError" class="error">{{ snapshotError }}</span>
    </section>

    <section v-if="activeTab === 'history'" class="dm-history-card">
      <div class="dm-history-head"><div><strong>历史行情与模型特征</strong><span>历史分母未核验前，缺价率只代表当前表内已有行</span></div>
        <span class="dm-status" :class="historyQuality?.training_ready ? 'fresh' : 'incomplete'">{{ historyQuality?.training_ready ? '可用于实验' : '实验阻止' }}</span>
      </div>
      <div v-if="historyQualityError" class="error">{{ historyQualityError }}</div>
      <div v-if="historyQuality" class="dm-quality-grid">
        <div class="dm-quality-item"><span>历史池分母</span><strong>{{ historyQuality.history_denominator_verified ? '已核验' : '待核验' }}</strong><small>{{ historyQuality.quote_denominator }}</small></div>
        <div class="dm-quality-item"><span>P1特征契约</span><strong>{{ historyQuality.panel_contract_current ? '当前版本' : '需要重建' }}</strong><small>{{ historyQuality.panel_path || '尚无面板报告' }}</small></div>
        <div class="dm-quality-item"><span>股票年度覆盖</span><strong>{{ historyQuality.stockyear_coverage_verified ? '已核验' : '待核验' }}</strong></div>
        <div class="dm-quality-item"><span>行业历史可见时间</span><strong>{{ historyQuality.industry_pit_verified ? '严格 PIT 已核验' : '尚未核验' }}</strong><small>{{ historyQuality.industry_vintage || '无版本信息' }}</small></div>
        <div class="dm-quality-item"><span>整体覆盖门槛</span><strong>{{ historyQuality.panel_coverage_pass ? '通过' : '未通过' }}</strong><small>要求核心字段有效覆盖至少 98%</small></div>
        <div class="dm-quality-item"><span>认证状态</span><strong>{{ historyQuality.certification_status === 'certified' ? '已认证' : '未认证' }}</strong><small>页面读取报告不等于实验准入</small></div>
      </div>
      <div v-if="historyQuality?.years?.length" class="dm-year-table-wrap">
        <table class="dm-year-table">
          <thead><tr><th>年份</th><th>现库行数</th><th>OHLC 缺失</th><th>OHLC 非法</th><th>流动性缺失</th><th>只有复权因子</th></tr></thead>
          <tbody><tr v-for="year in historyQuality.years" :key="year.year">
            <td>{{ year.year }}</td><td>{{ Number(year.rows ?? 0).toLocaleString() }}</td>
            <td>{{ (Number(year.price_missing_rate ?? 0) * 100).toFixed(2) }}%（{{ Number(year.price_missing ?? 0).toLocaleString() }}）</td>
            <td>{{ (Number(year.price_invalid_rate ?? 0) * 100).toFixed(2) }}%（{{ Number(year.price_invalid ?? 0).toLocaleString() }}）</td>
            <td>{{ Number(year.liquidity_missing ?? 0).toLocaleString() }}</td><td>{{ Number(year.factor_only ?? 0).toLocaleString() }}</td>
          </tr></tbody>
        </table>
      </div>
      <details v-if="historyQuality?.panel_missing_rates && Object.keys(historyQuality.panel_missing_rates).length" class="dm-quality-details">
        <summary>P1 特征缺失指示率</summary>
        <div v-for="(rate, field) in historyQuality.panel_missing_rates" :key="field"><span>{{ field }}</span><strong>{{ (rate * 100).toFixed(2) }}%</strong></div>
      </details>
      <div v-if="historyQuality && !historyQuality.history_denominator_verified" class="dm-hint-warn">当前报告不能代表历史可交易股票池覆盖率；请先运行历史预检并建立上市/退市/停牌分母。</div>
    </section>

    <!--
      Historical repair actions are deliberately batch based. No publish action
      is shown until the backend provides a validated, rollback-capable batch.
    -->
    <section v-if="activeTab === 'history'" class="dm-history-workflow">
      <div><strong>历史修复批次</strong><span>影子核对、冲突裁决和可回滚发布按批次执行</span></div>
      <div class="dm-history-gates">
        <span v-for="gate in historyGates" :key="gate.id" :class="['dm-gate', 'dm-gate--' + gate.status]">
          <b>{{ gate.step }}</b><span>{{ gate.label }}</span><small>{{ gate.note }}</small>
        </span>
      </div>
      <div class="dm-step-row"><button class="btn" :disabled="historyPreflightLoading" @click="runHistoryPreflight">{{ historyPreflightLoading ? '正在预检…' : '运行只读历史预检' }}</button><button class="btn btn-outline" :disabled="historyBaselineLoading" @click="freezeHistoryBaseline">{{ historyBaselineLoading ? '正在冻结基线…' : '冻结 G0 数据基线' }}</button><span v-if="historyPreflight?.checked_at">检查时间：{{ historyPreflight.checked_at }}</span></div>
      <div class="dm-shadow-batch">
        <strong>历史行业关系快照</strong>
        <p>按 BaoStock 每周历史版本采集全市场行业关系，保存查询日、来源更新日、原始文件和内容哈希。任务可中断后续跑，不覆盖当前行业关系。</p>
        <div class="dm-shadow-range"><label>开始周<input v-model="industryHistoryStart" type="date"></label><label>结束周<input v-model="industryHistoryEnd" type="date"></label></div>
        <div class="dm-step-row"><button class="btn" :disabled="industryHistoryBusy" @click="startIndustryHistory">{{ industryHistoryBusy ? '正在提交…' : '采集 / 续跑行业历史' }}</button><button class="btn btn-outline" :disabled="industryHistoryBusy" @click="loadIndustryHistory">刷新状态</button><button v-if="industryHistoryJobId" class="btn btn-danger" :disabled="industryHistoryBusy" @click="cancelIndustryHistory">请求停止</button><span>{{ industryHistoryMessage }}</span></div>
        <div v-if="industryHistoryStatus?.summary" class="dm-preflight-result" :class="industryHistoryStatus.pit_status === 'certified' ? '' : 'dm-preflight--blocked'">
          <strong>{{ industryHistoryStatus.summary.snapshots || 0 }} 个周快照 · {{ Number(industryHistoryStatus.summary.snapshot_rows || 0).toLocaleString() }} 快照行 · {{ industryHistoryStatus.summary.distinct_payloads || 0 }} 个不同内容版本</strong>
          <span>{{ industryHistoryStatus.summary.first_query_date || '无' }} 至 {{ industryHistoryStatus.summary.last_query_date || '无' }} · 来源更新日 {{ industryHistoryStatus.summary.first_source_update_date || '无' }} 至 {{ industryHistoryStatus.summary.last_source_update_date || '无' }}</span>
          <small>失败 {{ industryHistoryStatus.failures?.failed_weeks ?? 0 }} 周 · {{ industryHistoryStatus.note }} PIT：{{ industryHistoryStatus.pit_status || '未认证' }}</small>
        </div>
        <p class="dm-hint-warn">这是来源原样快照，不代表股票池覆盖率；历史响应包含查询日前已退市证券，后续必须与已核验的历史股票池分母取交集。来源只有更新日期，没有精确发布时间；暂按更新日后一天作为保守可用日期，仍需单独审核后才能用于正式历史实验。</p>
      </div>
      <div v-if="historyBaseline" class="dm-preflight-result dm-preflight--blocked">
        <strong>基线已冻结，但历史分母仍未认证</strong>
        <span>{{ historyBaseline.baseline_id }} · {{ historyBaseline.range?.start }} 至 {{ historyBaseline.range?.end }} · 聚合指纹 {{ historyBaseline.fingerprint }}</span>
        <span>身份记录 {{ historyBaseline.identity_count?.toLocaleString() }} 条；现库行情按年 {{ historyBaseline.quote_years?.length || 0 }} 组、股票年度 {{ historyBaseline.stock_years_count?.toLocaleString() || 0 }} 组。</span>
        <ul><li v-for="blocker in historyBaseline.denominator_blockers || []" :key="blocker">{{ blocker }}</li></ul>
        <small v-if="historyBaseline.artifact">工件：{{ historyBaseline.artifact }}</small>
      </div>
      <div class="dm-shadow-batch dm-denominator-batch">
        <strong>G0 历史证券池分母影子核算</strong>
        <p>用证券上市/退市有效日期、交易日历和全日停牌记录构造股票×交易日预期分母，再与现库逐日行情核对。先按年小范围验证；批次只保存影子工件，不改正式行情和 Qlib。</p>
        <div class="dm-shadow-range"><label>开始日期<input v-model="historyDenominatorStart" type="date"></label><label>结束日期<input v-model="historyDenominatorEnd" type="date"></label></div>
        <div class="dm-step-row"><button class="btn" :disabled="historyDenominatorBusy" @click="startHistoryDenominator">{{ historyDenominatorBusy ? '正在提交…' : '生成 G0 分母影子批次' }}</button><button v-if="historyDenominatorBatch?.batch_id" class="btn btn-outline" :disabled="historyDenominatorBusy" @click="refreshHistoryDenominator()">刷新批次</button><button v-if="['queued','processing'].includes(historyDenominatorBatch?.status)" class="btn btn-danger" :disabled="historyDenominatorBusy" @click="cancelHistoryDenominator">请求停止</button><span>{{ historyDenominatorMessage }}</span></div>
        <div v-if="historyDenominatorBatch?.batch_id" class="dm-preflight-result" :class="historyDenominatorBatch.denominator_verified ? '' : 'dm-preflight--blocked'">
          <strong>批次 {{ historyDenominatorBatch.batch_id }} · {{ historyDenominatorBatch.status }} · {{ historyDenominatorBatch.denominator_verified ? '分母结构校验通过' : '待核验/存在阻断' }}</strong>
          <span>{{ historyDenominatorBatch.date_range?.start }} 至 {{ historyDenominatorBatch.date_range?.end }} · 身份来源：{{ historyDenominatorBatch.basis }} · 严格 PIT：{{ historyDenominatorBatch.strict_pit_status_verified ? '已验证' : '未验证' }}</span>
          <div v-if="historyDenominatorBatch.year_summaries?.length" class="dm-year-table-wrap"><table class="dm-year-table"><thead><tr><th>年份</th><th>预期股票日</th><th>全日停牌</th><th>完整行情</th><th>缺行情行</th><th>有行缺价</th><th>阻断</th></tr></thead><tbody><tr v-for="year in historyDenominatorBatch.year_summaries" :key="year.year"><td>{{ year.year }}</td><td>{{ year.expected_stock_dates == null ? '—' : Number(year.expected_stock_dates).toLocaleString() }}</td><td>{{ year.suspended_stock_dates == null ? '—' : Number(year.suspended_stock_dates).toLocaleString() }}</td><td>{{ year.observed_price_complete == null ? '—' : Number(year.observed_price_complete).toLocaleString() }}</td><td>{{ year.missing_row_unresolved == null ? '—' : Number(year.missing_row_unresolved).toLocaleString() }}</td><td>{{ year.observed_row_missing_price == null ? '—' : Number(year.observed_row_missing_price).toLocaleString() }}</td><td>{{ year.source_blockers?.length ? year.source_blockers.join('；') : '无' }}</td></tr></tbody></table></div>
          <ul v-if="historyDenominatorBatch.denominator_blockers?.length"><li v-for="blocker in historyDenominatorBatch.denominator_blockers" :key="blocker">{{ blocker }}</li></ul>
          <small>Manifest：{{ historyDenominatorBatch.manifest_path || `data/historical_repair/denominator/${historyDenominatorBatch.batch_id}/manifest.json` }}</small>
          <p class="dm-hint-warn">此批次基于当前身份快照的上市/退市有效日期重建，不是严格历史 PIT 证券状态；“分母结构校验通过”不等于数据完整、可发布或可用于正式 OOS 实验。</p>
        </div>
      </div>
      <div class="dm-shadow-batch">
        <strong>G1 双源影子采集 / G2 分片发布</strong>
        <p>先生成股票×年度清单并完成双源采集。每个股票年度必须关联已认证的 G0 年度分母；预检通过后可发布该分片到正式行情库与 Qlib，自动保存 pre-image 和回滚凭据。</p>
        <label>G0 分母批次 ID<input v-model="historyDenominatorId" placeholder="先完成对应年度 G0；可从上方批次复制" /></label>
        <div class="dm-shadow-range">
          <label>载入已有 G1 批次 ID<input v-model="historyShadowLoadId" placeholder="输入 12 位批次 ID 后载入并续跑" /></label>
          <button class="btn btn-outline" :disabled="historyShadowBusy || !historyShadowLoadId.trim()" @click="loadHistoryShadowBatch">载入已有批次</button>
        </div>
        <label>股票代码（逗号、空格或换行分隔）<textarea v-model="historyShadowCodes" rows="3" placeholder="例如：000001, 600000"></textarea></label>
        <div class="dm-shadow-range"><label>开始日期<input v-model="historyShadowStart" type="date"></label><label>结束日期<input v-model="historyShadowEnd" type="date"></label></div>
        <div class="dm-step-row"><button class="btn" :disabled="historyShadowBusy" @click="prepareHistoryShadow">{{ historyShadowBusy ? '正在生成分片清单…' : '生成影子批次清单' }}</button><button class="btn btn-outline" :disabled="historyShadowBusy || !historyShadowBatch?.batch_id || historyShadowBatch.status === 'invalidated'" @click="startHistoryShadow">启动/续跑影子采集</button><button v-if="historyShadowBatch?.batch_id" class="btn btn-outline" :disabled="historyShadowBusy" @click="refreshHistoryShadow">刷新批次</button><button v-if="['queued','collecting'].includes(historyShadowBatch?.status)" class="btn btn-danger" :disabled="historyShadowBusy" @click="cancelHistoryShadow">请求停止</button><span>{{ historyShadowMessage }}</span></div>
        <div v-if="historyShadowBatch?.batch_id" class="dm-preflight-result" :class="historyShadowBatch.status === 'review_required' ? 'dm-preflight--blocked' : ''">
          <strong>批次 {{ historyShadowBatch.batch_id }} · {{ historyShadowBatch.status }}</strong>
          <span>{{ historyShadowBatch.work_items?.length ?? historyShadowBatch.work_items }} 个股票年度分片 · {{ historyShadowBatch.requested_codes?.length ?? historyShadowBatch.codes }} 只股票 · 数据源客户端版本 {{ JSON.stringify(historyShadowBatch.source_versions || {}) }}</span>
          <small>Manifest：{{ historyShadowBatch.manifest_path || historyShadowBatch.manifest || '批次已保存；采集完成后显示分片状态' }}</small>
          <div v-if="historyShadowBatch.work_items?.length" class="dm-shadow-items">
            <div v-for="item in historyShadowBatch.work_items" :key="`${item.code}-${item.year}`">
              <span>{{ item.code }} · {{ item.year }}</span><b>{{ item.status }}</b>
              <small v-if="item.comparison_status_counts">双源日线：{{ JSON.stringify(item.comparison_status_counts) }}；现库价格：{{ JSON.stringify(item.database_price_status_counts || {}) }}</small>
              <small v-else-if="item.error">{{ item.error }}</small>
              <div class="dm-step-row">
                <button v-if="item.files?.comparison" class="btn-sm btn-outline" @click="loadHistoryShadowConflicts(item)">{{ historyShadowConflicts[`${item.code}-${item.year}`] ? '收起差异行' : '查看未匹配日期' }}</button>
                <button v-if="item.status === 'collected_review'" class="btn-sm btn-outline" :disabled="historyShadowBusy || !historyDenominatorId" @click="prepareHistoryRelease(item)">{{ historyShadowBusy ? '正在预检…' : 'G2 发布预检' }}</button>
                <button v-if="canPrepareVerifiedConflictRepair(item)" class="btn-sm btn-outline" :disabled="historyShadowBusy || !historyDenominatorId" @click="prepareHistoryRelease(item, true)">{{ historyShadowBusy ? '正在核验双源…' : '双源一致后修复现库冲突' }}</button>
              </div>
              <div v-if="historicalReleases[`${item.code}-${item.year}`]" class="dm-preflight-result" :class="historicalReleases[`${item.code}-${item.year}`].status === 'ready' || historicalReleases[`${item.code}-${item.year}`].status === 'published' ? '' : 'dm-preflight--blocked'">
                <strong>G2 {{ historicalReleases[`${item.code}-${item.year}`].status }} · 发布 ID {{ historicalReleases[`${item.code}-${item.year}`].release_id }}</strong>
                <span>候选 {{ historicalReleases[`${item.code}-${item.year}`].assessment?.candidate_rows ?? '—' }} 行 · G0 预期 {{ historicalReleases[`${item.code}-${item.year}`].assessment?.expected_rows ?? '—' }} 行 · 备份 {{ historicalReleases[`${item.code}-${item.year}`].preimage_rows ?? '—' }} 行</span>
                <div v-if="historicalReleases[`${item.code}-${item.year}`].source_verified_conflict_replacement" class="dm-hint-warn" role="status">
                  冲突修复模式：逐日双源收盘价、成交量、成交额均通过核验；将更新现库 OHLC、成交量/额、涨跌幅和复权因子，并同步修复对应 Qlib 特征。发布前会保存 pre-image，可回滚。
                  <span>现库价格冲突 {{ historicalReleases[`${item.code}-${item.year}`].assessment?.price_conflict_rows ?? 0 }} 日 · 成交量/额匹配 {{ historicalReleases[`${item.code}-${item.year}`].source_quantity_evidence?.matched_rows ?? 0 }}/{{ historicalReleases[`${item.code}-${item.year}`].source_quantity_evidence?.candidate_rows ?? 0 }} 日</span>
                </div>
                <ul v-if="historicalReleases[`${item.code}-${item.year}`].blockers?.length" role="alert"><li v-for="blocker in historicalReleases[`${item.code}-${item.year}`].blockers" :key="blocker">{{ blocker }}</li></ul>
                <div class="dm-step-row">
                  <button v-if="historicalReleases[`${item.code}-${item.year}`].status === 'ready'" class="btn" :disabled="historyShadowBusy" @click="publishHistoryRelease(item)">{{ historicalReleases[`${item.code}-${item.year}`].source_verified_conflict_replacement ? '发布双源核验冲突修复' : '发布到正式行情库与 Qlib' }}</button>
                  <button v-if="historicalReleases[`${item.code}-${item.year}`].status === 'published'" class="btn btn-danger" :disabled="historyShadowBusy" @click="rollbackHistoryRelease(item)">回滚该发布</button>
                  <span>{{ historicalReleases[`${item.code}-${item.year}`].published_at || historicalReleases[`${item.code}-${item.year}`].error || '' }}</span>
                </div>
              </div>
              <div v-if="historyShadowConflicts[`${item.code}-${item.year}`]" class="dm-conflict-table">
                <strong>未匹配/停牌/冲突记录：{{ historyShadowConflicts[`${item.code}-${item.year}`].total_flagged }} 条（展示前 {{ historyShadowConflicts[`${item.code}-${item.year}`].rows.length }} 条） · G0 {{ historyShadowConflicts[`${item.code}-${item.year}`].denominator_batch_id || '未关联' }}</strong>
                <table><thead><tr><th>日期</th><th>G0分母判定</th><th>TuShare原始收盘</th><th>BaoStock原始收盘</th><th>现库折算未复权价</th><th>现库差值</th><th>双源分类</th><th>现库分类</th></tr></thead><tbody><tr v-for="(row, rowIndex) in historyShadowConflicts[`${item.code}-${item.year}`].rows" :key="rowIndex"><td>{{ (row.trade_date || row.date || row.date_key || '').slice(0, 10) }}</td><td>{{ row.g0_full_day_suspended ? '全日停牌' : row.g0_coverage_status || '无G0记录' }}{{ row.g0_partial_suspension ? ' · 部分停牌' : '' }}</td><td>{{ row.close_tushare ?? '—' }}</td><td>{{ row.close_baostock ?? '—' }}</td><td>{{ row.database_implied_raw_close ?? '—' }}</td><td>{{ row.database_raw_close_difference ?? '—' }}</td><td>{{ row.raw_close_status }}</td><td>{{ row.database_price_status }}</td></tr></tbody></table>
              </div>
            </div>
          </div>
          <p class="dm-hint-warn">影子采集不写正式数据。普通 G2 要求现库价格无冲突；冲突修复预检只有在 G0 认证、双源逐日收盘匹配、成交量/成交额交叉核验通过、Qlib 候选可回滚时才可发布。冲突修复会在发布前要求确认，并保存数据库与 Qlib pre-image。原始保存为 SDK 返回数据，不含 HTTP 原始报文或供应商历史修订版本号。</p>
        </div>
      </div>
      <div v-if="historyPreflight" class="dm-preflight-result" :class="'dm-preflight--' + historyPreflight.status">
        <strong>{{ historyPreflight.status === 'blocked' ? '预检完成：历史发布仍被阻止' : '预检完成' }}</strong>
        <span>{{ historyPreflight.summary }}</span>
        <ul v-if="historyPreflight.blockers?.length"><li v-for="blocker in historyPreflight.blockers" :key="blocker">{{ blocker }}</li></ul>
      </div>
      <div v-if="historyPreflight?.current_footprint" class="dm-preflight-facts">
        <div><span>行情现库范围</span><strong>{{ historyPreflight.current_footprint.quotes?.first_date || '无' }} — {{ historyPreflight.current_footprint.quotes?.last_date || '无' }}</strong><small>{{ historyPreflight.current_footprint.quotes?.stocks ?? 0 }} 只股票 · {{ historyPreflight.current_footprint.quotes?.trade_dates ?? 0 }} 个交易日 · {{ Number(historyPreflight.current_footprint.quotes?.rows ?? 0).toLocaleString() }} 行</small></div>
        <div><span>证券身份当前快照</span><strong>{{ historyPreflight.current_footprint.current_identity_by_status?.length ?? 0 }} 种状态记录</strong><small>{{ (historyPreflight.current_footprint.current_identity_by_status || []).map(item => `${item.status || '未知'} ${item.stocks}只`).join(' · ') || '暂无身份数据' }}；不能代替历史状态</small></div>
        <div><span>行业关系观察记录</span><strong>{{ Number(historyPreflight.current_footprint.industry_observations?.rows ?? 0).toLocaleString() }} 条</strong><small>{{ historyPreflight.current_footprint.industry_observations?.first_observed_date || '无' }} — {{ historyPreflight.current_footprint.industry_observations?.last_observed_date || '无' }}</small></div>
        <div><span>BaoStock 历史行业快照</span><strong>{{ historyPreflight.current_footprint.historical_industry_snapshots?.summary?.snapshots ?? 0 }} / {{ historyPreflight.current_footprint.historical_industry_snapshots?.expected_snapshots ?? '—' }} 周</strong><small>缺 {{ historyPreflight.current_footprint.historical_industry_snapshots?.missing_snapshot_count ?? '—' }} 周 · 失败 {{ historyPreflight.current_footprint.historical_industry_snapshots?.failures?.failed_weeks ?? 0 }} 周 · 仅数据留存，不代表 PIT 认证</small></div>
        <div><span>Qlib 当前目录</span><strong>{{ historyPreflight.qlib?.synced ? '已同步' : '未确认' }}</strong><small>{{ historyPreflight.qlib?.stocks || 0 }} 标的 · {{ historyPreflight.qlib?.dates || 0 }} 日 · 末日 {{ historyPreflight.qlib?.last_date || '未知' }}（未检查逐日截面）</small></div>
      </div>
      <p class="dm-hint-warn">全市场训练/实验准入仍被 G0 全池分母、历史行业 PIT、P1 覆盖和 Qlib 全截面校验门槛阻止；G2 单股年度修复不代表全市场历史已经完整。</p>
    </section>

    <section v-if="activeTab === 'history'" class="dm-audit-card">
      <div class="dm-audit-head"><div><strong>行情写入审计</strong><span>按股票、数据中心任务和日期检索；记录为追加写入，数据库禁止更新或删除</span></div><button class="btn-sm btn-outline" :disabled="quoteAuditLoading" @click="loadQuoteAudit">{{ quoteAuditLoading ? '查询中…' : '刷新记录' }}</button></div>
      <div class="dm-audit-filters">
        <label>股票代码<input v-model="quoteAuditFilters.code" placeholder="如 600000" @keyup.enter="searchQuoteAudit" /></label>
        <label>数据中心任务 ID<input v-model="quoteAuditFilters.task_id" placeholder="粘贴任务 ID" @keyup.enter="searchQuoteAudit" /></label>
        <label>开始日期<input v-model="quoteAuditFilters.start" type="date" /></label>
        <label>结束日期<input v-model="quoteAuditFilters.end" type="date" /></label>
        <button class="btn" :disabled="quoteAuditLoading" @click="searchQuoteAudit">查询</button>
        <button class="btn btn-outline" :disabled="quoteAuditLoading" @click="clearQuoteAuditFilters">清空</button>
      </div>
      <div v-if="quoteAuditError" class="error">{{ quoteAuditError }}</div>
      <div class="dm-audit-summary">共 {{ quoteAudit.total.toLocaleString() }} 条 · 第 {{ quoteAudit.page }} / {{ Math.max(1, Math.ceil(quoteAudit.total / quoteAudit.page_size)) }} 页</div>
      <div class="dm-audit-table-wrap"><table class="dm-audit-table">
        <thead><tr><th>写入时间</th><th>股票 / 日期</th><th>任务 ID</th><th>批次 ID</th><th>来源 / 复权</th><th>动作</th><th>输入指纹</th><th>旧值指纹</th><th>新值指纹</th></tr></thead>
        <tbody>
          <tr v-for="row in quoteAudit.items" :key="row.id">
            <td>{{ row.captured_at }}</td><td><strong>{{ row.code }}</strong><small>{{ row.trade_date }}</small></td>
            <td><button v-if="row.task_id" class="dm-audit-id" :title="row.task_id" @click="copyAuditId(row.task_id)">{{ row.task_id }}</button><span v-else>—</span></td>
            <td><button class="dm-audit-id" :title="row.batch_id" @click="copyAuditId(row.batch_id)">{{ row.batch_id }}</button></td>
            <td>{{ row.source }} · {{ row.adjustment_mode }}<small>{{ row.provider_version }} · 锚点 {{ row.anchor_date }}</small></td>
            <td><span class="dm-audit-action">{{ row.action }}</span></td>
            <td><code :title="row.input_payload_sha256 || ''">{{ row.input_payload_sha256 ? row.input_payload_sha256.slice(0, 12) : '—' }}</code></td>
            <td><code :title="row.old_payload_sha256 || ''">{{ row.old_payload_sha256 ? row.old_payload_sha256.slice(0, 12) : '—' }}</code></td>
            <td><code :title="row.new_payload_sha256 || ''">{{ row.new_payload_sha256 ? row.new_payload_sha256.slice(0, 12) : '—' }}</code></td>
          </tr>
          <tr v-if="!quoteAuditLoading && !quoteAudit.items.length"><td colspan="9" class="dm-audit-empty">没有匹配的审计记录</td></tr>
        </tbody>
      </table></div>
      <div class="dm-audit-pager"><button class="btn-sm btn-outline" :disabled="quoteAuditLoading || quoteAudit.page <= 1" @click="changeQuoteAuditPage(-1)">上一页</button><button class="btn-sm btn-outline" :disabled="quoteAuditLoading || quoteAudit.page * quoteAudit.page_size >= quoteAudit.total" @click="changeQuoteAuditPage(1)">下一页</button></div>
    </section>

    <!-- ═══ 分类卡片网格 ═══ -->
    <div v-if="overview && activeTab === 'overview'" class="dm-cats">
      <div v-for="cat in overview.categories" :key="cat.key"
           class="dm-cat" :class="'dm-cat--' + cat.health">
        <button type="button" class="dm-cat-head" :aria-expanded="expandedCategories.has(cat.key)" @click="toggleCategory(cat.key)">
          <div class="dm-cat-head-left">
            <span class="dm-cat-icon">{{ cat.icon }}</span>
            <div>
              <span class="dm-cat-name">{{ cat.name }}</span>
              <span class="dm-cat-count">{{ cat.tables.length }} 张表</span>
            </div>
          </div>
          <div class="dm-cat-head-right">
            <span class="dm-cat-tag" :class="cat.health">{{ healthLabel(cat.health) }}</span>
            <span class="dm-cat-arrow" :class="{ 'dm-cat-arrow--open': expandedCategories.has(cat.key) }">▾</span>
          </div>
        </button>

        <Transition name="dm-expand">
          <div v-if="expandedCategories.has(cat.key)" class="dm-cat-body">
            <div v-for="t in cat.tables" :key="t.table"
                 class="dm-table" :class="'dm-table--' + t.status">
              <div class="dm-table-main">
                <div class="dm-table-top">
                  <span class="dm-table-name">{{ t.desc }}</span>
                  <code class="dm-table-code">{{ t.table }}</code>
                  <span class="dm-table-freq">{{ t.frequency }}</span>
                </div>
                <div class="dm-table-meta">
                  <span>记录 <strong>{{ t.row_count.toLocaleString() }}</strong><span class="dm-table-min"> / {{ t.expected_min.toLocaleString() }}</span></span>
                  <span v-if="t.latest_date">最新 <strong>{{ t.latest_date }}</strong></span>
                </div>
                <div v-if="t.issues.length" class="dm-issues">
                  <span v-for="(iss, j) in t.issues" :key="j" class="dm-issue">{{ iss }}</span>
                </div>
              </div>
              <div class="dm-table-act">
                <span class="dm-status" :class="t.status">{{ statusLabel(t.status) }}</span>
                <button v-if="t.repairable && t.status !== 'fresh' && !['stock_identity', 'stock_index_membership', 'stock_dividend_events'].includes(t.table)"
                        class="btn-sm btn-outline" :disabled="repairBusy"
                        @click.stop="doRepair(t.table)">
                  {{ repairing.has(t.table) ? '修复中...' : '修复' }}
                </button>
              </div>
            </div>
          </div>
        </Transition>
      </div>
    </div>

    <!-- ═══ 一键修复 ═══ -->
    <div v-if="overview && activeTab === 'maintenance'" class="dm-repair-bar">
      <span class="dm-repair-hint">{{ repairMessage || `近期检查发现 ${warnCount} 项可修复；提交前先核对执行清单` }}</span>
      <button class="btn btn-warning" :disabled="repairBusy || repairPreviewLoading" @click="previewRepairs">
        {{ repairPreviewLoading ? '正在生成清单…' : '预览近期修复清单' }}
      </button>
    </div>
    <div v-if="repairPreview && activeTab === 'maintenance'" class="dm-preview-card">
      <div class="dm-preview-head"><strong>执行范围：{{ repairPreview.target_count }} 项</strong><span>目标交易日 {{ repairPreview.trade_date }} · 有效期 {{ repairPreview.expires_in_seconds }} 秒 · 不包含历史修复</span></div>
      <div v-if="!repairPreview.items?.length" class="dm-hint">近期数据无需修复。</div>
      <div v-for="item in repairPreview.items" :key="item.table" class="dm-preview-item"><strong>{{ item.desc }}</strong><code>{{ item.table }}</code><span>{{ item.repair_scope }}；来源：{{ item.source }}；写入：{{ item.write_target }}</span><small>当前：{{ statusLabel(item.status) }} · 最新：{{ item.latest_date || '无记录' }} · {{ item.row_count ?? '未知' }} 行 · 完成判据：{{ item.completion_check }} · 影响：{{ item.downstream }}{{ item.reason ? ` · 原因：${item.reason}` : '' }}</small></div>
      <button class="btn btn-warning" :disabled="repairBusy || !repairPreview.items?.length" @click="startRepairAll">{{ repairAllRunning ? '修复中…' : '按此清单开始修复' }}</button>
    </div>
    <div v-if="repairResults.length && activeTab === 'maintenance'" class="dm-repair-results">
      <div v-for="item in repairResults" :key="item.table" class="dm-repair-item">
        <span>{{ item.desc }}</span>
        <span class="dm-status" :class="item.status === 'completed' ? 'fresh' : item.status === 'error' ? 'error' : 'incomplete'">
          {{ item.restart_required ? '需重启后验证' : repairResultLabel(item.status) }}
        </span>
        <span class="dm-repair-count" v-if="item.after">
          {{ item.before?.row_count ?? '--' }} → {{ item.after.row_count ?? '--' }}
        </span>
        <span class="dm-repair-error" v-if="item.error">{{ item.error }}</span>
        <span class="dm-repair-error" v-else-if="item.status === 'partial'">{{ item.after?.reason }}</span>
      </div>
    </div>

    <!-- ═══ 采集与同步 ═══ -->
    <section v-if="activeTab === 'maintenance'" class="dm-ops">
      <h2 class="dm-ops-title">采集与同步</h2>

      <div class="fm-block">
        <div class="fm-block-title">股票身份与关系 / 事件与资金结构</div>
        <div class="dm-step-row">
          <button @click="collectContext('identity')" :disabled="contextRunning.identity" class="btn btn-outline">
            {{ contextRunning.identity ? '采集中...' : '更新股票身份与指数关系' }}
          </button>
          <button @click="collectContext('events')" :disabled="contextRunning.events" class="btn btn-outline">
            {{ contextRunning.events ? '采集中...' : '回补近一年分红事件' }}
          </button>
        </div>
        <div v-if="contextMsg" class="dm-prog">{{ contextMsg }}</div>
      </div>

      <div class="fm-block">
        <div class="fm-block-title">增量更新</div>
        <div class="dm-step-row">
          <select v-model="incrPool" class="dm-input">
            <option value="all">全部A股</option>
            <option value="csi_all">沪深300+中证500</option>
            <option value="csi300">仅沪深300</option>
            <option value="csi500">仅中证500</option>
          </select>
          <label class="dm-field-inline">回溯 <input v-model.number="incrLookback" type="number" min="1" max="30" class="dm-input dm-input--sm" /> 天</label>
          <label class="dm-field-inline">线程 <input v-model.number="incrThreads" type="number" min="4" max="64" class="dm-input dm-input--sm" /></label>
          <button @click="doIncremental" :disabled="incrRunning" class="btn" :class="{ loading: incrRunning }">
            {{ incrRunning ? '更新中...' : '开始增量更新' }}
          </button>
        </div>
        <div v-if="incrMsg" class="dm-prog">{{ incrMsg }}</div>
        <div v-if="incrementalResult" class="dm-prog">
          采集 {{ incrementalJob.collection_status || '未知' }} · Qlib同步 {{ incrementalJob.sync_status || '进行中' }}
          · 期望 {{ incrementalResult.expected }} / 成功 {{ incrementalResult.success }} / 失败 {{ incrementalResult.failed }} / 已有数据跳过 {{ incrementalResult.skipped }}
          · 实际日期 {{ incrementalResult.actual_date_range?.start || '无成功日期' }} 至 {{ incrementalResult.actual_date_range?.end || '无成功日期' }}
          <span v-if="incrementalResult.failed_codes?.length">· 失败样本 {{ incrementalResult.failed_codes.slice(0, 5).map(item => item.code).join('、') }}</span>
          <button
            v-if="incrementalJob.status === 'partial' && incrementalJob.collection_status === 'partial' && incrementalResult.failed_codes?.length"
            @click="retryFailedIncremental"
            :disabled="incrRunning"
            class="btn btn-outline"
          >仅重试失败证券</button>
        </div>
      </div>

      <div class="fm-block">
        <div class="fm-block-title">全量下载</div>
        <div class="dm-step-row">
          <button disabled class="btn btn-outline">全量下载已暂停</button>
          <span class="dm-hint">安全历史批次未接通，配置项暂不开放。</span>
        </div>
        <div class="dm-hint-warn">按当前存续名单筛选历史数据可能遗漏退市/ST证券；安全影子批次和版本发布尚未接通，当前不允许写入正式库。</div>
      </div>

      <div class="fm-block">
        <div class="fm-block-title">同步到量化数据集</div>
        <div class="dm-step-row">
          <button @click="doSync(false)" :disabled="syncing" class="btn">增量同步</button>
          <button disabled class="btn btn-outline">全量重建已暂停</button>
          <button @click="doValidateQlib" :disabled="syncing" class="btn btn-outline">验证量化数据</button>
        </div>
        <div v-if="syncMessage" class="dm-prog">{{ syncMessage }}</div>
        <div class="dm-index-sync">
          <div>
            <div class="dm-index-sync-title">基准指数</div>
            <div class="dm-hint">仅更新沪深300和上证指数的 Qlib 行情，不重建股票特征。</div>
          </div>
          <button @click="doSyncBenchmarks" :disabled="syncing || indexSyncing" class="btn btn-outline">
            {{ indexSyncing ? '指数同步中...' : '同步基准指数' }}
          </button>
        </div>
        <div v-if="indexSyncMsg" class="dm-prog">{{ indexSyncMsg }}</div>
        <div v-if="qlibCheck.length" class="dm-check-list">
          <div v-for="(c, i) in qlibCheck" :key="i" :class="['dm-check-line', c.ok ? 'dm-check--ok' : 'dm-check--fail']">
            {{ c.item }}: {{ c.msg }}
          </div>
        </div>
      </div>

      <div class="fm-block">
        <div class="fm-block-title">基本面数据</div>
        <div class="dm-step-row" style="margin-top:12px">
          <select v-model="fundPool" class="dm-input" style="width:110px">
            <option value="all">全A股</option>
            <option value="csi300">沪深300</option>
            <option value="csi500">中证500</option>
          </select>
          <button @click="doFundamentals" :disabled="fundRunning" class="btn-sm btn-outline">
            {{ fundRunning ? '采集中...' : '采集基本面' }}
          </button>
          <span v-if="fundMsg" class="dm-prog">{{ fundMsg }}</span>
        </div>
      </div>
    </section>

    <section v-if="activeTab === 'tasks'" class="dm-jobs">
      <div class="dm-jobs-head"><div><h2>任务与设置</h2><span>任务状态由服务端查询；离开页面或刷新后会继续恢复跟踪。</span></div><button class="btn-sm btn-outline" @click="loadJobs">刷新任务</button></div>
      <div v-if="!jobs.length" class="msg">暂无任务记录</div>
      <article v-for="job in jobs" :key="job.job_id" class="dm-job-card">
        <div class="dm-job-main"><strong>{{ job.title || job.description || job.name || job.job_id }}</strong><code>{{ job.job_id }}</code><span>{{ job.message || job.progress?.message || job.status }}</span></div>
        <div class="dm-job-meta"><span class="dm-status" :class="jobStatusClass(job.status)">{{ jobStatusLabel(job.status) }}</span><span>{{ job.progress?.stage || job.stage || job.source || '任务' }}</span><span v-if="job.progress?.total">{{ job.progress.current || 0 }} / {{ job.progress.total }}</span><time>{{ job.created_at ? new Date(typeof job.created_at === 'number' ? job.created_at * 1000 : job.created_at).toLocaleString() : '' }}</time></div>
        <details v-if="job.error || job.sync_error || job.collection_result || job.sync_result || job.progress?.results?.length || job.result"><summary>查看分阶段结果与错误</summary><pre>{{ JSON.stringify({ collection: job.collection_result, qlib_sync: job.sync_result, sync_error: job.sync_error, error: job.error, result: job.result || job.progress?.results }, null, 2) }}</pre></details>
      </article>
      <div class="dm-scheduler-settings"><h3>定时采集</h3><div v-if="schedulerError" class="error">{{ schedulerError }}</div><div class="dm-task-grid"><div class="dm-task-card" v-for="task in schedulerTasks" :key="task.key"><div class="dm-task-top"><span class="dm-task-name">{{ task.name }}</span><span class="dm-task-status" :class="task.running ? 'dm-task-status--on' : 'dm-task-status--off'">{{ task.running ? '● 运行中' : '○ 已停止' }}</span></div><div class="dm-task-info">{{ task.extra }}</div><div v-if="task.next" class="dm-task-next">下次/最近：{{ task.next }}</div><div class="dm-task-btns"><button v-if="!task.running" @click="toggleScheduler(task.key, true)" class="btn-sm btn-success">启动</button><button v-if="task.running" @click="toggleScheduler(task.key, false)" class="btn-sm btn-danger">停止</button><button @click="triggerScheduler" class="btn-sm btn-outline">立即触发</button></div></div></div></div>
    </section>
  </div>
  <Teleport to="body">
    <div v-if="releaseConfirmation" class="dm-confirm-backdrop" @click.self="cancelReleaseConfirmation">
      <section class="dm-confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="dm-confirm-title" tabindex="-1" @keydown.esc.stop.prevent="cancelReleaseConfirmation">
        <h2 id="dm-confirm-title">确认发布双源核验修复</h2>
        <p><strong>{{ releaseConfirmation.key }}</strong> 将按双源一致的候选覆盖现库行情冲突，并同步更新 Qlib。</p>
        <p>收盘价、成交量、成交额已逐日核验 <strong>{{ releaseConfirmation.matched }}/{{ releaseConfirmation.expected }}</strong> 行。数据库和 Qlib 均可回滚。</p>
        <div class="dm-confirm-actions">
          <button ref="releaseCancelButton" class="btn btn-outline" :disabled="historyShadowBusy" @click="cancelReleaseConfirmation">取消</button>
          <button class="btn btn-danger" :disabled="historyShadowBusy" @click="confirmHistoryRelease">{{ historyShadowBusy ? '发布中…' : '确认发布' }}</button>
        </div>
      </section>
    </div>
  </Teleport>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { createDataSnapshot } from '../api/strategyProfiles.js'
import dataApi from '../api/data'
import qlibApi from '../api/qlib'

const route = useRoute()
const router = useRouter()
const releaseConfirmation = ref(null)
const releaseCancelButton = ref(null)
const tabs = [
  { id: 'overview', label: '总览' },
  { id: 'maintenance', label: '日常维护' },
  { id: 'history', label: '历史修复' },
  { id: 'tasks', label: '任务与设置' },
]
const activeTab = ref(tabs.some(tab => tab.id === route.query.tab) ? route.query.tab : 'overview')
watch(() => route.query.tab, value => { activeTab.value = tabs.some(tab => tab.id === value) ? value : 'overview' })
watch(activeTab, value => { if (value === 'history') loadQuoteAudit() })
function setActiveTab(tab) {
  activeTab.value = tab
  router.replace({ query: { ...route.query, tab: tab === 'overview' ? undefined : tab } })
}

const historyGates = ref([
  { id: 'g0', step: 'G0', label: '年度历史池分母', note: '逐年核验；全池仍未认证', status: 'blocked' },
  { id: 'g1', step: 'G1', label: '影子采集与冲突核对', note: '双源采集已接入', status: 'ready' },
  { id: 'g2', step: 'G2', label: '可回滚分片发布', note: '逐片预检后发布', status: 'ready' },
  { id: 'g3', step: 'G3', label: 'Qlib / P1 / 实验准入', note: '等待上游', status: 'blocked' },
])
const historyPreflightLoading = ref(false)
const historyPreflight = ref(null)
const historyBaselineLoading = ref(false)
const historyBaseline = ref(null)
const historyDenominatorStart = ref('2023-01-01')
const historyDenominatorEnd = ref('2023-12-31')
const historyDenominatorBusy = ref(false)
const historyDenominatorBatch = ref(null)
const historyDenominatorId = ref(localStorage.getItem('data-center-g0-batch') || '')
const industryHistoryStart = ref('2018-12-24')
const industryHistoryEnd = ref((() => { const date = new Date(); date.setDate(date.getDate() - ((date.getDay() + 6) % 7)); return date.toISOString().slice(0, 10) })())
const industryHistoryBusy = ref(false)
const industryHistoryStatus = ref(null)
const industryHistoryJobId = ref(localStorage.getItem('data-center-industry-history-job') || '')
const industryHistoryMessage = ref('可从上次完成位置续跑；不会修改正式行业关系')
watch(industryHistoryJobId, value => {
  if (value) localStorage.setItem('data-center-industry-history-job', value)
  else localStorage.removeItem('data-center-industry-history-job')
})
watch(historyDenominatorId, value => {
  if (value.trim()) localStorage.setItem('data-center-g0-batch', value.trim())
  else localStorage.removeItem('data-center-g0-batch')
})
const historyDenominatorMessage = ref('建议先完成单年验证，再扩展历史区间')
const historyShadowCodes = ref('')
const historyShadowStart = ref('2023-01-01')
const historyShadowEnd = ref('2023-12-31')
const historyShadowBusy = ref(false)
const historyShadowBatch = ref(null)
const historyShadowLoadId = ref(localStorage.getItem('data-center-history-shadow-batch') || '')
const historyShadowMessage = ref('单批上限 60 个股票年度分片')
const historyShadowConflicts = reactive({})
const historicalReleases = reactive({})
const jobs = ref([])
const jobTimers = new Map()
const jobStorageKey = 'data-center-jobs-v1'

const overview = ref(null)
const overviewLoading = ref(false)
const overviewError = ref('')
const historyQuality = ref(null)
const historyQualityError = ref('')
const quoteAuditFilters = reactive({ code: '', task_id: '', start: '', end: '' })
const quoteAudit = ref({ items: [], total: 0, page: 1, page_size: 50 })
const quoteAuditLoading = ref(false)
const quoteAuditError = ref('')
const snapshotDate = ref(new Date().toISOString().slice(0, 10))
const snapshot = ref(null)
const snapshotLoading = ref(false)
const snapshotError = ref('')

async function checkSnapshot() {
  snapshotLoading.value = true; snapshotError.value = ''
  try { const result = await createDataSnapshot({ as_of: snapshotDate.value, min_history_days: 500 }); snapshot.value = result?.data ?? result }
  catch (e) { snapshotError.value = e.message }
  finally { snapshotLoading.value = false }
}
async function loadQuoteAudit(page = 1) {
  quoteAuditLoading.value = true
  quoteAuditError.value = ''
  try {
    const params = { page, page_size: quoteAudit.value.page_size }
    for (const [key, value] of Object.entries(quoteAuditFilters)) if (value) params[key] = value
    const response = await dataApi.getQuoteAudit(params)
    quoteAudit.value = response.data ?? response
  } catch (e) { quoteAuditError.value = e.message || '读取行情写入审计失败' }
  finally { quoteAuditLoading.value = false }
}
function searchQuoteAudit() { loadQuoteAudit(1) }
function clearQuoteAuditFilters() {
  Object.assign(quoteAuditFilters, { code: '', task_id: '', start: '', end: '' })
  loadQuoteAudit(1)
}
function changeQuoteAuditPage(delta) { loadQuoteAudit(quoteAudit.value.page + delta) }
function copyAuditId(value) { if (value) navigator.clipboard.writeText(value) }
function copySnapshot() { if (snapshot.value?.snapshot_id) navigator.clipboard.writeText(snapshot.value.snapshot_id) }
const expandedCategories = reactive(new Set())
const repairing = reactive(new Set())
const repairAllRunning = ref(false)
const repairMessage = ref('')
const repairResults = ref([])
const repairPreview = ref(null)
const repairPreviewLoading = ref(false)
const repairBusy = computed(() => repairAllRunning.value || repairing.size > 0)

const overallHealth = computed(() => {
  if (!overview.value) return { health_pct: 0, fresh: 0, stale: 0, incomplete: 0, missing: 0, error: 0, total: 0 }
  return overview.value.overall_health
})

const healthColor = computed(() => {
  const pct = overallHealth.value.recent_update_rate ?? overallHealth.value.health_pct
  if (pct >= 80) return ''
  if (pct >= 50) return 'dm-bar-val--warn'
  return 'dm-bar-val--bad'
})

const healthDot = computed(() => {
  const pct = overallHealth.value.recent_update_rate ?? overallHealth.value.health_pct
  if (pct >= 80) return 'dm-dot--ok'
  if (pct >= 50) return 'dm-dot--warn'
  return 'dm-dot--bad'
})

const warnCount = computed(() => {
  if (!overview.value) return 0
  let c = 0
  for (const cat of overview.value.categories) {
    for (const t of cat.tables) {
      if (t.status !== 'fresh' && t.repairable) c++
    }
  }
  return c
})

function statusLabel(s) {
  const map = { fresh: '正常', stale: '过期', missing: '缺失', incomplete: '不完整', error: '异常', unavailable: '来源已停发', restart_required: '需重启后验证' }
  return map[s] || s
}

function healthLabel(h) {
  const map = { healthy: '正常', warning: '警告', error: '异常' }
  return map[h] || h
}

function toggleCategory(key) {
  if (expandedCategories.has(key)) expandedCategories.delete(key)
  else expandedCategories.add(key)
}

async function refreshOverview() {
  overviewLoading.value = true
  overviewError.value = ''
  try {
    const r = await dataApi.getOverview(true)
    overview.value = r.data
  } catch (e) {
    overviewError.value = e.message || '加载失败'
  } finally {
    overviewLoading.value = false
  }
  try {
    const r = await dataApi.getHistoryQuality()
    historyQuality.value = r.data ?? r
    historyQualityError.value = ''
  } catch (e) {
    historyQuality.value = null
    historyQualityError.value = e.message || '历史质量检查失败'
  }
}

async function runHistoryPreflight() {
  historyPreflightLoading.value = true
  try {
    const response = await dataApi.historyPreflight()
    historyPreflight.value = response.data ?? response
  } catch (e) {
    historyPreflight.value = { status: 'blocked', summary: e.message || '预检请求失败', blockers: ['无法从服务端读取历史质量状态'], writes_data: false }
  } finally {
    historyPreflightLoading.value = false
  }
}

async function loadIndustryHistory() {
  industryHistoryBusy.value = true
  try {
    const response = await dataApi.getIndustryHistorySnapshots({ start_date: industryHistoryStart.value, end_date: industryHistoryEnd.value })
    industryHistoryStatus.value = response.data ?? response
  } catch (e) {
    industryHistoryMessage.value = e.message || '读取历史行业快照状态失败'
  } finally {
    industryHistoryBusy.value = false
  }
}

async function startIndustryHistory() {
  industryHistoryBusy.value = true
  try {
    const response = await dataApi.startIndustryHistorySnapshots({ start_date: industryHistoryStart.value, end_date: industryHistoryEnd.value })
    const job = response.data ?? response
    industryHistoryMessage.value = job.message || `已提交 ${job.requested_snapshots || 0} 个周快照采集`
    if (job.job_id) {
      industryHistoryJobId.value = job.job_id
      registerJob(job.job_id, 'task', '历史行业快照', 'history-industry')
    }
  } catch (e) {
    industryHistoryMessage.value = e.response?.data?.detail || e.message || '启动历史行业采集失败'
  } finally {
    industryHistoryBusy.value = false
  }
}

async function cancelIndustryHistory() {
  if (!industryHistoryJobId.value) return
  industryHistoryBusy.value = true
  try {
    const response = await dataApi.cancelIndustryHistorySnapshots(industryHistoryJobId.value)
    const result = response.data ?? response
    industryHistoryMessage.value = result.status === 'cancel_requested' ? '已请求停止；当前周快照完成后保存检查点' : '当前任务已结束或任务 ID 不存在'
  } catch (e) {
    industryHistoryMessage.value = e.message || '请求停止失败'
  } finally {
    industryHistoryBusy.value = false
  }
}

async function freezeHistoryBaseline() {
  historyBaselineLoading.value = true
  try {
    const response = await dataApi.createHistoryBaseline()
    historyBaseline.value = response.data ?? response
  } catch (e) {
    historyBaseline.value = { status: 'error', denominator_blockers: [e.message || '基线冻结失败'] }
  } finally {
    historyBaselineLoading.value = false
  }
}

async function startHistoryDenominator() {
  historyDenominatorBusy.value = true
  try {
    const response = await dataApi.startHistoryDenominatorShadow({ start: historyDenominatorStart.value, end: historyDenominatorEnd.value })
    const job = response.data ?? response
    if (job.status === 'busy') {
      historyDenominatorMessage.value = job.message || '已有 G0 分母任务运行中'
      if (job.batch_id) await refreshHistoryDenominator(job.batch_id)
      return
    }
    historyDenominatorBatch.value = { batch_id: job.batch_id, job_id: job.job_id, status: job.status, date_range: { start: historyDenominatorStart.value, end: historyDenominatorEnd.value }, basis: 'effective_date_reconstruction', strict_pit_status_verified: false }
    localStorage.setItem('data-center-g0-batch', job.batch_id)
    historyDenominatorId.value = job.batch_id
    historyDenominatorMessage.value = 'G0 分母影子任务已提交；正在采集身份、日历和停牌证据'
    if (job.job_id) registerJob(job.job_id, 'task', `G0 分母核算 ${job.batch_id}`, 'history-denominator')
    await refreshHistoryDenominator(job.batch_id)
  } catch (e) {
    historyDenominatorMessage.value = e.message || 'G0 分母影子任务提交失败'
  } finally {
    historyDenominatorBusy.value = false
  }
}

async function refreshHistoryDenominator(batchId = historyDenominatorBatch.value?.batch_id) {
  if (!batchId) return
  try {
    const response = await dataApi.getHistoryDenominatorShadow(batchId)
    historyDenominatorBatch.value = response.data ?? response
    historyDenominatorId.value = batchId
    historyDenominatorMessage.value = `G0 批次状态：${historyDenominatorBatch.value.status}`
  } catch (e) {
    historyDenominatorMessage.value = e.message || '读取 G0 分母批次失败'
  }
}

async function cancelHistoryDenominator() {
  const batchId = historyDenominatorBatch.value?.batch_id
  if (!batchId) return
  historyDenominatorBusy.value = true
  try {
    const response = await dataApi.cancelHistoryDenominatorShadow(batchId)
    const result = response.data ?? response
    historyDenominatorMessage.value = result.status === 'cancel_requested' ? '已请求停止；当前数据源请求或年度核算结束后会保存部分批次' : '当前没有运行中的 G0 任务'
  } catch (e) {
    historyDenominatorMessage.value = e.message || '停止请求失败'
  } finally {
    historyDenominatorBusy.value = false
  }
}

async function prepareHistoryShadow() {
  historyShadowBusy.value = true
  try {
    const codes = historyShadowCodes.value.split(/[\s,，;；]+/).map(code => code.trim()).filter(Boolean)
    const response = await dataApi.previewHistoryShadowBatch({ codes, start: historyShadowStart.value, end: historyShadowEnd.value, denominator_batch_id: historyDenominatorId.value.trim() })
    historyShadowBatch.value = response.data ?? response
    historyShadowLoadId.value = historyShadowBatch.value.batch_id
    localStorage.setItem('data-center-history-shadow-batch', historyShadowBatch.value.batch_id)
    historyShadowMessage.value = '清单已创建；确认范围后再启动双源采集'
  } catch (e) {
    historyShadowMessage.value = e.message || '生成影子批次失败'
  } finally {
    historyShadowBusy.value = false
  }
}

async function loadHistoryShadowBatch() {
  const batchId = historyShadowLoadId.value.trim()
  if (!/^[a-f0-9]{12}$/i.test(batchId)) {
    historyShadowMessage.value = '批次 ID 必须是 12 位十六进制字符'
    return
  }
  historyShadowBusy.value = true
  try {
    const response = await dataApi.getHistoryShadowBatch(batchId)
    const batch = response.data ?? response
    if (!batch?.batch_id || batch.status === 'not_found') throw new Error('没有找到该历史影子批次')
    historyShadowBatch.value = batch
    historyShadowLoadId.value = batch.batch_id
    localStorage.setItem('data-center-history-shadow-batch', batch.batch_id)
    historyShadowMessage.value = `已载入批次 ${batch.batch_id}；检查分片状态后可续跑未完成项`
  } catch (e) {
    historyShadowMessage.value = e.message || '载入历史影子批次失败'
  } finally {
    historyShadowBusy.value = false
  }
}

function canPrepareVerifiedConflictRepair(item) {
  const rawConflicts = Number(item.comparison_status_counts?.raw_close_conflict || 0)
  const databaseConflicts = Number(item.database_price_status_counts?.database_price_conflict || 0)
  return rawConflicts === 0 && (item.status === 'collected_review' || (item.status === 'quarantined_conflict' && databaseConflicts > 0))
}

async function prepareHistoryRelease(item, replaceConflicts = false) {
  const key = `${item.code}-${item.year}`
  historyShadowBusy.value = true
  try {
    const response = await dataApi.prepareHistoryRelease(historyShadowBatch.value.batch_id, item.code, item.year, historyDenominatorId.value.trim(), replaceConflicts)
    historicalReleases[key] = response.data ?? response
    historyShadowMessage.value = historicalReleases[key].status === 'ready'
      ? replaceConflicts ? `${key} 双源冲突修复预检通过，检查数量证据后发布` : `${key} 通过 G2 发布预检`
      : `${key} 未通过 G2 预检，查看阻断项`
  } catch (e) {
    historyShadowMessage.value = e.response?.data?.detail || e.message || `G2 预检失败：${key}`
  } finally {
    historyShadowBusy.value = false
  }
}

async function publishHistoryRelease(item) {
  const key = `${item.code}-${item.year}`
  const release = historicalReleases[key]
  const releaseId = release?.release_id
  if (!releaseId) return
  if (release.source_verified_conflict_replacement) {
    const matched = release.source_quantity_evidence?.matched_rows ?? 0
    const expected = release.source_quantity_evidence?.candidate_rows ?? 0
    releaseConfirmation.value = { key, releaseId, matched, expected }
    await nextTick()
    releaseCancelButton.value?.focus()
    return
  }
  await executeHistoryRelease(key, releaseId)
}

function cancelReleaseConfirmation() {
  if (historyShadowBusy.value) return
  releaseConfirmation.value = null
}

async function confirmHistoryRelease() {
  const pending = releaseConfirmation.value
  if (!pending || historyShadowBusy.value) return
  releaseConfirmation.value = null
  await executeHistoryRelease(pending.key, pending.releaseId)
}

async function executeHistoryRelease(key, releaseId) {
  historyShadowBusy.value = true
  try {
    const response = await dataApi.publishHistoryRelease(releaseId)
    historicalReleases[key] = response.data ?? response
    historyShadowMessage.value = `${key} 已发布；请核对数据库、Qlib 和审计记录`
    const status = await dataApi.getHistoryRelease(releaseId)
    historicalReleases[key] = status.data ?? status
  } catch (e) {
    historyShadowMessage.value = e.response?.data?.detail || e.message || `发布失败：${key}`
  } finally {
    historyShadowBusy.value = false
  }
}

async function rollbackHistoryRelease(item) {
  const key = `${item.code}-${item.year}`
  const releaseId = historicalReleases[key]?.release_id
  if (!releaseId) return
  historyShadowBusy.value = true
  try {
    const response = await dataApi.rollbackHistoryRelease(releaseId)
    historicalReleases[key] = response.data ?? response
    historyShadowMessage.value = `${key} 已回滚；请核对数据库、Qlib 和审计记录`
  } catch (e) {
    historyShadowMessage.value = e.response?.data?.detail || e.message || `回滚失败：${key}`
  } finally {
    historyShadowBusy.value = false
  }
}

async function startHistoryShadow() {
  if (!historyShadowBatch.value?.batch_id) return
  historyShadowBusy.value = true
  try {
    const response = await dataApi.startHistoryShadowBatch(historyShadowBatch.value.batch_id)
    historyShadowLoadId.value = historyShadowBatch.value.batch_id
    localStorage.setItem('data-center-history-shadow-batch', historyShadowBatch.value.batch_id)
    const job = response.data ?? response
    if (job.status === 'busy') {
      historyShadowMessage.value = job.message || '已有影子采集任务运行中'
      return
    }
    historyShadowMessage.value = job.message || `采集任务状态：${job.status}`
    if (job.job_id) registerJob(job.job_id, 'task', `历史影子采集 ${job.batch_id}`, 'history-shadow')
    const manifestResponse = await dataApi.getHistoryShadowBatch(historyShadowBatch.value.batch_id)
    historyShadowBatch.value = manifestResponse.data ?? manifestResponse
  } catch (e) {
    historyShadowMessage.value = e.message || '启动影子批次失败'
  } finally {
    historyShadowBusy.value = false
  }
}

async function refreshHistoryShadow() {
  const batchId = historyShadowBatch.value?.batch_id || localStorage.getItem('data-center-history-shadow-batch')
  if (!batchId) return
  historyShadowBusy.value = true
  try {
    const response = await dataApi.getHistoryShadowBatch(batchId)
    historyShadowBatch.value = response.data ?? response
    historyShadowMessage.value = `影子批次状态：${historyShadowBatch.value.status}`
  } catch (e) {
    historyShadowMessage.value = e.message || '刷新影子批次失败'
  } finally {
    historyShadowBusy.value = false
  }
}

async function loadHistoryShadowConflicts(item) {
  const key = `${item.code}-${item.year}`
  if (historyShadowConflicts[key]) {
    delete historyShadowConflicts[key]
    return
  }
  try {
    const denominatorBatchId = historyDenominatorBatch.value?.batch_id || localStorage.getItem('data-center-g0-batch')
    const response = await dataApi.getHistoryShadowConflicts(historyShadowBatch.value.batch_id, item.code, item.year, 2000, denominatorBatchId)
    historyShadowConflicts[key] = response.data ?? response
  } catch (e) {
    historyShadowMessage.value = e.message || `读取 ${key} 差异记录失败`
  }
}

async function cancelHistoryShadow() {
  if (!historyShadowBatch.value?.batch_id) return
  historyShadowBusy.value = true
  try {
    const response = await dataApi.cancelHistoryShadowBatch(historyShadowBatch.value.batch_id)
    const result = response.data ?? response
    historyShadowMessage.value = result.status === 'cancel_requested' ? '已请求停止；当前股票年度分片结束后会保存检查点' : '当前没有运行中的影子任务'
  } catch (e) {
    historyShadowMessage.value = e.message || '停止请求失败'
  } finally {
    historyShadowBusy.value = false
  }
}

async function doRepair(table) {
  repairing.add(table)
  repairMessage.value = `正在提交 ${table} 修复任务...`
  repairResults.value = []
  try {
    const r = await dataApi.repairTable(table)
    const job = r.data || r
    if (!job.job_id) throw new Error(job.message || '未返回修复任务ID')
    registerJob(job.job_id, 'task', `逐表修复 ${table}`, table)
  } catch (e) {
    repairMessage.value = `修复失败: ${e.message || '未知错误'}`
    repairing.delete(table)
  }
}

async function previewRepairs() {
  repairPreviewLoading.value = true
  try {
    const response = await dataApi.repairPreview()
    repairPreview.value = response.data ?? response
    repairMessage.value = repairPreview.value.target_count ? '请核对下面的近期修复范围后再提交' : '近期数据无需修复'
  } catch (e) {
    repairMessage.value = `生成修复清单失败: ${e.message || '未知错误'}`
  } finally {
    repairPreviewLoading.value = false
  }
}

async function startRepairAll() {
  if (!repairPreview.value?.preview_id) return
  repairAllRunning.value = true
  repairResults.value = []
  repairMessage.value = '正在提交预览中的修复清单...'
  try {
    const r = await dataApi.repairAll(repairPreview.value.preview_id)
    const job = r.data ?? r
    if (!job.job_id) throw new Error(job.message || '未返回修复任务ID')
    registerJob(job.job_id, 'task', '修复近期异常')
    repairPreview.value = null
  } catch (e) {
    repairMessage.value = `修复失败: ${e.message || '未知错误'}`
    repairAllRunning.value = false
  }
}

function repairResultLabel(status) {
  return { completed: '已修复', partial: '仍不完整', error: '失败' }[status] || status
}

// ── 采集操作 ──
const syncing = ref(false)
const syncMessage = ref('')
const indexSyncing = ref(false)
const indexSyncMsg = ref('')
const incrPool = ref('all')
const incrLookback = ref(5)
const incrThreads = ref(32)
const incrRunning = ref(false)
const incrMsg = ref('')
const incrementalJob = computed(() => jobs.value.find(job => job.key === 'incr') || null)
const incrementalResult = computed(() => incrementalJob.value?.collection_result || null)
const fullRunning = ref(false)
const fullMsg = ref('')
const fundPool = ref('all')
const fundRunning = ref(false)
const fundMsg = ref('')
const qlibCheck = ref([])
const contextRunning = reactive({ identity: false, events: false })
const contextMsg = ref('')

async function collectContext(kind) {
  contextRunning[kind] = true
  contextMsg.value = kind === 'identity' ? '正在提交股票身份与指数关系更新...' : '正在提交分红事件回补...'
  try {
    const response = await dataApi.collectContext(kind)
    const jobId = response?.data?.job_id || response?.job_id
    if (!jobId) throw new Error(response?.data?.message || '未返回任务ID')
    registerJob(jobId, 'task', kind === 'identity' ? '采集股票身份与关系' : '回补分红事件', kind)
  } catch (e) {
    contextRunning[kind] = false
    contextMsg.value = `采集失败: ${e.response?.data?.detail || e.message}`
  }
}

onBeforeUnmount(() => {
  for (const timer of jobTimers.values()) clearTimeout(timer)
  jobTimers.clear()
})

function persistJobs() {
  try { localStorage.setItem(jobStorageKey, JSON.stringify(jobs.value.slice(0, 100))) } catch { /* storage may be unavailable */ }
}

function registerJob(jobId, source, title, key = null) {
  const existing = jobs.value.find(job => job.job_id === jobId)
  const job = { ...(existing || {}), job_id: jobId, source, title, key, status: 'pending', created_at: Date.now() / 1000 }
  jobs.value = [job, ...jobs.value.filter(item => item.job_id !== jobId)].slice(0, 100)
  persistJobs()
  pollJob(jobId)
}

async function pollJob(jobId) {
  const current = jobs.value.find(job => job.job_id === jobId)
  if (!current) return
  try {
    const response = current.source === 'qlib' ? await qlibApi.getSyncStatus(jobId) : await dataApi.getRepairStatus(jobId)
    const serverJob = response?.data ?? response
    const status = serverJob.status || serverJob.state || 'running'
    const updated = { ...current, ...serverJob, job_id: jobId, source: current.source, title: current.title, key: current.key, status }
    jobs.value = [updated, ...jobs.value.filter(item => item.job_id !== jobId)]
    persistJobs()
    const message = serverJob.progress?.message || serverJob.message || ''
    const terminal = ['completed', 'partial', 'failed', 'error', 'cancelled', 'interrupted', 'stalled', 'unavailable'].includes(status)
    if (current.key === 'history-shadow') {
      historyShadowMessage.value = message || `影子采集任务：${status}`
      if (terminal) {
        const batchId = current.title?.match(/[a-f0-9]{12}$/)?.[0]
        if (batchId) {
          const manifestResponse = await dataApi.getHistoryShadowBatch(batchId)
          historyShadowBatch.value = manifestResponse.data ?? manifestResponse
          historyShadowMessage.value = `批次状态：${historyShadowBatch.value.status}`
        }
      }
    }
    if (current.key === 'history-denominator') {
      historyDenominatorMessage.value = message || `G0 分母核算任务：${status}`
      if (terminal) {
        const batchId = current.title?.match(/[a-f0-9]{12}$/)?.[0]
        if (batchId) await refreshHistoryDenominator(batchId)
      }
    }
    if (current.key === 'history-industry') {
      industryHistoryMessage.value = message || `历史行业采集任务：${status}`
      if (terminal) {
        industryHistoryJobId.value = ''
      }
      await loadIndustryHistory()
    }
    if (current.key === 'identity' || current.key === 'events') {
      contextMsg.value = message || (status === 'completed' ? '采集完成' : '数据采集中…')
      if (terminal) contextRunning[current.key] = false
    } else if (current.title?.startsWith('逐表修复')) {
      if (terminal) repairing.delete(current.key)
      repairMessage.value = message || `修复任务状态：${status}`
      repairResults.value = serverJob.progress?.results || serverJob.result?.results || []
    } else if (current.key === 'repair-all' || current.title === '修复近期异常') {
      repairAllRunning.value = !['completed', 'partial', 'failed', 'error', 'cancelled', 'interrupted', 'stalled'].includes(status)
      repairMessage.value = message || `近期修复状态：${status}`
      repairResults.value = serverJob.progress?.results || serverJob.result?.results || []
    }
    if (current.key === 'incr') { incrRunning.value = status === 'pending' || status === 'running'; incrMsg.value = message || `增量更新：${status}` }
    if (current.key === 'full') { fullRunning.value = false; fullMsg.value = message || `全量任务：${status}` }
    if (current.key === 'sync') { syncing.value = status === 'pending' || status === 'running'; if (!syncing.value) syncMessage.value = message || `Qlib同步：${status}` }
    if (current.key === 'benchmarks') { indexSyncing.value = status === 'pending' || status === 'running'; indexSyncMsg.value = message || `基准指数同步：${status}` }
    if (current.key === 'fundamentals') { fundRunning.value = status === 'pending' || status === 'running'; fundMsg.value = message || `基本面采集：${status}` }
    if (terminal) {
      jobTimers.delete(jobId)
      if (['completed', 'partial'].includes(status)) await refreshOverview()
      return
    }
  } catch (error) {
    const updated = { ...current, status: 'unavailable', message: `暂时无法读取任务状态：${error.message || '请求失败'}` }
    jobs.value = [updated, ...jobs.value.filter(item => item.job_id !== jobId)]
    persistJobs()
    if (current.key === 'identity' || current.key === 'events') contextRunning[current.key] = false
    if (current.key === 'repair-all' || current.title === '修复近期异常') repairAllRunning.value = false
    if (current.key && current.key !== 'repair-all') repairing.delete(current.key)
    if (current.key === 'incr') incrRunning.value = false
    if (current.key === 'sync') syncing.value = false
    if (current.key === 'benchmarks') indexSyncing.value = false
    if (current.key === 'fundamentals') fundRunning.value = false
    jobTimers.delete(jobId)
    return
  }
  jobTimers.set(jobId, setTimeout(() => pollJob(jobId), 2500))
}

async function loadJobs() {
  let stored = []
  try { stored = JSON.parse(localStorage.getItem(jobStorageKey) || '[]') } catch { stored = [] }
  try {
    const response = await dataApi.getJobs(50)
    const serverJobs = response?.data?.jobs || response?.data || response
    const merged = new Map(stored.filter(job => job?.job_id).map(job => [job.job_id, job]))
    for (const serverJob of Array.isArray(serverJobs) ? serverJobs : []) {
      const saved = merged.get(serverJob.job_id) || {}
      const source = serverJob.source || (jobIdIsTask(serverJob.job_id) ? 'task' : 'qlib')
      merged.set(serverJob.job_id, { ...saved, ...serverJob, source, title: saved.title || serverJob.title || serverJob.name })
    }
    jobs.value = [...merged.values()].sort((a, b) => (b.created_at || 0) - (a.created_at || 0)).slice(0, 100)
  } catch {
    jobs.value = stored
  }
  for (const job of jobs.value) {
    if (['pending', 'running', 'unavailable', 'cancel_requested'].includes(job.status) && !jobTimers.has(job.job_id)) pollJob(job.job_id)
  }
  persistJobs()
}

function jobIdIsTask(jobId) {
  return ['context', 'repair', 'fund'].some(prefix => jobId?.startsWith(prefix))
}

function jobStatusLabel(status) {
  return ({ pending: '排队中', running: '运行中', completed: '完成', partial: '部分完成', failed: '失败', error: '失败', cancelled: '已取消', interrupted: '进程中断', stalled: '疑似停滞', unavailable: '暂不可读取', cancel_requested: '取消处理中' })[status] || status || '未知'
}

function jobStatusClass(status) {
  if (status === 'completed') return 'fresh'
  if (['failed', 'error', 'stalled', 'interrupted'].includes(status)) return 'error'
  if (['partial', 'unavailable', 'cancel_requested'].includes(status)) return 'incomplete'
  return 'stale'
}

async function doIncremental() {
  incrRunning.value = true; incrMsg.value = '启动增量更新...'
  try {
    const r = await qlibApi.collectIncremental(incrPool.value, incrLookback.value, incrThreads.value)
    const job = r.data || r
    if (job.status === 'completed') { incrRunning.value = false; incrMsg.value = job.message || '更新完成'; refreshOverview() }
    else if (job.job_id) { registerJob(job.job_id, 'qlib', 'Qlib 增量采集', 'incr') }
    else { incrRunning.value = false; incrMsg.value = job.message || '已提交' }
  } catch (e) { incrRunning.value = false; incrMsg.value = '更新失败: ' + (e.message || '未知错误') }
}

async function retryFailedIncremental() {
  const parent = incrementalJob.value
  if (!parent?.job_id || !Array.isArray(incrementalResult.value?.failed_codes) || !incrementalResult.value.failed_codes.length) return
  incrRunning.value = true
  incrMsg.value = '正在提交仅针对失败证券的重试...'
  try {
    const response = await qlibApi.retryFailedIncremental(parent.job_id, incrThreads.value)
    const job = response?.data ?? response
    if (!job.job_id) throw new Error(job.message || '未返回重试任务ID')
    registerJob(job.job_id, 'qlib', 'Qlib增量采集失败重试', 'incr')
  } catch (e) {
    incrRunning.value = false
    incrMsg.value = `失败证券重试未启动：${e.response?.data?.detail || e.message || '未知错误'}`
  }
}

async function doFullCollect() {
  fullRunning.value = false
  fullMsg.value = '全量下载已暂停：当前名单筛选不能安全重建历史股票池。'
}

async function doSync(full) {
  if (full) {
    syncMessage.value = '全量重建已暂停：候选目录、逐日截面验证和安全发布尚未接入。'
    return
  }
  syncing.value = true
  syncMessage.value = '正在提交 Qlib 增量同步…'
  try {
    const r = await qlibApi.syncData(full)
    const job = r.data || r
    if (job.job_id) registerJob(job.job_id, 'qlib', 'Qlib 增量同步', 'sync')
    else { syncing.value = false; syncMessage.value = job.message || (job.status === 'disabled' ? '全量同步已暂停' : '服务端未返回任务ID') }
  } catch (e) { syncing.value = false; syncMessage.value = `增量同步失败：${e.message || '未知错误'}` }
}

async function doSyncBenchmarks() {
  indexSyncing.value = true
  indexSyncMsg.value = '正在提交指数同步...'
  try {
    const r = await qlibApi.syncBenchmarks()
    const job = r.data || r
    if (job.job_id) {
      registerJob(job.job_id, 'qlib', '同步基准指数', 'benchmarks')
    } else {
      indexSyncing.value = false
      indexSyncMsg.value = job.message || '基准指数同步已提交。'
      refreshOverview()
    }
  } catch (e) {
    indexSyncing.value = false
    indexSyncMsg.value = '基准指数同步失败: ' + (e.message || '未知错误')
  }
}

async function doValidateQlib() {
  try {
    const r = await qlibApi.validateQlibData()
    qlibCheck.value = r.data?.checks || r.checks || []
  } catch (e) { qlibCheck.value = [{ item: '验证', msg: '失败: ' + (e.message || '未知'), ok: false }] }
}

async function doFundamentals() {
  fundRunning.value = true; fundMsg.value = '采集中...'
  try {
    const r = await qlibApi.collectFundamentals(fundPool.value)
    const job = r.data ?? r
    if (job.job_id) registerJob(job.job_id, 'task', '采集基本面数据', 'fundamentals')
    else { fundRunning.value = false; fundMsg.value = job.message || '服务端未返回任务ID' }
  } catch (e) { fundRunning.value = false; fundMsg.value = '采集失败: ' + (e.message || '未知错误') }
}

// ── 定时采集 ──
const schedulerStatuses = ref({})
const schedulerError = ref('')

const schedulerTasks = computed(() => {
  const s = schedulerStatuses.value
  if (!s || !Object.keys(s).length) return [{ key: 'collection', name: '每日行情采集', running: false, extra: '状态未知' }]
  return [{ key: 'collection', name: '每日行情采集', running: !!s.running,
    extra: `${s.data_source || 'tencent'} · ${s.stock_pool || 'all'} · ${s.thread_count || 32}线程`,
    next: s.next_collection || s.last_collection || null }]
})

async function loadSchedulerStatuses() {
  try { const r = await qlibApi.getSchedulerStatus('collection'); schedulerStatuses.value = r.data || r || {}; schedulerError.value = '' }
  catch (e) { schedulerError.value = `读取定时采集状态失败：${e.message || '请求异常'}` }
}

async function toggleScheduler(key, start) {
  try { if (start) await qlibApi.startScheduler('collection'); else await qlibApi.stopScheduler('collection'); await loadSchedulerStatuses() }
  catch (e) { schedulerError.value = `定时采集操作失败：${e.message || '请求异常'}` }
}

async function triggerScheduler() {
  try { await qlibApi.triggerScheduler('collection'); await loadSchedulerStatuses() }
  catch (e) { schedulerError.value = `触发定时采集失败：${e.message || '请求异常'}` }
}

onMounted(() => {
  refreshOverview(); loadSchedulerStatuses(); loadJobs()
  loadIndustryHistory()
  if (activeTab.value === 'history') loadQuoteAudit()
  const batchId = localStorage.getItem('data-center-g0-batch')
  if (batchId) refreshHistoryDenominator(batchId)
  const shadowBatchId = localStorage.getItem('data-center-history-shadow-batch')
  if (shadowBatchId) {
    dataApi.getHistoryShadowBatch(shadowBatchId).then(response => { historyShadowBatch.value = response.data ?? response }).catch(() => {})
  }
})
</script>

<style scoped>
.dm-root {
  max-width: var(--content-max-width);
  margin: 0 auto;
  padding: 0 var(--content-padding) var(--space-8);
}

.dm-subtitle {
  font-size: var(--fs-sm);
  color: var(--text-muted);
  margin-top: 2px;
}
.dm-tabs { display:flex; gap:6px; overflow-x:auto; border-bottom:1px solid var(--border); margin:10px 0 16px; }
.dm-tab { flex:0 0 auto; min-height:42px; padding:8px 14px; border:0; border-bottom:2px solid transparent; background:transparent; color:var(--text-muted); cursor:pointer; }
.dm-tab--active { color:var(--primary); border-bottom-color:var(--primary); font-weight:600; }
.dm-state-summary { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:10px 0 16px; }
.dm-state-summary article { display:grid; gap:5px; min-width:0; padding:12px; border:1px solid var(--border); border-radius:var(--radius); background:var(--card-bg); }
.dm-state-summary article span,.dm-state-summary article small { color:var(--text-muted); font-size:var(--fs-xs); }
.dm-state-summary article strong { font-size:var(--fs-lg); }
.dm-tab:focus-visible,.dm-cat-head:focus-visible { outline:2px solid var(--primary); outline-offset:2px; }
.dm-preview-card,.dm-history-workflow,.dm-jobs { padding:16px; margin:12px 0; border:1px solid var(--border); border-radius:var(--radius); background:var(--card-bg); }
.dm-audit-card { padding:16px; margin:12px 0; border:1px solid var(--border); border-radius:var(--radius); background:var(--card-bg); }
.dm-audit-head { display:flex; justify-content:space-between; align-items:flex-start; gap:12px; margin-bottom:14px; }
.dm-audit-head > div { display:grid; gap:4px; }
.dm-audit-head span,.dm-audit-summary { color:var(--text-muted); font-size:var(--fs-sm); }
.dm-audit-filters { display:grid; grid-template-columns:repeat(4,minmax(140px,1fr)) auto auto; align-items:end; gap:10px; margin-bottom:12px; }
.dm-audit-filters label { display:grid; gap:5px; color:var(--text-secondary); font-size:var(--fs-sm); }
.dm-audit-filters input { width:100%; min-height:38px; padding:6px 9px; border:1px solid var(--border); border-radius:var(--radius-sm); color:var(--text); background:var(--input-bg,var(--card-bg)); }
.dm-audit-table-wrap { overflow-x:auto; margin-top:8px; }
.dm-audit-table { width:100%; min-width:1040px; border-collapse:collapse; font-size:var(--fs-sm); }
.dm-audit-table th,.dm-audit-table td { padding:9px 10px; border-bottom:1px solid var(--border); text-align:left; vertical-align:top; white-space:nowrap; }
.dm-audit-table th { color:var(--text-muted); font-weight:600; }
.dm-audit-table td small { display:block; margin-top:3px; color:var(--text-muted); }
.dm-audit-table code { color:var(--text-secondary); }
.dm-audit-id { max-width:150px; overflow:hidden; border:0; padding:0; background:transparent; color:var(--primary); text-overflow:ellipsis; white-space:nowrap; cursor:pointer; }
.dm-audit-action { display:inline-flex; padding:3px 7px; border-radius:var(--radius-sm); background:var(--surface-muted,var(--bg-secondary)); color:var(--text-secondary); }
.dm-audit-empty { padding:24px !important; text-align:center !important; color:var(--text-muted); }
.dm-audit-pager { display:flex; justify-content:flex-end; gap:8px; margin-top:12px; }
.dm-preview-head,.dm-jobs-head { display:flex; justify-content:space-between; align-items:flex-start; gap:12px; margin-bottom:12px; }
.dm-preview-head span,.dm-jobs-head span { color:var(--text-muted); font-size:var(--fs-sm); }
.dm-preview-item,.dm-job-card { display:grid; grid-template-columns:minmax(150px,1fr) auto; gap:5px 12px; padding:12px 0; border-top:1px solid var(--border); }
.dm-preview-item code,.dm-job-main code { color:var(--text-muted); overflow-wrap:anywhere; }
.dm-preview-item span,.dm-preview-item small { grid-column:1/-1; color:var(--text-muted); }
.dm-preview-card > .btn { margin-top:12px; }
.dm-history-workflow > div:first-child { display:grid; gap:4px; }
.dm-history-workflow > div:first-child span { color:var(--text-muted); font-size:var(--fs-sm); }
.dm-history-gates { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; margin:14px 0; }
.dm-gate { display:grid; gap:4px; padding:12px; border:1px solid var(--border); border-radius:var(--radius-sm); }
.dm-gate b { color:var(--primary); }.dm-gate small { color:var(--text-muted); }
.dm-gate--blocked { border-color:color-mix(in srgb,var(--warning) 45%,var(--border)); }
.dm-preflight-result { display:grid; gap:6px; padding:12px; margin:12px 0; border-radius:var(--radius-sm); background:var(--bg-secondary); }
.dm-preflight-result ul { margin:4px 0 0; padding-left:20px; }
.dm-preflight--blocked { border-left:3px solid var(--warning); }
.dm-shadow-batch { display:grid; gap:10px; margin:14px 0; padding:14px; border:1px solid var(--border); border-radius:var(--radius-sm); background:var(--bg-secondary); }
.dm-shadow-batch p { margin:0; color:var(--text-muted); font-size:var(--fs-sm); }
.dm-shadow-batch label { display:grid; gap:6px; color:var(--text-muted); font-size:var(--fs-xs); }
.dm-shadow-batch textarea,.dm-shadow-batch input { width:100%; padding:9px 10px; color:var(--text-primary); background:var(--bg-primary); border:1px solid var(--border); border-radius:var(--radius-sm); font:inherit; }
.dm-shadow-range { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
.dm-shadow-items { display:grid; gap:6px; max-height:260px; overflow:auto; }
.dm-shadow-items > div { display:grid; grid-template-columns:1fr auto; gap:4px 10px; padding:8px; border:1px solid var(--border); border-radius:var(--radius-sm); }
.dm-shadow-items small { grid-column:1/-1; color:var(--text-muted); overflow-wrap:anywhere; }
.dm-conflict-table { grid-column:1/-1; max-width:100%; overflow:auto; }
.dm-conflict-table table { width:100%; border-collapse:collapse; font-size:var(--fs-xs); }
.dm-conflict-table th,.dm-conflict-table td { padding:6px 8px; border-bottom:1px solid var(--border); text-align:left; white-space:nowrap; }
.dm-confirm-backdrop { position:fixed; inset:0; z-index:2000; display:grid; place-items:center; padding:20px; background:color-mix(in srgb,var(--bg-primary) 62%,transparent); }
.dm-confirm-dialog { width:min(520px,100%); padding:22px; border:1px solid var(--border); border-radius:var(--radius); background:var(--card-bg); box-shadow:var(--shadow-lg,0 18px 48px rgba(0,0,0,.28)); }
.dm-confirm-dialog h2 { margin:0 0 14px; font-size:var(--fs-lg); }
.dm-confirm-dialog p { color:var(--text-secondary); line-height:1.6; }
.dm-confirm-actions { display:flex; justify-content:flex-end; gap:10px; margin-top:20px; }
.dm-confirm-dialog button:focus-visible { outline:2px solid var(--primary); outline-offset:2px; }
.dm-preflight-facts { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; margin-top:12px; }
.dm-preflight-facts > div { display:grid; gap:4px; min-width:0; padding:10px; border:1px solid var(--border); border-radius:var(--radius-sm); }
.dm-preflight-facts span,.dm-preflight-facts small { color:var(--text-muted); font-size:var(--fs-xs); }
.dm-preflight-facts strong { overflow-wrap:anywhere; }
.dm-job-main { display:grid; grid-template-columns:1fr auto; gap:5px 12px; min-width:0; }
.dm-job-main span { grid-column:1/-1; overflow-wrap:anywhere; }
.dm-job-meta { display:flex; align-items:center; justify-content:flex-end; flex-wrap:wrap; gap:10px; color:var(--text-muted); font-size:var(--fs-xs); }
.dm-job-card details { grid-column:1/-1; }.dm-job-card pre { max-height:320px; overflow:auto; white-space:pre-wrap; overflow-wrap:anywhere; }
.dm-scheduler-settings { margin-top:20px; }.dm-scheduler-settings h3 { margin:0 0 12px; }

/* ── 仪表条 ── */
.dm-bar {
  display: flex;
  align-items: center;
  gap: 0;
  padding: 10px 0;
  margin-bottom: var(--space-3);
  border-bottom: 1px solid var(--border);
  overflow-x: auto;
}
.dm-bar-item { display: flex; align-items: baseline; gap: 6px; padding: 0 16px; flex-shrink: 0; }
.dm-bar-item:first-child { padding-left: 0; }
.dm-bar-val {
  font-family: var(--font-data);
  font-size: var(--fs-lg);
  font-weight: 600;
  color: var(--text);
  font-variant-numeric: tabular-nums;
}
.dm-bar-val--ok { color: var(--success); }
.dm-bar-val--warn { color: var(--warning); }
.dm-bar-val--bad { color: var(--danger); }
.dm-bar-val--muted { color: var(--text-muted); }
.dm-bar-label {
  font-size: var(--fs-caption);
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.3px;
}
.dm-bar-sep { width: 1px; height: 24px; background: var(--border); flex-shrink: 0; }
.dm-bar-status { font-size: var(--fs-sm); padding: 2px 8px; border-radius: var(--radius-full); font-weight: 500; }
.dm-bar-status--on { background: rgba(52,211,153,.12); color: var(--success); }
.dm-bar-status--off { background: rgba(100,116,139,.12); color: var(--text-muted); }
.dm-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.dm-dot--ok { background: var(--success); }
.dm-dot--warn { background: var(--warning); }
.dm-dot--bad { background: var(--danger); }

/* ── 分类卡片 ── */
.dm-cats {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
  gap: 12px;
  margin-bottom: var(--space-4);
}
.dm-cat {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  transition: border-color var(--transition-fast);
}
.dm-cat--warning { border-color: rgba(244,167,66,.3); }
.dm-cat--error { border-color: rgba(239,68,68,.3); }

.dm-cat-head {
  display: flex; align-items: center; justify-content: space-between;
  width:100%; padding: 14px 16px; cursor: pointer; user-select: none;
  border:0; background:transparent; color:inherit; text-align:left; font:inherit;
}
.dm-cat-head:hover { background: rgba(255,255,255,.02); }
.dm-cat-head-left { display: flex; align-items: center; gap: 12px; }
.dm-cat-icon { font-size: 24px; }
.dm-cat-name { font-size: var(--fs-lg); font-weight: 600; color: var(--text); }
.dm-cat-count { font-size: var(--fs-caption); color: var(--text-muted); display: block; }
.dm-cat-head-right { display: flex; align-items: center; gap: 10px; }
.dm-cat-tag {
  font-size: var(--fs-caption); padding: 2px 8px;
  border-radius: var(--radius-full); font-weight: 500;
}
.dm-cat-tag.healthy { background: rgba(52,211,153,.12); color: var(--success); }
.dm-cat-tag.warning { background: rgba(244,167,66,.12); color: var(--warning); }
.dm-cat-tag.error { background: rgba(239,68,68,.12); color: var(--danger); }
.dm-cat-arrow { font-size: 14px; color: var(--text-muted); transition: transform var(--transition-fast); }
.dm-cat-arrow--open { transform: rotate(180deg); }

/* ── 展开的表详情 ── */
.dm-cat-body { border-top: 1px solid var(--border); padding: 8px 12px 12px; }
.dm-table {
  display: flex; align-items: flex-start; justify-content: space-between;
  padding: 10px 12px; margin-bottom: 6px;
  background: var(--card-bg); border-radius: var(--radius-sm);
  border-left: 3px solid transparent;
}
.dm-table:last-child { margin-bottom: 0; }
.dm-table--fresh { border-left-color: var(--success); }
.dm-table--stale { border-left-color: var(--warning); }
.dm-table--missing, .dm-table--error { border-left-color: var(--danger); }
.dm-table--incomplete { border-left-color: var(--warning); }

.dm-table-main { flex: 1; min-width: 0; }
.dm-table-top { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.dm-table-name { font-size: var(--fs-body); font-weight: 600; color: var(--text); }
.dm-table-code {
  font-size: var(--fs-caption); font-family: var(--font-mono);
  color: var(--text-muted); background: var(--bg-secondary);
  padding: 1px 6px; border-radius: 3px;
}
.dm-table-freq {
  font-size: var(--fs-caption); color: var(--text-muted);
  padding: 1px 6px; border: 1px solid var(--border); border-radius: var(--radius-sm);
}
.dm-table-meta { display: flex; gap: 16px; margin-bottom: 4px; font-size: var(--fs-sm); color: var(--text-muted); }
.dm-table-meta strong { color: var(--text-secondary); font-variant-numeric: tabular-nums; }
.dm-table-min { color: var(--text-muted); }
.dm-issues { display: flex; flex-wrap: wrap; gap: 4px; }
.dm-issue {
  font-size: var(--fs-caption); padding: 1px 6px;
  background: rgba(239,68,68,.08); color: var(--danger); border-radius: 3px;
}
.dm-table-act { display: flex; flex-direction: column; align-items: flex-end; gap: 6px; flex-shrink: 0; margin-left: 12px; }
.dm-status {
  font-size: var(--fs-caption); padding: 2px 8px;
  border-radius: var(--radius-full); font-weight: 500;
}
.dm-status.fresh { background: rgba(52,211,153,.12); color: var(--success); }
.dm-status.stale { background: rgba(244,167,66,.12); color: var(--warning); }
.dm-status.missing, .dm-status.error { background: rgba(239,68,68,.12); color: var(--danger); }
.dm-status.incomplete { background: rgba(244,167,66,.12); color: var(--warning); }
.dm-status.unavailable { background: var(--bg-secondary); color: var(--text-muted); }

/* ── expand transition ── */
.dm-expand-enter-active, .dm-expand-leave-active { transition: all .2s ease; overflow: hidden; }
.dm-expand-enter-from, .dm-expand-leave-to { opacity: 0; max-height: 0; }
.dm-expand-enter-to, .dm-expand-leave-from { opacity: 1; max-height: 1200px; }

/* ── 一键修复 ── */
.dm-repair-bar {
  display: flex; align-items: center; justify-content: flex-end;
  gap: 12px; padding: 12px 0; margin-bottom: var(--space-4);
  border-top: 1px solid var(--border);
}
.dm-repair-hint { font-size: var(--fs-sm); color: var(--text-muted); }
.dm-repair-results {
  display: grid; gap: 6px; margin: -8px 0 var(--space-4); padding: 10px 12px;
  border: 1px solid var(--border); border-radius: var(--radius); background: var(--card-bg);
}
.dm-repair-item { display: flex; align-items: center; gap: 10px; font-size: var(--fs-sm); }
.dm-repair-count { color: var(--text-muted); font-variant-numeric: tabular-nums; }
.dm-repair-error { color: var(--danger); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ── 采集与同步 ── */
.dm-ops {
  border-top: 1px solid var(--border);
  padding-top: var(--space-4);
}
.dm-ops-title {
  font-size: var(--fs-xl);
  font-weight: 700;
  color: var(--text);
  margin-bottom: 16px;
}

.dm-step-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.dm-index-sync { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-top: 12px; padding: 12px; border: 1px solid var(--border); border-radius: var(--radius-sm); }
.dm-index-sync-title { font-size: var(--fs-sm); font-weight: 600; }
.dm-field-inline { display: flex; align-items: center; gap: 4px; font-size: var(--fs-sm); color: var(--text-muted); }
.dm-field-label { font-size: var(--fs-sm); color: var(--text-muted); }

.dm-input {
  padding: 5px 8px; border: 1px solid var(--border); border-radius: var(--radius-sm);
  background: var(--bg); color: var(--text); font-family: var(--font); font-size: var(--fs-sm);
}
.dm-input:focus { border-color: var(--primary); outline: none; }
.dm-input--sm { width: 50px; }

.dm-prog { margin-top: 6px; font-size: var(--fs-sm); color: var(--text-muted); }
.dm-hint-warn { margin-top: 8px; font-size: var(--fs-caption); color: var(--warning); }

.dm-check-list { margin-top: 8px; font-size: var(--fs-sm); }
.dm-check-line { padding: 2px 0; color: var(--text-muted); }
.dm-check--ok { color: var(--success); }
.dm-check--fail { color: var(--danger); }

/* ── 定时任务卡片 ── */
.dm-task-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
.dm-task-card {
  padding: 14px; background: var(--card-bg); border-radius: var(--radius);
  border: 1px solid var(--border); display: flex; flex-direction: column; gap: 6px;
}
.dm-task-top { display: flex; justify-content: space-between; align-items: center; }
.dm-task-name { font-size: var(--fs-base); font-weight: 600; color: var(--text); }
.dm-task-status { font-size: var(--fs-caption); padding: 2px 8px; border-radius: var(--radius-full); }
.dm-task-status--on { background: rgba(52,211,153,.12); color: var(--success); }
.dm-task-status--off { background: rgba(100,116,139,.12); color: var(--text-muted); }
.dm-task-info { font-size: var(--fs-sm); color: var(--text-muted); }
.dm-task-next { font-size: var(--fs-caption); color: var(--text-muted); }
.dm-task-btns { display: flex; gap: 6px; margin-top: 4px; }

/* ── fm-block (复用模式) ── */
.fm-block {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: var(--space-4);
  margin-bottom: var(--space-3);
}
.fm-block-title {
  font-size: var(--fs-caption);
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 10px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}
.dm-snapshot { display:flex; align-items:center; gap:14px; flex-wrap:wrap; padding:14px 18px; margin:12px 0; border:1px solid var(--border); border-radius:var(--radius); background:var(--card-bg); }
.dm-snapshot > div { display:grid; gap:3px; }.dm-snapshot > div span,.dm-snapshot label { color:var(--text-muted); font-size:var(--fs-xs); }.dm-snapshot dl { width:100%; display:grid; grid-template-columns:90px minmax(0,1fr) 90px minmax(0,1fr); gap:7px; margin:0; }.dm-snapshot dt { color:var(--text-muted); }.dm-snapshot dd { margin:0; min-width:0; }.dm-snapshot code { display:block; overflow-wrap:anywhere; cursor:pointer; color:var(--primary); }
@media(max-width:760px){.dm-snapshot dl{grid-template-columns:90px minmax(0,1fr)}}
@media(max-width:760px){
  .dm-audit-filters{grid-template-columns:repeat(2,minmax(0,1fr))}
  .dm-audit-head{align-items:flex-start;flex-direction:column}
  .dm-state-summary{grid-template-columns:repeat(2,minmax(0,1fr))}
  .dm-history-gates{grid-template-columns:repeat(2,minmax(0,1fr))}
  .dm-preflight-facts{grid-template-columns:1fr}
  .dm-shadow-range{grid-template-columns:1fr}
  .dm-job-card{grid-template-columns:1fr}
  .dm-job-meta{justify-content:flex-start}
  .dm-preview-head,.dm-jobs-head{align-items:flex-start;flex-direction:column}
}
</style>
