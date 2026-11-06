<template>
  <div>
    <StrategyProfileBanner :profile="profile" title="回测策略版本" />
    <p class="text-gray ft-sm page-desc">
        按市场状态自适应切换方向模型，叠加择时仓位管理。仅验证量化方向模型的选股能力，不含多信号融合。
        回测假设同上。
    </p>

    <!-- 前视偏差警告 -->
    <div v-if="lookaheadWarning" class="card lookahead-warning">
      <div class="warning-title">⚠️ 前视偏差风险</div>
      <div class="warning-body">{{ lookaheadWarning }}</div>
      <div class="warning-detail" v-if="trainingInfo">
        模型训练截止:
        <span v-if="trainingInfo.qlib_fit_end">量化模型归一化 {{ trainingInfo.qlib_fit_end }}</span>
        <span v-if="trainingInfo.qlib_train_end"> · 量化模型训练集 {{ trainingInfo.qlib_train_end }}</span>
        <span v-if="trainingInfo.timing_train_end"> · 择时模型 {{ trainingInfo.timing_train_end }}</span>
      </div>
    </div>

    <!-- 配置卡片 -->
    <div class="card config-card" :class="{ 'profile-locked': profile }">
      <div v-if="profile" class="locked-note">除样本外子区间外，其余参数由策略版本冻结；修改参数必须复制为新草稿。</div>
      <div class="settings-stack">
        <section class="settings-section settings-section--core">
          <div class="settings-section__header">
            <div>
              <div class="settings-section__title">回测范围</div>
              <div class="settings-section__desc">先固定样本区间、股票池和真实调仓节奏。</div>
            </div>
            <span class="settings-section__badge">{{ config.freq === 'week' ? '周频' : config.freq }}</span>
          </div>
      <div class="settings-grid">
        <label class="settings-field">开始日期 <input v-model="config.start" type="date" /></label>
        <label class="settings-field">结束日期 <input v-model="config.end" type="date" /></label>
        <label class="settings-field">股票池
          <select v-model="config.pool">
            <option v-for="o in poolOptions" :key="o.value" :value="o.value">{{ o.label }}</option>
          </select>
        </label>
        <label class="settings-field">持仓数量 <input v-model.number="config.top_n" type="number" min="5" max="50" /></label>
        <label class="settings-field">调仓频率
          <select v-model="config.freq">
            <option value="day">每日</option>
            <option value="2d">双日</option>
            <option value="3d">三日</option>
            <option value="4d">四日</option>
            <option value="week">每周</option>
            <option value="biweek">双周</option>
            <option value="3w">三周</option>
            <option value="month">每月</option>
          </select>
        </label>
      </div>
      <div class="settings-group">
        <span class="settings-group__label">代码前缀</span>
        <div class="settings-choice-row">
        <label v-for="o in CODE_PREFIX_OPTIONS" :key="o.value" class="settings-choice">
          <input type="checkbox" :value="o.value" v-model="config.codePrefixes" />
          {{ o.label }}
        </label>
        <span class="text-gray ft-xs" v-if="!config.codePrefixes.length">不选 = 全市场</span>
        </div>
      </div>
        </section>

      <!-- 方向模型配置 -->
        <section class="settings-section">
          <div class="settings-section__header">
            <div>
              <div class="settings-section__title">模型路由</div>
              <div class="settings-section__desc">选择可参与路由的方向模型；市场状态决定每期实际使用哪一个。</div>
            </div>
            <span class="settings-section__badge">{{ config.directions.length }} 个方向</span>
          </div>
        <div class="settings-grid settings-grid--wide model-mode-grid">
          <label class="settings-option">
            <input type="checkbox" v-model="config.fixed_model" />
            <span><strong>固定单模型</strong><small>整段回测只使用同一个已训练模型，原方向路由设置会保留但暂停。</small></span>
          </label>
          <label class="settings-field">固定模型
            <select v-model="config.fixed_model_id" :disabled="!config.fixed_model">
              <option :value="null">请选择已训练模型</option>
              <option v-for="model in availableModels" :key="modelId(model)" :value="modelId(model)">
                {{ modelLabel(model) }}
              </option>
            </select>
            <span class="settings-field__hint">支持原有单模型、双树模型、截面与时序融合模型及专家子样本模型。</span>
          </label>
        </div>
        <div class="settings-choice-row">
          <label v-for="d in directionOptions" :key="d.value" class="settings-choice" :class="{ 'is-paused': config.fixed_model }">
            <input type="checkbox" :checked="config.directions.includes(d.value)"
              :disabled="config.fixed_model" @change="toggleDirection(d.value)" />
            {{ d.label }}
          </label>
        </div>
        <div class="settings-field__hint model-direction-hint">{{ directionHint }}</div>
        <div class="settings-options settings-group">
          <label class="settings-option">
            <input type="checkbox" :checked="config.use_vol_switch && !config.fixed_model" :disabled="config.use_rule_based || config.fixed_model" @change="config.use_vol_switch = $event.target.checked" />
            <span><strong>波动率切换{{ config.fixed_model ? '（已暂停）' : '' }}</strong><small>{{ config.fixed_model ? '固定单模型期间不参与模型选择' : '高波路由超卖反弹，低波路由趋势动量。' }}</small></span>
          </label>
          <label class="settings-option">
            <input type="checkbox" v-model="config.neutralize_qlib" />
            <span><strong>市值/行业中性化</strong><small>剥离市值和行业暴露，会改变模型排序；诊断基线默认关闭。</small></span>
          </label>
          <label class="settings-option">
            <input type="checkbox" v-model="config.macro_bias" />
            <span><strong>宏观行业加分</strong><small>对宏观利好行业统一加分，会改变模型排序；默认关闭。</small></span>
          </label>
          <label class="settings-option">
            <input type="checkbox" v-model="config.macro_regime" />
            <span><strong>市场状态置信度宏观修正</strong><small>只调整各期市场状态置信度，不给个股加分；诊断基线默认关闭。</small></span>
          </label>
          <label class="settings-option">
            <input type="checkbox" v-model="config.decision_day_filter" />
            <span><strong>决策日硬过滤并补足目标数量</strong><small>按排名剔除特别处理、停牌、涨停、低流动性、低价和次新股，再向后补足。</small></span>
          </label>
          <label class="settings-option">
            <input type="checkbox" :checked="config.sector_fusion && !routingPaused" :disabled="routingPaused" @change="config.sector_fusion = $event.target.checked" />
            <span><strong>板块背离融合{{ routingPaused ? '（已暂停）' : '' }}</strong><small>{{ routingPauseText || '用板块背离修正方向得分' }}</small></span>
          </label>
        </div>
        <details v-if="!config.fixed_model" class="model-routing-details">
          <summary>状态识别与模型映射</summary>
          <div class="settings-options routing-advanced">
            <label class="settings-option">
              <input type="checkbox" v-model="config.use_hmm" :disabled="config.use_rule_based" />
              <span><strong>HMM 6 状态</strong><small>使用状态序列特征辅助识别市场环境。</small></span>
            </label>
            <label class="settings-option">
              <input type="checkbox" v-model="config.use_rule_based" />
              <span><strong>确定性规则</strong><small>不使用学习型状态识别，改用固定规则。</small></span>
            </label>
            <label class="settings-option">
              <input type="checkbox" v-model="config.use_crf" :disabled="config.use_rule_based" />
              <span><strong>CRF 状态平滑</strong><small>抑制相邻周期的状态频繁跳变。</small></span>
            </label>
          </div>
        <StateModelMapping
          v-model="stateModelMap"
          :use-vol-switch="config.use_vol_switch"
          :models="trainedModels"
        />
        </details>
        <div v-else class="settings-field__hint model-direction-hint">市场状态仍用于仓位、止盈止损和结果诊断，但不会切换选股模型。</div>
        </section>

        <section class="settings-section">
          <div class="settings-section__header">
            <div>
              <div class="settings-section__title">仓位管理</div>
              <div class="settings-section__desc">控制每期投入比例与闲置资金处理方式。</div>
            </div>
            <span class="settings-section__badge">{{ config.full_position ? '固定95%仓位' : config.use_timing ? '动态仓位' : '固定95%仓位' }}</span>
          </div>
          <div class="settings-options">
            <label class="settings-option">
              <input type="checkbox" v-model="config.use_timing" :disabled="config.full_position" />
              <span><strong>择时仓位</strong><small>由择时模型在 20%–100% 之间动态调节。</small></span>
            </label>
            <label class="settings-option">
              <input type="checkbox" v-model="config.full_position" />
              <span><strong>固定95%仓位回测</strong><small>忽略择时输出，保留5%目标暴露。</small></span>
            </label>
            <label class="settings-option">
              <input type="checkbox" v-model="config.cash_yield" />
              <span><strong>闲置现金计息</strong><small>未投入资金按债基收益口径计入净值。</small></span>
            </label>
          </div>
          <div class="settings-field__hint option-hint">{{ optionHint }}</div>
        </section>

        <section class="settings-section">
          <div class="settings-section__header">
            <div>
              <div class="settings-section__title">退出与交易约束</div>
              <div class="settings-section__desc">止盈止损、冷冻期、费率与仓位上限集中在这里。</div>
            </div>
            <span class="settings-section__badge">{{ stopSourceLabels[config.stopSource] }}</span>
          </div>
        <div class="config-row risk-row">
          <span class="section-label">止盈止损</span>
          <span class="ltv-mode-switch">
            <button v-for="s in ['off','fixed','regime','dynamic','atr']" :key="s" type="button"
                    :class="{ 'ltv-mode--active': config.stopSource === s }"
                    @click="config.stopSource = s">{{ stopSourceLabels[s] }}</button>
          </span>
        </div>
        <div v-if="config.stopSource === 'fixed'" class="config-row risk-row">
          <span class="section-label">阶梯止损</span>
          <label class="risk-item">预警减半
            <input v-model.number="config.stopLossWarnPct" type="number" step="1" min="0" max="20" style="width:50px" />
            <span class="unit">%</span>
          </label>
          <label class="risk-item">清仓
            <input v-model.number="config.stopLossFullPct" type="number" step="1" min="0" max="20" style="width:50px" />
            <span class="unit">%</span>
          </label>
          <span class="sep">|</span>
          <span class="section-label">阶梯止盈</span>
          <label class="risk-item">减半锁利
            <input v-model.number="config.takeProfitHalfPct" type="number" step="1" min="0" max="50" style="width:50px" />
            <span class="unit">%</span>
          </label>
          <label class="risk-item">清仓收割
            <input v-model.number="config.takeProfitFullPct" type="number" step="1" min="0" max="50" style="width:50px" />
            <span class="unit">%</span>
          </label>
        </div>
        <div v-if="config.stopSource === 'dynamic'" class="config-row risk-row">
          <span class="section-label">动态回撤</span>
          <label class="risk-item">回看窗口
            <input v-model.number="config.dynamicLookback" type="number" step="1" min="5" max="60" style="width:50px" />
            <span class="unit">日</span>
          </label>
          <label class="risk-item">回撤幅度
            <input v-model.number="config.dynamicTrail" type="number" step="1" min="1" max="30" style="width:50px" />
            <span class="unit">%</span>
          </label>
        </div>
        <div v-if="config.stopSource === 'atr'" class="config-row risk-row">
          <span class="section-label">ATR 止损</span>
          <label class="risk-item">周期
            <input v-model.number="config.atrPeriod" type="number" step="1" min="5" max="30" style="width:50px" />
            <span class="unit">日</span>
          </label>
          <label class="risk-item">倍数
            <input v-model.number="config.atrMult" type="number" step="0.5" min="1" max="5" style="width:50px" />
            <span class="unit">×ATR</span>
          </label>
          <span class="text-gray ft-xs">止损线 = 入场价 − 倍数 × ATR</span>
        </div>
        <div class="config-row risk-row">
          <span class="section-label">冷冻期</span>
          <label class="risk-item">止损后
            <input v-model.number="config.cooldownStopWeeks" type="number" step="1" min="1" max="12" style="width:45px" placeholder="2" />
            <span class="unit">周不买</span>
          </label>
          <label class="risk-item">止盈后
            <input v-model.number="config.cooldownProfitWeeks" type="number" step="1" min="1" max="12" style="width:45px" placeholder="1" />
            <span class="unit">周不买</span>
          </label>
          <span class="text-gray ft-xs">清仓后 N 周内不再买入，留空 = 随频率默认</span>
        </div>

      <details class="config-details">
        <summary class="details-summary">费用与风控参数</summary>
        <div class="config-row risk-row">
          <span class="section-label">费率 (bp)</span>
          <label class="risk-item">佣金 <input v-model.number="config.commissionBps" type="number" step="1" min="0" max="20" style="width:45px" /></label>
          <label class="risk-item">滑点 <input v-model.number="config.slippageBps" type="number" step="1" min="0" max="20" style="width:45px" /></label>
          <label class="risk-item">印花税 <input v-model.number="config.stampBps" type="number" step="1" min="0" max="20" style="width:45px" /></label>
          <span class="sep">|</span>
          <span class="section-label">仓位上限</span>
          <label class="risk-item">单只 <input v-model.number="config.maxPositionPct" type="number" step="5" min="5" max="50" style="width:45px" /><span class="unit">%</span></label>
          <label class="risk-item">行业 <input v-model.number="config.maxSectorPct" type="number" step="5" min="10" max="60" style="width:45px" /><span class="unit">%</span></label>
        </div>
      </details>

      <details v-if="config.stopSource === 'regime'" class="config-details" open>
        <summary class="details-summary">仓位基数 & 止盈止损 (按市场状态独立设置)</summary>
        <div style="display:grid;grid-template-columns:70px 70px 65px 65px 70px 70px;gap:1px 6px;margin-top:6px;font-size:10px;color:var(--text-muted);padding:0 8px;">
          <span></span><span>仓位</span><span>止损预警</span><span>止损清仓</span><span>止盈减半</span><span>止盈清仓</span>
        </div>
        <div v-for="(label, key) in ratioLabels" :key="key"
             style="display:grid;grid-template-columns:70px 70px 65px 65px 70px 70px;gap:2px 6px;align-items:center;padding:3px 8px;margin-bottom:2px;border-radius:4px;font-size:12px;"
             :style="{ background: key==='0'||key==='3'?'rgba(37,99,235,.06)':key==='1'||key==='4'?'rgba(239,68,68,.04)':'rgba(245,158,11,.04)' }">
          <span style="font-weight:600;color:var(--text);">{{ label }}</span>
          <span style="display:flex;align-items:center;gap:2px;">
            <input v-model.number="ratioConfig[key]" type="number" min="0.05" max="2.0" step="0.05"
              style="width:48px;-moz-appearance:textfield;" class="no-spin" />
            <span style="font-size:10px;color:var(--primary);">{{ (ratioConfig[key]*100).toFixed(0) }}%</span>
          </span>
          <input v-model.number="stopConfig[key].warn" type="number" min="1" max="20" step="1" style="width:55px;" />
          <input v-model.number="stopConfig[key].full" type="number" min="1" max="30" step="1" style="width:55px;" />
          <input v-model.number="stopConfig[key].profitHalf" type="number" min="5" max="50" step="1" style="width:55px;" />
          <input v-model.number="stopConfig[key].profitFull" type="number" min="5" max="50" step="1" style="width:55px;" />
        </div>
        <div style="margin-top:6px;display:flex;gap:6px;padding:0 8px;">
          <button type="button" @click="loadRatioConfig" class="btn" style="font-size:11px;">重新加载</button>
          <button type="button" @click="saveRatioConfig" class="btn primary" style="font-size:11px;">保存配置</button>
        </div>
      </details>
        </section>
      </div>

      <div class="settings-ledger">
        <div class="settings-ledger__summary">
          <span class="settings-ledger__label">本次运行</span>
          <span class="settings-ledger__chip">{{ config.start }} → {{ config.end }}</span>
          <span class="settings-ledger__chip">{{ poolOptions.find(o => o.value === config.pool)?.label || config.pool }}</span>
          <span class="settings-ledger__chip">Top {{ config.top_n }}</span>
          <span class="settings-ledger__chip">{{ config.freq === 'week' ? '周频' : config.freq }}</span>
          <span class="settings-ledger__chip">{{ stopSourceLabels[config.stopSource] }}</span>
          <span v-if="config.fixed_model" class="settings-ledger__chip">固定单模型 · {{ fixedModelName }}</span>
          <span v-else-if="config.use_vol_switch" class="settings-ledger__chip">波动率切换</span>
          <span v-if="config.neutralize_qlib" class="settings-ledger__chip">市值/行业中性化</span>
          <span v-if="config.macro_bias" class="settings-ledger__chip">宏观行业加分</span>
          <span v-if="config.macro_regime" class="settings-ledger__chip">状态置信度宏观修正</span>
          <span v-if="config.decision_day_filter" class="settings-ledger__chip">决策日硬过滤</span>
        </div>
        <button class="btn-run" :disabled="running" @click="startBacktest">
          {{ running ? '回测中...' : '开始回测' }}
        </button>
      </div>
    </div>

    <!-- 进度条 -->
    <div v-if="running" class="card progress-card">
      <div class="card-title">⏳ 回测进度</div>
      <div class="progress-bar-bg"><div class="progress-bar" :style="{ width: pollProgress + '%' }"></div></div>
      <div class="progress-text">{{ progressText }} <span v-if="elapsedSec">· {{ fmtElapsed(elapsedSec) }}</span></div>
    </div>

    <div v-if="error" class="error">{{ error }}</div>

    <!-- 回测结果 -->
    <template v-if="result">
      <!-- 运行配置摘要 -->
      <div class="card config-summary" style="margin-top:12px;">
        <span class="summary-chip" v-if="result.summary?.timing">
          ⏱ 择时仓位 {{ (result.summary.timing.avg * 100).toFixed(0) }}%
          <small>（{{ result.summary.timing.n_reduced }}期降低仓位）</small>
        </span>
        <span class="summary-chip">{{ poolOptions.find(o => o.value === config.pool)?.label || config.pool }}</span>
        <span class="summary-chip">Top-{{ config.top_n }}</span>
        <span class="summary-chip">{{ result.portfolio_contract?.sizing === 'equal' ? '等权' : '分数加权' }} · {{ ((result.portfolio_contract?.target_exposure ?? 0.95) * 100).toFixed(0) }}%目标暴露</span>
        <span class="summary-chip">{{ (config.use_rule_based ? '确定性规则' : config.use_hmm ? 'HMM' : '规则') + '体制' }}</span>
        <span class="summary-chip">{{ config.full_position ? '固定95%仓位' : config.use_timing ? '择时开' : '固定95%仓位' }}</span>
        <span v-if="result.score_processing?.neutralize_qlib_applied" class="summary-chip">市值/行业中性化</span>
        <span v-else-if="result.score_processing?.neutralize_qlib" class="summary-chip">中性化未生效</span>
        <span v-if="result.score_processing?.macro_sector_bias" class="summary-chip">宏观行业加分</span>
        <span v-if="result.score_processing?.macro_regime" class="summary-chip">状态置信度宏观修正</span>
        <span v-if="result.score_processing?.decision_day_filter" class="summary-chip">决策日硬过滤</span>
        <span v-if="result.model_routing?.mode !== 'fixed' && config.sector_fusion && !routingPaused" class="summary-chip">板块背离融合</span>
        <span v-if="config.use_crf" class="summary-chip">CRF平滑</span>
        <span class="summary-chip">{{ result.summary?.n_periods }}期</span>
      </div>

      <!-- 指标卡片 -->
      <div class="card" style="margin-top:12px;">
        <div class="card-title">📊 回测指标</div>
        <div class="metrics-grid">
          <div class="metric-item" v-for="m in metrics" :key="m.label">
            <div class="metric-value" :class="m.color">{{ m.value }}</div>
            <div class="metric-label">{{ m.label }}</div>
          </div>
        </div>
      </div>

      <!-- 摘要信息 -->
      <div class="card" v-if="result.direction_distribution || result.regime_breakdown || result.model_routing" style="margin-top:12px;">
        <div class="card-title">📋 摘要</div>
        <div class="summary-grid">
          <div v-if="result.direction_distribution" class="summary-block">
            <div class="summary-label">方向分布</div>
            <div class="summary-value">{{ formatDict(result.direction_distribution) }}</div>
          </div>
          <div v-if="result.regime_breakdown" class="summary-block">
            <div class="summary-label">体制收益分解</div>
            <div class="summary-value">{{ formatDict(result.regime_breakdown) }}</div>
          </div>
          <div v-if="result.model_routing" class="summary-block">
            <div class="summary-label">状态模型实际使用</div>
            <div class="summary-value">{{ formatDict(result.model_routing.usage) }}</div>
          </div>
        </div>
      </div>

      <!-- 权益曲线 -->
      <div v-if="equityDates.length" class="card" style="margin-top:12px;">
        <div class="card-title">📈 权益曲线</div>
        <div ref="chartRef" class="chart-box"></div>
      </div>

      <!-- 逐期记录 -->
      <div v-if="periods.length" class="card" style="margin-top:12px;">
        <div class="card-title">
          📜 逐期记录 ({{ periods.length }} 期)
          <span class="ft-xs text-gray" style="margin-left:8px">
            超额胜率 {{ excessWinRate }} · 平均仓位 {{ avgTimingRatio }}
          </span>
        </div>
        <div class="table-wrap">
          <table class="data-table">
            <thead>
              <tr>
                <th>日期</th><th>体制</th><th>方向</th><th>模型状态</th><th>置信度</th><th>仓位</th>
                <th>前5名</th><th>收益</th><th>基准</th><th>超额</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="p in paginatedPeriods" :key="p.date"
                :class="{ 'row-stop-loss': p.n_holdings < config.top_n }">
                <td>{{ p.date }}</td>
                <td><span :class="'regime-tag regime-' + p.regime">{{ REGIME_LABELS[p.regime] || p.regime }}</span></td>
                <td>{{ dirLabel(p.direction) }}</td>
                <td>{{ p.model_state || '-' }}</td>
                <td>{{ p.confidence != null ? (p.confidence * 100).toFixed(0) + '%' : '-' }}</td>
                <td>
                  <span v-if="p.timing_ratio != null && p.timing_ratio < 1"
                    class="timing-lo">{{ (p.timing_ratio * 100).toFixed(0) }}%</span>
                  <span v-else>-</span>
                </td>
                <td class="top5-cell" :title="(p.top_stocks || []).join(', ')">{{ (p.top_stocks || []).join(', ') }}</td>
                <td :class="p.period_return >= 0 ? 'up' : 'down'">{{ fmtPct(p.period_return) }}</td>
                <td :class="p.benchmark_return >= 0 ? 'up' : 'down'">{{ fmtPct(p.benchmark_return) }}</td>
                <td :class="(p.period_return - p.benchmark_return) >= 0 ? 'up' : 'down'">
                  {{ fmtPct(p.period_return - p.benchmark_return) }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <div v-if="periodPages > 1" class="pagination">
          <button :disabled="periodPage <= 1" @click="periodPage--">上一页</button>
          <span>{{ periodPage }} / {{ periodPages }}</span>
          <button :disabled="periodPage >= periodPages" @click="periodPage++">下一页</button>
        </div>
      </div>
    </template>

    <div v-if="!running && !result && !error" class="card empty-hint">
      <AppEmpty text="配置参数后点击「开始回测」" />
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, watch, nextTick, onUnmounted, onMounted } from 'vue'
import api from '../api/index.js'
import { REGIME_LABELS, CODE_PREFIX_OPTIONS } from '../constants/index.js'
import AppEmpty from '../components/AppEmpty.vue'
import StateModelMapping from '../components/StateModelMapping.vue'
import StrategyProfileBanner from '../components/StrategyProfileBanner.vue'
import { useChart } from '../composables/useChart.js'
import { todayStr, dateStr } from '../utils/date.js'
import { cssVar } from '../utils/marketFormat.js'
import { modelRefPayload, stateModelsPayload } from '../utils/modelRouting.js'

const { chartRef, render, dispose } = useChart()
const props = defineProps({ profile: { type: Object, default: null } })

const poolOptions = [
  { value: 'stable_history_500d', label: '稳定历史池500D' },
  { value: 'csi300', label: '沪深300' },
  { value: 'csi300+csi500', label: '沪深300+中证500' },
  { value: 'all', label: '全A股' },
  { value: 'small_cap', label: '小市值 (中证1000+国证2000)' },
  { value: 'sector_rotation', label: '板块轮动（前5行业）' },
  { value: 'concept_rotation', label: '概念轮动（前5概念）' },
]

const directionOptions = [
  { value: 'trend_momentum', label: '趋势动量' },
  { value: 'volume_breakout', label: '量价突破' },
  { value: 'oversold_rebound', label: '超卖反弹' },
  { value: 'value_defense', label: '价值防御' },
]

const DIR_LABELS = { trend_momentum: '趋势动量', volume_breakout: '量价突破', oversold_rebound: '超卖反弹', value_defense: '价值防御' }
function dirLabel(d) { return DIR_LABELS[d] || d }

const today = todayStr()
const oneYearAgo = dateStr(new Date(Date.now() - 365 * 24 * 60 * 60 * 1000))

const trainingInfo = ref(null)
const trainedModels = ref([])
const stateModelMap = ref({})
const availableModels = computed(() => trainedModels.value.filter(model =>
  !model.disabled && model.experiment_id !== 'timing' && model.run_id &&
  model.label_mode === 'tradable_open_to_open'))

function modelId(model) {
  return `${model.experiment_id}/${model.run_id}`
}

function modelLabel(model) {
  const ic = model.ic == null ? '' : ` · 信息系数 ${Number(model.ic).toFixed(4)}`
  return `${model.display_name || model.model_class || model.run_id.slice(0, 8)}${ic}`
}
const lookaheadWarning = computed(() => {
  if (!trainingInfo.value?.safe_start) return ''
  if (config.start >= trainingInfo.value.safe_start) return ''
  const parts = []
  if (trainingInfo.value.qlib_fit_end && config.start <= trainingInfo.value.qlib_fit_end) {
    parts.push(`量化模型标准分归一化统计量截止 ${trainingInfo.value.qlib_fit_end}，回测起始之前的特征标准化会泄露未来信息`)
  }
  if (trainingInfo.value.qlib_train_end && config.start <= trainingInfo.value.qlib_train_end) {
    parts.push(`量化模型训练集截止 ${trainingInfo.value.qlib_train_end}，模型参数已见过回测期的数据`)
  }
  if (trainingInfo.value.timing_train_end && config.use_timing && !config.full_position && config.start <= trainingInfo.value.timing_train_end) {
    parts.push(`择时模型训练截止 ${trainingInfo.value.timing_train_end}，回测期内择时信号使用了回测期训练的模型`)
  }
  if (!parts.length) {
    parts.push(`建议回测起始日期不早于 ${trainingInfo.value.safe_start}，否则可能存在前视偏差`)
  }
  return parts.join('；') + '。'
})

onMounted(async () => {
  const [cutoffs, models] = await Promise.allSettled([
    api.get('/backtest/training-cutoffs'),
    api.get('/qlib/trained-models'),
  ])
  if (cutoffs.status === 'fulfilled') {
    trainingInfo.value = (cutoffs.value?.data || cutoffs.value) ?? null
  }
  if (models.status === 'fulfilled') {
    trainedModels.value = (models.value?.data || models.value) || []
  }
})

const config = reactive({
  start: oneYearAgo,
  end: today,
  pool: 'stable_history_500d',
  top_n: 5,
  freq: 'week',
  sizing: 'equal',
  target_exposure: 0.95,
  max_one_way_turnover: 0.30,
  directions: ['trend_momentum'],
  fixed_model: false,
  fixed_model_id: null,
  use_hmm: false,
  use_rule_based: false,
  use_timing: false,
  full_position: true,
  neutralize_qlib: false,
  macro_bias: false,
  macro_regime: false,
  sector_fusion: false,
  decision_day_filter: true,
  use_crf: false,
  use_vol_switch: false,
  stopLossPct: null,
  takeProfitPct: null,
  stopLossWarnPct: 5,
  stopLossFullPct: 10,
  takeProfitHalfPct: 15,
  takeProfitFullPct: 30,
  stopSource: 'off',
  dynamicLookback: 20,
  dynamicTrail: 5,
  atrPeriod: 14,
  atrMult: 2.0,
  commissionBps: 3,
  slippageBps: 1,
  stampBps: 5,
  maxPositionPct: 20,
  maxSectorPct: 30,
  codePrefixes: ['600,601,603,605', '000,001,002,003'],
  cash_yield: false,
  cooldownStopWeeks: null,
  cooldownProfitWeeks: null,
})

const stopSourceLabels = { off: '关闭', fixed: '固定百分比', regime: '按市场状态', dynamic: '动态回撤', atr: '平均真实波幅止损' }

// 仓位基数配置
const ratioConfig = reactive({ 0: 1.50, 1: 0.15, 2: 0.50, 3: 1.50, 4: 0.15, 5: 0.50 })
const stopConfig = reactive({
  0: { warn: 3, full: 5,  profitHalf: 15, profitFull: 30 },
  1: { warn: 3, full: 5,  profitHalf: 10, profitFull: 20 },
  2: { warn: 3, full: 5,  profitHalf: 10, profitFull: 20 },
  3: { warn: 5, full: 10, profitHalf: 15, profitFull: 30 },
  4: { warn: 5, full: 10, profitHalf: 15, profitFull: 30 },
  5: { warn: 5, full: 10, profitHalf: 15, profitFull: 30 },
})
const ratioLabels = { 0: '高波快涨', 1: '高波快跌', 2: '高波震荡', 3: '稳步上涨', 4: '稳步下跌', 5: '低波盘整' }

async function loadRatioConfig() {
  try { const r = await api.get('/timing/regime/ratio-config')
    if (r?.data) {
      Object.assign(ratioConfig, r.data.ratios)
      if (r.data.stops) for (const k in r.data.stops) Object.assign(stopConfig[k]||{}, r.data.stops[k])
    }
  } catch(e) { alert('加载失败') }
}
async function saveRatioConfig() {
  try { await api.post('/timing/regime/ratio-config', { ratios: { ...ratioConfig }, stops: { ...stopConfig } })
    alert('已保存')
  } catch(e) { alert('保存失败') }
}

function toggleDirection(val) {
  const idx = config.directions.indexOf(val)
  if (idx >= 0) config.directions.splice(idx, 1)
  else config.directions.push(val)
}

const directionHint = computed(() => {
  if (config.fixed_model) return '方向模型路由已暂停；取消固定单模型后恢复原选择'
  const n = config.directions.length
  if (n === 0) return '至少选择一个方向'
  return `已选 ${n} 个方向，分别加载对应最优模型`
})

const routingPaused = computed(() => config.fixed_model || config.use_vol_switch)
const routingPauseText = computed(() => {
  if (config.fixed_model) return '固定单模型开启时不可引入其他方向模型'
  if (config.use_vol_switch) return '波动率切换开启时不可同时使用'
  return ''
})
const fixedModelName = computed(() => {
  const model = availableModels.value.find(item => modelId(item) === config.fixed_model_id)
  return model ? (model.display_name || model.model_class || model.run_id.slice(0, 8)) : '未选择'
})

const optionHint = computed(() => {
  const parts = []
  if (config.use_hmm) parts.push('HMM 识别稳步上涨/阴跌等 6 种状态')
  else parts.push('规则分类器 4 状态')
  if (config.full_position) parts.push('固定95%仓位(忽略择时)')
  else if (config.use_timing) parts.push('择时模型动态调节仓位 20%~100%')
  else parts.push('固定95%仓位运作')
  return parts.join('，')
})

const running = ref(false)
const error = ref('')
const result = ref(null)
const pollProgress = ref(0)
const progressText = ref('')
const elapsedSec = ref(0)
const periodPage = ref(1)
const periodPageSize = 20
let pollTimer = null
let elapsedInterval = null
let pollToken = 0

const metrics = computed(() => {
  const s = result.value?.summary
  if (!s) return []
  const fmtP = (v) => v != null ? (v * 100).toFixed(1) + '%' : '-'
  const fmtN = (v, d) => v != null ? v.toFixed(d) : '-'
  return [
    { label: '累计收益', value: fmtP(s.total_return), color: s.total_return >= 0 ? 'up' : 'down' },
    { label: '年化收益', value: fmtP(s.annual_return), color: s.annual_return >= 0 ? 'up' : 'down' },
    { label: '年化基准', value: fmtP(s.annual_benchmark) },
    { label: '年化超额', value: fmtP(s.annual_excess), color: s.annual_excess >= 0 ? 'up' : 'down' },
    { label: '夏普比率', value: fmtN(s.sharpe, 2) },
    { label: '索提诺', value: fmtN(s.sortino, 2) },
    { label: '卡玛比率', value: fmtN(s.calmar, 2) },
    { label: '超额卡玛', value: fmtN(s.excess_calmar, 2) },
    { label: '最大回撤', value: fmtP(s.max_drawdown), color: 'down' },
    { label: '胜率', value: fmtP(s.win_rate) },
    { label: '超额胜率', value: fmtP(s.excess_win_rate) },
    { label: '周度单边换手', value: fmtP(s.max_one_way_turnover) },
    { label: '盈亏比', value: fmtN(s.profit_factor, 2) },
    { label: '最佳周', value: fmtP(s.best_week), color: 'up' },
    { label: '最差周', value: fmtP(s.worst_week), color: 'down' },
    { label: '交易次数', value: s.n_trades ?? '-' },
  ]
})

const periods = computed(() => result.value?.periods || [])
const periodPages = computed(() => Math.max(1, Math.ceil(periods.value.length / periodPageSize)))
const paginatedPeriods = computed(() => {
  const start = (periodPage.value - 1) * periodPageSize
  return periods.value.slice(start, start + periodPageSize)
})
const excessWinRate = computed(() => {
  const ps = periods.value
  if (!ps.length) return '-'
  const n = ps.filter(p => (p.period_return - p.benchmark_return) > 0).length
  return (n / ps.length * 100).toFixed(0) + '%'
})
const avgTimingRatio = computed(() => {
  const ps = periods.value
  if (!ps.length) return '-'
  const sum = ps.reduce((s, p) => s + (p.timing_ratio || 1), 0)
  return (sum / ps.length * 100).toFixed(0) + '%'
})

const equityDates = computed(() => (result.value?.equity_curve || []).map(p => p.date || ''))
const equityValues = computed(() => (result.value?.equity_curve || []).map(p => p.value || 0))
const benchmarkValues = computed(() => (result.value?.benchmark_curve || []).map(p => p.value || 0))

function fmtPct(v) {
  if (v == null) return '-'
  return (v * 100).toFixed(2) + '%'
}

function fmtElapsed(s) {
  if (s < 60) return s + 's'
  const m = Math.floor(s / 60)
  const sec = s % 60
  return m + 'm' + sec + 's'
}

function formatDict(obj) {
  if (!obj) return '-'
  const parts = Object.entries(obj).map(([k, v]) => {
    const label = REGIME_LABELS[k] || DIR_LABELS[k] || k
    if (typeof v === 'object' && v !== null) {
      const avg = v.avg_return != null ? `${(v.avg_return * 100).toFixed(2)}%` : ''
      const wr = v.win_rate != null ? `胜率${(v.win_rate * 100).toFixed(0)}%` : ''
      const cnt = v.count != null ? `${v.count}期` : ''
      return `${label}: ${avg} ${wr} ${cnt}`
    }
    if (typeof v === 'number') {
      if (Number.isInteger(v) || v > 1) return `${label}: ${v}期`
      return `${label}: ${(v * 100).toFixed(1)}%`
    }
    return `${label}: ${v}`
  })
  return parts.join('  |  ')
}

async function startBacktest() {
  if (!props.profile && config.fixed_model && !config.fixed_model_id) {
    error.value = '固定单模型模式下，请选择一个已训练模型'; return
  }
  if (!props.profile && !config.fixed_model && !config.directions.length) {
    error.value = '至少选择一个策略方向'; return
  }
  running.value = true; error.value = ''; result.value = null
  pollProgress.value = 0; elapsedSec.value = 0
  progressText.value = '提交回测任务...'
  periodPage.value = 1
  pollToken++
  if (pollTimer) clearTimeout(pollTimer)
  if (elapsedInterval) clearInterval(elapsedInterval)
  dispose()

  elapsedInterval = setInterval(() => { elapsedSec.value++ }, 1000)

  try {
    let r, jobId
    const payload = props.profile ? {
      start: config.start,
      end: config.end,
      profile_id: props.profile.profile_id,
      profile_version: props.profile.version,
    } : {
        start: config.start,
        end: config.end,
        pool: config.pool,
        top_n: config.top_n,
        freq: config.freq,
        sizing: config.sizing,
        target_exposure: config.target_exposure,
        max_one_way_turnover: config.max_one_way_turnover,
        directions: config.directions,
        use_hmm: config.use_hmm,
        use_rule_based: config.use_rule_based,
        use_timing: config.use_timing,
        full_position: config.full_position,
        cash_yield: config.cash_yield,
        macro_bias: config.macro_bias,
        macro_regime: config.macro_regime,
        neutralize_qlib: config.neutralize_qlib,
        sector_fusion: config.sector_fusion && !routingPaused.value && !Object.keys(stateModelMap.value).length,
        decision_day_filter: config.decision_day_filter,
        use_crf: config.use_crf,
        use_vol_switch: config.fixed_model ? false : config.use_vol_switch,
        state_models: config.fixed_model ? {} : stateModelsPayload(stateModelMap.value, config.use_vol_switch),
        fixed_model: config.fixed_model ? modelRefPayload(config.fixed_model_id) : null,
        cooldown_stop_weeks: config.cooldownStopWeeks || null,
        cooldown_profit_weeks: config.cooldownProfitWeeks || null,
        code_prefixes: config.codePrefixes.length ? config.codePrefixes.flatMap(p => p.split(',')) : null,
        stop_loss_warn: config.stopSource === 'fixed' && config.stopLossWarnPct ? -(config.stopLossWarnPct / 100) : null,
        stop_loss_full: config.stopSource === 'fixed' && config.stopLossFullPct ? -(config.stopLossFullPct / 100) : null,
        take_profit_half: config.stopSource === 'fixed' && config.takeProfitHalfPct ? (config.takeProfitHalfPct / 100) : null,
        take_profit_full: config.stopSource === 'fixed' && config.takeProfitFullPct ? (config.takeProfitFullPct / 100) : null,
        stop_source: config.stopSource,
        dynamic_lookback: config.dynamicLookback,
        dynamic_trail: (config.dynamicTrail / 100) || null,
        atr_period: config.atrPeriod,
        atr_mult: config.atrMult,
        commission_rate: (config.commissionBps / 10000) || null,
        slippage_rate: (config.slippageBps / 10000) || null,
        stamp_duty: (config.stampBps / 10000) || null,
        max_position_pct: (config.maxPositionPct / 100) || null,
        max_sector_pct: (config.maxSectorPct / 100) || null,
      }
    r = await api.post('/backtest/direction', payload)
    jobId = r.data?.job_id || r.job_id
    if (!jobId) { error.value = '未获取到任务ID'; stopRunning(); return }
    pollJob(jobId)
  } catch (e) { error.value = '提交失败: ' + e.message; stopRunning() }
}

function stopRunning() {
  running.value = false
  if (elapsedInterval) { clearInterval(elapsedInterval); elapsedInterval = null }
}

function pollJob(jobId) {
  let attempts = 0
  const token = pollToken

  async function poll() {
    attempts++
    const delay = attempts <= 12 ? 5000 : attempts <= 30 ? 10000 : 15000
    pollProgress.value = Math.min(95, 5 + Math.log2(attempts) * 10)
    const mins = Math.floor(elapsedSec.value / 60)
    progressText.value = `回测运行中 (${mins}分)`

    try {
      const url = `/backtest/direction/${jobId}`
      const job = await api.get(url)
      if (token !== pollToken) return
      const data = job.data || job
      if (data.status === 'done') {
        result.value = data
        pollProgress.value = 100
        progressText.value = `回测完成 · ${periods.value.length} 期`
        stopRunning()
        await nextTick()
        renderChart()
        return
      }
      if (data.status === 'stalled') {
        progressText.value = `${data.message || '后台仍在运行，暂未收到心跳'} (${mins}分)`
      }
      if (data.status === 'error' || data.status === 'failed') {
        error.value = '回测失败: ' + (data.error || '未知错误')
        stopRunning()
        return
      }
    } catch (e) {
      // 继续轮询
    }

    if (token !== pollToken) return
    pollTimer = setTimeout(poll, delay)
  }

  poll()
}

onUnmounted(() => {
  pollToken++
  if (pollTimer) clearTimeout(pollTimer)
  if (elapsedInterval) clearInterval(elapsedInterval)
})

function renderChart() {
  if (!equityDates.value.length) return
  render(() => {
    const dates = equityDates.value
    const eq = equityValues.value
    const bm = benchmarkValues.value
    let minVal = Infinity, maxVal = -Infinity
    for (const v of [...eq, ...bm]) {
      if (v < minVal) minVal = v
      if (v > maxVal) maxVal = v
    }
    return {
      tooltip: { trigger: 'axis' },
      legend: { data: ['策略权益', '基准'], top: 0 },
      grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
      xAxis: { type: 'category', data: dates, axisLabel: { fontSize: 10, rotate: 30 } },
      yAxis: {
        type: 'value', scale: true, min: Math.floor(minVal * 0.95 * 100) / 100,
        axisLabel: { fontSize: 11, formatter: (v) => v.toFixed(2) },
      },
      series: [
        {
          name: '策略权益', type: 'line', data: eq, smooth: true,
          lineStyle: { color: cssVar('--up'), width: 2 },
          areaStyle: { color: 'rgba(239,68,68,0.08)' },
        },
        {
          name: '基准', type: 'line', data: bm, smooth: true,
          lineStyle: { color: '#3b82f6', width: 2, type: 'dashed' },
          areaStyle: { color: 'rgba(59,130,246,0.05)' },
        },
      ],
    }
  })
}
</script>

<style scoped>
.card { margin: 16px 32px; }
.page-desc { margin: 0 32px 12px 32px; }
.error { margin-left: 32px; margin-right: 32px; }

.mode-tabs { display: flex; gap: 0; margin-bottom: 16px; border-radius: var(--radius-sm); overflow: hidden; border: 1px solid var(--border) }
.mode-tab { flex: 1; padding: 8px 12px; border: none; background: var(--bg); color: var(--text-muted); font-size: var(--fs-sm); cursor: pointer; font-family: var(--font); transition: background .2s, color .2s }
.mode-tab.active { background: var(--primary); color: var(--on-primary); font-weight: 600 }
.mode-tab:hover:not(.active) { background: var(--card-bg); color: var(--text) }

.config-card { padding: 20px }
.config-row { display: flex; align-items: flex-end; gap: 14px; flex-wrap: wrap; margin-bottom: 14px }
.config-row label { display: flex; flex-direction: column; gap: 4px; font-size: var(--fs-sm); color: var(--text-muted); flex: 1; min-width: 120px }
.config-row select, .config-row input { width: 100% }

.direction-row, .option-row, .prefix-row, .risk-row { align-items: center }
.model-mode-grid .settings-option small, .model-mode-grid .settings-field__hint { color: var(--text-secondary); }
.settings-choice.is-paused { opacity: .55; cursor: not-allowed; }
.model-direction-hint, .option-hint { margin-top: 8px; }
.model-routing-details { margin-top: 12px; border-top: 1px solid var(--border); }
.model-routing-details > summary { min-height: 44px; display: flex; align-items: center; color: var(--text-secondary); cursor: pointer; font-size: var(--fs-sm); }
.routing-advanced { margin: 4px 0 12px; }
.risk-row .sep { color: var(--text-muted); margin: 0 8px }
.risk-item { display: flex; align-items: center; gap: 2px; font-size: var(--fs-sm); }
.risk-item .unit { color: var(--text-muted); font-size: var(--fs-xs); }
.config-details { margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border-color) }
.details-summary { cursor: pointer; color: var(--text-muted); font-size: var(--fs-sm); user-select: none }
.details-summary:hover { color: var(--text-primary) }
.section-label { font-size: var(--fs-sm); color: var(--text-muted); font-weight: 600; margin-right: 4px }

.dimension-item { font-size: var(--fs-sm); color: var(--text-secondary); display: flex; align-items: center; gap: 4px; cursor: pointer; flex: none }

.btn-run { width: 100%; padding: 10px; border: none; border-radius: var(--radius-sm); background: var(--primary); color: var(--on-primary); font-size: var(--fs-base); font-weight: 600; cursor: pointer; margin-top: 8px; font-family: var(--font) }
.btn-run:disabled { opacity: .5; cursor: not-allowed }
.btn-run:hover:not(:disabled) { background: var(--primary-hover) }
.settings-ledger .btn-run { width: auto; min-width: 180px; margin: 0; }

.progress-card { padding: 20px }
.progress-bar-bg { height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; margin-bottom: 8px }
.progress-bar { height: 100%; background: var(--primary); border-radius: 3px; transition: width .3s }
.progress-text { font-size: var(--fs-sm); color: var(--text-secondary); text-align: center }

.config-summary { padding: 10px 16px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center }
.summary-chip { font-size: var(--fs-sm); padding: 3px 10px; border-radius: 10px; background: var(--bg); border: 1px solid var(--border); color: var(--text-secondary); white-space: nowrap }
.summary-chip small { opacity: .6 }

.metrics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 12px; margin-top: 8px }
.metric-item { padding: 10px 12px; background: var(--bg); border-radius: var(--radius-sm); text-align: center }
.metric-value { font-size: var(--fs-lg); font-weight: 700; color: var(--text) }
.metric-value.up { color: var(--up) }
.metric-value.down { color: var(--down) }
.metric-label { font-size: var(--fs-xs); color: var(--text-muted); margin-top: 4px }

.summary-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 4px }
.summary-block { padding: 10px 14px; background: var(--bg); border-radius: var(--radius-sm) }
.summary-label { font-size: var(--fs-xs); color: var(--text-muted); margin-bottom: 4px }
.summary-value { font-size: var(--fs-sm); color: var(--text-secondary); word-break: break-all }

.chart-box { width: 100%; height: 400px; margin-top: 8px }

.table-wrap { padding: 0; overflow-x: auto }
.data-table { width: 100%; border-collapse: collapse; font-size: var(--fs-sm) }
.data-table th { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); color: var(--text-muted); font-weight: 600; font-size: var(--fs-xs); white-space: nowrap }
.data-table td { padding: 6px 10px; border-bottom: 1px solid rgba(42,42,80,.4); color: var(--text) }
.data-table tr:hover td { background: rgba(59,130,246,.04) }
.data-table tr.row-stop-loss td { background: rgba(239,68,68,.03) }
.top5-cell { font-size: 11px; max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap }

.regime-tag { font-size: 11px; padding: 1px 6px; border-radius: 8px }
.regime-volatile_up { color: #ef4444; background: rgba(239,68,68,.1) }
.regime-volatile_down { color: #22c55e; background: rgba(34,197,94,.1) }
.regime-volatile_flat { color: #8b5cf6; background: rgba(139,92,246,.1) }
.regime-steady_up { color: #2563eb; background: rgba(37,99,235,.1) }
.regime-steady_down { color: #6b7280; background: rgba(107,114,128,.1) }
.regime-steady_flat { color: #f59e0b; background: rgba(245,158,11,.1) }

.timing-lo { color: #f59e0b; font-weight: 600 }

.pagination { display: flex; justify-content: center; align-items: center; gap: 14px; padding: 10px 0; font-size: var(--fs-sm); color: var(--text-muted) }
.pagination button { padding: 4px 12px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--card-bg); color: var(--text); cursor: pointer; font-family: var(--font); font-size: var(--fs-sm) }
.pagination button:disabled { opacity: .4; cursor: not-allowed }
.pagination button:hover:not(:disabled) { border-color: var(--primary); color: var(--primary) }

.empty-hint { padding: 60px 20px; text-align: center }
.profile-locked .settings-section,.profile-locked .settings-group,.profile-locked .cost-section,.profile-locked .risk-section { pointer-events:none; opacity:.66; }
.locked-note { margin-bottom:12px; padding:9px 12px; border:1px solid var(--primary); border-radius:6px; color:var(--primary); font-size:var(--fs-sm); }

.lookahead-warning { padding: 14px 20px; background: rgba(245,158,11,.08); border: 1px solid rgba(245,158,11,.25); border-radius: var(--radius); margin-bottom: 12px }
.warning-title { font-weight: 700; font-size: var(--fs-base); color: #f59e0b; margin-bottom: 6px }
.warning-body { font-size: var(--fs-sm); color: var(--text-secondary); line-height: 1.6 }
.warning-detail { font-size: var(--fs-xs); color: var(--text-muted); margin-top: 6px }
.warning-detail span { margin-right: 6px }

.up { color: var(--up) }
.down { color: var(--down) }
.text-gray { color: var(--text-muted) }
.ft-xs { font-size: var(--fs-xs) }
.ft-sm { font-size: var(--fs-sm) }
.no-spin::-webkit-inner-spin-button, .no-spin::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }

/* stop mode switch */
.ltv-mode-switch { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--border); border-radius: var(--radius-sm); overflow: hidden; margin-left: 4px; }
.ltv-mode-switch button { min-height: 36px; padding: 0 12px; border: none; background: var(--card-bg); color: var(--text-secondary); cursor: pointer; border-right: 1px solid var(--border); font-family: var(--font); }
.ltv-mode-switch button:last-child { border-right: none; }
.ltv-mode--active { background: var(--primary, #3b82f6) !important; color: var(--on-primary) !important; }

@media (max-width: 760px) {
  .card, .page-desc, .error { margin-left: 16px; margin-right: 16px; }
  .risk-row { align-items: flex-start; }
  .ltv-mode-switch { width: 100%; }
  .ltv-mode-switch button { flex: 1 1 90px; border-bottom: 1px solid var(--border); }
  .settings-ledger .btn-run { width: 100%; }
}
</style>
