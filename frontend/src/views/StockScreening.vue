<template>
  <div class="screening-page">
    <h1>{{ mode === 'backtest' ? '策略回测' : mode === 'liveTrading' ? '量化调仓' : mode === 'alerts' ? '风险告警' : '选股' }}</h1>
    <div class="tabs" role="tablist" aria-label="选股工作区">
      <button role="tab" :aria-selected="mode === 'screening'" :class="['tab', { active: mode === 'screening' }]" @click="mode = 'screening'">选股</button>
      <button role="tab" :aria-selected="mode === 'backtest'" :class="['tab', { active: mode === 'backtest' }]" @click="mode = 'backtest'">策略回测</button>
      <button role="tab" :aria-selected="mode === 'liveTrading'" :class="['tab', { active: mode === 'liveTrading' }]" @click="mode = 'liveTrading'">量化调仓</button>
      <button role="tab" :aria-selected="mode === 'alerts'" :class="['tab', { active: mode === 'alerts' }]" @click="mode = 'alerts'">风险告警</button>
    </div>

    <div class="profile-control">
      <StrategyProfileBanner :profile="activeProfile" :title="mode === 'liveTrading' ? '实盘策略版本' : '当前策略版本'" />
      <label>运行版本
        <select v-model="selectedProfileKey">
          <option value="">旧版研究模式（不可晋级、不可新增买入）</option>
          <option v-for="row in selectableProfiles" :key="`${row.profile_id}-${row.version}`" :value="`${row.profile_id}/${row.version}`">
            {{ row.name }} · 第{{ row.version }}版 · {{ strategyStatusText(row.status) }}
          </option>
        </select>
      </label>
    </div>

    <template v-if="mode === 'screening'">

    <!-- 预设 -->
    <div class="preset-bar" v-if="mode === 'screening'">
      <span class="preset-label">预设:</span>
      <button v-for="p in presets" :key="p.name"
        class="preset-btn" :class="{ active: presetActive === p.name }"
        @click="loadPreset(p)">{{ p.name }}</button>
      <button class="preset-btn preset-save" @click="saveCurrentPreset">+ 保存当前</button>
    </div>

    <!-- Zone 0: 市场状态 + 策略方向 -->
    <div v-if="currentRegime" class="regime-banner" :class="'regime-' + currentRegime.regime">
      <div class="regime-main">
        <span class="regime-dot"></span>
        <span class="regime-label">{{ REGIME_LABELS[currentRegime.regime] || currentRegime.regime }}</span>
        <span class="regime-conf">置信度 {{ (currentRegime.confidence * 100).toFixed(0) }}%</span>
        <span v-if="macroAction" :class="['regime-macro', 'macro-act-' + macroAction.action]">{{ macroAction.text }}</span>
        <span class="regime-dir" v-if="currentRegime.direction">
          → {{ dirLabel(currentRegime.direction) }} ({{ currentRegime.direction_factors || 0 }}因子)
        </span>
      </div>
      <div class="regime-advice" v-if="regimeAdvice">{{ regimeAdvice }}</div>
    </div>

    <!-- 宏观指标条 -->
    <div v-if="macroOverview" class="macro-bar">
      <div class="macro-meta">
        宏观as-of {{ macroOverview.as_of || '--' }}（查询截点，不等于发布日期） · 可用 {{ macroOverview.coverage?.accepted ?? 0 }}/{{ macroOverview.coverage?.requested ?? 0 }} 项 · {{ macroOverview.data_envelope?.status === 'unavailable' ? '数据不可用' : '部分来源发布日期未记录' }}
      </div>
      <span v-if="macroOverview.indicators['制造业PMI']" class="macro-chip"
        :class="macroOverview.indicators['制造业PMI'].trend">
        PMI {{ macroOverview.indicators['制造业PMI'].value }}点 · {{ macroDateLabel(macroOverview.indicators['制造业PMI']) }} {{ macroOverview.indicators['制造业PMI'].source_date || '未知' }}
      </span>
      <span v-if="macroOverview.indicators['M2同比']" class="macro-chip"
        :class="macroOverview.indicators['M2同比'].trend">
        M2 {{ macroOverview.indicators['M2同比'].value }}% · {{ macroDateLabel(macroOverview.indicators['M2同比']) }} {{ macroOverview.indicators['M2同比'].source_date || '未知' }}
      </span>
      <span v-if="macroOverview.indicators['CPI同比']" class="macro-chip"
        :class="macroOverview.indicators['CPI同比'].trend">
        CPI {{ macroOverview.indicators['CPI同比'].value }}% · {{ macroDateLabel(macroOverview.indicators['CPI同比']) }} {{ macroOverview.indicators['CPI同比'].source_date || '未知' }}
      </span>
      <span v-if="macroOverview.indicators['LPR5年']" class="macro-chip">
        LPR5Y {{ macroOverview.indicators['LPR5年'].value }}% · {{ macroDateLabel(macroOverview.indicators['LPR5年']) }} {{ macroOverview.indicators['LPR5年'].source_date || '未知' }}
      </span>
      <span :class="['macro-chip', 'risk-' + (macroOverview.risk_level || 'unknown')]">
        宏观风险 {{ macroOverview.risk_score == null ? '未知' : (macroOverview.risk_score * 100).toFixed(0) }}{{ macroOverview.risk_score_status === 'partial' ? '（部分来源）' : '' }}
      </span>
      <span v-if="macroOverview.regime_bias" class="macro-chip macro-note">
        宏观偏向: {{ regimeBiasHint }}
      </span>
    </div>

    <div v-if="error" class="error" role="alert" tabindex="-1">{{ error }}</div>

    <!-- Zone A: 选股配置 -->
    <div class="card config-card" :class="{ 'profile-locked': activeProfile }" :aria-disabled="!!activeProfile">
      <div v-if="activeProfile" class="locked-note">关键参数由策略版本冻结。需要修改时，请前往“量化实验室 → 策略版本”复制为新草稿。</div>
      <div class="settings-stack">
        <section class="settings-section settings-section--core">
          <div class="settings-section__header">
            <div>
              <div class="settings-section__title">模型与方向</div>
              <div class="settings-section__desc">先确定排名依据，再决定由市场状态自动路由还是手动指定策略。</div>
            </div>
            <span class="settings-section__badge">核心设置</span>
          </div>
          <div class="settings-grid settings-grid--wide">
            <label class="settings-field">主预测源
              <select v-model="config.alphaSource">
                <option v-for="o in alphaSources" :key="o.value" :value="o.value">{{ o.label }}</option>
              </select>
              <span class="settings-field__hint">{{ currentAlphaHint }}</span>
            </label>
            <label class="settings-field">策略方向
              <select v-model="config.direction">
                <option value="auto">自动（跟随市场状态）</option>
                <option value="trend_momentum">趋势动量</option>
                <option value="volume_breakout">量价突破</option>
                <option value="oversold_rebound">超卖反弹</option>
                <option value="value_defense">价值防御</option>
              </select>
              <span class="settings-field__hint">{{ dirHint }}</span>
            </label>
          </div>

      <!-- Qlib模式：模型选择 -->
      <div v-if="(config.alphaSource === 'qlib' || config.alphaSource === 'regime')" class="settings-group">
        <div class="settings-grid settings-grid--wide model-mode-grid">
        <label class="settings-field">模型
          <select v-model="config.qlibModelId">
            <option :value="null">自动（按策略方向匹配，IC最优）</option>
            <option v-for="m in trainedModels.filter(m => m.label_mode === 'tradable_open_to_open' && !m.disabled && !['timing', 'regime', 'regime_v2'].includes(m.model_type) && !['timing', 'regime', 'regime_v2'].includes(m.experiment_id))" :key="m.run_id" :value="m.experiment_id + '/' + m.run_id">
              {{ m.display_name || m.model_class || '?' }} · IC {{ m.ic != null ? m.ic.toFixed(4) : '?' }}
            </option>
          </select>
          <span class="settings-field__hint">现场预测：最新交易日数据 → 下一交易日</span>
        </label>
        <label class="settings-option">
          <input type="checkbox" v-model="config.fixedModel" />
          <span>
            <strong>固定单模型</strong>
            <small>只使用上方选中的模型，不做高低波或状态模型切换。</small>
          </span>
        </label>
        <label class="settings-option">
          <input type="checkbox" :checked="config.sectorFusion && !modelRoutingPaused" :disabled="(config.alphaSource !== 'qlib' && config.alphaSource !== 'regime') || modelRoutingPaused" @change="config.sectorFusion = $event.target.checked" />
          <span>
            <strong>板块背离融合{{ modelRoutingPaused ? '（已暂停）' : '' }}</strong>
            <small>{{ modelRoutingPauseText || '用板块背离信号修正模型排序' }}</small>
          </span>
        </label>
        </div>
        <div v-if="infoModel" class="model-info-card">
          <div class="model-info-line" v-if="config.fixedModel">
            <span class="model-info-tag tag-manual">固定单模型</span>
          </div>
          <div class="model-info-line" v-else-if="!config.qlibModelId">
            <span class="model-info-tag tag-auto">{{ dirLabel(resolvedDirection) }} → 自动匹配</span>
          </div>
          <div class="model-info-line" v-else>
            <span class="model-info-tag tag-manual">手动选择</span>
          </div>
          <div class="model-info-line">
            <span class="model-info-label">模型</span>
            <span>{{ infoModel.display_name || infoModel.model_class || '?' }}</span>
          </div>
          <div class="model-info-line">
            <span class="model-info-label">因子集</span>
            <span>{{ infoModel.factor_label || infoModel.factor_set || '?' }} ({{ infoModel.instruments || '?' }})</span>
          </div>
          <div class="model-info-line">
            <span class="model-info-label">IC</span>
            <span>{{ infoModel.ic != null ? infoModel.ic.toFixed(4) : '?' }}
              <template v-if="infoModel.lgb_ic != null || infoModel.xgb_ic != null || infoModel.lstm_ic != null">
                · LGB {{ infoModel.lgb_ic?.toFixed(4) || '?' }}
                <template v-if="infoModel.xgb_ic != null"> · XGB {{ infoModel.xgb_ic.toFixed(4) }}</template>
                <template v-if="infoModel.lstm_ic != null"> · 时序网络信息系数 {{ infoModel.lstm_ic.toFixed(4) }}</template>
              </template>
            </span>
          </div>
          <div class="model-info-line">
            <span class="model-info-label">训练</span>
            <span>{{ infoModel.train_period || '?' }} · 标签{{ infoModel.label_period || '?' }}日</span>
          </div>
          <div class="model-info-line">
            <span class="model-info-label">Run ID</span>
            <span class="text-mono">{{ infoModel.experiment_id?.slice(-8) || '?' }}/{{ infoModel.run_id?.slice(-8) || '?' }}</span>
          </div>
        </div>
        <div v-if="!trainedModels.length" class="msg" style="margin-top:4px;width:100%">
          暂无可用的开盘到开盘量化学习模型。请先去<router-link to="/qlib">量化实验室</router-link>训练模型。
        </div>
        <details v-if="!config.fixedModel" class="model-routing-details">
          <summary>状态模型路由</summary>
          <StateModelMapping
            v-model="stateModelMap"
            :use-vol-switch="config.useVolSwitch"
            :models="trainedModels"
          />
        </details>
        <div v-else class="settings-field__hint model-direction-hint">状态识别仍可用于仓位和风控，但不会更换选股模型。</div>
      </div>

        </section>

        <section class="settings-section">
          <div class="settings-section__header">
            <div>
              <div class="settings-section__title">筛选范围</div>
              <div class="settings-section__desc">限定候选市场与结果规模；代码前缀不选时覆盖整个股票池。</div>
            </div>
            <span class="settings-section__badge">保留 {{ config.topN }} 只</span>
          </div>
          <div class="settings-grid">
            <label class="settings-field">股票池
              <select v-model="config.pool" @change="onPoolChange">
                <option v-for="o in poolOptions" :key="o.value" :value="o.value">{{ o.label }}</option>
              </select>
            </label>
            <label class="settings-field">保留数量
              <input v-model.number="config.topN" type="number" min="10" max="200" />
              <span class="settings-field__hint">用于最终排名展示，不改变股票池本身。</span>
            </label>
          </div>

      <!-- 代码前缀 -->
      <div class="settings-group">
        <span class="settings-group__label">代码前缀</span>
        <div class="settings-choice-row">
          <label v-for="p in CODE_PREFIX_OPTIONS" :key="p.value" class="settings-choice">
            <input type="checkbox" :value="p.value" :checked="config.codePrefixes.includes(p.value)" @change="togglePrefix(p.value)" />
            {{ p.label }}
          </label>
          <span v-if="!config.codePrefixes.length" class="settings-field__hint">当前为全部代码</span>
        </div>
      </div>

      <!-- 板块选择器 -->
      <div v-if="config.pool === 'hot_sectors'" class="sector-section">
        <div class="sector-row">
          <label>热门板块数 <input v-model.number="hotSectorCount" type="number" min="3" max="30" /></label>
          <button class="btn-sm" @click="loadHotSectors" :disabled="loadingSectors">{{ loadingSectors ? '加载中...' : '加载' }}</button>
        </div>
        <div v-if="hotSectors.length" class="sector-tags">
          <span v-for="s in hotSectors" :key="s.name" :class="['sector-tag', { selected: selectedHotSectors.includes(s.name) }]" @click="toggleHotSector(s.name)">
            {{ s.name }} <small :class="(s.change ?? 0) > 0 ? 'up' : 'down'">{{ (s.change ?? 0) > 0 ? '+' : '' }}{{ (s.change ?? 0).toFixed(1) }}%</small>
          </span>
        </div>
        <div v-if="selectedHotSectors.length" class="sector-selected">已选 {{ selectedHotSectors.length }} 个板块</div>
      </div>

      <div v-if="config.pool === 'custom'" class="sector-section">
        <div class="sector-row">
          <input v-model="sectorSearch" placeholder="搜索行业板块..." class="sector-search" />
          <button class="btn-sm" @click="loadSectors" :disabled="loadingSectors">{{ loadingSectors ? '...' : '加载行业板块' }}</button>
        </div>
        <div v-if="filteredSectors.length" class="sector-list">
          <label v-for="s in filteredSectors" :key="s.name" class="sector-item">
            <input type="checkbox" :checked="selectedSectors.includes(s.name)" @change="toggleSector(s.name, 'custom')" />
            {{ s.name }} <span class="sector-count">{{ s.stock_count || '?' }}只</span>
          </label>
        </div>
        <div v-if="selectedSectors.length" class="sector-selected">已选 {{ selectedSectors.length }} 个板块</div>
      </div>

      <div v-if="config.pool === 'concept'" class="sector-section">
        <div class="sector-row">
          <input v-model="sectorSearch" placeholder="搜索概念板块..." class="sector-search" />
          <button class="btn-sm" @click="loadConceptSectors" :disabled="loadingSectors">{{ loadingSectors ? '...' : '加载概念板块' }}</button>
        </div>
        <div v-if="filteredConceptSectors.length" class="sector-list">
          <label v-for="s in filteredConceptSectors" :key="s.name" class="sector-item">
            <input type="checkbox" :checked="selectedConceptSectors.includes(s.name)" @change="toggleSector(s.name, 'concept')" />
            {{ s.name }} <span class="sector-count">{{ s.stock_count || '?' }}只</span>
          </label>
        </div>
        <div v-if="selectedConceptSectors.length" class="sector-selected">已选 {{ selectedConceptSectors.length }} 个板块</div>
      </div>
        </section>

      <!-- 辅助参考维度（仅展示） -->
        <details class="settings-section settings-disclosure">
          <summary>
            <span>增强与展示</span>
            <span class="settings-section__badge">{{ enhancementCount }} 项启用</span>
          </summary>
          <div class="settings-disclosure__body">
      <div class="aux-section settings-group">
        <span class="section-label">辅助参考（仅展示，不影响排序）</span>
        <div class="settings-choice-row">
          <label v-for="a in auxDimensions" :key="a.key" class="settings-choice">
            <input type="checkbox" v-model="auxVisible[a.key]" />
            {{ a.label }}
          </label>
        </div>
      </div>

      <!-- AI复核 + 宏观偏好 -->
      <div class="settings-options settings-group">
        <label class="settings-option">
          <input type="checkbox" v-model="config.useDebate" />
          <span><strong>智能深度复核</strong><small>对前5名进行多模型复核，约1至3分钟。</small></span>
        </label>
        <label class="settings-option">
          <input type="checkbox" v-model="config.neutralizeQlib" />
          <span><strong>市值/行业中性化</strong><small>属于量化预测分处理，会改变模型内部排序；诊断基线默认关闭。</small></span>
        </label>
        <label class="settings-option">
          <input type="checkbox" v-model="config.macroBias" />
          <span><strong>宏观行业加分</strong><small>对宏观利好行业统一加分，会改变量化模型排名；默认关闭。</small></span>
        </label>
        <label class="settings-option">
          <input type="checkbox" v-model="config.decisionDayFilter" />
          <span><strong>决策日硬过滤并补足目标数量</strong><small>按排名剔除特别处理、停牌、涨停、低流动性、低价和次新股，再向后补足。</small></span>
        </label>
        <label class="settings-option">
          <input type="checkbox" v-model="config.macroRegime" />
          <span><strong>市场状态置信度宏观修正</strong><small>只调整状态置信度，不给个股加分，也不改变排序。</small></span>
        </label>
        <label class="settings-option">
          <input type="checkbox" v-model="config.smoothRegime" />
          <span><strong>状态平滑（CRF）</strong><small>抑制市场状态在相邻交易日频繁跳变；默认关闭。</small></span>
        </label>
        <label class="settings-option">
          <input type="checkbox" :checked="config.useVolSwitch && !config.fixedModel" :disabled="config.fixedModel" @change="config.useVolSwitch = $event.target.checked" />
          <span><strong>波动率切换{{ config.fixedModel ? '（已暂停）' : '' }}</strong><small>{{ config.fixedModel ? '固定单模型期间不参与模型选择' : '高波使用超卖反弹，低波使用趋势动量。' }}</small></span>
        </label>
      </div>
          </div>
        </details>
      </div>

      <div class="settings-ledger">
        <div class="settings-ledger__summary">
          <span class="settings-ledger__label">本次运行</span>
          <span class="settings-ledger__chip">{{ currentAlphaLabel }}</span>
          <span class="settings-ledger__chip">{{ config.fixedModel ? '固定单模型' : (config.direction === 'auto' ? '自动路由' : dirLabel(config.direction)) }}</span>
          <span class="settings-ledger__chip">{{ poolOptions.find(o => o.value === config.pool)?.label || config.pool }}</span>
          <span class="settings-ledger__chip">Top {{ config.topN }}</span>
          <span v-if="config.useVolSwitch && !config.fixedModel" class="settings-ledger__chip">波动率切换</span>
          <span v-if="config.neutralizeQlib" class="settings-ledger__chip">市值/行业中性化</span>
          <span v-if="config.macroBias" class="settings-ledger__chip">宏观行业加分</span>
          <span v-if="config.macroRegime" class="settings-ledger__chip">状态置信度宏观修正</span>
          <span v-if="config.decisionDayFilter" class="settings-ledger__chip">决策日硬过滤</span>
        </div>
        <button class="btn-run" :disabled="running" @click="startScreening">
          {{ running ? '运行中...' : '开始选股' }}
        </button>
      </div>
    </div>

    <!-- Zone B: 进度 -->
    <div v-if="running" class="card progress-card">
      <div class="card-title">筛选进度</div>
      <div class="stepper">
        <div v-for="s in stages" :key="s.key" :class="['step', { done: s.done, active: s.active }]">
          <span class="step-dot">{{ s.done ? '✓' : s.active ? '●' : '○' }}</span>
          <span class="step-label">{{ s.label }}</span>
        </div>
      </div>
      <div class="progress-bar-bg"><div class="progress-bar" :style="{ width: progressPct + '%' }"></div></div>
      <div class="progress-text">{{ progressText }}</div>
    </div>

    <!-- Zone C: 结果 -->
    <div v-if="screeningResult?.vol_switch_empty" class="card empty-hint" role="status">
      {{ screeningResult.message || '中波区间，当前保持空仓，不生成买入候选。' }}
    </div>
    <template v-else-if="screeningResult?.candidates?.length">
      <!-- 筛选元数据 -->
      <div v-if="screeningResult.prediction || screeningResult.direction || screeningResult.model_routing || screeningResult.score_processing || screeningResult.style || screeningResult.etf_flow_bias || screeningResult.pool_info?.top_sectors || screeningResult.pool_info?.top_concepts" class="screening-meta card" role="status" aria-live="polite">
        <span v-if="screeningResult.prediction" :class="['meta-tag', 'freshness', { stale: !screeningResult.prediction.fresh }]">
          模型数据：{{ screeningResult.prediction.prediction_date }}
        </span>
        <span v-if="screeningResult.prediction" class="meta-tag direction">
          决策日：{{ screeningResult.prediction.decision_date }}
        </span>
        <span v-if="screeningResult.prediction?.model_id" class="meta-tag direction" :title="screeningResult.prediction.model_id">
          实际模型：{{ screeningResult.prediction.model_id.split('/').pop().slice(0, 8) }}
        </span>
        <span v-if="screeningResult.prediction?.coverage_rate != null" class="meta-tag direction">
          模型覆盖：{{ (screeningResult.prediction.coverage_rate * 100).toFixed(2) }}%
          ({{ screeningResult.prediction.coverage_count }}/{{ screeningResult.prediction.universe_count }})
        </span>
        <span
          v-for="([source, coverage]) in Object.entries(screeningResult.auxiliary_coverage || {}).filter(([, value]) => value.rate < 0.5)"
          :key="source"
          class="meta-tag coverage-warning"
        >
          ⚠ {{ AUX_COVERAGE_LABELS[source] || source }}覆盖：{{ (coverage.rate * 100).toFixed(1) }}%
          ({{ coverage.covered }}/{{ coverage.total }})
        </span>
        <span v-if="screeningResult.model_routing && !screeningResult.prediction?.model_id" class="meta-tag direction">
          实际模型：{{ screeningResult.model_routing.state }} → {{ screeningResult.model_routing.run_id?.slice(0, 8) }}
        </span>
        <span v-else-if="screeningResult.model_routing" class="meta-tag direction">
          路由状态：{{ screeningResult.model_routing.state }}
        </span>
        <span v-if="screeningResult.score_processing" class="meta-tag direction">
          市值/行业中性化：{{ neutralizeStatusText(screeningResult.score_processing) }}
        </span>
        <span v-if="screeningResult.score_processing" class="meta-tag direction">
          排序：量化模型主分；辅助指标不参与排名
        </span>
        <span v-if="screeningResult.score_processing?.macro_sector_bias" class="meta-tag direction">
          宏观行业加分：开
        </span>
        <span v-if="screeningResult.score_processing?.macro_regime" class="meta-tag direction">
          状态置信度宏观修正：开
        </span>
        <span v-if="screeningResult.score_processing?.decision_day_filter" class="meta-tag direction">
          决策日硬过滤：开
        </span>
        <span v-if="screeningResult.direction" class="meta-tag direction">
          🧭 方向精选：{{ DIRECTION_LABELS[screeningResult.direction] || screeningResult.direction }}
        </span>
        <span v-if="screeningResult.style" class="meta-tag style">
          📐 {{ screeningResult.style.style === 'large_cap' ? '大盘风格' : '小盘风格' }}
          ({{ screeningResult.style.dominant_index }}
          RSRS={{ screeningResult.style.score?.toFixed(4) }}
          <template v-if="screeningResult.style.adjustments">
            {{ Object.entries(screeningResult.style.adjustments).map(([k,v]) => k+'×'+v).join(' ') }}
          </template>)
        </span>
        <span v-if="screeningResult.etf_flow_bias" class="meta-tag etf">
          💰 ETF流入板块：{{ Object.keys(screeningResult.etf_flow_bias).slice(0, 5).join('、') }}
        </span>
        <span v-if="screeningResult.pool_info?.top_sectors" class="meta-tag rotation">
          Top行业：{{ screeningResult.pool_info.top_sectors.map(s => s[0] + '(' + s[1]?.toFixed(2) + ')').join(' → ') }}
        </span>
        <span v-if="screeningResult.pool_info?.top_concepts" class="meta-tag rotation">
          🔥 Top概念：{{ screeningResult.pool_info.top_concepts.map(s => s[0] + '(' + s[1]?.toFixed(2) + ')').join(' → ') }}
        </span>
      </div>
      <!-- 结果 Tab -->
      <div class="card result-card">
        <div class="result-tabs">
          <button :class="['result-tab', { active: activeResultTab === 'main' }]" @click="activeResultTab = 'main'">
            {{ currentAlphaLabel }}排序
          </button>
          <button v-if="config.useDebate && debateResults.length" :class="['result-tab', { active: activeResultTab === 'debate' }]" @click="activeResultTab = 'debate'">
            AI复核 ({{ debateResults.length }})
          </button>
          <button :class="['result-tab', { active: activeResultTab === 'combined' }]" @click="activeResultTab = 'combined'">
            辅助参考
          </button>
          <div class="result-tab-spacer"></div>
          <button class="btn-save" @click="openSaveDialog" :disabled="saving || screeningResult.status !== 'done' || !screeningResult.result_id">💾 保存此选股结果</button>
        </div>

        <!-- 主排序表 -->
        <div v-if="activeResultTab === 'main'" class="table-wrap">
          <div v-if="fusionSummary" class="fusion-summary">🔀 {{ fusionSummary }}</div>
          <table class="data-table">
            <thead>
              <tr>
                <th>#</th><th>代码</th><th>名称</th>
                <th class="sort-col" title="截面相对排名分，不是预期收益率">相对排名分</th>
                <th v-if="auxVisible.fundamental">基本面</th>
                <th v-if="auxVisible.flow" title="评分-1~1：近5日主力净流入占成交额比均值(5%≈满分)+趋势修正；后面为近5日主力净流入合计金额">资金流向</th>
                <th v-if="auxVisible.north_flow">北向资金</th>
                <th v-if="auxVisible.sentiment" title="近7日新闻情绪均值×置信度+30日趋势修正，范围-1~1：>0.2偏多 / -0.1~0.1中性 / <-0.2偏空；置信度<0.35不计分">情绪</th>
                <th v-if="auxVisible.etf_flow" title="行业ETF近5日资金净流入加分">ETF流</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(c, i) in paginatedMain" :key="c.code" @click="toggleExpand(c.code)">
                <td>{{ (page - 1) * pageSize + i + 1 }}</td>
                <td>{{ c.code }}</td>
                <td>{{ c.name }}
                  <span v-if="c.sector_divergent" class="tag tag-divergent" :title="qlibAttrText(c)">🔀板块逆势</span>
                  <span v-if="c.model_strong" class="tag tag-escort">⚡{{ dirLabel(c.model_strong) }}强推</span>
                </td>
                <td :class="mainScore(c) > 0 ? 'up' : 'down'" :title="scoreTraceText(c)">{{ mainScore(c).toFixed(3) }}</td>
                <td v-if="auxVisible.fundamental" :class="(c.fundamental_score||0) > 0 ? 'up' : 'down'">{{ c.fundamental_score?.toFixed(2) || '--' }}</td>
                <td v-if="auxVisible.flow" :class="(c.flow_score||0) > 0 ? 'up' : 'down'">{{ c.flow_score?.toFixed(2) || '--' }}<small v-if="c.flow_net_5d != null" class="flow-amt">{{ fmtFlowAmt(c.flow_net_5d) }}</small></td>
                <td v-if="auxVisible.north_flow" :class="(c.north_flow_score||0) > 0 ? 'up' : 'down'">{{ c.north_flow_score?.toFixed(2) || '--' }}</td>
                <td v-if="auxVisible.sentiment" :class="(c.sentiment_score||0) > 0 ? 'up' : 'down'">{{ c.sentiment_score?.toFixed(2) || '--' }}<small v-if="c.sentiment_score != null" class="sent-label">{{ sentLabel(c.sentiment_score) }}</small></td>
                <td v-if="auxVisible.etf_flow" :class="(c.etf_flow_score||0) > 0 ? 'up' : 'down'">{{ c.etf_flow_score?.toFixed(2) || '--' }}</td>
                <td><button class="btn-sm" @click.stop="goStock(c.code)">详情</button></td>
              </tr>
            </tbody>
          </table>
          <div v-if="totalPages > 1" class="pagination">
            <button :disabled="page <= 1" @click="page--">上一页</button>
            <span>{{ page }} / {{ totalPages }}</span>
            <button :disabled="page >= totalPages" @click="page++">下一页</button>
          </div>
        </div>

        <!-- AI复核 Tab -->
        <div v-if="activeResultTab === 'debate'" class="table-wrap">
          <table class="data-table">
            <thead>
              <tr><th>#</th><th>代码</th><th>名称</th><th>机会评分</th><th>风险评分</th><th>推荐等级</th><th>操作</th></tr>
            </thead>
            <tbody>
              <template v-for="(c, i) in debateResults" :key="c.code">
                <tr @click="toggleExpand(c.code)" style="cursor:pointer">
                  <td>{{ i + 1 }}</td>
                  <td>{{ c.code }}</td>
                  <td>{{ c.name }}</td>
                  <td :class="(c.opportunity_score||0) >= 7 ? 'up' : ''">{{ c.opportunity_score || '--' }}/10</td>
                  <td :class="(c.risk_score||0) <= 3 ? 'down' : (c.risk_score||0) >= 7 ? 'up' : ''">{{ c.risk_score || '--' }}/10</td>
                  <td><span :class="['tag', levelClass(c.recommendation_level)]">{{ c.recommendation_level || '--' }}</span></td>
                  <td><button class="btn-sm" @click.stop="goStock(c.code)">详情</button></td>
                </tr>
                <tr v-if="expandedCode === c.code" class="expand-row">
                  <td :colspan="7">
                    <div class="debate-detail">
                      <div class="debate-header">
                        <span class="debate-title">📋 AI 复核完整报告</span>
                        <button class="btn-sm" @click.stop="expandedCode = null">收起 ▲</button>
                      </div>
                      <div v-if="c.debate_detail?.final" class="debate-section">
                        <h4>🏛️ 最终决议</h4>
                        <pre class="debate-text">{{ c.debate_detail.final }}</pre>
                      </div>
                      <div v-if="c.debate_summary && !c.debate_detail?.final" class="debate-section">
                        <h4>🏛️ 最终决议</h4>
                        <pre class="debate-text">{{ c.debate_summary }}</pre>
                      </div>
                      <div v-if="c.debate_detail?.agents?.length" class="debate-section">
                        <h4>🤖 各Agent分析报告</h4>
                        <div v-for="(ag, ai) in c.debate_detail.agents" :key="ai" class="agent-block">
                          <h5>{{ ag.name }}</h5>
                          <pre class="debate-text">{{ ag.content }}</pre>
                        </div>
                      </div>
                      <div v-if="c.signal_breakdown" class="debate-section">
                        <h4>📊 信号分解</h4>
                        <div class="signal-grid">
                          <div v-for="(v, k) in c.signal_breakdown" :key="k" class="signal-item">
                            <span class="signal-name">{{ k }}</span>
                            <span :class="['signal-dir', v.direction === 'bullish' ? 'up' : v.direction === 'bearish' ? 'down' : '']">{{ v.direction }}</span>
                            <span class="signal-score">{{ v.score }}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </td>
                </tr>
              </template>
            </tbody>
          </table>
        </div>

        <!-- 辅助参考 Tab -->
        <div v-if="activeResultTab === 'combined'" class="table-wrap">
          <p class="combined-hint">保持量化模型原始名次，仅展开基本面、资金流向、北向资金、情绪和交易型开放式指数基金资金流；这些字段不加权、不替换、不重排。</p>
          <table class="data-table">
            <thead>
              <tr>
                <th>#</th><th>代码</th><th>名称</th><th title="截面相对排名分，不是预期收益率">相对排名分</th>
                <th v-if="auxVisible.fundamental">基本面</th>
                <th v-if="auxVisible.flow" title="评分-1~1：近5日主力净流入占成交额比均值(5%≈满分)+趋势修正；后面为近5日主力净流入合计金额">资金流向</th>
                <th v-if="auxVisible.north_flow">北向</th>
                <th v-if="auxVisible.sentiment" title="近7日新闻情绪均值×置信度+30日趋势修正，范围-1~1：>0.2偏多 / -0.1~0.1中性 / <-0.2偏空；置信度<0.35不计分">情绪</th>
                <th v-if="auxVisible.etf_flow" title="行业ETF近5日资金净流入加分">ETF流</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(c, i) in paginatedCombined" :key="c.code" @click="toggleExpand(c.code)">
                <td>{{ (page - 1) * pageSize + i + 1 }}</td>
                <td>{{ c.code }}</td>
                <td>{{ c.name }}</td>
                <td :class="c.l0_score > 0 ? 'up' : 'down'">{{ c.l0_score?.toFixed(3) || '--' }}</td>
                <td v-if="auxVisible.fundamental" :class="(c.fundamental_score||0) > 0 ? 'up' : 'down'">{{ c.fundamental_score?.toFixed(2) || '--' }}</td>
                <td v-if="auxVisible.flow" :class="(c.flow_score||0) > 0 ? 'up' : 'down'">{{ c.flow_score?.toFixed(2) || '--' }}<small v-if="c.flow_net_5d != null" class="flow-amt">{{ fmtFlowAmt(c.flow_net_5d) }}</small></td>
                <td v-if="auxVisible.north_flow" :class="(c.north_flow_score||0) > 0 ? 'up' : 'down'">{{ c.north_flow_score?.toFixed(2) || '--' }}</td>
                <td v-if="auxVisible.sentiment" :class="(c.sentiment_score||0) > 0 ? 'up' : 'down'">{{ c.sentiment_score?.toFixed(2) || '--' }}<small v-if="c.sentiment_score != null" class="sent-label">{{ sentLabel(c.sentiment_score) }}</small></td>
                <td v-if="auxVisible.etf_flow" :class="(c.etf_flow_score||0) > 0 ? 'up' : 'down'">{{ c.etf_flow_score?.toFixed(2) || '--' }}</td>
                <td><button class="btn-sm" @click.stop="goStock(c.code)">详情</button></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- 保存对话框 -->
      <div v-if="showSaveDialog" class="modal-overlay" @click.self="showSaveDialog = false">
        <div class="modal-card">
          <h3>保存选股结果</h3>
          <p class="modal-desc">保存后可在「量化调仓」中使用此排名进行调仓</p>
          <label class="modal-label">名称</label>
          <input v-model="saveName" class="modal-input" placeholder="例如：2026-01-15 周调仓选股" @keyup.enter="doSave" />
          <label class="modal-label">备注（可选）</label>
          <input v-model="saveNote" class="modal-input" placeholder="选股参数说明..." />
          <div v-if="saveMessage" :class="['modal-msg', saveError ? 'msg-error' : 'msg-ok']">{{ saveMessage }}</div>
          <div class="modal-actions">
            <button class="btn-cancel" @click="showSaveDialog = false">取消</button>
            <button class="btn-confirm" @click="doSave" :disabled="saving || !saveName.trim()">
              {{ saving ? '保存中...' : '确认保存' }}
            </button>
          </div>
        </div>
      </div>
    </template>

    <div v-if="!running && !screeningResult && !error" class="card empty-hint">
      <AppEmpty text="配置参数后点击「开始选股」" />
    </div>
    </template>

    <!-- ═══ 量化调仓 ═══ -->
    <template v-if="mode === 'liveTrading'">
      <LiveTradingView />
    </template>

    <!-- ═══ 策略回测 ═══ -->
    <template v-if="mode === 'backtest'">
      <ScreeningBacktestView :profile="activeProfile" />
    </template>

    <!-- ═══ 风险告警 ═══ -->
    <template v-if="mode === 'alerts'">
      <AlertView />
    </template>

  </div>
</template>

<script setup>
import { ref, reactive, computed, watch, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { useAsyncTaskTracker } from '../stores/asyncTaskTracker.js'
import { POOL_OPTIONS, SIGNAL_LABELS, REGIME_LABELS, CODE_PREFIX_OPTIONS } from '../constants/index.js'
import AppEmpty from '../components/AppEmpty.vue'
import StrategyProfileBanner from '../components/StrategyProfileBanner.vue'
import StateModelMapping from '../components/StateModelMapping.vue'
import ScreeningBacktestView from './ScreeningBacktestView.vue'
import LiveTradingView from './LiveTradingView.vue'
import { todayStr } from '../utils/date.js'
import AlertView from './AlertView.vue'
import { runScreening, getScreeningResult, getLatestScreening, getHotSectors, getAvailableSectors, getConceptSectors } from '../api/screening.js'
import { saveScreening } from '../api/rebalance.js'
import { predictTiming, getBreadth, trainTiming, getTrainStatus } from '../api/timing.js'
import api from '../api'
import { listStrategyProfiles, getChampionProfile } from '../api/strategyProfiles.js'
import { stateModelsPayload } from '../utils/modelRouting.js'
import { strategyStatusText } from '../utils/strategyTerms.js'

const router = useRouter()

const DIRECTION_LABELS = {
  trend_momentum: '趋势动量',
  volume_breakout: '量价突破',
  oversold_rebound: '超卖反弹',
  value_defense: '价值防御',
}

const AUX_COVERAGE_LABELS = {
  industry: '行业/板块名',
  fundamental: '基本面',
  flow: '资金流向',
  north_flow: '北向资金',
  sentiment: '情绪',
  etf_flow: 'ETF板块资金',
}

const mode = ref('screening')

const alphaSources = [
  { value: 'qlib', label: '量化模型预测', hint: '量化模型是唯一排序依据；目标名单产生后再计算辅助参考' },
  { value: 'regime', label: '市场状态模型路由', hint: '市场状态只负责选择量化模型，最终仍只按量化预测分排序' },
]

const auxDimensions = [
  { key: 'fundamental', label: '基本面', scoreKey: 'fundamental_score' },
  { key: 'flow', label: '资金流向', scoreKey: 'flow_score' },
  { key: 'north_flow', label: '北向资金', scoreKey: 'north_flow_score' },
  { key: 'sentiment', label: '情绪分析', scoreKey: 'sentiment_score' },
  { key: 'etf_flow', label: 'ETF资金流', scoreKey: 'etf_flow_score' },
]

const poolOptions = [
  ...POOL_OPTIONS,
  { value: 'hot_sectors', label: '热门板块' },
  { value: 'custom', label: '自定义行业板块' },
  { value: 'concept', label: '自定义概念板块' },
]

// --- 状态 ---
const config = reactive({ alphaSource: 'qlib', topN: 50, pool: 'stable_history_500d', useDebate: false, neutralizeQlib: false, macroBias: false, macroRegime: false, decisionDayFilter: true, smoothRegime: false, useVolSwitch: false, qlibModelId: null, fixedModel: false, codePrefixes: ['600,601,603,605', '000,001,002,003'], direction: 'auto', sectorFusion: false })
const strategyProfiles = ref([])
const selectedProfileKey = ref('')
const selectableProfiles = computed(() => strategyProfiles.value.filter(row => {
  if (mode.value === 'liveTrading') return row.status === 'champion'
  return ['challenger', 'paper', 'champion'].includes(row.status) || row.config?.factors?.research_only
}))
const activeProfile = computed(() => {
  const [id, version] = selectedProfileKey.value.split('/')
  return selectableProfiles.value.find(row => row.profile_id === id && String(row.version) === version) || null
})

// ── 预设 ──
const PRESET_KEY = 'screening_presets'
const DEFAULT_PRESETS = [
  { name: '沪深300精选', config: { pool: 'csi300', topN: 20, alphaSource: 'qlib', direction: 'auto', macroBias: false, sectorFusion: true, useDebate: false } },
  { name: '全市场动量', config: { pool: 'all', topN: 50, alphaSource: 'qlib', direction: 'trend_momentum', macroBias: false, sectorFusion: true, useDebate: false } },
  { name: '中证500波段', config: { pool: 'csi500', topN: 30, alphaSource: 'qlib', direction: 'volume_breakout', macroBias: false, sectorFusion: false, useDebate: false } },
]
const presetActive = ref('')

function loadPresets() {
  try {
    const raw = localStorage.getItem(PRESET_KEY)
    return raw ? JSON.parse(raw) : [...DEFAULT_PRESETS]
  } catch { return [...DEFAULT_PRESETS] }
}
const presets = ref(loadPresets())

function loadPreset(p) {
  const cfg = p.config
  Object.keys(cfg).forEach(k => { if (k in config) config[k] = cfg[k] })
  if (config.alphaSource !== 'regime') config.alphaSource = 'qlib'
  presetActive.value = p.name
}

function saveCurrentPreset() {
  const name = prompt('预设名称 (如 "我的周频策略"):')
  if (!name) return
  const snapshot = { ...config }
  const list = presets.value.filter(p => p.name !== name)
  list.push({ name, config: snapshot })
  presets.value = list
  presetActive.value = name
  localStorage.setItem(PRESET_KEY, JSON.stringify(list))
}
const auxVisible = reactive({ fundamental: true, flow: true, north_flow: true, sentiment: true, etf_flow: true })

const currentRegime = ref(null)
const macroOverview = ref(null)
const trainedModels = ref([])
const stateModelMap = ref({})
const error = ref('')
const running = ref(false)
const screeningResult = ref(null)
const jobId = ref('')
const tracker = useAsyncTaskTracker()
const expandedCode = ref(null)
const showSaveDialog = ref(false)
const saveName = ref('')
const saveNote = ref('')
const saving = ref(false)
const saveMessage = ref('')
const saveError = ref(false)
const activeResultTab = ref('main')
const page = ref(1)
const pageSize = 100

// --- 择时信号状态 ---
const timingSignal = ref(null)
const timingBreadth = ref(null)
const timingLoading = ref(false)
const timingTraining = ref(false)
const timingTrainJobId = ref('')
let timingTrainPollTimer = null
let timingTrainPollCount = 0

// --- 择时信号 ---
async function refreshTiming() {
  timingLoading.value = true
  try {
    const [sig, br] = await Promise.allSettled([predictTiming(), getBreadth()])
    if (sig.status === 'fulfilled') timingSignal.value = sig.value?.data ?? sig.value
    if (br.status === 'fulfilled') timingBreadth.value = br.value?.data ?? br.value
  } catch (e) { console.error('[Timing] refresh 异常:', e) }
  timingLoading.value = false
}

async function startTimingTraining() {
  timingTraining.value = true; timingTrainPollCount = 0
  try {
    const res = await trainTiming()
    const jid = res?.data?.job_id || res?.job_id
    if (jid) { timingTrainJobId.value = jid; pollTimingTrain() }
    else { timingTraining.value = false }
  } catch (e) {
    alert('训练失败: ' + (e?.message || String(e)))
    timingTraining.value = false
  }
}

async function pollTimingTrain() {
  if (!timingTrainJobId.value) { timingTraining.value = false; return }
  try {
    const r = await getTrainStatus(timingTrainJobId.value)
    const data = r?.data || r
    if (data.status === 'completed') { timingTraining.value = false; refreshTiming() }
    else if (data.status === 'failed') { alert('训练失败: ' + (data.error || '未知')); timingTraining.value = false }
    else { if (timingTrainPollTimer) clearTimeout(timingTrainPollTimer)
      timingTrainPollCount++
      const delay = timingTrainPollCount <= 3 ? 3000 : timingTrainPollCount <= 6 ? 5000 : timingTrainPollCount <= 10 ? 8000 : 15000
      timingTrainPollTimer = setTimeout(pollTimingTrain, delay) }
  } catch (e) { timingTraining.value = false }
}

// 板块
const hotSectors = ref([])
const selectedHotSectors = ref([])
const hotSectorCount = ref(10)
const sectors = ref([])
const conceptSectors = ref([])
const selectedSectors = ref([])
const selectedConceptSectors = ref([])
const sectorSearch = ref('')
const loadingSectors = ref(false)

let pollTimer = null
let pollCount = 0
let pollToken = 0
onMounted(async () => {
  try {
    if (sessionStorage.getItem('rebalance-plan-commit:LOCAL-LIVE')) mode.value = 'liveTrading'
  } catch {}
  const task = tracker.getTask('screening')
  const resumeJobId = task?.status === 'running' && task.jobId ? String(task.jobId) : ''
  if (resumeJobId) {
    jobId.value = resumeJobId
    running.value = true
    error.value = ''
    pollResult()
  }
  try {
    const [latest, regime, macro] = await Promise.all([
      resumeJobId ? Promise.resolve(null) : getLatestScreening().catch(() => null),
      api.get('/regime/current').catch(() => null),
      api.get('/macro/overview').catch(() => null),
    ])
    if (latest && !resumeJobId && !running.value) screeningResult.value = latest.data || latest
    if (regime) currentRegime.value = regime.data || regime
    if (macro) macroOverview.value = macro.data || macro
  } catch (e) { /* ignore */ }
  loadModels()
  refreshTiming()
  try {
    const [rows, active] = await Promise.all([listStrategyProfiles(), getChampionProfile()])
    strategyProfiles.value = rows?.data ?? rows ?? []
    const champion = active?.data ?? active
    if (champion) selectedProfileKey.value = `${champion.profile_id}/${champion.version}`
  } catch (e) { console.warn('[Profile] 加载失败:', e.message) }
})
onUnmounted(() => { pollToken++; if (pollTimer) clearTimeout(pollTimer); if (timingTrainPollTimer) clearTimeout(timingTrainPollTimer) })

async function loadModels() {
  try {
    const r = await api.get('/qlib/trained-models')
    trainedModels.value = (r?.data || r) || []
    if (config.qlibModelId) {
      const eligible = trainedModels.value.some(model =>
        model.label_mode === 'tradable_open_to_open' &&
        `${model.experiment_id}/${model.run_id}` === config.qlibModelId)
      if (!eligible) config.qlibModelId = null
    }
  } catch (e) { /* ignore */ }
}

// --- 计算属性 ---
const currentAlphaHint = computed(() => alphaSources.find(a => a.value === config.alphaSource)?.hint || '')
const currentAlphaLabel = computed(() => alphaSources.find(a => a.value === config.alphaSource)?.label || '')
const enhancementCount = computed(() => [
  config.useDebate,
  config.neutralizeQlib,
  config.macroBias,
  config.decisionDayFilter,
  config.macroRegime,
  config.smoothRegime,
  config.useVolSwitch,
].filter(Boolean).length)

const macroAction = computed(() => {
  const m = currentRegime.value?.details?.macro
  if (!m?.action) return null
  const labels = { volatile_up: '上涨', volatile_down: '下跌', volatile_flat: '高波动', steady_up: '上涨', steady_down: '下跌', steady_flat: '震荡', trending_up: '上涨', trending_down: '下跌', ranging: '震荡', volatile: '高波动' }
  if (m.action === 'confirm') return { action: 'confirm', text: '✓ 宏观确认' }
  const favor = m.macro_favor || (m.bias && Object.keys(m.bias).length ? Object.entries(m.bias).sort((a, b) => b[1] - a[1])[0][0] : null)
  if (m.action === 'weaken') return { action: 'weaken', text: `⚠ 宏观弱化 · 置信度×0.85` }
  if (m.action === 'weaken_strong') return { action: 'weaken_strong', text: `⚠⚠ 宏观强弱化 · 置信度×0.65 · 宏观偏向${labels[favor] || '?'}` }
  if (m.action === 'flip') return { action: 'weaken_strong', text: `⚠⚠ 宏观背离 · 技术面为${labels[m.from] || m.from}` }
  return null
})

const resolvedDirection = computed(() => {
  if (config.direction === 'auto') return currentRegime.value?.direction || null
  return config.direction
})

const autoMatchedModel = computed(() => {
  if ((config.alphaSource !== 'qlib' && config.alphaSource !== 'regime') || config.qlibModelId || !resolvedDirection.value) return null
  const matched = trainedModels.value.filter(m =>
    m.label_mode === 'tradable_open_to_open' &&
    m.factor_label === `精选:${resolvedDirection.value}` && !m.disabled)
  if (!matched.length) return null
  return matched.reduce((best, m) => ((m.ic ?? -Infinity) > (best.ic ?? -Infinity) ? m : best))
})

const infoModel = computed(() => {
  if ((config.alphaSource !== 'qlib' && config.alphaSource !== 'regime') || !trainedModels.value.length) return null
  const mid = config.qlibModelId
  if (mid) {
    return trainedModels.value.find(m => m.experiment_id + '/' + m.run_id === mid) || null
  }
  return autoMatchedModel.value || null
})

const modelRoutingPaused = computed(() => config.useVolSwitch || config.fixedModel)
const modelRoutingPauseText = computed(() => {
  if (config.fixedModel) return '固定单模型开启时不可引入其他方向模型'
  if (config.useVolSwitch) return '波动率切换开启时不可同时使用'
  return ''
})

const fusionMetaText = computed(() => {
  if (config.alphaSource !== 'qlib' || !config.sectorFusion || modelRoutingPaused.value) return ''
  return '背离板块股票用对应方向模型融合打分（每个方向约+4分钟）'
})

const fusionSummary = computed(() => {
  const sf = screeningResult.value?.sector_fusion
  if (!sf?.divergent || !Object.keys(sf.divergent).length) return ''
  const labels = { volatile_up: '↑', volatile_down: '↓', volatile_flat: '⚡', steady_up: '↑', steady_down: '↓', steady_flat: '〰', trending_up: '↑', trending_down: '↓', ranging: '〰', volatile: '⚡' }
  const parts = Object.entries(sf.divergent).map(([ind, d]) =>
    `${ind.replace(/^[A-Z]\d+/, '')}${labels[d.regime] || ''}(${dirLabel(d.direction)})`)
  return `检测到 ${parts.length} 个背离板块，双模型融合：${parts.join('、')}`
})

function qlibAttrText(c) {
  const a = c.qlib_attr
  if (!a) return ''
  return `板块逆势: ${a.sector} ${a.sector_regime} conf=${a.confidence}\n` +
    `融合 = 市场模型 ${((1 - a.w_sector) * 100).toFixed(0)}% × ${a.market_pct} + ` +
    `${dirLabel(a.direction)}模型 ${(a.w_sector * 100).toFixed(0)}% × ${a.sector_pct} − 分歧惩罚 ${a.penalty}` +
    (c.qlib_score_before_fusion != null ? `\n融合前相对分: ${c.qlib_score_before_fusion}` : '') +
    (c.raw_model_score != null ? `\n模型原始输出: ${c.raw_model_score}` : '')
}

function scoreTraceText(c) {
  const parts = ['该分数只表示当日截面相对排名，不是预期收益率']
  if (c.model_input_score != null && c.raw_model_score == null) {
    const semantic = c.model_input_semantics === 'daily_cross_sectional_rank_fusion'
      ? '模型日度截面秩融合输入'
      : '模型输入（语义未确认）'
    parts.push(`${semantic}: ${c.model_input_score} (rank ${c.model_input_rank})`)
  }
  if (c.raw_model_score != null) parts.push(`模型原始输出: ${c.raw_model_score} (rank ${c.raw_rank})`)
  if (c.neutralized_score != null) parts.push(`中性化阶段输出: ${c.neutralized_score} (rank ${c.neutralized_rank})`)
  if (c.rank_percentile != null) parts.push(`当日分位: ${(c.rank_percentile * 100).toFixed(1)}%`)
  parts.push(`最终排名分: ${mainScore(c).toFixed(6)} (rank ${c.final_rank ?? c.rank ?? '--'})`)
  const fusion = qlibAttrText(c)
  if (fusion) parts.push(fusion)
  return parts.join('\n')
}

function neutralizeStatusText(processing) {
  const requested = processing.neutralize_qlib_requested ?? processing.neutralize_qlib
  if (!requested) return '关闭'
  const labels = {
    applied: '已应用',
    partial: '部分应用',
    no_effect_missing_exposures: '暴露数据不足，未生效',
    fallback: '失败降级',
    skipped_small_universe: '样本过少，未应用',
    unknown_legacy_cache: '历史缓存，状态未知',
  }
  return labels[processing.neutralize_status] || '请求开启'
}

const regimeAdvice = computed(() => {
  const r = currentRegime.value
  if (!r?.suggested_weights) return ''
  const top = Object.entries(r.suggested_weights).filter(([, v]) => v > 0.1).sort((a, b) => b[1] - a[1]).slice(0, 3)
  return top.length ? '当前建议侧重：' + top.map(([k]) => SIGNAL_LABELS[k] || k).join(' + ') : ''
})

const regimeBiasHint = computed(() => {
  const b = macroOverview.value?.regime_bias
  if (!b) return '无'
  const labels = { volatile_up: '上涨', volatile_down: '下跌', volatile_flat: '高波动', steady_up: '上涨', steady_down: '下跌', steady_flat: '震荡', trending_up: '上涨', trending_down: '下跌', ranging: '震荡', volatile: '高波动' }
  const entries = Object.entries(b)
  if (!entries.length) return '无'
  const top = entries.sort((a, b) => b[1] - a[1])[0]
  return labels[top[0]] || top[0]
})

const DIR_LABELS = { trend_momentum: '趋势动量', volume_breakout: '量价突破', oversold_rebound: '超卖反弹', value_defense: '价值防御' }
const DIR_DESC = {
  trend_momentum: '上涨/牛市趋势中动量+成长因子主导 (48因子/6类)',
  volume_breakout: '横盘等放量突破+量价背离+资金异动 (45因子/5类)',
  oversold_rebound: '高波动中识别超卖+恐慌放量+聪明钱抄底 (20因子/4类)',
  value_defense: '下跌市中低估值+高质量+稳定现金流防御 (48因子/6类)',
}

function dirLabel(d) { return DIR_LABELS[d] || d }
function macroDateLabel(indicator) {
  return indicator?.date_semantics === 'report_period_label' ? '报告期' : '数据日'
}
function fmtFlowAmt(wan) {
  const sign = wan > 0 ? '+' : ''
  if (Math.abs(wan) >= 10000) return ` ${sign}${(wan / 10000).toFixed(1)}亿`
  return ` ${sign}${Math.round(wan)}万`
}
function sentLabel(score) {
  if (score == null) return ''
  if (score >= 0.2) return '利好'
  if (score > 0.05) return '偏多'
  if (score > -0.05) return '中性'
  if (score > -0.2) return '偏空'
  return '利空'
}
const dirHint = computed(() => {
  if (config.direction === 'auto') {
    const r = currentRegime.value
    if (r?.direction) return `当前市场 → ${dirLabel(r.direction)}`
    return '根据市场状态自动切换策略方向'
  }
  return DIR_DESC[config.direction] || ''
})

// 主排序分
function mainScore(c) {
  return c.qlib_score || 0
}

const mainSorted = computed(() => {
  const list = [...(screeningResult.value?.candidates || [])]
  return list.sort((a, b) => mainScore(b) - mainScore(a))
})
const debateResults = computed(() => {
  const rec = screeningResult.value?.recommended || []
  const cands = screeningResult.value?.candidates || []
  const merged = [...rec]
  for (const c of cands) {
    if ((c.debate_summary || c.risk_score != null) && !merged.find(r => r.code === c.code)) {
      merged.push(c)
    }
  }
  return merged
})
const combinedSorted = computed(() => mainSorted.value)

const totalPages = computed(() => Math.max(1, Math.ceil(mainSorted.value.length / pageSize)))
const paginatedMain = computed(() => mainSorted.value.slice((page.value - 1) * pageSize, page.value * pageSize))
const paginatedCombined = computed(() => combinedSorted.value.slice((page.value - 1) * pageSize, page.value * pageSize))

const stages = computed(() => {
  const done = screeningResult.value?.status === 'done'
  return [
    { key: 'l0', label: '评分', done, active: !done },
    { key: 'l1', label: '策略排序', done, active: false },
    { key: 'l2', label: 'AI复核', done: done && config.useDebate, active: !done && config.useDebate },
  ]
})
const progressPct = computed(() => {
  const p = screeningResult.value?.progress
  return p?.total ? Math.round((p.current / p.total) * 100) : 0
})
const progressText = computed(() => {
  const p = screeningResult.value?.progress
  if (!p) return '准备中...'
  return p.stage === 'l2' ? 'AI复核辩论中...' : `处理中 ${p.current || 0}/${p.total || '?'}`
})

// 板块过滤
const filteredSectors = computed(() => {
  if (!sectorSearch.value) return sectors.value
  const q = sectorSearch.value.toLowerCase()
  return sectors.value.filter(s => s.name.toLowerCase().includes(q))
})
const filteredConceptSectors = computed(() => {
  if (!sectorSearch.value) return conceptSectors.value
  const q = sectorSearch.value.toLowerCase()
  return conceptSectors.value.filter(s => s.name.toLowerCase().includes(q))
})

// --- 方法 ---
async function loadHotSectors() {
  loadingSectors.value = true
  try { const r = await getHotSectors(hotSectorCount.value); hotSectors.value = (r?.data || r)?.sectors || [] } catch (e) { /* ignore */ }
  loadingSectors.value = false
}
function toggleHotSector(name) {
  const idx = selectedHotSectors.value.indexOf(name)
  if (idx >= 0) selectedHotSectors.value.splice(idx, 1); else selectedHotSectors.value.push(name)
}
async function loadSectors() {
  loadingSectors.value = true
  try { const r = await getAvailableSectors(); sectors.value = (r?.data || r) || [] } catch (e) { /* ignore */ }
  loadingSectors.value = false
}
async function loadConceptSectors() {
  loadingSectors.value = true
  try { const r = await getConceptSectors(); conceptSectors.value = (r?.data || r) || [] } catch (e) { /* ignore */ }
  loadingSectors.value = false
}
function toggleSector(name, type) {
  const arr = type === 'custom' ? selectedSectors : selectedConceptSectors
  const idx = arr.value.indexOf(name)
  if (idx >= 0) arr.value.splice(idx, 1); else arr.value.push(name)
}
function togglePrefix(val) {
  const idx = config.codePrefixes.indexOf(val)
  if (idx >= 0) config.codePrefixes.splice(idx, 1)
  else config.codePrefixes.push(val)
}

function onPoolChange() {
  selectedHotSectors.value = []; selectedSectors.value = []; selectedConceptSectors.value = []; sectorSearch.value = ''
  if (config.pool === 'hot_sectors' && !hotSectors.value.length) loadHotSectors()
  if (config.pool === 'custom' && !sectors.value.length) loadSectors()
  if (config.pool === 'concept' && !conceptSectors.value.length) loadConceptSectors()
}

function buildParams() {
  const isDeep = config.useDebate
  const qlibModelId = config.qlibModelId
    || (autoMatchedModel.value ? autoMatchedModel.value.experiment_id + '/' + autoMatchedModel.value.run_id : null)
  if (config.fixedModel && !config.qlibModelId) {
    throw new Error('固定单模型模式下，请选择一个具体的已训练模型')
  }

  const p = {
    pool: config.pool, top_n: config.topN, debate_top: 5,
    skip_l1: false, skip_l2: !isDeep, skip_debate: !isDeep,
    skip_l0: false, skip_qlib: false,
    skip_tech: true, skip_momentum: true, skip_volatility: true,
    skip_size_turnover: true, skip_illiq: true, skip_overnight_gap: true,
    skip_sent: false, skip_fundamental: false, skip_flow: false, skip_north_flow: false,
    skip_wb_industry: true, skip_wb_divergence: true, skip_wb_moneyflow: true,
    source_weights: { qlib: 1.0 },
    tech_preset: 'all', regime: 'auto', async_mode: true,
    hot_sector_count: hotSectorCount.value,
    code_prefixes: config.codePrefixes.length ? config.codePrefixes.flatMap(p => p.split(',')) : null,
    qlib_model: qlibModelId ? qlibModelId.split('/') : null,
    sector_fusion: config.sectorFusion && !modelRoutingPaused.value && !Object.keys(stateModelMap.value).length,
    macro_bias: config.macroBias,
    neutralize_qlib: config.neutralizeQlib,
    decision_day_filter: config.decisionDayFilter,
    macro_regime: config.macroRegime,
    smooth_regime: config.smoothRegime,
    use_vol_switch: config.fixedModel ? false : config.useVolSwitch,
    state_models: config.fixedModel ? {} : stateModelsPayload(stateModelMap.value, config.useVolSwitch),
    profile_id: activeProfile.value?.profile_id || null,
    profile_version: activeProfile.value?.version || null,
  }

  if (config.pool === 'hot_sectors' && selectedHotSectors.value.length) {
    p.custom_sectors = selectedHotSectors.value; p.pool = 'custom'
  }
  if (config.pool === 'custom' && selectedSectors.value.length) p.custom_sectors = selectedSectors.value
  if (config.pool === 'concept' && selectedConceptSectors.value.length) p.concept_sectors = selectedConceptSectors.value

  return p
}

async function startScreening() {
  running.value = true; error.value = ''; screeningResult.value = null
  jobId.value = ''; activeResultTab.value = 'main'; page.value = 1
  pollCount = 0
  pollToken++
  if (pollTimer) clearTimeout(pollTimer)
  try {
    const res = await runScreening(buildParams())
    if (res?.job_id) { jobId.value = res.job_id; tracker.register('screening', res.job_id, { type: '选股筛选' }); pollResult() }
    else if (res?.data?.job_id) { jobId.value = res.data.job_id; tracker.register('screening', res.data.job_id, { type: '选股筛选' }); pollResult() }
    else { screeningResult.value = res?.data || res; stopRunning() }
  } catch (e) { error.value = e.message; stopRunning() }
}

function _pollInterval() {
  pollCount++
  if (pollCount <= 3) return 3000
  if (pollCount <= 6) return 5000
  if (pollCount <= 10) return 8000
  if (pollCount <= 15) return 15000
  return 30000
}
async function pollResult() {
  if (!jobId.value) return
  const token = pollToken
  const requestedJobId = jobId.value
  try {
    const res = await getScreeningResult(requestedJobId)
    if (token !== pollToken || jobId.value !== requestedJobId) return
    const r = res?.data || res
    if (r.status === 'error' || r.status === 'interrupted') {
      error.value = r.error || (r.status === 'interrupted' ? '选股任务已中断，请重新运行' : '筛选出错')
      screeningResult.value = null
      stopRunning()
      return
    }
    if (r.status === 'done') {
      if (String(r.job_id) !== requestedJobId || String(r.result_id) !== requestedJobId) {
        error.value = '选股结果与当前任务 ID 不匹配，已阻止展示和保存'
        screeningResult.value = null
      } else {
        screeningResult.value = r
      }
      stopRunning()
      return
    }
    if (!['pending', 'running', 'finalizing'].includes(r.status)) {
      error.value = `选股任务返回未知状态: ${r.status || 'empty'}`
      screeningResult.value = null
      stopRunning()
      return
    }
    screeningResult.value = r
    pollTimer = setTimeout(pollResult, _pollInterval())
  } catch (e) { if (token === pollToken) { error.value = e.message; stopRunning() } }
}
function stopRunning() { running.value = false; jobId.value = ''; tracker.unregister('screening') }

function toggleExpand(code) { expandedCode.value = expandedCode.value === code ? null : code }
function levelClass(lv) {
  if (lv?.includes('强烈')) return 'tag-high'
  if (lv?.includes('推荐')) return 'tag-mid'
  return 'tag-low'
}
function goStock(code) { router.push(`/stock/${code}`) }

function openSaveDialog() {
  if (screeningResult.value?.status !== 'done' || !screeningResult.value?.result_id || !screeningResult.value?.candidates?.length) return
  const ds = todayStr()
  saveName.value = `${ds} 选股 Top-${screeningResult.value.candidates.length}`
  saveNote.value = `池:${config.pool} 源:${config.alphaSource}`
  saveMessage.value = ''
  saveError.value = false
  showSaveDialog.value = true
}

async function doSave() {
  const resultId = screeningResult.value?.result_id
  if (!saveName.value.trim() || screeningResult.value?.status !== 'done' || !resultId) return
  saving.value = true
  saveMessage.value = ''
  try {
    const res = await saveScreening(resultId, saveName.value.trim(), saveNote.value.trim())
    const data = res?.data || res
    saveMessage.value = `✅ 已保存！共 ${data.candidate_count || '?'} 只候选，可在「量化调仓」中使用`
    saveError.value = false
    setTimeout(() => { showSaveDialog.value = false }, 2000)
  } catch (e) {
    saveMessage.value = `❌ 保存失败: ${e.message || e}`
    saveError.value = true
  } finally {
    saving.value = false
  }
}
</script>

<style scoped>
.screening-page { padding: 0 }
.card { margin: 16px 32px; }
.screening-meta { display: flex; gap: 16px; flex-wrap: wrap; padding: 10px 20px;
  background: var(--bg2, #1a1a2e); border-radius: 8px; font-size: 13px; }
.screening-meta .meta-tag { display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 10px; border-radius: 4px; }
.screening-meta .direction { background: rgba(59,130,246,.15); color: #60a5fa; }
.screening-meta .freshness { background: color-mix(in srgb, var(--success) 15%, transparent); color: var(--success); }
.screening-meta .freshness.stale { background: color-mix(in srgb, var(--danger) 15%, transparent); color: var(--danger); }
.screening-meta .coverage-warning { background: color-mix(in srgb, var(--warning) 15%, transparent); color: var(--warning); }
.screening-meta .style { background: rgba(139,92,246,.15); color: #a78bfa; }
.screening-meta .etf { background: rgba(16,185,129,.15); color: #34d399; }
.screening-meta .rotation { background: rgba(245,158,11,.15); color: #fbbf24; }
.error { margin-left: 32px; margin-right: 32px; }
.tune-trigger { margin-left: 32px; margin-right: 32px; }
.screening-page h1 { color: var(--text); font-size: var(--fs-2xl); font-weight: 700; margin: 0 0 12px 0; padding: 28px 32px 0; letter-spacing: 0 }

/* Zone 0 */
.regime-banner { padding: 14px 20px; border-radius: var(--radius); margin: 16px 32px; display: flex; align-items: center; gap: 20px; flex-wrap: wrap }
.regime-banner.regime-volatile_up { background: rgba(239,68,68,.08); border: 1px solid rgba(239,68,68,.2) }
.regime-banner.regime-volatile_down { background: rgba(34,197,94,.08); border: 1px solid rgba(34,197,94,.2) }
.regime-banner.regime-volatile_flat { background: rgba(139,92,246,.08); border: 1px solid rgba(139,92,246,.2) }
.regime-banner.regime-steady_up { background: rgba(37,99,235,.08); border: 1px solid rgba(37,99,235,.25) }
.regime-banner.regime-steady_down { background: rgba(107,114,128,.08); border: 1px solid rgba(107,114,128,.2) }
.regime-banner.regime-steady_flat { background: rgba(245,158,11,.08); border: 1px solid rgba(245,158,11,.2) }
.regime-main { display: flex; align-items: center; gap: 8px }
.regime-dot { width: 8px; height: 8px; border-radius: 50% }
.regime-volatile_up .regime-dot { background: var(--danger) }
.regime-volatile_down .regime-dot { background: var(--success) }
.regime-volatile_flat .regime-dot { background: #8b5cf6 }
.regime-steady_up .regime-dot { background: #2563eb }
.regime-steady_down .regime-dot { background: #6b7280 }
.regime-steady_flat .regime-dot { background: var(--warning) }
.regime-label { font-weight: 700; font-size: var(--fs-lg) }
.regime-volatile_up .regime-label { color: var(--danger) }
.regime-volatile_down .regime-label { color: var(--success) }
.regime-volatile_flat .regime-label { color: #8b5cf6 }
.regime-steady_up .regime-label { color: #2563eb }
.regime-steady_down .regime-label { color: #6b7280 }
.regime-steady_flat .regime-label { color: var(--warning) }
.regime-conf { font-size: var(--fs-sm); color: var(--text-muted) }
.regime-macro { font-size: var(--fs-sm); padding: 2px 10px; border-radius: 10px }
.check-mini { display: flex; align-items: center; gap: 4px; font-size: var(--fs-sm); color: var(--text-muted); white-space: nowrap; padding-bottom: 2px }
.model-info-card { width: 100%; padding: 10px 14px; margin-top: 4px; background: rgba(148,163,184,.06); border: 1px solid rgba(148,163,184,.15); border-radius: var(--radius); display: grid; grid-template-columns: 1fr 1fr; gap: 4px 16px }
.model-info-line { display: flex; align-items: center; gap: 8px; font-size: var(--fs-sm); color: var(--text-secondary) }
.model-mode-grid .settings-option small, .model-mode-grid .settings-field__hint { color: var(--text-secondary); }
.model-info-label { min-width: 36px; color: var(--text-muted); font-size: 11px; text-transform: uppercase }
.model-info-tag { font-size: 11px; padding: 0 8px; border-radius: 10px; line-height: 18px }
.tag-auto { color: #8b5cf6; background: rgba(139,92,246,.12) }
.tag-manual { color: var(--text-muted); background: rgba(148,163,184,.12) }
.text-mono { font-family: monospace; font-size: 11px; opacity: .75 }
.model-routing-details { margin-top: 12px; border-top: 1px solid var(--border); }
.model-routing-details > summary { min-height: 42px; display: flex; align-items: center; color: var(--text-secondary); cursor: pointer; font-size: var(--fs-sm); }
.model-routing-details .state-model-map { margin-top: 0; }
.fusion-summary { padding: 8px 12px; margin-bottom: 8px; font-size: var(--fs-sm); color: var(--text-secondary); background: rgba(139,92,246,.08); border: 1px solid rgba(139,92,246,.2); border-radius: var(--radius) }
.tag-divergent { margin-left: 6px; font-size: 11px; padding: 1px 6px; border-radius: 8px; color: #8b5cf6; background: rgba(139,92,246,.12); cursor: help }
.tag-escort { margin-left: 6px; font-size: 11px; padding: 1px 6px; border-radius: 8px; color: var(--warning); background: rgba(245,158,11,.12) }
.flow-amt { margin-left: 4px; opacity: .75; font-size: 11px }
.sent-label { margin-left: 4px; opacity: .8; font-size: 11px }
.macro-act-confirm { color: var(--text-secondary); background: rgba(148,163,184,.15) }
.macro-act-weaken { color: var(--warning); background: rgba(245,158,11,.15) }
.macro-act-weaken_strong { color: #ef4444; background: rgba(239,68,68,.15) }
.regime-advice { font-size: var(--fs-sm); color: var(--text-secondary) }

/* Macro bar */
.macro-bar { display: flex; flex-wrap: wrap; gap: 8px; padding: 10px 32px; margin-bottom: 2px }
.macro-meta { flex-basis: 100%; color: var(--text-muted); font-size: var(--fs-xs); }
.macro-chip { font-size: var(--fs-xs); padding: 3px 10px; border-radius: 10px; background: var(--bg-secondary); border: 1px solid var(--border); color: var(--text-secondary); white-space: nowrap }
.macro-chip.up { border-color: rgba(239,68,68,.25); color: var(--up) }
.macro-chip.down { border-color: rgba(34,197,94,.25); color: var(--down) }
.macro-chip.risk-warning { border-color: #f59e0b; color: #f59e0b }
.macro-chip.risk-danger { border-color: var(--danger); color: var(--danger); background: rgba(239,68,68,.08) }
.macro-chip.risk-normal { border-color: var(--success); color: var(--success) }
.macro-note { color: var(--text-muted); font-style: italic }

/* Zone A */
.config-card { padding: 20px }
.profile-control { padding:0 32px; display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
.profile-control > :first-child { flex:1; min-width:320px; }
.profile-control label { display:grid; gap:4px; color:var(--text-muted); font-size:var(--fs-xs); }
.profile-control select { min-width:300px; }
.profile-locked .settings-stack { pointer-events:none; opacity:.68; }
.locked-note { margin-bottom:12px; padding:9px 12px; border:1px solid var(--primary); border-radius:6px; color:var(--primary); font-size:var(--fs-sm); }
.config-main { display: flex; align-items: flex-end; gap: 14px; margin-bottom: 14px; flex-wrap: wrap }
.config-main label { display: flex; flex-direction: column; gap: 4px; font-size: var(--fs-sm); color: var(--text-muted); flex: 1; min-width: 200px }
.config-main select { width: 100% }
.alpha-hint { font-size: var(--fs-xs); color: var(--text-muted); padding-bottom: 2px; flex-shrink: 0 }
.config-row { display: flex; align-items: flex-end; gap: 14px; flex-wrap: wrap; margin-bottom: 14px }
.config-row label { display: flex; flex-direction: column; gap: 4px; font-size: var(--fs-sm); color: var(--text-muted); flex: 1; min-width: 120px }
.config-row select, .config-row input { width: 100% }

.factor-section { margin-bottom: 14px; padding: 12px; background: var(--bg); border-radius: var(--radius-sm) }
.section-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px }
.section-actions { display: flex; gap: 6px }
.section-label { font-size: var(--fs-sm); color: var(--text-muted); font-weight: 600; display: block; margin-bottom: 8px }
.factor-list { max-height: 220px; overflow-y: auto; padding: 4px 0 }
.factor-group { margin-bottom: 8px }
.factor-group:last-child { margin-bottom: 0 }
.factor-group-header { font-size: var(--fs-xs); color: var(--text-muted); font-weight: 600; padding: 2px 0 4px 0; border-bottom: 1px solid var(--border); margin-bottom: 4px; display: flex; justify-content: space-between }
.factor-group-items { display: flex; flex-wrap: wrap; gap: 4px 12px }

.prefix-section { margin-bottom: 14px; padding: 10px 12px; background: var(--bg); border-radius: var(--radius-sm) }
.sector-section { margin-bottom: 14px; padding-top: 12px; border-top: 1px solid var(--border) }
.sector-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px }
.sector-search { flex: 1; padding: 4px 8px; font-size: var(--fs-sm); max-width: 500px }
.sector-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 6px }
.sector-tag { font-size: var(--fs-xs); padding: 3px 8px; border-radius: 10px; background: var(--bg-secondary); border: 1px solid var(--border); cursor: pointer; user-select: none; color: var(--text-secondary) }
.sector-tag:hover { border-color: var(--primary) }
.sector-tag.selected { background: rgba(59,130,246,.15); border-color: var(--primary); color: var(--primary) }
.sector-tag small { font-size: 10px }
.sector-list { max-height: 180px; overflow-y: auto; display: flex; flex-wrap: wrap; gap: 4px }
.sector-item { font-size: var(--fs-sm); color: var(--text-secondary); display: flex; align-items: center; gap: 4px; cursor: pointer; min-width: 130px; padding: 2px 0 }
.sector-count { font-size: var(--fs-xs); color: var(--text-muted) }
.sector-selected { font-size: var(--fs-sm); color: var(--primary); margin-top: 4px }

.aux-section { margin-bottom: 10px }
.dimension-grid { display: flex; flex-wrap: wrap; gap: 14px }
.dimension-item { font-size: var(--fs-sm); color: var(--text-secondary); display: flex; align-items: center; gap: 4px; cursor: pointer }

.debate-row { display: flex; align-items: center; gap: 10px; margin-bottom: 14px }

.btn-run { width: 100%; padding: 10px; border: none; border-radius: var(--radius-sm); background: var(--primary); color: var(--on-primary); font-size: var(--fs-base); font-weight: 600; cursor: pointer; margin-top: 8px; font-family: var(--font) }
.btn-run:disabled { opacity: .5; cursor: not-allowed }
.btn-run:hover:not(:disabled) { background: var(--primary-hover) }
.settings-ledger .btn-run { width: auto; min-width: 180px; margin: 0; }
.btn-sm { padding: 3px 10px; font-size: var(--fs-xs); font-family: var(--font); background: var(--bg-secondary); border: 1px solid var(--border); border-radius: var(--radius-sm); color: var(--text-secondary); cursor: pointer }
.btn-sm:hover { border-color: var(--primary); color: var(--primary) }
.btn-sm:disabled { opacity: .4; cursor: not-allowed }

@media (max-width: 760px) {
  .settings-ledger .btn-run { width: 100%; }
  .screening-page h1 { padding: 20px 12px 0; }
  .tabs { padding: 0 12px; overflow-x: auto; scrollbar-width: thin; }
  .tab { flex: 0 0 auto; min-height: 44px; padding: 10px 16px; }
  .profile-control { padding: 0 12px; align-items: stretch; }
  .profile-control > :first-child { flex-basis: 100%; min-width: 0; }
  .profile-control label, .profile-control select { width: 100%; min-width: 0; }
  .preset-bar { padding: 8px 12px; }
  .regime-banner { margin: 16px 12px; padding: 12px; gap: 10px; }
  .regime-main { flex-wrap: wrap; }
  .regime-advice { flex-basis: 100%; }
}

/* Zone B */
.progress-card { padding: 20px }
.stepper { display: flex; justify-content: center; gap: 40px; margin-bottom: 14px }
.step { display: flex; align-items: center; gap: 6px; font-size: var(--fs-md); color: var(--text-muted) }
.step.done .step-dot { color: var(--success) }
.step.active .step-dot { color: var(--primary); animation: pulse 1s infinite }
.step.done { color: var(--text-secondary) }
.step.active { color: var(--text) }
.progress-bar-bg { height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; margin-bottom: 8px }
.progress-bar { height: 100%; background: var(--primary); border-radius: 3px; transition: width .5s }
.progress-text { font-size: var(--fs-sm); color: var(--text-secondary); text-align: center }
@keyframes pulse { 0%, 100% { opacity: 1 } 50% { opacity: .4 } }

/* Zone C */
.result-card { padding: 0 }
.result-tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border); padding: 0 16px }
.result-tab { padding: 10px 16px; border: none; border-bottom: 2px solid transparent; background: transparent; color: var(--text-muted); font-size: var(--fs-md); cursor: pointer; font-family: var(--font); white-space: nowrap }
.result-tab:hover { color: var(--text) }
.result-tab.active { color: var(--primary); border-bottom-color: var(--primary) }
.result-tab-spacer { flex: 1 }
.btn-save { padding: 6px 16px; border: 1px solid var(--primary); border-radius: 6px; background: var(--primary); color: var(--on-primary); font-size: 13px; font-weight: 600; cursor: pointer; font-family: var(--font); white-space: nowrap; transition: opacity .2s }
.btn-save:hover { opacity: .85 }
.btn-save:disabled { opacity: .5; cursor: not-allowed }

/* Save Modal */
.modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,.5); display: flex; align-items: center; justify-content: center; z-index: 1000 }
.modal-card { background: var(--bg); border: 1px solid var(--border); border-radius: 12px; padding: 24px; width: 420px; max-width: 90vw }
.modal-card h3 { margin: 0 0 8px; font-size: 18px }
.modal-desc { color: var(--text-muted); font-size: 13px; margin: 0 0 16px }
.modal-label { display: block; font-size: 13px; margin-bottom: 4px; color: var(--text-muted) }
.modal-input { width: 100%; padding: 8px 12px; border: 1px solid var(--border); border-radius: 6px; background: var(--bg2); color: var(--text); font-size: 14px; margin-bottom: 12px; font-family: var(--font) }
.modal-msg { font-size: 13px; padding: 8px 0 }
.msg-ok { color: #34d399 }
.msg-error { color: #f87171 }
.modal-actions { display: flex; gap: 12px; justify-content: flex-end; margin-top: 16px }
.btn-cancel { padding: 8px 20px; border: 1px solid var(--border); border-radius: 6px; background: transparent; color: var(--text); cursor: pointer; font-family: var(--font); font-size: 14px }
.btn-confirm { padding: 8px 20px; border: none; border-radius: 6px; background: var(--primary); color: var(--on-primary); font-weight: 600; cursor: pointer; font-family: var(--font); font-size: 14px }
.btn-confirm:disabled { opacity: .5; cursor: not-allowed }

.table-wrap { padding: 16px }
.sort-col { color: var(--primary) }
.combined-hint { font-size: var(--fs-sm); color: var(--text-muted); margin-bottom: 12px; padding: 8px 12px; background: var(--bg); border-radius: var(--radius-sm) }

.data-table { width: 100%; border-collapse: collapse; font-size: var(--fs-md) }
.data-table th { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); color: var(--text-muted); font-weight: 600; font-size: var(--fs-sm); white-space: nowrap }
.data-table td { padding: 7px 10px; border-bottom: 1px solid rgba(42,42,80,.4); color: var(--text); cursor: pointer }
.data-table tr:hover td { background: rgba(59,130,246,.04) }

.pagination { display: flex; justify-content: center; align-items: center; gap: 14px; padding: 10px 0; font-size: var(--fs-sm); color: var(--text-muted) }
.pagination button { padding: 4px 12px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--card-bg); color: var(--text); cursor: pointer; font-family: var(--font); font-size: var(--fs-sm) }
.pagination button:disabled { opacity: .4; cursor: not-allowed }
.pagination button:hover:not(:disabled) { border-color: var(--primary); color: var(--primary) }

.tune-trigger { margin-top: 16px; text-align: center }
.empty-hint { padding: 60px 20px; text-align: center }

.up { color: var(--up) }
.down { color: var(--down) }
.text-gray { color: var(--text-muted) }
.ft-xs { font-size: var(--fs-xs) }
.ft-sm { font-size: var(--fs-sm) }

/* 周频调仓 */
.tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border); padding: 0 32px; margin-bottom: 0 }
.tab { padding: 10px 20px; border: none; background: transparent; color: var(--text-muted); font-size: var(--fs-md); cursor: pointer; border-bottom: 2px solid transparent; margin-bottom: -1px; transition: all var(--transition); font-family: var(--font) }
.tab:hover { color: var(--text) }
.tab.active { color: var(--primary); border-bottom-color: var(--primary) }

/* ── 预设栏 ── */
.preset-bar { display:flex; align-items:center; gap:6px; padding:8px 32px; border-bottom:1px solid var(--border); flex-wrap:wrap }
.preset-label { font-size:var(--fs-xs); color:var(--text-muted); margin-right:4px }
.preset-btn { padding:4px 14px; border:1px solid var(--border); border-radius:14px; background:transparent; color:var(--text-secondary); font-size:var(--fs-xs); cursor:pointer; font-family:var(--font); transition:all var(--transition); white-space:nowrap }
.preset-btn:hover { border-color:var(--primary); color:var(--text) }
.preset-btn.active { background:rgba(107,159,212,.15); border-color:var(--primary); color:var(--primary) }
.preset-save { border-style:dashed }

.metric-badge { display: inline-block; padding: 4px 14px; margin: 4px 6px 4px 0; background: var(--bg-secondary); border-radius: 20px; font-size: var(--fs-sm); color: var(--text) }

.tag-buy { background: rgba(239,68,68,.12); color: var(--danger) !important }
.tag-sell { background: rgba(34,197,94,.12); color: var(--success) !important }

.constraint-list { display: flex; flex-direction: column; gap: 6px; margin-top: 8px }
.constraint-item { display: flex; align-items: center; gap: 8px; font-size: var(--fs-sm) }
.constraint-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0 }
.constraint-dot.pass { background: var(--success) }
.constraint-dot.fail { background: var(--warning) }
.constraint-name { color: var(--text); min-width: 100px }
.constraint-value { color: var(--text-secondary) }
.constraint-note { color: var(--warning); font-size: var(--fs-xs) }

.text-report { background: var(--bg-secondary); padding: 16px; border-radius: var(--radius); font-size: var(--fs-sm); line-height: 1.6; white-space: pre-wrap; max-height: 600px; overflow-y: auto; font-family: var(--font-mono, 'Courier New', monospace) }
.text-muted { color: var(--text-muted); font-size: var(--fs-sm) }
.text-danger { color: var(--danger); font-size: var(--fs-sm) }
.timing-indicator { padding: 10px 16px; background: rgba(245,158,11,.1); border: 1px solid rgba(245,158,11,.25); border-radius: var(--radius); margin-bottom: 12px; font-size: 13px; color: var(--text-secondary); }

/* ===== 组合管理: 折叠区段 ===== */
.portfolio-section { margin: 16px 32px; border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; background: var(--card-bg); }
.section-header { display: flex; align-items: center; justify-content: space-between; padding: 14px 20px; cursor: pointer; user-select: none; background: var(--bg2, #1a1a2e); transition: background .2s; }
.section-header:hover { background: rgba(59,130,246,.06); }
.section-title { font-weight: 600; font-size: var(--fs-base); color: var(--text); }
.section-badge { font-size: var(--fs-sm); padding: 2px 12px; border-radius: 12px; font-weight: 600; margin-left: auto; margin-right: 16px; }
.section-toggle { font-size: var(--fs-xs); color: var(--text-muted); padding: 2px 10px; border: 1px solid var(--border); border-radius: 4px; }
.section-body { padding: 16px 20px; }
.section-body :deep(.timing-card) { margin-bottom: 0; }
.section-body-flush { padding: 0; }

/* ===== 成本 & 风控 ===== */
.cost-section, .risk-section { margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--border); }
.cost-row { display: flex; align-items: flex-end; gap: 14px; flex-wrap: wrap; margin-top: 8px; }
.cost-row label { display: flex; flex-direction: column; gap: 4px; font-size: var(--fs-sm); color: var(--text-muted); }
.cost-row input { padding: 4px 6px; border: 1px solid var(--border); border-radius: 4px; background: var(--bg); color: var(--text); font-size: var(--fs-sm); font-family: var(--font); width: 100px; }
.cost-hint { font-size: var(--fs-xs); color: var(--text-muted); padding-bottom: 4px; }

/* ===== AI复核展开详情 ===== */
.expand-row td { padding: 0; background: rgba(0,0,0,.15); }
.debate-detail { padding: 16px 20px; max-width: 100%; }
.debate-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.debate-title { font-weight: 700; font-size: var(--fs-base); color: var(--text); }
.debate-section { margin-bottom: 14px; }
.debate-section h4 { font-size: var(--fs-sm); color: var(--primary); margin: 0 0 6px 0; font-weight: 600; }
.debate-section h5 { font-size: var(--fs-sm); color: var(--text-secondary); margin: 8px 0 4px 0; font-weight: 600; border-left: 3px solid var(--primary); padding-left: 8px; }
.debate-text { white-space: pre-wrap; font-size: var(--fs-sm); line-height: 1.7; color: var(--text-secondary); background: var(--bg); padding: 10px 14px; border-radius: 6px; max-height: 400px; overflow-y: auto; font-family: var(--font); border: 1px solid var(--border); }
.agent-block { margin-bottom: 8px; }
.signal-grid { display: flex; flex-wrap: wrap; gap: 8px; }
.signal-item { display: flex; align-items: center; gap: 6px; padding: 4px 10px; background: var(--bg); border-radius: 6px; font-size: var(--fs-xs); border: 1px solid var(--border); }
.signal-name { color: var(--text-muted); min-width: 60px; }
.signal-dir { padding: 1px 6px; border-radius: 8px; font-size: 10px; }
.signal-dir.up { color: var(--danger); background: rgba(239,68,68,.12); }
.signal-dir.down { color: var(--success); background: rgba(34,197,94,.12); }
.signal-score { font-weight: 700; color: var(--text); }

</style>
