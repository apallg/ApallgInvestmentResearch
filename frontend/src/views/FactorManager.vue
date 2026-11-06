<template>
  <div class="fm-root">
    <div class="page-header">
      <h1 class="page-title">因子管理</h1>
    </div>

    <div class="fm-tabs" role="tablist" aria-label="因子管理视图">
      <button role="tab" :aria-selected="activeTab === 'rolling'" :class="['fm-tab', { active: activeTab === 'rolling' }]" @click="activeTab = 'rolling'">滚动筛选</button>
      <button role="tab" :aria-selected="activeTab === 'architecture'" :class="['fm-tab', { active: activeTab === 'architecture' }]" @click="activeTab = 'architecture'; loadArchitectureHistory()">核心/卫星</button>
      <button role="tab" :aria-selected="activeTab === 'train'" :class="['fm-tab', { active: activeTab === 'train' }]" @click="activeTab = 'train'">滚动训练</button>
      <button role="tab" :aria-selected="activeTab === 'pool'" :class="['fm-tab', { active: activeTab === 'pool' }]" @click="activeTab = 'pool'; loadPool()">
        因子池 <span class="fm-tab-badge">{{ poolCount }}</span>
      </button>
      <button role="tab" :aria-selected="activeTab === 'curated'" :class="['fm-tab', { active: activeTab === 'curated' }]" @click="activeTab = 'curated'; loadCuratedSets()">
        精选集 <span class="fm-tab-badge" v-if="activeCuratedSets.length">{{ activeCuratedSets.length }}</span>
      </button>
      <button role="tab" :aria-selected="activeTab === 'direction'" :class="['fm-tab', { active: activeTab === 'direction' }]" @click="activeTab = 'direction'; loadDirections()">
        策略方向
      </button>
      <button role="tab" :aria-selected="activeTab === 'genetic'" :class="['fm-tab', { active: activeTab === 'genetic' }]" @click="activeTab = 'genetic'; loadCuratedSets(); loadMiningModels()">
        遗传挖矿
      </button>
    </div>

    <div v-if="activeTab === 'architecture'" class="fm-section">
      <div class="fm-hint">
        流程为：生成滚动证据 → 核心组合优化 → 最终样本外验证。精简和均衡候选由后台自动生成，页面只展示当前结果；最终通过后才能生成卫星。
      </div>
        <div class="fm-form-block">
          <div class="fm-form-title">历史任务</div>
          <div class="fm-form-row">
            <label class="fm-field">历史任务
              <select v-model="architectureHistoryRunId" @change="loadArchitectureRun(architectureHistoryRunId)">
                <option value="">选择历史任务</option>
                <option v-for="r in architectureHistory" :key="r.run_id" :value="r.run_id">
                  {{ r.run_id }} · {{ r.status }} · {{ r.selection_cutoff || '—' }} · {{ formatArchitectureTime(r.updated_at) }}
                </option>
              </select>
            </label>
            <label class="fm-field">任务ID
              <input type="text" v-model="architectureLoadRunId" placeholder="bdba67f459d2b928178s504cf" />
            </label>
            <button class="btn-sm" :disabled="!architectureLoadRunId" @click="loadArchitectureRun(architectureLoadRunId)">加载</button>
          </div>
        </div>
      <div v-if="architectureMsg" class="msg">{{ architectureMsg }}</div>
      <div class="fm-form-block">
        <div class="fm-form-title">证据窗口</div>
        <div class="fm-form-row">
          <label class="fm-field">因子来源
            <select v-model="architectureCfg.factor_source">
              <option value="pool">因子池活跃因子</option>
              <option value="public">公开因子集</option>
            </select>
          </label>
          <label v-if="architectureCfg.factor_source === 'public'" class="fm-field">因子集
            <select v-model="architectureCfg.source">
              <option value="Alpha158">Alpha158</option>
              <option value="Alpha101">Alpha101</option>
              <option value="Alpha191">Alpha191</option>
            </select>
          </label>
          <label class="fm-field">股票池
            <select v-model="architectureCfg.pool">
              <option value="csi300">沪深300</option>
              <option value="csi500">中证500</option>
              <option value="stable_history_500d">稳定历史池500D</option>
            </select>
          </label>
          <label class="fm-field">首个测试锚点 <input type="date" v-model="architectureCfg.start" /></label>
          <label class="fm-field">最后测试锚点 <input type="date" v-model="architectureCfg.end" /></label>
          <label class="fm-field">证据截止日 <input type="date" v-model="architectureCfg.selection_cutoff" /></label>
          <label class="fm-field">交易标签
            <select v-model="architectureCfg.label_mode" disabled>
              <option value="tradable_open_to_open">可交易：t+1开盘→t+11开盘</option>
            </select>
          </label>
        </div>
        <div class="fm-form-row fm-form-row--end">
          <span class="fm-hint fm-hint--inline">60月训练 · 6月验证 · 3月测试 · 每段均匀24点 · 21日隔离 · 标签周期10日</span>
          <button class="btn" :disabled="architectureRunning" @click="startArchitecture">
            {{ architectureRunning ? '生成证据中…' : '开始核心筛选' }}
          </button>
          <button v-if="architectureRunning && architectureRun?.run_id" class="btn-sm" @click="cancelArchitecture">取消任务</button>
        </div>
      </div>
      <div v-if="architectureProgress" class="fm-progress">
        <div class="fm-prog-head">
          <span class="fm-prog-label">{{ architectureProgress.message || architectureProgress.stage }}</span>
          <span>{{ architectureProgress.current || 0 }}/{{ architectureProgress.total || 0 }}</span>
        </div>
        <div class="fm-prog-bar"><div class="fm-prog-fill" :style="{ width: architecturePct + '%' }"></div></div>
      </div>
      <div v-if="architectureRun" class="fm-result">
        <h4 class="fm-result-title">当前因子集</h4>
        <table class="fm-table">
          <thead><tr><th>角色</th><th>状态</th><th>因子数</th><th>操作</th></tr></thead>
          <tbody><tr v-for="item in architectureVisibleSets" :key="item.set_id">
            <td>{{ item.role === 'core' ? '最终核心' : item.role }}</td><td>{{ architectureSetStatus(item) }}</td><td>{{ item.factor_keys?.length || 0 }}</td>
            <td>
              <button v-if="item.status !== 'validated'" class="btn-sm" :disabled="architectureSubRunning || !item.factor_keys?.length" @click="startArchitectureAb(item, 10)">
                {{ architectureSetAction(item) }}
              </button>
              <button v-if="item.role === 'core' && item.status === 'validated'" class="btn-sm" :disabled="architectureSubRunning" @click="startArchitectureResidual">生成卫星</button>
            </td>
          </tr></tbody>
        </table>
        <div v-if="architectureVisibleSets.some(item => item.role === 'trend' || item.role === 'rebound')" style="margin-top: 10px;">
          <button class="btn-sm" :disabled="architecturePublishing" @click="publishArchitectureSatellites">
            {{ architecturePublishing ? '发布中…' : '📦 发布为成品因子集' }}
          </button>
          <span v-if="architecturePublishMsg" class="fm-hint fm-hint--inline" style="margin-left: 8px;">{{ architecturePublishMsg }}</span>
        </div>
      </div>
    </div>

    <!-- ═══════════════════════ 滚动筛选 ═══════════════════════ -->
    <div v-if="activeTab === 'rolling'" class="fm-section">
      <div class="fm-hint">
        时序滚动筛选：每期只用「当期及之前」的历史数据做五层校验（严格样本外，杜绝数据窥探）。
        含市值+行业中性化、近期加权 IC、边际贡献 L8。开启「写回因子池」后，每期通过因子集会更新动态因子池。
      </div>

      <div v-if="rollMsg" class="msg">{{ rollMsg }}</div>

      <!-- 表单：三段式分组 -->
      <div class="fm-form-block">
        <div class="fm-form-title">数据与范围</div>
        <div class="fm-form-row">
          <label class="fm-field">因子来源
            <select v-model="roll.factor_source">
              <option value="public">公开因子集</option>
              <option value="pool">因子池活跃因子</option>
              <option value="manual">手动指定</option>
            </select>
          </label>
          <label class="fm-field" v-if="roll.factor_source === 'public'">因子集
            <select v-model="roll.source">
              <option value="Alpha158">Alpha158</option>
              <option value="Alpha101">Alpha101</option>
              <option value="Alpha191">Alpha191</option>
            </select>
          </label>
          <label class="fm-field" v-if="roll.factor_source === 'manual'">因子keys
            <input type="text" v-model="roll.keysText" placeholder="a158_roc20,a191_mom_05" />
          </label>
          <label class="fm-field">股票池
            <select v-model="roll.pool" @change="onRollPoolChange">
              <option value="csi300">沪深300 (快)</option>
              <option value="csi500">中证500</option>
              <option value="csi300+csi500">沪深300+中证500</option>
              <option value="stable_history_500d">稳定历史池500D</option>
              <option value="small_cap">小市值 (中证1000+国证2000)</option>
              <option value="all">全市场 (慢)</option>
              <option value="hot_sectors">热门板块</option>
              <option value="custom">自定义行业板块</option>
              <option value="concept">自定义概念板块</option>
            </select>
          </label>
          <label class="fm-field fm-field--inline">代码前缀
            <span class="fm-prefix-row">
              <label class="fm-check" v-for="pf in PREFIX_OPTS" :key="pf[0][0]">
                <input type="checkbox" :value="pf[0][0]" v-model="roll.prefixKeys" /> {{ pf[1] }}
              </label>
              <span v-if="!roll.prefixKeys.length" class="text-muted" style="font-size:var(--fs-caption)">不选=全部</span>
            </span>
          </label>
        </div>

        <!-- 板块选择器 -->
        <div v-if="roll.pool === 'hot_sectors'" class="fm-sector-row">
          <label class="fm-field">热门板块数 <input v-model.number="rollHotSectorCount" type="number" min="3" max="30" class="fm-input-narrow" /></label>
          <button class="btn-sm" @click="loadRollHotSectors" :disabled="loadingRollSectors">{{ loadingRollSectors ? '...' : '加载' }}</button>
          <span v-if="rollHotSectors.length" class="text-muted" style="font-size:var(--fs-caption)">已选 {{ rollSelectedHotSectors.length }}</span>
        </div>
        <div v-if="roll.pool === 'hot_sectors' && rollHotSectors.length" class="fm-chip-row">
          <span v-for="s in rollHotSectors" :key="s.name"
            :class="['fm-chip', { 'fm-chip--sel': rollSelectedHotSectors.includes(s.name) }]"
            @click="toggleRollHotSector(s.name)">{{ s.name }}</span>
        </div>
        <div v-if="roll.pool === 'custom' || roll.pool === 'concept'" class="fm-sector-row">
          <input v-model="rollSectorSearch" :placeholder="roll.pool === 'concept' ? '搜索概念板块...' : '搜索行业板块...'" class="fm-input-search" />
          <button class="btn-sm" @click="loadRollSectors" :disabled="loadingRollSectors">{{ loadingRollSectors ? '...' : '加载' }}</button>
          <span v-if="rollSelectedSectors.length" class="text-primary" style="font-size:var(--fs-caption)">已选 {{ rollSelectedSectors.length }} 个</span>
        </div>
        <div v-if="(roll.pool === 'custom' || roll.pool === 'concept') && filteredRollSectors.length" class="fm-chip-row fm-chip-row--scroll">
          <label v-for="s in filteredRollSectors" :key="s.name" class="fm-check">
            <input type="checkbox" :checked="rollSelectedSectors.includes(s.name)" @change="toggleRollSector(s.name)" /> {{ s.name }}
          </label>
        </div>
      </div>

      <div class="fm-form-block">
        <div class="fm-form-title">筛选参数</div>
        <div class="fm-form-row">
          <label class="fm-field">起始日 <input type="date" v-model="roll.start" /></label>
          <label class="fm-field">结束日 <input type="date" v-model="roll.end" /></label>
          <label class="fm-field">步长(月) <input type="number" v-model.number="roll.step_months" min="1" max="12" class="fm-input-narrow" /></label>
          <label class="fm-field">回看(月) <input type="number" v-model.number="roll.lookback_months" min="6" max="60" class="fm-input-narrow" /></label>
          <label class="fm-field">模式
            <select v-model="roll.mode">
              <option value="full">完整五层</option>
              <option value="fast">快速二层</option>
            </select>
          </label>
          <label class="fm-field">标签周期 <input type="number" v-model.number="roll.label_horizon" class="fm-input-narrow" disabled /> 日</label>
          <label class="fm-field">Purge <input type="number" v-model.number="roll.purge_gap" class="fm-input-narrow" disabled /> 交易日</label>
          <label class="fm-check"><input type="checkbox" v-model="roll.point_in_time" disabled /> 严格时点一致性</label>
        </div>
        <div class="fm-hint">60月训练 · 6月验证 · 3月测试 · 3月步长 · T+1开盘→T+11开盘</div>
      </div>

      <div class="fm-form-block">
        <div class="fm-form-title">执行</div>
        <div class="fm-form-row fm-form-row--end">
          <label class="fm-check"><input type="checkbox" v-model="roll.write_pool" /> 写回因子池</label>
          <label v-if="roll.write_pool" class="fm-field fm-field--inline">投票阈值 <input type="number" v-model.number="roll.min_periods_frac" min="0" max="1" step="0.05" class="fm-input-narrow" /></label>
          <span class="fm-hint fm-hint--inline">多锚点自动使用共享映射多进程</span>
          <button class="btn" @click="startRolling" :disabled="rolling">
            {{ rolling ? '运行中…' : '开始滚动筛选' }}
          </button>
        </div>
      </div>

      <!-- 进度 -->
      <div v-if="rolling || rollResult" class="fm-progress">
        <div class="fm-prog-head">
          <span class="fm-prog-label">
            滚动筛选 {{ rollProgress?.anchor_total || 0 }} 个锚点
            <span v-if="rollElapsed" class="fm-prog-time">{{ rollElapsed }}</span>
          </span>
          <span class="fm-prog-pct">{{ rollPct }}%</span>
        </div>
        <div class="fm-prog-bar"><div class="fm-prog-fill" :style="{ width: rollPct + '%' }"></div></div>
        <div v-if="rollProgress?.backing_mode" class="fm-prog-meta">
          数据模式 {{ rollProgress.backing_mode === 'memmap' ? '只读共享映射' : rollProgress.backing_mode }}
          <span v-if="rollProgress.mapped_mb != null">· 映射文件 {{ rollProgress.mapped_mb }} MB</span>
        </div>

        <div v-if="anchorsDetail && Object.keys(anchorsDetail).length" class="fm-anchors">
          <div v-for="(ad, akey) in anchorsDetail" :key="akey" class="fm-anchor" :class="'fm-anchor--' + (ad.status || 'running')">
            <div class="fm-anchor-head">
              <span class="fm-anchor-dot">
                <span v-if="ad.status === 'completed' || ad.stage === 'done'">&#10003;</span>
                <span v-else class="fm-anchor-spin">&#8635;</span>
              </span>
              <code class="fm-anchor-date">{{ akey }}</code>
              <span class="fm-anchor-passed" v-if="ad.passed_count != null">{{ ad.passed_count }} 通过</span>
            </div>
            <div v-if="ad.stage && ad.stage !== 'done'" class="fm-anchor-stage">
              {{ stageLabelMap[ad.stage] || ad.stage }}
              <span v-if="ad.total"> {{ ad.current }}/{{ ad.total }}</span>
            </div>
            <div v-if="ad.stage && ad.stage !== 'done' && ad.total" class="fm-anchor-bar">
              <div class="fm-anchor-fill" :style="{ width: (ad.total ? ad.current/ad.total*100 : 0) + '%' }"></div>
            </div>
          </div>
        </div>

        <div v-else-if="rollProgress" class="fm-prog-line">
          <span v-if="rollProgress.anchor_total">
            锚点 {{ rollProgress.anchor_idx }}/{{ rollProgress.anchor_total }} ({{ rollProgress.anchor }})
            · {{ stageLabel }}
            {{ rollProgress.current }}/{{ rollProgress.total }}
            <span v-if="rollProgress.factor"> · {{ rollProgress.factor }}</span>
          </span>
          <span v-else>{{ stageLabel }}：{{ rollProgress.current }}/{{ rollProgress.total }}</span>
        </div>
      </div>

      <div v-if="rollResult" class="fm-result">
        <h4 class="fm-result-title">各期通过因子（final_active 已写入动态池：{{ rollResult.final_active?.length || 0 }} 个）</h4>
        <table class="fm-table">
          <thead><tr><th>锚点日期</th><th>通过数</th><th>通过因子</th></tr></thead>
          <tbody>
            <tr v-for="a in rollResult.anchors" :key="a">
              <td><code>{{ a }}</code></td>
              <td>{{ rollResult.per_anchor[a]?.passed?.length || 0 }}</td>
              <td class="fm-cell-wrap" :title="(rollResult.per_anchor[a]?.passed || []).join(', ')">
                {{ (rollResult.per_anchor[a]?.passed || []).join(', ') || '—' }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- ═══════════════════════ 滚动训练 ═══════════════════════ -->
    <div v-if="activeTab === 'train'" class="fm-section">
      <div class="card">
        <div class="card-title">
          滚动时序训练
          <span class="text-muted" style="font-size:var(--fs-body);font-weight:400;margin-left:8px">
            按季度锚点滚动 · 每轮独立实验 · 全期样本外拼接评估
          </span>
        </div>

        <div class="fm-form-title">模型配置</div>
        <div class="fm-form-row fm-form-row--grid">
          <label class="fm-field">因子集 <select v-model="trainCfg.factor_set" @change="onTrainFactorSetChange">
            <option>Alpha158</option><option>Alpha360</option>
            <option value="__pool__">📦 因子池因子</option>
            <option value="__curated__">精选因子集</option>
          </select></label>
          <label class="fm-field">股票池 <select v-model="trainCfg.pool">
            <option>csi300</option><option>csi500</option><option>csi300+csi500</option><option>all</option><option value="stable_history_500d">稳定历史池500D</option>
          </select></label>
          <div v-if="trainCfg.factor_set === '__pool__'" class="fm-field-span">
            <span class="text-muted" style="font-size:var(--fs-sm)">动态因子池活跃因子 {{ activeCount }} 个（训练时后端自动去同源）</span>
          </div>
          <div v-if="trainCfg.factor_set === '__curated__'" class="fm-field-span">
            <label class="fm-field">精选集 <select v-model="trainCfg.use_curated" style="width:200px">
              <option value="latest">最新版本</option>
              <option v-for="s in activeCuratedSets" :key="s.name" :value="s.name">{{ s.name }} ({{ s.count }}因子, {{ s.created?.slice(0,10) }}) · {{ s.production_eligible ? '生产准入' : s.qualification === 'research_only' ? '仅研究' : '资格未核验' }}</option>
            </select></label>
          </div>
          <label class="fm-check fm-field-span"><input type="checkbox" v-model="trainCfg.train_dual" /> 双树模型集成（按信息系数加权融合）</label>
          <label class="fm-check fm-field-span"><input type="checkbox" v-model="trainCfg.enable_crossover" /> ✕ 因子交叉（方差排名前15的因子两两相乘，生成105个新特征）</label>
          <label class="fm-field">标签标准化 <select v-model="trainCfg.label_norm">
            <option value="csrank">截面排名 (csrank — 抗厚尾)</option>
            <option value="cszscore">截面标准分（旧默认）</option>
            <option value="none">不标准化</option>
          </select></label>
          <label class="fm-field">标签周期 <input v-model.number="trainCfg.label_horizon" type="number" min="1" max="60" class="fm-input-narrow" /> 日</label>
        </div>

        <div class="fm-form-title">高级选项</div>
        <div class="fm-form-row fm-form-row--grid">
          <label class="fm-field">早停指标 <select v-model="trainCfg.early_stop_metric">
            <option value="rank_ic">排名信息系数（排序相关，推荐）</option>
            <option value="mse">均方误差（回归损失，旧默认）</option>
          </select></label>
          <label class="fm-field">超参方案 <select v-model="trainCfg.param_profile">
            <option value="default">默认 (depth=7/leaves=127/λ=0.5)</option>
            <option value="conservative">保守 (depth=5/leaves=63/λ=0.8 — 稳定池)</option>
            <option value="aggressive">激进 (depth=8/leaves=200/λ=0.15)</option>
          </select></label>
        </div>
        <div class="fm-checks-row">
          <label class="fm-check"><input type="checkbox" v-model="trainCfg.use_reweighter" /> ⚖ 流动性加权+次新剔除</label>
          <label class="fm-check"><input type="checkbox" v-model="trainCfg.with_linear" /> 📐 Ridge线性基线</label>
          <label class="fm-check"><input type="checkbox" v-model="trainCfg.with_catboost" /> 🐱 类别提升第三模型</label>
        </div>
        <div class="fm-form-row">
          <span class="fm-field-label">代码前缀</span>
          <label class="fm-check" v-for="pf in PREFIX_OPTS" :key="pf[0][0]">
            <input type="checkbox" :value="pf[1]" v-model="trainPrefixLabels" /> {{ pf[1] }}
          </label>
          <span v-if="!trainPrefixLabels.length" class="text-muted" style="font-size:var(--fs-caption)">不选=全部</span>
        </div>

        <label v-if="trainCfg.factor_set === '__pool__'" class="fm-banner">
          <input type="checkbox" v-model="trainCfg.filter_mode" disabled />
          <strong>🔍 嵌入式二次筛选</strong>
          <span class="text-muted" style="font-size:var(--fs-sm);margin-left:4px">— 历史快照滚动训练不会将样本外结果回写当前因子池</span>
        </label>

        <div class="fm-form-title">滚动窗口设置</div>
        <div class="fm-form-row fm-form-row--grid">
          <label class="fm-field">起始锚点 <input v-model="trainCfg.start_anchor" placeholder="第一个样本外测试段起点" /></label>
          <label class="fm-field">结束锚点 <input v-model="trainCfg.end_anchor" placeholder="空=今天" /></label>
          <label class="fm-field">滚动步长 <select v-model.number="trainCfg.step_months">
            <option :value="1">1个月</option><option :value="3">3个月 (季度)</option><option :value="6">6个月</option>
          </select></label>
          <label class="fm-field">训练窗口 <select v-model.number="trainCfg.train_years">
            <option :value="3">3年</option><option :value="5">5年</option><option :value="8">8年</option>
          </select></label>
          <label class="fm-field">验证窗口 <select v-model.number="trainCfg.valid_months">
            <option :value="3">3个月</option><option :value="6">6个月 (保持早停阈值稳定)</option><option :value="12">12个月</option>
          </select></label>
        </div>

        <div class="fm-preview">
          <div class="fm-preview-title">📐 第一轮示例（训练/验证段均在锚点之前，测试段 = 锚点→锚点+步长）</div>
          <div class="fm-preview-bar">
            <span class="fm-preview-tag fm-preview-tag--train">训练 {{ trainFirstRound.train }}</span>
            <span class="fm-preview-arr">→</span>
            <span class="fm-preview-tag fm-preview-tag--valid">验证 {{ trainFirstRound.valid }}</span>
            <span class="fm-preview-arr">→</span>
            <span class="fm-preview-tag fm-preview-tag--test">测试 {{ trainFirstRound.test }}</span>
          </div>
          <div v-if="trainFirstRound.nRounds !== null" class="text-muted" style="font-size:var(--fs-sm);margin-top:4px">
            预计 {{ trainFirstRound.nRounds }} 轮（{{ trainCfg.start_anchor }} ~ {{ trainFirstRound.lastAnchor }}）
          </div>
        </div>

        <button @click="startTraining" :disabled="training" class="btn" style="margin-top:16px">
          {{ training ? '滚动训练中...' : '开始滚动训练' }}
        </button>

        <div v-if="training" class="fm-progress" style="margin-top:14px">
          <div class="fm-prog-head">
            <span class="fm-prog-label">{{ trainStatusText }}</span>
            <span class="fm-prog-pct">{{ trainPct }}%</span>
          </div>
          <div class="fm-prog-bar"><div class="fm-prog-fill" :style="{ width: trainPct + '%' }"></div></div>
        </div>
        <p v-if="trainError && !training" class="error">{{ trainError }}</p>

        <div v-if="trainResult" class="fm-train-result">
          <h3 class="fm-result-title">{{ trainResult.n_trained }} / {{ trainResult.n_planned }} 轮完成</h3>

          <!-- OOS 指标 -->
          <div v-if="trainResult.oos && trainResult.oos.ic_mean != null" class="fm-metrics-strip">
            <div class="fm-metric">
              <div class="data-value data-value--lg" style="color:var(--success)">{{ (trainResult.oos.ic_mean * 100).toFixed(2) }}%</div>
              <div class="fm-metric-label">全期样本外平均信息系数</div>
            </div>
            <div class="fm-metric">
              <div class="data-value data-value--lg" style="color:var(--success)">{{ (trainResult.oos.ric_mean * 100).toFixed(2) }}%</div>
              <div class="fm-metric-label">全期样本外排名信息系数</div>
            </div>
            <div class="fm-metric">
              <div class="data-value data-value--lg" style="color:var(--success)">{{ trainResult.oos.icir?.toFixed(2) || '--' }}</div>
              <div class="fm-metric-label">全期样本外信息系数稳定性</div>
            </div>
            <div v-if="trainResult.oos.ic_win_rate_m != null" class="fm-metric">
              <div class="data-value data-value--lg">{{ (trainResult.oos.ic_win_rate_m * 100).toFixed(0) }}%</div>
              <div class="fm-metric-label">月度IC胜率</div>
            </div>
            <div class="fm-metric">
              <div class="data-value data-value--lg">{{ trainResult.oos.n_days || '--' }}</div>
              <div class="fm-metric-label">评估交易日</div>
            </div>
          </div>

          <!-- 筛选结果 -->
          <div v-if="trainResult.filter_result" class="fm-filter-result">
            <h3 style="color:var(--success);margin:0 0 8px">✅ 因子池已更新</h3>
            <div class="fm-metrics-strip" style="background:none;padding:0">
              <div class="fm-metric">
                <div class="data-value data-value--lg" style="color:var(--success)">{{ trainResult.filter_result.n_passed }}</div>
                <div class="fm-metric-label">保留因子 (active=true)</div>
              </div>
              <div class="fm-metric">
                <div class="data-value data-value--lg" style="color:var(--danger)">{{ trainResult.filter_result.n_failed }}</div>
                <div class="fm-metric-label">淘汰因子 (active=false)</div>
              </div>
            </div>
            <details v-if="trainResult.filter_result.failed.length" style="margin-top:8px">
              <summary class="fm-detail-summary" style="color:var(--danger)">淘汰因子明细 ({{ trainResult.filter_result.failed.length }} 个)</summary>
              <div v-for="key in trainResult.filter_result.failed" :key="key" class="fm-failed-item">❌ {{ key }}</div>
            </details>
          </div>

          <!-- 因子重要性排名 -->
          <div v-if="sortedImportance.length" class="fm-importance">
            <h3 style="margin:0 0 8px">📊 因子重要性排名（{{ sortedImportance.length }} 个因子）</h3>
            <div style="max-height:400px;overflow-y:auto">
              <table class="fm-table">
                <thead><tr>
                  <th>#</th><th>因子</th><th>平均Split</th><th>胜率</th><th>波动率</th><th>轮次</th><th>状态</th>
                </tr></thead>
                <tbody>
                  <tr v-for="(item, idx) in sortedImportance" :key="item[0]"
                      :class="{ 'fm-row--eliminated': eliminatedSet.has(item[0]) }">
                    <td class="td-right">{{ idx + 1 }}</td>
                    <td class="fm-cell-name" :title="item[0]">{{ item[0] }}</td>
                    <td>{{ item[1].mean_split.toFixed(4) }}</td>
                    <td>{{ (item[1].win_rate * 100).toFixed(0) }}%</td>
                    <td>{{ item[1].volatility.toFixed(2) }}</td>
                    <td>{{ item[1].n_rounds }}</td>
                    <td>
                      <span v-if="eliminatedSet.has(item[0])" class="tag tag-high">淘汰</span>
                      <span v-else class="tag tag-low">保留</span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <!-- 各轮详情 -->
          <details v-if="trainResult.rounds?.length" class="fm-rounds">
            <summary class="fm-detail-summary">各轮训练详情 ({{ trainResult.rounds.length }} 轮)</summary>
            <div v-for="(r, i) in trainResult.rounds" :key="i" class="fm-model-card">
              <div class="fm-model-head">
                <span class="fm-model-name">
                  {{ r.anchor }}
                  <span v-if="r.error" class="tag tag-high">失败</span>
                  <span v-if="r.degraded?.length" class="tag tag-high">降级: {{ r.degraded.join(', ') }}</span>
                </span>
              </div>
              <div v-if="r.factor_selection_cutoff" style="font-size:var(--fs-sm);margin:6px 0;color:var(--text-secondary)">
                因子快照：{{ r.factor_selection_source }}；训练数据截止 {{ r.factor_selection_cutoff }}；
                快照生效 {{ r.factor_selection_effective_date }}；{{ r.custom_factor_keys?.length || 0 }} 个因子
                <details v-if="r.custom_factor_keys?.length" style="margin-top:4px">
                  <summary class="fm-detail-summary">查看本轮实际因子与指纹</summary>
                  <div style="word-break:break-word">{{ r.custom_factor_keys.join(', ') }}</div>
                  <code v-if="r.factor_selection_fingerprint">{{ r.factor_selection_fingerprint }}</code>
                </details>
              </div>
              <div v-if="!r.error" class="fm-model-metrics">
                <span class="fm-mm"><span class="fm-mm-label">IC</span><span class="fm-mm-val">{{ r.ic_mean ? (r.ic_mean * 100).toFixed(2) + '%' : '--' }}</span></span>
                <span class="fm-mm"><span class="fm-mm-label">排名信息系数</span><span class="fm-mm-val">{{ r.ric_mean ? (r.ric_mean * 100).toFixed(2) + '%' : '--' }}</span></span>
                <span class="fm-mm"><span class="fm-mm-label">稳定性</span><span class="fm-mm-val">{{ r.icir?.toFixed(2) || '--' }}</span></span>
                <span class="fm-mm"><span class="fm-mm-label">LGB W</span><span class="fm-mm-val">{{ r.lgb_weight ? (r.lgb_weight * 100).toFixed(0) + '%' : '--' }}</span></span>
                <span class="fm-mm"><span class="fm-mm-label">XGB W</span><span class="fm-mm-val">{{ r.xgb_weight ? (r.xgb_weight * 100).toFixed(0) + '%' : '--' }}</span></span>
              </div>
              <div v-else class="error" style="font-size:var(--fs-sm)">{{ r.error }}</div>
            </div>
          </details>
        </div>
      </div>
    </div>

    <!-- ═══════════════════════ 因子池 ═══════════════════════ -->
    <div v-if="activeTab === 'pool'" class="fm-section">
      <div v-if="poolMsg" class="msg">{{ poolMsg }}</div>

      <div class="fm-toolbar">
        <span class="fm-toolbar-stat">共 {{ poolFactors.length }} 个因子，活跃 {{ activeCount }} 个</span>
        <input v-model="poolSearch" placeholder="搜索因子名/标签..." class="fm-search" />
        <select v-model="poolFilter" class="fm-select-sm">
          <option value="all">全部</option>
          <option value="active">活跃</option>
          <option value="inactive">停用</option>
          <option value="white_box">白盒</option>
          <option value="traditional">传统</option>
        </select>
        <button class="btn-sm btn-success" @click="toggleAllFactors(true)">启用全部</button>
        <button class="btn-sm btn-warning" @click="toggleAllFactors(false)">停用全部</button>
        <button class="btn-sm btn-danger" @click="deleteSelected" :disabled="!selected.length">删除选中 ({{ selected.length }})</button>
        <button class="btn-sm btn-outline" @click="loadPool">刷新</button>
      </div>

      <table class="fm-table" v-if="pagedFactors.length">
        <thead>
          <tr>
            <th style="width:30px"><input type="checkbox" @change="toggleAll" :checked="allSelected" /></th>
            <th>因子名</th>
            <th>标签</th>
            <th>类别</th>
            <th>参数</th>
            <th>拥挤</th>
            <th>状态</th>
            <th style="width:160px">操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="f in pagedFactors" :key="f.key" :class="{ 'fm-row--inactive': !f.active }">
            <td><input type="checkbox" :value="f.key" v-model="selected" /></td>
            <td><code>{{ f.key }}</code></td>
            <td>{{ f.label }}</td>
            <td><span :class="['tag', f.category==='white_box'?'tag-mid':'tag-low']">{{ f.category === 'white_box' ? '白盒' : f.category === 'traditional' ? '传统' : f.category }}</span></td>
            <td class="fm-cell-wrap">{{ formatParams(f.params) }}</td>
            <td>
              <span v-if="crowdingMap[f.key]" :class="['fm-dot', 'fm-dot--' + crowdingMap[f.key]]" :title="'拥挤度: ' + crowdingMap[f.key]"></span>
              <span v-else class="fm-dot fm-dot--unknown" title="未检测"></span>
            </td>
            <td>
              <span class="tag" :class="f.active ? 'tag-low' : 'tag-high'" style="cursor:pointer" @click="toggleFactor(f.key, !f.active)">
                {{ f.active ? '活跃' : '停用' }}
              </span>
            </td>
            <td>
              <button class="btn-mini" @click="runPortfolioSort(f.key)" :disabled="sortRunning === f.key">{{ sortRunning === f.key ? '...' : '分组回测' }}</button>
              <button class="btn-mini btn-mini--danger" @click="deleteFactor(f.key)">删除</button>
            </td>
          </tr>
          <tr v-if="sortResult" class="fm-sort-row">
            <td colspan="8">
              <div class="fm-sort-card">
                <strong>{{ sortResult.key }}</strong> 分组回测 ({{ sortResult.n_dates }}期):
                Q1 {{ sortResult.q1?.toFixed(3) }} → Q2 {{ sortResult.q2?.toFixed(3) }} → Q3 {{ sortResult.q3?.toFixed(3) }} → Q4 {{ sortResult.q4?.toFixed(3) }} → Q5 {{ sortResult.q5?.toFixed(3) }}
                | 多空年化 {{ (sortResult.annual_return*100).toFixed(1) }}% | 夏普 {{ sortResult.sharpe?.toFixed(2) }} | 单调性 {{ (sortResult.monotonicity*100).toFixed(0) }}%
                <button class="btn-mini" @click="sortResult = null">✕</button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>

      <div class="fm-toolbar" style="margin-top:8px">
        <button class="btn-sm btn-outline" @click="loadCrowding" :disabled="loadingCrowding">{{ loadingCrowding ? '检测中...' : '检测拥挤度' }}</button>
        <button class="btn-sm btn-outline" @click="runFamaMacBeth" :disabled="fmRunning">{{ fmRunning ? 'FM回归中...' : 'Fama-MacBeth 定价检验' }}</button>
        <span v-if="fmResult" class="fm-toolbar-stat" style="margin-left:12px">
          显著因子: {{ fmResult.filter(r=>r.significant).length }}/{{ fmResult.length }}
          <button class="btn-mini" @click="fmResult=null">✕</button>
        </span>
      </div>

      <div class="fm-pager" v-if="totalPages > 1">
        <button @click="poolPage=1" :disabled="poolPage===1">首页</button>
        <button @click="poolPage--" :disabled="poolPage===1">上一页</button>
        <span>第 {{ poolPage }} / {{ totalPages }} 页 (共 {{ filteredFactors.length }} 个)</span>
        <button @click="poolPage++" :disabled="poolPage===totalPages">下一页</button>
        <button @click="poolPage=totalPages" :disabled="poolPage===totalPages">末页</button>
      </div>
      <AppEmpty v-else-if="!poolFactors.length" text="因子池为空" action-label="去滚动筛选" @action="activeTab = 'rolling'" />
    </div>

    <!-- ═══════════════════════ 精选集 ═══════════════════════ -->
    <div v-if="activeTab === 'curated'" class="fm-section">
      <div v-if="curatedMsg" class="msg" :class="{ error: curatedError }">{{ curatedMsg }}</div>

      <div class="fm-toolbar">
        <span class="fm-toolbar-stat">{{ activeCuratedSets.length ? `${activeCuratedSets.length} 套精选集` : '暂无精选集' }}</span>
        <button class="btn" @click="startCurate" :disabled="curating">
          {{ curating ? `精选中... ${curatePct}%` : '生成精选集' }}
        </button>
        <button class="btn-sm btn-outline" @click="loadCuratedSets">刷新</button>
      </div>

      <div v-if="curating" class="fm-progress" style="margin-bottom:12px">
        <div class="fm-prog-line">{{ curateMsg }}</div>
        <div class="fm-prog-bar"><div class="fm-prog-fill" :style="{width:curatePct+'%'}"></div></div>
      </div>

      <table class="fm-table" v-if="activeCuratedSets.length">
        <thead>
          <tr>
            <th>名称</th><th>创建时间</th><th>因子数</th><th>用途资格</th><th>跨类最大相关</th><th style="width:120px">操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="s in activeCuratedSets" :key="s.name">
            <td><code>{{ s.name }}</code></td>
            <td>{{ s.created?.slice(0, 16) || '-' }}</td>
            <td>{{ s.count }}</td>
            <td>{{ s.production_eligible ? '生产准入' : s.qualification === 'research_only' ? '研究候选·不可生产' : '资格未核验' }}</td>
            <td>{{ s.max_cross_corr != null ? s.max_cross_corr.toFixed(3) : '-' }}</td>
            <td>
              <button class="btn-mini btn-mini--primary" @click="viewCuratedSet(s.name)">详情</button>
              <button class="btn-mini btn-mini--danger" @click="deleteCuratedSet(s.name)">删除</button>
            </td>
          </tr>
        </tbody>
      </table>

      <!-- 精选集详情弹窗 -->
      <div v-if="curatedDetail" class="card" style="margin-top:16px">
        <div class="flex-between" style="margin-bottom:12px">
          <h3 style="margin:0">📋 {{ curatedDetail.name }}</h3>
          <button class="btn-sm btn-outline" @click="curatedDetail = null">关闭</button>
        </div>
        <div class="text-muted" style="font-size:var(--fs-sm);margin-bottom:12px">
          创建: {{ curatedDetail.created?.slice(0, 16) }} |
          总因子: {{ curatedDetail.stats?.total || curatedDetail.flat_list?.length || 0 }} |
          用途资格: {{ curatedDetail.production_eligible ? '生产准入' : curatedDetail.qualification === 'research_only' ? '研究候选·不可生产' : '资格未核验' }} |
          跨类最大相关: {{ curatedDetail.stats?.max_cross_corr?.toFixed(3) || '-' }}
        </div>
        <div v-if="curatedDetail.factors" style="margin-bottom:12px">
          <h4 class="fm-form-title">8大类分布</h4>
          <div class="fm-cat-grid">
            <div v-for="(keys, cat) in curatedDetail.factors" :key="cat" class="fm-cat-card">
              <div class="fm-cat-name">{{ cat }}</div>
              <div class="data-value data-value--lg">{{ Array.isArray(keys) ? keys.length : 0 }}</div>
              <div class="fm-cat-keys">{{ Array.isArray(keys) ? keys.slice(0,5).join(', ') + (keys.length>5?'...':'') : '-' }}</div>
            </div>
          </div>
        </div>
        <div v-if="curatedDetail.config" class="text-muted" style="font-size:var(--fs-caption)">
          配置: 目标 {{ curatedDetail.config.target_range?.join('-') }} |
          类内阈值 {{ curatedDetail.config.within_corr_threshold || 0.65 }} |
          跨类阈值 {{ curatedDetail.config.cross_corr_threshold || 0.70 }}
        </div>
      </div>
    </div>

    <!-- ═══════════════════════ 策略方向 ═══════════════════════ -->
    <div v-if="activeTab === 'direction'" class="fm-section">
      <div class="fm-hint">按策略方向自动精选因子。趋势动量=追涨、反转防御=抄底、量价突破=放量信号。</div>

      <div class="fm-regime-bar">
        <span class="text-muted" style="font-size:var(--fs-body)">当前市场状态:</span>
        <span v-if="regimeLoading" class="text-muted" style="font-size:var(--fs-body)">加载中...</span>
        <span v-else-if="currentRegime" class="fm-regime-tag" :style="regimeTagStyle">{{ regimeLabel(currentRegime.regime) }}</span>
        <span v-else class="text-muted" style="font-size:var(--fs-body)">未获取 (需回填数据)</span>
        <button class="btn-sm btn-outline" @click="loadRegimeStatus" :disabled="regimeLoading">刷新状态</button>
      </div>

      <div class="fm-toolbar" style="margin-bottom:12px">
        <button class="btn" @click="recalcAllIcir" :disabled="recalcRunning">
          {{ recalcRunning ? `重算中... ${recalcPct}%` : '重新计算全部因子稳定性' }}
        </button>
        <span v-if="recalcResult" class="text-success" style="font-size:var(--fs-sm);margin-left:8px">{{ recalcResult }}</span>
      </div>
      <div v-if="dirMsg" class="msg">{{ dirMsg }}</div>

      <div class="fm-dir-grid">
        <div v-for="d in directions" :key="d.key" class="fm-dir-card">
          <h4 class="fm-dir-name">{{ d.icon }} {{ d.label }}</h4>
          <p class="fm-dir-desc">{{ d.desc }}</p>
          <div class="fm-dir-cats">{{ d.categories?.join(', ') }}</div>
          <div v-if="d.status === 'loading'" class="text-muted" style="font-size:var(--fs-sm)">加载中...</div>
          <div v-else class="fm-dir-stat">
            <span>{{ d.count || 0 }} 个因子</span>
            <span v-if="d.avg_icir != null && d.avg_icir > 0" class="text-muted" style="margin-left:8px;font-size:var(--fs-caption)">平均稳定性 {{ d.avg_icir.toFixed(4) }}</span>
            <span v-else-if="d.avg_icir === 0" class="text-success" style="margin-left:8px;font-size:var(--fs-caption)">(均匀采样)</span>
          </div>
          <div v-if="!d.pinned" class="fm-dir-actions">
            <button class="btn-sm" @click="doCurateDirection(d.key)" :disabled="d.status === 'loading'">
              {{ d.status === 'loading' ? '...' : '更新精选' }}
            </button>
            <button class="btn-sm btn-outline" @click="doBackfillIcir(d.key)" :disabled="d.status === 'loading'">回填稳定性</button>
          </div>
          <div v-else class="fm-dir-actions">
            <span class="text-muted" style="font-size:var(--fs-sm)">研究候选；可用于隔离研究训练，不具备生产准入资格</span>
          </div>
          <div v-if="d.factors?.length" class="fm-dir-factors">
            {{ d.factors.slice(0,20).join(', ') }}{{ d.factors.length > 20 ? '...' : '' }}
          </div>
        </div>
      </div>
    </div>

    <!-- ═══════════════════════ 遗传挖矿 ═══════════════════════ -->
    <div v-if="activeTab === 'genetic'" class="fm-section">
      <div class="fm-hint">
        因子挖掘闭环：候选生成 → 训练筛选 → 现有层级校验 → 5折滚动样本外验证 → 增量评估。
        所有表达式会先与生产因子池和历史挖掘档案去重；通过后只进入“待人工晋级”，不会自动写入生产因子池。
      </div>
      <div v-if="geneticMsg" class="msg" :class="{ error: geneticError }">{{ geneticMsg }}</div>

      <div class="fm-form-block">
        <div class="fm-form-title">数据与范围</div>
        <div class="fm-form-row">
          <label class="fm-field">股票池
            <select v-model="gp.pool" @change="resetMiningModel">
              <option value="csi300">沪深300</option>
              <option value="csi500">中证500</option>
              <option value="csi300+csi500">沪深300+中证500</option>
              <option value="stable_history_500d">稳定历史池500D</option>
              <option value="all">全市场（慢）</option>
            </select>
          </label>
          <label class="fm-field">训练起始 <input type="date" v-model="gp.train_start" /></label>
          <label class="fm-field">训练结束 <input type="date" v-model="gp.train_end" @change="resetMiningModel" /></label>
          <label class="fm-field">验证起始 <input type="date" v-model="gp.valid_start" /></label>
          <label class="fm-field">验证结束 <input type="date" v-model="gp.valid_end" /></label>
        </div>
      </div>

      <div class="fm-form-block">
        <div class="fm-form-title">候选来源</div>
        <div class="fm-form-row">
          <label class="fm-field">生成方式
            <select v-model="gp.candidate_mode">
              <option value="gp">仅遗传编程（默认）</option>
              <option value="hybrid">遗传编程与大语言模型</option>
              <option value="llm">仅大语言模型</option>
            </select>
          </label>
          <template v-if="gp.candidate_mode !== 'gp'">
            <label class="fm-field">大语言模型提供商
              <select v-model="gp.llm_provider"><option value="deepseek">DeepSeek</option><option value="qwen">通义千问</option></select>
            </label>
            <label class="fm-field">提议数 <input type="number" v-model.number="gp.llm_candidate_count" min="1" max="50" class="fm-input-narrow" /></label>
            <label class="fm-field">生成随机度 <input type="number" v-model.number="gp.llm_temperature" min="0" max="1" step="0.1" class="fm-input-narrow" /></label>
            <label class="fm-field fm-field-wide">研究主题（可选）
              <textarea v-model="gp.llm_research_focus" maxlength="500" rows="2" placeholder="例如：寻找低换手、对波动率状态更稳健的量价因子"></textarea>
            </label>
          </template>
        </div>
        <div v-if="gp.candidate_mode !== 'gp'" class="fm-hint fm-hint--inline">
          大语言模型只生成表达式与研究假设，不会看到最终样本外或回测结果；所有候选仍执行相同验证和人工晋级。
        </div>
      </div>

      <div v-if="gp.candidate_mode !== 'llm'" class="fm-form-block">
        <div class="fm-form-title">遗传编程参数</div>
        <div class="fm-form-row">
          <label class="fm-field">种群 <input type="number" v-model.number="gp.population" min="20" class="fm-input-narrow" /></label>
          <label class="fm-field">代数 <input type="number" v-model.number="gp.generations" min="1" class="fm-input-narrow" /></label>
          <label class="fm-field">最大深度 <input type="number" v-model.number="gp.max_depth" min="2" max="8" class="fm-input-narrow" /></label>
          <label class="fm-field">候选数量 <input type="number" v-model.number="gp.top_n" min="1" class="fm-input-narrow" /></label>
          <label class="fm-field">前瞻天数 <input type="number" v-model.number="gp.forward_days" min="5" class="fm-input-narrow" /></label>
          <label class="fm-field">评估日步长 <input type="number" v-model.number="gp.step" min="1" class="fm-input-narrow" /></label>
          <label class="fm-field">随机种子 <input type="number" v-model.number="gp.seed" min="0" class="fm-input-narrow" /></label>
        </div>
      </div>

      <div class="fm-form-block">
        <div class="fm-form-title">验证策略</div>
        <div class="fm-form-row fm-form-row--end">
          <label class="fm-check"><input type="checkbox" v-model="gp.risk_neutralize" /> 五因子中性化（动量/换手/波动）</label>
          <label class="fm-field">验证配置
            <select v-model="gp.validation_profile"><option value="default">默认验证</option></select>
          </label>
          <label class="fm-field">基线
            <select v-model="gp.baseline_set" @change="resetMiningModel">
              <option value="current_curated">当前精选集</option>
              <option v-for="s in activeCuratedSets" :key="s.name" :value="s.name">{{ s.name }}（{{ s.count }}因子）</option>
            </select>
          </label>
          <label class="fm-field fm-field-wide">量化基线模型
            <select v-model="gp.baseline_model_key" @focus="loadMiningModels">
              <option value="auto">自动匹配可用模型</option>
              <option v-for="m in miningModels" :key="miningModelKey(m)" :value="miningModelKey(m)" :disabled="!m.compatible">
                {{ miningModelLabel(m) }}
              </option>
            </select>
          </label>
          <button type="button" class="btn-sm btn-outline" @click="loadMiningModels" :disabled="miningModelsLoading">
            {{ miningModelsLoading ? '加载中…' : '刷新模型' }}
          </button>
          <label class="fm-field">最终样本外起始 <input type="date" v-model="gp.final_oos_start" /></label>
          <label class="fm-field">最终样本外结束 <input type="date" v-model="gp.final_oos_end" /></label>
        </div>
        <div v-if="selectedMiningModel" class="fm-hint fm-hint--inline">
          已固定 {{ selectedMiningModel.experiment_id }}/{{ selectedMiningModel.run_id }}；
          数据截止 {{ selectedMiningModel.observed_end || '--' }}；信息系数 {{ formatIc(selectedMiningModel.ic) }}；
          基线覆盖 {{ formatIc(selectedMiningModel.baseline_coverage) }}。
          <span v-if="selectedMiningModel.compatibility_warnings?.length">{{ selectedMiningModel.compatibility_warnings.join('；') }}</span>
        </div>
        <div v-else-if="miningModelsLoaded" class="fm-hint fm-hint--inline">
          {{ compatibleMiningModels.length }} 个模型满足当前股票池、基线和训练截止日约束；自动模式将选择覆盖度最高的模型。
        </div>
        <div v-if="miningModelsError" class="msg error">{{ miningModelsError }}</div>
      </div>

      <div class="fm-form-block">
        <div class="fm-form-title">生产对照实验配置</div>
        <div class="fm-form-row">
          <label class="fm-field">调仓频率
            <select v-model="gp.backtest_freq"><option value="week">周频</option><option value="biweek">双周</option><option value="month">月频</option></select>
          </label>
          <label class="fm-field">持仓数 <input type="number" v-model.number="gp.backtest_top_n" min="1" class="fm-input-narrow" /></label>
          <label class="fm-field">候选权重 <input type="number" v-model.number="gp.overlay_weight" min="0" max="0.5" step="0.01" class="fm-input-narrow" /></label>
          <label class="fm-check"><input type="checkbox" v-model="gp.backtest_use_timing" /> 使用生产择时</label>
          <label class="fm-check"><input type="checkbox" v-model="gp.backtest_macro_bias" /> 使用宏观偏置</label>
        </div>
      </div>

      <div class="fm-form-block">
        <div class="fm-form-title">执行模式</div>
        <div class="fm-form-row">
          <label class="fm-field">模式
            <select v-model="gp.execution_mode">
              <option value="single">单次完整挖掘</option>
              <option value="continuous">空闲时持续挖掘</option>
            </select>
          </label>
          <template v-if="gp.execution_mode === 'continuous'">
            <label class="fm-field">最多批次 <input type="number" v-model.number="gp.max_batches" min="0" class="fm-input-narrow" /></label>
            <label class="fm-field">最多小时 <input type="number" v-model.number="gp.max_hours" min="0" step="0.5" class="fm-input-narrow" /></label>
            <label class="fm-field">处理器使用率上限% <input type="number" v-model.number="gp.idle_cpu_percent" min="10" max="100" class="fm-input-narrow" /></label>
            <label class="fm-field">最少空闲内存（吉字节）<input type="number" v-model.number="gp.min_free_memory_gb" min="1" step="1" class="fm-input-narrow" /></label>
          </template>
        </div>
        <div v-if="gp.execution_mode === 'continuous'" class="fm-hint fm-hint--inline">
          0批次且0小时表示持续运行到手工终止。每批使用新种子，并跳过历史已见表达式；冻结最终样本外只留给单次晋级评估。
        </div>
      </div>

      <div class="fm-form-block">
        <div class="fm-form-title">执行操作</div>
        <div class="fm-form-row fm-form-row--end">
          <button class="btn" @click="startGeneticMine" :disabled="geneticMining">
            {{ geneticMining ? '挖掘中…' : '开始挖掘' }}
          </button>
          <button v-if="geneticResumeAvailable && ['interrupted', 'failed'].includes(geneticStatus)" class="btn btn-outline" @click="resumeGeneticMine" :disabled="geneticMining">
            从已验证代际检查点恢复
          </button>
          <button v-if="geneticMining" class="btn btn-outline" @click="cancelGeneticMine">取消任务</button>
          <button v-if="geneticStatus === 'stalled'" class="btn btn-outline" @click="refreshGenetic">刷新状态</button>
          <button class="btn btn-outline" @click="loadMiningArchive" :disabled="miningArchiveLoading">{{ miningArchiveLoading ? '加载中…' : '查看全局档案' }}</button>
        </div>
      </div>

      <div v-if="geneticMining || geneticProgress" class="fm-progress">
        <div class="fm-prog-head">
          <span class="fm-prog-label">
            {{ geneticStageText }}
            <span v-if="geneticElapsed" class="fm-prog-time">{{ geneticElapsed }}</span>
          </span>
          <span class="fm-prog-pct">{{ geneticPct }}%</span>
        </div>
        <div class="fm-prog-bar"><div class="fm-prog-fill" :style="{ width: geneticPct + '%' }"></div></div>
        <div class="fm-prog-meta">
          候选 {{ geneticProgress?.candidate_count || 0 }} · 通过 {{ geneticProgress?.passed_count || 0 }}
          · 训练最佳 {{ formatIc(geneticProgress?.best_train_ic) }} · 样本外最佳 {{ formatIc(geneticProgress?.best_oos_ic) }}
          <span v-if="geneticProgress?.generation_source">· 来源 {{ sourceText(geneticProgress.generation_source) }}</span>
          <span v-if="geneticProgress?.last_date">· 最后日期 {{ geneticProgress.last_date }}</span>
        </div>
        <div v-if="geneticProgress?.batch_index != null" class="fm-prog-meta">
          持续批次 {{ geneticProgress.batch_index }}<span v-if="geneticProgress.total">/{{ geneticProgress.total }}</span>
          · 全局唯一表达式 {{ geneticProgress.unique_candidate_count || 0 }}
          <span v-if="geneticProgress.resource_state">· CPU {{ geneticProgress.resource_state.cpu_percent }}% · 空闲内存 {{ geneticProgress.resource_state.free_memory_gb }} GB</span>
        </div>
        <div v-if="geneticProgress?.snapshot_mode || geneticProgress?.rss_mb" class="fm-prog-meta">
          数据模式 {{ geneticProgress?.snapshot_mode === 'panel' ? '共享研究面板' : '兼容快照' }}
          <span v-if="geneticProgress?.panel_mb != null">· 面板 {{ geneticProgress.panel_mb }} MB</span>
          <span v-if="geneticProgress?.rss_mb != null">· 当前内存 {{ geneticProgress.rss_mb }} MB</span>
          <span v-if="geneticProgress?.peak_rss_mb != null">· 峰值 {{ geneticProgress.peak_rss_mb }} MB</span>
        </div>
        <div v-if="geneticStatus === 'stalled'" class="msg error">超过 10 分钟没有心跳，任务可能停滞；可刷新确认或取消。</div>
      </div>

      <div v-if="geneticResult" class="fm-result">
        <h4 class="fm-result-title">挖矿结果（{{ geneticCandidates.length }} 个候选，状态：{{ geneticStatusText }}）</h4>
        <div v-if="geneticResult.data_version" class="fm-hint">
          生产对照实验：{{ geneticResult.data_version.final_oos_start }} ~ {{ geneticResult.data_version.final_oos_end }}；
          数据 cutoff：{{ geneticResult.data_version.snapshot_cutoff }}；
          候选选择数：{{ geneticResult.selection_control?.candidate_count || 0 }}；
          <span v-if="geneticResult.backtest?.baseline_context?.model_info">
            量化模型：{{ geneticResult.backtest.baseline_context.model_info.run_id?.slice(0, 8) }}
            （{{ geneticResult.backtest.baseline_context.model_selection_mode === 'manual' ? '手工选择' : '自动匹配' }}）；
          </span>
          <span v-if="geneticResult.data_version.overlaps_validation">当前样本外区间与验证区间重叠，该结果不能申请晋级；下一次任务应使用冻结后的新区间。</span>
        </div>
        <div v-if="geneticResult.promotion_conclusion === 'no_candidate_passed'" class="msg error">
          流程已完成，但没有合格候选。滚动样本外评估 {{ geneticResult.result_summary?.walkforward_evaluated_count || 0 }} 个，生产对照实验评估 {{ geneticResult.result_summary?.strategy_evaluated_count || 0 }} 个。
        </div>
        <table class="fm-table">
          <thead>
            <tr><th>序号</th><th>候选</th><th>来源</th><th>表达式</th><th>复杂度</th><th>训练信息系数</th><th>样本外信息系数</th><th>稳定性</th><th>证据</th><th>操作</th></tr>
          </thead>
          <tbody>
            <tr v-for="(f, i) in geneticCandidates" :key="f.candidate_key || f.key">
              <td class="td-right">{{ i + 1 }}</td>
              <td><code>{{ f.candidate_key || f.key }}</code></td>
              <td><span class="tag tag-high">{{ sourceText(f.source) }}</span></td>
              <td class="fm-cell-wrap" :title="f.expr">{{ f.expr }}</td>
              <td>{{ f.complexity ?? '--' }}</td>
              <td>{{ formatIc(f.train_metrics?.ic ?? f.train_ic) }}</td>
              <td :style="{ color: (f.oos_metrics?.oos_ic ?? f.valid_ic ?? 0) >= 0 ? 'var(--success)' : 'var(--danger)' }">{{ formatIc(f.oos_metrics?.oos_ic ?? f.valid_ic) }}</td>
              <td>{{ f.oos_metrics?.icir != null ? Number(f.oos_metrics.icir).toFixed(2) : '--' }}</td>
              <td>
                <span class="tag" :class="f.status === 'awaiting_promotion' ? 'tag-low' : 'tag-high'">{{ candidateStatusText(f.status, f.in_pool) }}</span>
                <details v-if="f.oos_metrics?.folds_detail?.length || f.validation_layers?.layers?.length || f.strategy_metrics || f.detail"><summary>详情</summary><div class="fm-detail"><span v-if="proposalSummary(f)">{{ proposalSummary(f) }}<br /></span>折 {{ f.oos_metrics?.passed_folds || 0 }}/{{ f.oos_metrics?.folds || 0 }} · 衰减 {{ f.oos_metrics?.decay != null ? (f.oos_metrics.decay * 100).toFixed(1) + '%' : '--' }}<br />{{ validationSummary(f) }}<br /><span v-if="candidateFailureSummary(f)">{{ candidateFailureSummary(f) }}<br /></span>{{ abSummary(f) }}<br />{{ gateSummary(f) }}<div v-if="f.detail" class="fm-subdetail">已加载评分 {{ detailScoreCount(f) }} 条 · 回测 {{ detailBacktestCount(f) }} 条<br /><span v-for="item in detailWalkforwardScores(f)" :key="item.fold_index">折 {{ item.fold_index ?? '-' }}：{{ detailScoreSummary(item) }}<br /></span><span v-if="detailDailyIc(f).length">最近每日信息系数：<span v-for="point in detailDailyIc(f)" :key="point.date">{{ point.date }} {{ formatIc(point.ic) }}；</span></span></div></div></details>
              </td>
              <td class="fm-actions-cell">
                <button class="btn-sm btn-outline" @click="loadCandidateDetail(f)" :disabled="geneticDetailLoading[f.candidate_key || f.key]">{{ geneticDetailLoading[f.candidate_key || f.key] ? '加载中…' : '查看详情' }}</button>
                <button class="btn-sm btn-outline" @click="trainCandidate(f)" :disabled="candidateTraining">训练模型</button>
                <button v-if="f.status === 'awaiting_promotion'" class="btn-sm" @click="promoteCandidate(f)">申请晋级</button>
                <button v-if="f.status === 'awaiting_promotion'" class="btn-sm btn-outline" @click="rejectCandidate(f)">拒绝</button>
              </td>
            </tr>
          </tbody>
        </table>
        <div v-if="candidateTraining || candidateTrainingResult" class="fm-progress" style="margin-top:14px">
          <div class="fm-prog-head"><span>{{ candidateTrainingText }}</span><span v-if="candidateTrainingProgress?.factor_count">{{ candidateTrainingProgress.factor_count }} 个输入因子</span></div>
          <div v-if="candidateTrainingResult" class="fm-prog-meta">
            基线信息系数 {{ formatIc(candidateTrainingResult.baseline_ic_mean) }} · 增强信息系数 {{ formatIc(candidateTrainingResult.ic_mean) }}
            · 增量 {{ formatIc(candidateTrainingResult.ic_delta) }} · 稳定性 {{ candidateTrainingResult.icir == null ? '--' : Number(candidateTrainingResult.icir).toFixed(2) }} · 未激活
          </div>
        </div>
      </div>
      <details v-if="miningArchive.length" class="fm-result">
        <summary>全局防重复档案（展示 {{ miningArchive.length }} 条）</summary>
        <table class="fm-table" style="margin-top:10px">
          <thead><tr><th>候选</th><th>表达式</th><th>首次任务</th><th>出现次数</th><th>最佳训练适应度</th><th>最佳样本外信息系数</th><th>状态</th></tr></thead>
          <tbody><tr v-for="item in miningArchive" :key="item.expr_fingerprint">
            <td><code>{{ item.candidate_key }}</code></td><td class="fm-cell-wrap">{{ item.expr }}</td>
            <td><code>{{ item.first_run_id }}</code></td><td>{{ item.seen_count }}</td>
            <td>{{ item.best_train_fitness == null ? '--' : Number(item.best_train_fitness).toFixed(4) }}</td>
            <td>{{ formatIc(item.best_oos_ic) }}</td><td>{{ candidateStatusText(item.best_status) }}</td>
          </tr></tbody>
        </table>
      </details>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, watch, onMounted, onUnmounted } from 'vue'
import AppEmpty from '../components/AppEmpty.vue'
import api from '../api'
import { getHotSectors, getAvailableSectors, getConceptSectors } from '../api/screening.js'
import { CODE_PREFIX_GROUPS } from '../constants/index.js'
import { pollSleep } from '../utils/poll.js'
import { useAsyncTaskTracker } from '../stores/asyncTaskTracker.js'

const tracker = useAsyncTaskTracker()

const activeTab = ref('rolling')
const poolFactors = ref([])
const poolMsg = ref('')
const selected = ref([])
const poolSearch = ref('')
const poolFilter = ref('all')
const poolPage = ref(1)
const PAGE_SIZE = 50
const crowdingMap = ref({})
const loadingCrowding = ref(false)
const sortRunning = ref(null)
const sortResult = ref(null)
const fmRunning = ref(false)
const fmResult = ref(null)

const filteredFactors = computed(() => {
  let list = poolFactors.value
  const s = poolSearch.value.toLowerCase()
  if (s) list = list.filter(f => f.key.toLowerCase().includes(s) || (f.label||'').toLowerCase().includes(s))
  if (poolFilter.value === 'active') list = list.filter(f => f.active)
  else if (poolFilter.value === 'inactive') list = list.filter(f => !f.active)
  else if (poolFilter.value === 'white_box') list = list.filter(f => f.category === 'white_box')
  else if (poolFilter.value === 'traditional') list = list.filter(f => f.category === 'traditional')
  return list
})
const totalPages = computed(() => Math.max(1, Math.ceil(filteredFactors.value.length / PAGE_SIZE)))
const pagedFactors = computed(() => {
  const start = (poolPage.value - 1) * PAGE_SIZE
  return filteredFactors.value.slice(start, start + PAGE_SIZE)
})

// ── 滚动筛选 ──
const PREFIX_OPTS = CODE_PREFIX_GROUPS
const roll = ref({
  factor_source: 'public', source: 'Alpha158', keysText: '', pool: 'stable_history_500d',
  prefixKeys: [],
  start: '2022-06-30', end: '2025-06-30',
  step_months: 3, lookback_months: 60, mode: 'full', write_pool: false, min_periods_frac: 0.5,
  label_horizon: 10, purge_gap: 11, point_in_time: true,
})
const rolling = ref(false)
const rollMsg = ref('')
const rollHotSectors = ref([])
const rollSelectedHotSectors = ref([])
const rollHotSectorCount = ref(10)
const rollSectors = ref([])
const rollConceptSectors = ref([])
const rollSelectedSectors = ref([])
const rollSectorSearch = ref('')
const loadingRollSectors = ref(false)
const filteredRollSectors = computed(() => {
  if (!rollSectorSearch.value) return roll.value.pool === 'concept' ? rollConceptSectors.value : rollSectors.value
  const q = rollSectorSearch.value.toLowerCase()
  const src = roll.value.pool === 'concept' ? rollConceptSectors.value : rollSectors.value
  return src.filter(s => s.name.toLowerCase().includes(q))
})
const rollProgress = ref(null)
const rollResult = ref(null)
let rollTimer = null
let rollStartTime = null

const architectureCfg = reactive({
  factor_source: 'pool', source: 'Alpha158', pool: 'stable_history_500d',
  start: '2022-01-01', end: '2024-12-31', selection_cutoff: '2025-06-30',
  label_horizon: 10, label_mode: 'tradable_open_to_open',
})
const architectureRunning = ref(false)
const architecturePublishing = ref(false)
const architecturePublishMsg = ref('')
const architectureSubRunning = ref(false)
const architectureMsg = ref('')
const architectureProgress = ref(null)
const architectureRun = ref(null)
const architectureHistory = ref([])
const architectureHistoryRunId = ref('')
const architectureLoadRunId = ref('')
let architectureTimer = null
const architectureVisibleSets = computed(() => {
  const sets = architectureRun.value?.sets || []
  const coreSets = sets.filter(item => item.role === 'core')
  const finalSetId = architectureRun.value?.result?.final_core_set_id
  const core = coreSets.find(item => item.set_id === finalSetId)
    || coreSets.find(item => item.status === 'validated')
    || [...coreSets].filter(item => item.status === 'screened').sort((a, b) => (b.version_no || 0) - (a.version_no || 0))[0]
    || [...coreSets].sort((a, b) => (b.version_no || 0) - (a.version_no || 0))[0]
  const currentSatelliteIds = new Set(
    (architectureRun.value?.result?.satellite_sets || []).map(item => item.set_id)
  )
  const satellites = currentSatelliteIds.size
    ? sets.filter(item => item.role !== 'core' && currentSatelliteIds.has(item.set_id))
    : sets.filter(item => item.role !== 'core')
  return [...(core ? [core] : []), ...satellites]
})
const architecturePct = computed(() => {
  const progress = architectureProgress.value
  if (progress?.progress_fraction != null) return Math.min(100, Math.round(progress.progress_fraction * 100))
  return progress?.total ? Math.min(100, Math.round((progress.current || 0) / progress.total * 100)) : 0
})
function architectureSetStatus(item) {
  if (item.status === 'validated') return '最终通过'
  if (item.status === 'screened') return item.version_no === 1 && !item.metrics?.model_ab ? '待组合优化' : '待最终验证'
  if (item.status === 'rejected') return '最终未通过'
  return item.status
}
function architectureSetAction(item) {
  if (item.role !== 'core') return '模型对照实验'
  return item.version_no === 1 && !item.metrics?.model_ab ? '核心组合优化' : '最终样本外验证'
}
function formatArchitectureTime(value) {
  if (!value) return ''
  const d = new Date(value)
  if (Number.isNaN(d.getTime())) return String(value)
  const pad = n => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}

async function loadArchitectureHistory() {
  try {
    const response = await api.get('/factor/architecture/history')
    const result = response?.data || response
    architectureHistory.value = Array.isArray(result) ? result : (result?.items || [])
    const completed = architectureHistory.value.filter(r => r.status === 'completed')
    if (completed.length) {
      const latest = completed[0]
      architectureHistoryRunId.value = latest.run_id
      await loadArchitectureRun(latest.run_id)
    }
  } catch (e) {
    architectureMsg.value = e.response?.data?.detail || e.message
  }
}

async function loadArchitectureRun(runId) {
  if (!runId) return
  if (architectureTimer) clearInterval(architectureTimer)
  architectureTimer = null
  architectureMsg.value = ''
  architectureRunning.value = false
  try {
    const response = await api.get(`/factor/architecture/${runId}`)
    const result = response?.data || response
    architectureRun.value = result
    architectureProgress.value = result?.progress || null
    architectureHistoryRunId.value = runId
    const activeSubtask = result?.active_subtask
    if (activeSubtask?.status === 'interrupted') {
      architectureProgress.value = activeSubtask.progress || null
      architectureMsg.value = '上次卫星任务因后端重启而中断，请重新点击“生成卫星”'
    }
    if (activeSubtask && ['pending', 'running', 'stalled', 'cancel_requested'].includes(activeSubtask.status)) {
      architectureSubRunning.value = true
      architectureProgress.value = activeSubtask.progress || null
      architectureMsg.value = `已恢复卫星任务 ${activeSubtask.job_id} 的进度跟踪`
      pollArchitectureSubtask(activeSubtask.job_id)
        .catch(e => { architectureMsg.value = e.message || '卫星任务中断' })
        .finally(() => { architectureSubRunning.value = false })
      return
    }
    if (['completed', 'failed', 'cancelled'].includes(result?.status)) {
      architectureMsg.value = result.status === 'completed'
        ? `已加载任务 ${runId}，可执行模型对照实验`
        : `任务状态：${result.status}`
    } else {
      architectureRunning.value = true
      architectureMsg.value = `已加载运行中任务 ${runId}`
      pollArchitecture(runId)
    }
  } catch (e) {
    architectureMsg.value = e.response?.data?.detail || e.message
  }
}

async function startArchitecture() {
  architectureRunning.value = true
  architectureRun.value = null
  architectureMsg.value = ''
  try {
    const response = await api.post('/factor/architecture/run', architectureCfg)
    const result = response?.data || response
    if (!result?.run_id) throw new Error('未返回 run_id')
    architectureMsg.value = `证据任务 ${result.run_id} 已提交`
    pollArchitecture(result.run_id)
  } catch (e) {
    architectureRunning.value = false
    architectureMsg.value = e.response?.data?.detail || e.message
  }
}

function pollArchitecture(runId) {
  if (architectureTimer) clearInterval(architectureTimer)
  const fetchStatus = async () => {
    try {
      const response = await api.get(`/factor/architecture/${runId}`)
      const result = response?.data || response
      architectureRun.value = result
      architectureProgress.value = result?.progress || null
      if (['completed', 'failed', 'cancelled'].includes(result?.status)) {
        clearInterval(architectureTimer)
        architectureRunning.value = false
        architectureMsg.value = result.status === 'completed'
          ? '核心滚动证据已完成，请执行模型对照实验'
          : result.status === 'cancelled' ? '任务已取消' : `失败：${result.error || '未知错误'}`
      }
    } catch (e) { /* 网络抖动忽略 */ }
  }
  fetchStatus()
  architectureTimer = setInterval(fetchStatus, 3000)
}

async function cancelArchitecture() {
  try {
    await api.post(`/factor/architecture/${architectureRun.value.run_id}/cancel`)
    architectureMsg.value = '已请求取消，正在停止工作进程…'
  } catch (e) {
    architectureMsg.value = e.response?.data?.detail || e.message
  }
}

async function startArchitectureAb(item, randomCount = 10) {
  architectureSubRunning.value = true
  const validationScope = item.role === 'core' && !(item.version_no === 1 && !item.metrics?.model_ab)
    ? 'fixed_oos'
    : 'development'
  architectureMsg.value = `${item.set_id} ${validationScope === 'fixed_oos' ? '固定样本外验证' : '模型对照实验'}已提交`
  try {
    const response = await api.post(`/factor/architecture/${architectureRun.value.run_id}/model-ab`, {
      role: item.role,
      set_id: item.set_id,
      random_count: item.role === 'core' ? randomCount : 10,
      validation_scope: validationScope,
    })
    await pollArchitectureSubtask((response?.data || response).task_id)
  } catch (e) {
    architectureMsg.value = e.response?.data?.detail || e.message
  } finally {
    architectureSubRunning.value = false
  }
}

async function startArchitectureResidual() {
  architectureSubRunning.value = true
  architectureMsg.value = '卫星残差筛选已提交'
  try {
    const response = await api.post(`/factor/architecture/${architectureRun.value.run_id}/residual`, {})
    await pollArchitectureSubtask((response?.data || response).task_id)
  } catch (e) {
    architectureMsg.value = e.response?.data?.detail || e.message
  } finally {
    architectureSubRunning.value = false
  }
}

async function publishArchitectureSatellites() {
  if (!architectureRun.value?.run_id) return
  architecturePublishing.value = true
  architecturePublishMsg.value = ''
  try {
    const response = await api.post(`/factor/architecture/${architectureRun.value.run_id}/publish-satellites`)
    const payload = response?.data || response
    const published = payload?.published || []
    architecturePublishMsg.value = published.length
      ? `已发布: ${published.map(x => `${x.label}(${x.count})`).join('、')}`
      : '没有可发布的因子集'
    await loadCuratedSets()
    await loadDirections()
  } catch (e) {
    architecturePublishMsg.value = e.response?.data?.detail || e.message
  } finally {
    architecturePublishing.value = false
  }
}

async function pollArchitectureSubtask(taskId) {
  if (!taskId) throw new Error('未返回 task_id')
  while (true) {
    const response = await api.get(`/factor/architecture-task/${taskId}`)
    const task = response?.data || response
    architectureProgress.value = task?.progress || null
    if (task?.status === 'completed') {
      const refreshed = await api.get(`/factor/architecture/${architectureRun.value.run_id}`)
      architectureRun.value = refreshed?.data || refreshed
      architectureMsg.value = task?.progress?.message || '任务完成，因子集状态已刷新'
      return
    }
    if (task?.status === 'failed') throw new Error(task.error || '任务失败')
    if (task?.status === 'cancelled') {
      architectureMsg.value = '卫星任务已取消，请重新点击“生成卫星”'
      return
    }
    if (task?.status === 'interrupted') {
      architectureMsg.value = task?.progress?.message || '卫星任务因后端重启而中断，请重新提交'
      return
    }
    await pollSleep(3000)
  }
}

// ── 滚动训练 ──
const trainCfg = reactive({
  model_name: 'lightgbm_rank', factor_set: '__pool__', pool: 'stable_history_500d',
  train_dual: false, enable_crossover: false,
  label_norm: 'csrank', early_stop_metric: 'rank_ic', param_profile: 'conservative',
  use_reweighter: false, with_linear: false, with_catboost: false,
  use_curated: 'latest', label_horizon: 10,
  start_anchor: '2023-01-01', end_anchor: '',
  step_months: 3, train_years: 5, valid_months: 6,
  filter_mode: false,
})
const training = ref(false)
const trainResult = ref(null)
const trainError = ref('')
const trainPrefixLabels = ref([])

const trainFirstRound = computed(() => {
  const offsetMonths = (dateStr, months) => {
    const d = new Date(dateStr); d.setMonth(d.getMonth() + months); return d.toISOString().slice(0, 10)
  }
  const offsetYears = (dateStr, years) => {
    const d = new Date(dateStr); d.setFullYear(d.getFullYear() + years); return d.toISOString().slice(0, 10)
  }
  const dayBefore = (dateStr) => {
    const d = new Date(dateStr); d.setDate(d.getDate() - 1); return d.toISOString().slice(0, 10)
  }
  const step = trainCfg.step_months || 3
  const trainY = trainCfg.train_years || 5
  const validM = trainCfg.valid_months || 6
  const start = trainCfg.start_anchor || '2023-01-01'
  const end = trainCfg.end_anchor || new Date().toISOString().slice(0, 10)
  const validStart = offsetMonths(start, -validM)
  const trainStart = offsetYears(validStart, -(trainY - validM / 12))
  const testEnd = offsetMonths(start, step)
  let lastAnchor = ''; let nRounds = null
  try {
    let cur = new Date(start), endD = new Date(end), count = 0
    while (cur <= endD) { count++; cur.setMonth(cur.getMonth() + step) }
    lastAnchor = new Date(cur.getTime() - step * 30.44 * 86400 * 1000).toISOString().slice(0, 10)
    nRounds = count
  } catch {}
  return {
    train: trainStart + ' ~ ' + dayBefore(validStart),
    valid: validStart + ' ~ ' + dayBefore(start),
    test: start + ' ~ ' + dayBefore(testEnd),
    lastAnchor, nRounds,
  }
})

const sortedImportance = computed(() => {
  const summary = trainResult.value?.importance_summary
  if (!summary) return []
  return Object.entries(summary).sort((a, b) => b[1].mean_split - a[1].mean_split)
})

const eliminatedSet = computed(() => {
  return new Set(trainResult.value?.filter_result?.failed || [])
})

const trainStatusText = computed(() => {
  if (!trainError.value) return ''
  if (trainResult.value) return '训练完成'
  return trainError.value
})

const trainPct = computed(() => {
  if (trainResult.value) return 100
  const m = trainError.value?.match(/(\d+)\/(\d+)/)
  if (m) return Math.round((parseInt(m[1]) / parseInt(m[2])) * 100)
  return 0
})

function onTrainFactorSetChange() {
  if (trainCfg.factor_set === '__pool__' || trainCfg.factor_set === '__curated__') {
    trainCfg.filter_mode = false
  }
  if (trainCfg.factor_set === '__curated__') {
    try { api.get('/factor/curated-sets').then(r => { curatedSets.value = r?.data || [] }) } catch {}
  }
}

async function startTraining() {
  training.value = true; trainResult.value = null; trainError.value = ''
  const startTime = Date.now()
  try {
    const cfg = { ...trainCfg }
    if (cfg.factor_set === '__pool__') {
      cfg.use_factor_pool = true; cfg.use_curated = null
      cfg.factor_set = 'Alpha158'; cfg.custom_factor_keys = null
    } else if (cfg.factor_set === '__curated__') {
      cfg.use_curated = cfg.use_curated || 'latest'
      cfg.use_factor_pool = false
      cfg.factor_set = 'Alpha158'; cfg.custom_factor_keys = null
    } else {
      cfg.use_factor_pool = false; cfg.use_curated = null
      cfg.custom_factor_keys = null
    }
    const labelToPrefixes = Object.fromEntries(PREFIX_OPTS.map(p => [p[1], p[0]]))
    const prefixes = trainPrefixLabels.value.flatMap(label => labelToPrefixes[label] || [])
    if (prefixes.length) cfg.code_prefixes = prefixes
    if (!cfg.end_anchor) cfg.end_anchor = ''
    if (!cfg.model_name) cfg.model_name = 'lightgbm_rank'
    const r = await api.post('/qlib/models/rolling-train', cfg)
    const jobId = r.data.job_id
    trainError.value = `滚动训练已启动: ${jobId}，轮询中...`
    for (let i = 0; i < 1440; i++) {
      await pollSleep(10000)
      try {
        const job = await api.get(`/qlib/models/train/${jobId}`)
        const status = job.data.status
        const elapsed = Math.floor((Date.now() - startTime) / 60000)
        if (status === 'done') {
          trainResult.value = job.data.result
          trainError.value = ''
          break
        } else if (status === 'error') {
          trainError.value = `训练失败: ${job.data.error}`
          break
        } else {
          const p = job.data.progress || 0
          const t = job.data.total || '?'
          const a = job.data.current_anchor || ''
          trainError.value = `轮询中... ${elapsed}分钟 | 进度 ${p}/${t} | 锚点：${a}`
        }
      } catch { trainError.value = `轮询中... ${elapsed}分钟` }
    }
  } catch(e) {
    const detail = e.response?.data?.detail
    trainError.value = '启动失败: ' + (typeof detail === 'string' ? detail : detail?.message || JSON.stringify(detail) || e.message)
  }
  training.value = false
}

const stageLabelMap = { init: '准备中', shared_panel: '构建共享映射面板', master: '指标构建', snapshot: '构建日期视图', factor: '因子校验', cluster: '聚类去重', save_db: '保存结果', sync_pool: '同步因子池', parallel: '并行计算', done: '完成' }

const anchorsDetail = computed(() => {
  const p = rollProgress.value
  if (!p?.anchors || !Object.keys(p.anchors).length) return null
  const sorted = Object.entries(p.anchors).sort(([a], [b]) => a.localeCompare(b))
  return Object.fromEntries(sorted)
})

const rollElapsed = computed(() => {
  if (!rollStartTime || !rolling.value) return ''
  const sec = Math.round((Date.now() - rollStartTime) / 1000)
  if (sec < 60) return `${sec}秒`
  if (sec < 3600) return `${Math.floor(sec/60)}分${sec%60}秒`
  return `${Math.floor(sec/3600)}时${Math.floor((sec%3600)/60)}分`
})

const stageLabel = computed(() => {
  const s = rollProgress.value?.stage
  if (!s) return ''
  return stageLabelMap[s] || s
})

const rollPct = computed(() => {
  const p = rollProgress.value
  if (!p) return 0
  if (p.anchors) {
    const entries = Object.values(p.anchors)
    if (!entries.length) return 0
    const done = entries.filter(e => e.status === 'completed' || e.stage === 'done').length
    const stageBaseMap = { master: 0, snapshot: 0.05, factor: 0.20, cluster: 0.90, save_db: 0.95 }
    const stageWeightMap = { master: 0.05, snapshot: 0.15, factor: 0.70, cluster: 0.05, save_db: 0.05 }
    const running = entries.filter(e => e.stage && e.stage !== 'done').reduce((sum, e) => {
      const base = stageBaseMap[e.stage] || 0
      const sw = stageWeightMap[e.stage] || 0.05
      return sum + base + (e.total ? (e.current / e.total) * sw : 0)
    }, 0)
    return Math.min(99, Math.round(((done + running) / entries.length) * 100))
  }
  if (p.anchor_total) {
    const perAnchor = 1 / p.anchor_total
    const done = (p.anchor_idx - 1) * perAnchor
    const frac = p.total ? (p.current / p.total) : 0
    const stageW = p.stage === 'snapshot' ? 0.5 : p.stage === 'factor' ? 0.4 : 0.05
    const stageBase = p.stage === 'snapshot' ? 0 : p.stage === 'factor' ? 0.5 : 0.9
    const within = (stageBase + frac * stageW) * perAnchor
    return Math.min(100, Math.round((done + within) * 100))
  }
  return p.total ? Math.round(p.current / p.total * 100) : 0
})

async function startRolling() {
  rolling.value = true
  rollMsg.value = ''
  rollResult.value = null
  rollProgress.value = null
  rollStartTime = Date.now()
  try {
    const r0 = roll.value
    const payload = {
      factor_source: r0.factor_source, pool: r0.pool,
      start: r0.start, end: r0.end, step_months: r0.step_months,
      lookback_months: r0.lookback_months, mode: r0.mode,
      write_pool: r0.write_pool, min_periods_frac: r0.min_periods_frac,
      label_horizon: r0.label_horizon, purge_gap: r0.purge_gap,
      train_months: 60, valid_months: 6, test_months: 3,
    }
    if (r0.prefixKeys.length) {
      payload.code_prefixes = r0.prefixKeys.map(k => PREFIX_OPTS.find(p => p[0][0] === k)[0]).flat()
    }
    if (r0.factor_source === 'public') payload.source = r0.source
    else if (r0.factor_source === 'manual') {
      const keys = r0.keysText.split(',').map(s => s.trim()).filter(Boolean)
      if (!keys.length) throw new Error('请填写因子 keys')
      payload.factor_keys = keys
    }
    if (r0.pool === 'hot_sectors' && rollSelectedHotSectors.value.length) {
      payload.custom_sectors = rollSelectedHotSectors.value; payload.pool = 'custom'
    }
    if (r0.pool === 'custom' && rollSelectedSectors.value.length) payload.custom_sectors = rollSelectedSectors.value
    if (r0.pool === 'concept' && rollSelectedSectors.value.length) payload.concept_sectors = rollSelectedSectors.value
    const r = await api.post('/factor/rolling-screen', payload)
    const jobId = r?.data?.job_id
    if (!jobId) throw new Error('未返回 job_id')
    tracker.register('factor-rolling', jobId, { type: '滚动筛选' })
    rollMsg.value = `任务已提交 (${r.data.n_factors} 因子)，运行中…`
    pollRolling(jobId)
  } catch (e) {
    rolling.value = false
    rollMsg.value = e.message
  }
}

function pollRolling(jobId) {
  rollTimer = setInterval(async () => {
    try {
      const r = await api.get(`/factor/rolling-screen/${jobId}`)
      const t = r?.data || {}
      rollProgress.value = t.progress || null
      if (t.status === 'completed') {
        clearInterval(rollTimer)
        rolling.value = false
        rollResult.value = t.result
        rollMsg.value = '滚动筛选完成' + (roll.value.write_pool ? '，已写回动态因子池' : '')
        tracker.unregister('factor-rolling')
      } else if (t.status === 'failed') {
        clearInterval(rollTimer)
        rolling.value = false
        rollMsg.value = '失败：' + (t.error || '未知错误')
        tracker.unregister('factor-rolling')
      }
    } catch (e) { /* 网络抖动忽略 */ }
  }, 3000)
}

function onRollPoolChange() {
  rollSelectedHotSectors.value = []; rollSelectedSectors.value = []; rollSectorSearch.value = ''
  if (roll.value.pool === 'hot_sectors' && !rollHotSectors.value.length) loadRollHotSectors()
  if (roll.value.pool === 'custom' && !rollSectors.value.length) loadRollSectors()
  if (roll.value.pool === 'concept' && !rollConceptSectors.value.length) loadRollSectors()
}
async function loadRollHotSectors() {
  loadingRollSectors.value = true
  try { const r = await getHotSectors(rollHotSectorCount.value); rollHotSectors.value = r?.sectors || (r?.data || r)?.sectors || [] } catch (e) { /* ignore */ }
  loadingRollSectors.value = false
}
function toggleRollHotSector(name) {
  const idx = rollSelectedHotSectors.value.indexOf(name)
  if (idx >= 0) rollSelectedHotSectors.value.splice(idx, 1); else rollSelectedHotSectors.value.push(name)
}
async function loadRollSectors() {
  loadingRollSectors.value = true
  try {
    if (roll.value.pool === 'concept') {
      const r = await getConceptSectors(); rollConceptSectors.value = (r?.data || r) || []
    } else {
      const r = await getAvailableSectors(); rollSectors.value = (r?.data || r) || []
    }
  } catch (e) { /* ignore */ }
  loadingRollSectors.value = false
}
function toggleRollSector(name) {
  const idx = rollSelectedSectors.value.indexOf(name)
  if (idx >= 0) rollSelectedSectors.value.splice(idx, 1); else rollSelectedSectors.value.push(name)
}

watch([poolSearch, poolFilter], () => { poolPage.value = 1 })
let sortPollTimer = null
let fmPollTimer = null
onUnmounted(() => {
  if (rollTimer) clearInterval(rollTimer)
  if (architectureTimer) clearInterval(architectureTimer)
  if (crowdingPollTimer) clearTimeout(crowdingPollTimer)
  if (sortPollTimer) clearTimeout(sortPollTimer)
  if (fmPollTimer) clearTimeout(fmPollTimer)
})

onMounted(() => {
  loadPool()
    loadArchitectureHistory()
  const rollTask = tracker.getTask('factor-rolling')
  if (rollTask && rollTask.status === 'running') {
    rolling.value = true
    rollMsg.value = '正在恢复滚动筛选任务...'
    pollRolling(rollTask.jobId)
  }
  const curateTask = tracker.getTask('factor-curate')
  if (curateTask && curateTask.status === 'running') {
    curating.value = true
    curateMsg.value = '正在恢复精选集生成...'
    pollCurate(curateTask.jobId)
  }
})

const poolCount = ref(0)
const activeCount = computed(() => poolFactors.value.filter(f => f.active).length)
const allSelected = computed(() => poolFactors.value.length > 0 && selected.value.length === poolFactors.value.length)

async function toggleAllFactors(active) {
  const label = active ? '启用' : '停用'
  if (!confirm(`确定要${label}全部 ${poolFactors.value.length} 个因子吗？`)) return
  try {
    await api.post('/factor/pool-toggle', { active, keys: null })
    poolMsg.value = `已${label}全部因子`
    await loadPool()
  } catch (e) { poolMsg.value = `失败: ${e.message}` }
}

function toggleAll(e) {
  selected.value = e.target.checked ? poolFactors.value.map(f => f.key) : []
}

function formatParams(p) {
  if (!p || !Object.keys(p).length) return '-'
  return Object.entries(p).map(([k,v]) => `${k}=${v}`).join(', ')
}

async function loadPool() {
  try {
    const r = await api.get('/factor/pool/all')
    const factors = r?.data?.factors || []
    poolFactors.value = factors
    poolCount.value = factors.length
    selected.value = []
  } catch (e) { poolMsg.value = e.message }
}

async function toggleFactor(key, active) {
  try {
    await api.post('/tuner/factor-pool/toggle', { key, active })
    const f = poolFactors.value.find(x => x.key === key)
    if (f) f.active = active
    poolMsg.value = `"${key}" 已${active ? '启用' : '停用'}`
  } catch (e) { poolMsg.value = e.message }
}

async function deleteFactor(key) {
  if (!confirm(`确定删除因子 "${key}"？此操作不可恢复。`)) return
  try {
    await api.delete(`/tuner/factor-pool/${key}`)
    poolFactors.value = poolFactors.value.filter(f => f.key !== key)
    selected.value = selected.value.filter(k => k !== key)
    poolCount.value = poolFactors.value.length
    poolMsg.value = `"${key}" 已删除`
  } catch (e) { poolMsg.value = e.message }
}

async function deleteSelected() {
  if (!confirm(`确定删除选中的 ${selected.value.length} 个因子？`)) return
  for (const key of [...selected.value]) {
    try { await api.delete(`/tuner/factor-pool/${key}`) } catch {}
  }
  loadPool()
}

// ── 拥挤度 ──
let crowdingPollTimer = null
async function loadCrowding() {
  loadingCrowding.value = true
  poolMsg.value = ''
  try {
    const active = poolFactors.value.filter(f => f.active).map(f => f.key)
    if (!active.length) { poolMsg.value = '无活跃因子'; loadingCrowding.value = false; return }
    const r = await api.post('/factor/crowding-check', { factor_keys: active, pool: 'stable_history_500d' })
    const jobId = r?.data?.job_id
    if (!jobId) { poolMsg.value = '提交拥挤度检测失败'; loadingCrowding.value = false; return }
    const poll = async () => {
      const res = await api.get(`/factor/crowding-check/${jobId}`)
      const j = res?.data || res
      if (j.status === 'done') {
        const map = {}
        for (const cr of (j.result || [])) {
          if (cr) map[cr.factor] = cr.status
        }
        crowdingMap.value = map
        loadingCrowding.value = false
        const crowded = j.result.filter(r => r.status === 'crowded').length
        const warning = j.result.filter(r => r.status === 'warning').length
        poolMsg.value = `拥挤度检测完成: 🔴${crowded} 🟡${warning} 🟢${j.result.length - crowded - warning}`
      } else if (j.status === 'error') {
        poolMsg.value = '拥挤度检测失败: ' + j.error
        loadingCrowding.value = false
      } else {
        crowdingPollTimer = setTimeout(poll, 3000)
      }
    }
    crowdingPollTimer = setTimeout(poll, 3000)
  } catch (e) { poolMsg.value = '拥挤度检测失败: ' + e.message; loadingCrowding.value = false }
}

// ── Portfolio Sort ──
async function runPortfolioSort(key) {
  sortRunning.value = key
  sortResult.value = null
  try {
    const r = await api.post('/factor/portfolio-sort', {
      factor_keys: [key], pool: 'stable_history_500d', eval_weeks: 12, forward_days: 20,
    })
    if (r?.data?.job_id) {
      const poll = async () => {
        const res = await api.get(`/factor/portfolio-sort/${r.data.job_id}`)
        const j = res?.data || res
        if (j.status === 'done' && j.result?.length) {
          const s = j.result[0]
          const qr = s.q_returns || {}
          sortResult.value = {
            key, n_dates: s.n_dates,
            q1: qr['0'], q2: qr['1'], q3: qr['2'], q4: qr['3'], q5: qr['4'],
            annual_return: s.annual_return, sharpe: s.sharpe, monotonicity: s.monotonicity,
          }
          sortRunning.value = null
        } else if (j.status === 'error') {
          poolMsg.value = `分组回测失败: ${j.error}`
          sortRunning.value = null
        } else {
          sortPollTimer = setTimeout(poll, 3000)
        }
      }
      sortPollTimer = setTimeout(poll, 3000)
    } else {
      sortRunning.value = null
      poolMsg.value = '分组回测启动失败'
    }
  } catch (e) {
    poolMsg.value = '分组回测失败: ' + e.message
    sortRunning.value = null
  }
}

// ── Fama-MacBeth ──
async function runFamaMacBeth() {
  fmRunning.value = true
  fmResult.value = null
  try {
    const keys = poolFactors.value.filter(f => f.active).slice(0, 10).map(f => f.key)
    const r = await api.post('/factor/fama-macbeth', {
      factor_keys: keys, pool: 'stable_history_500d', eval_weeks: 52, forward_days: 20,
    })
    if (r?.data?.job_id) {
      const poll = async () => {
        const res = await api.get(`/factor/fama-macbeth/${r.data.job_id}`)
        const j = res?.data || res
        if (j.status === 'done') {
          fmResult.value = j.result || []
          fmRunning.value = false
        } else if (j.status === 'error') {
          poolMsg.value = `FM回归失败: ${j.error}`
          fmRunning.value = false
        } else {
          fmPollTimer = setTimeout(poll, 5000)
        }
      }
      fmPollTimer = setTimeout(poll, 5000)
    } else {
      fmRunning.value = false
    }
  } catch (e) {
    poolMsg.value = 'FM回归失败: ' + e.message
    fmRunning.value = false
  }
}

// ── 精选集 ──
const curatedSets = ref([])
const activeDirections = ['trend_momentum', 'volume_breakout', 'oversold_rebound', 'value_defense',
                           'core_base', 'trend_satellite', 'rebound_satellite',
                           'core_trend_satellite', 'core_rebound_satellite']
const activeCuratedSets = computed(() => curatedSets.value.filter(s => activeDirections.includes(s.name)))
const curatedMsg = ref('')
const curatedError = ref(false)
const curating = ref(false)
const curatePct = ref(0)
const curateMsg = ref('')
const curatedDetail = ref(null)
let curateTimer = null

async function loadCuratedSets() {
  try {
    const r = await api.get('/factor/curated-sets')
    curatedSets.value = r?.data || []
  } catch (e) { curatedMsg.value = '加载失败: ' + e.message; curatedError.value = true }
}

async function startCurate() {
  curating.value = true
  curatePct.value = 0
  curatedMsg.value = ''
  curatedError.value = false
  try {
    const r = await api.post('/factor/curate', {
      pool: 'all', target_lower: 60, target_upper: 80, eval_weeks: 12,
    })
    const jobId = r?.data?.job_id
    if (!jobId) throw new Error('未返回 job_id')
    tracker.register('factor-curate', jobId, { type: '精选集生成' })
    curateMsg.value = '任务已提交...'
    pollCurate(jobId)
  } catch (e) {
    curating.value = false
    curatedMsg.value = e.message
    curatedError.value = true
  }
}

function pollCurate(jobId) {
  curateTimer = setInterval(async () => {
    try {
      const r = await api.get(`/factor/curate-progress/${jobId}`)
      const t = r?.data || {}
      if (t.progress != null) {
        curatePct.value = t.progress
        curateMsg.value = t.message || ''
      }
      if (t.status === 'completed') {
        clearInterval(curateTimer)
        curating.value = false
        curatedMsg.value = `精选完成: ${t.result?.input_count} → ${t.result?.output_count} 因子`
        loadCuratedSets()
        tracker.unregister('factor-curate')
      } else if (t.status === 'failed') {
        clearInterval(curateTimer)
        curating.value = false
        curatedMsg.value = '失败: ' + (t.error || '未知')
        curatedError.value = true
        tracker.unregister('factor-curate')
      }
    } catch {}
  }, 3000)
}

async function viewCuratedSet(name) {
  try {
    const r = await api.get(`/factor/curated-sets/${name}`)
    curatedDetail.value = r?.data
  } catch (e) { curatedMsg.value = '加载详情失败: ' + e.message }
}

async function deleteCuratedSet(name) {
  if (!confirm(`确定删除精选集 "${name}"？`)) return
  try {
    await api.delete(`/factor/curated-sets/${name}`)
    curatedSets.value = curatedSets.value.filter(s => s.name !== name)
    if (curatedDetail.value?.name === name) curatedDetail.value = null
    curatedMsg.value = `"${name}" 已删除`
  } catch (e) { curatedMsg.value = e.message; curatedError.value = true }
}

// ── 市场状态 + 策略方向 ──
const currentRegime = ref(null)
const regimeLoading = ref(false)

const REGIME_LABELS = {
  volatile_up: '📈 高波快涨', volatile_down: '📉 高波快跌',
  volatile_flat: '🌊 高波震荡', steady_up: '🐂 稳步上涨',
  steady_down: '🐻 稳步下跌', steady_flat: '📊 低波盘整',
}
function regimeLabel(r) { return REGIME_LABELS[r] || r }

const REGIME_COLORS = {
  volatile_up: 'var(--success)', volatile_down: 'var(--danger)',
  volatile_flat: '#e74c3c', steady_up: '#2ecc71',
  steady_down: '#e67e22', steady_flat: 'var(--warning)',
}
const regimeTagStyle = computed(() => {
  if (!currentRegime.value) return {}
  return {
    background: REGIME_COLORS[currentRegime.value.regime] || 'var(--primary)',
    color: '#fff', padding: '2px 10px', borderRadius: '12px',
    fontSize: '13px', fontWeight: 600,
  }
})

async function loadRegimeStatus() {
  regimeLoading.value = true
  try {
    const r = await api.get('/factor/regime-status')
    currentRegime.value = r?.data?.current
  } catch (e) { /* ignore */ }
  finally { regimeLoading.value = false }
}

// ── 策略方向精选 ──
const dirMsg = ref('')
const recalcRunning = ref(false)
const recalcPct = ref(0)
const recalcResult = ref('')
const directions = ref([
  { key: 'trend_momentum', icon: '📈', label: '趋势动量', desc: '上涨/牛市趋势中动量+成长因子主导', categories: ['价格动量','趋势强度','成长','规模','高阶特征','成交量特征'], count: 0, factors: [], avg_icir: null, status: '' },
  { key: 'volume_breakout', icon: '📊', label: '量价突破', desc: '横盘等放量突破+量价背离+资金异动', categories: ['成交量特征','流动性','趋势强度','价格动量','技术形态'], count: 0, factors: [], avg_icir: null, status: '' },
  { key: 'oversold_rebound', icon: '🎯', label: '超卖反弹', desc: '高波动中识别超卖+恐慌放量+聪明钱抄底，博短期反弹', categories: ['波动率特征','流动性','成交量特征','反转效应'], count: 0, factors: [], avg_icir: null, status: '' },
  { key: 'value_defense', icon: '🛡️', label: '价值防御', desc: '下跌市中低估值+高质量+稳定现金流防御', categories: ['估值','盈利','现金流','杠杆','股息','经营'], count: 0, factors: [], avg_icir: null, status: '' },
  { key: 'trend_satellite', icon: '🛰️', label: '趋势卫星', desc: '核心/卫星架构成品：低波动趋势状态下的风格因子', categories: [], count: 0, factors: [], avg_icir: null, status: '', pinned: true },
  { key: 'rebound_satellite', icon: '🛰️', label: '反弹卫星', desc: '核心/卫星架构成品：高波动超卖状态下的风格因子', categories: [], count: 0, factors: [], avg_icir: null, status: '', pinned: true },
])

async function loadDirections() {
  loadRegimeStatus()
  for (const d of directions.value) {
    try {
      const r = await api.get(`/factor/curated-sets/${d.key}`)
      const info = r?.data
      if (info) {
        d.count = info.stats?.total || info.flat_list?.length || 0
        d.factors = info.flat_list || []
        d.avg_icir = info.stats?.avg_icir
        d.status = 'loaded'
      }
    } catch { d.status = 'empty' }
  }
}

async function doCurateDirection(dirKey) {
  const d = directions.value.find(x => x.key === dirKey)
  if (!d) return
  d.status = 'loading'
  dirMsg.value = `正在生成 ${d.label} 精选集...`
  try {
    const r = await api.post('/factor/curate-direction', { direction: dirKey })
    d.count = r?.data?.count || 0
    d.factors = r?.data?.factors || []
    d.status = 'loaded'
    dirMsg.value = `${d.label}: ${d.count} 个因子`
    try {
      const info = await api.get(`/factor/curated-sets/${dirKey}`)
      if (info?.data) {
        d.avg_icir = info.data.stats?.avg_icir
        d.count = info.data.stats?.total || info.data.flat_list?.length || d.count
        d.factors = info.data.flat_list || d.factors
      }
    } catch {}
  } catch (e) {
    d.status = 'error'
    dirMsg.value = `生成失败: ${e.message}`
  }
}

async function doBackfillIcir(dirKey) {
  const d = directions.value.find(x => x.key === dirKey)
  if (!d) return
  d.status = 'loading'
  dirMsg.value = `正在回填 ${d.label} 信息系数稳定性数据（可能需要几分钟）...`
  try {
    await api.post(`/factor/backfill-icir/${dirKey}`)
    dirMsg.value = `${d.label}：信息系数稳定性回填完成，可重新更新精选`
    d.status = 'loaded'
  } catch (e) {
    d.status = 'error'
    dirMsg.value = `回填失败: ${e.message}`
  }
}

async function recalcAllIcir() {
  recalcRunning.value = true
  recalcPct.value = 0
  recalcResult.value = ''
  dirMsg.value = ''
  try {
    recalcPct.value = 30
    const r = await api.post('/factor/recalc-all-icir')
    recalcPct.value = 100
    const d = r?.data || {}
    recalcResult.value = d.message || `完成! 更新 ${d.updated} 个, 跳过 ${d.skipped} 个`
  } catch (e) {
    dirMsg.value = `全部信息系数稳定性重算失败：${e.response?.data?.detail || e.message}`
  }
  recalcRunning.value = false
}

onUnmounted(() => { if (curateTimer) clearInterval(curateTimer) })

// ── 遗传挖矿 ──
const gp = reactive({
  pool: 'csi300', train_start: '2020-01-01', train_end: '2024-06-30',
  valid_start: '2024-07-01', valid_end: '2025-06-30',
  population: 200, generations: 20, max_depth: 5, top_n: 10, forward_days: 20, step: 5,
  risk_neutralize: true, validation_profile: 'default', baseline_set: 'current_curated', seed: 42,
  baseline_model_key: 'auto',
  backtest_profile: 'production_screening', backtest_freq: 'week', backtest_top_n: 10,
  backtest_use_timing: true, backtest_macro_bias: true, backtest_sector_fusion: null,
  overlay_weight: 0.15, final_oos_start: '', final_oos_end: '',
  candidate_mode: 'gp', llm_provider: 'deepseek', llm_candidate_count: 10,
  llm_temperature: 0.2, llm_max_tokens: 4096, llm_prompt_profile: 'factor_hypothesis_v1',
  llm_research_focus: '',
  execution_mode: 'single', max_batches: 0, max_hours: 0,
  idle_cpu_percent: 70, min_free_memory_gb: 6, resource_poll_seconds: 15,
})
const geneticMining = ref(false)
const geneticMsg = ref('')
const geneticError = ref(false)
const geneticProgress = ref(null)
const geneticResult = ref(null)
const geneticStatus = ref('idle')
const geneticRunId = ref(null)
const geneticResumeAvailable = ref(false)
const geneticTaskMode = ref('single')
const geneticDetailLoading = reactive({})
const miningModels = ref([])
const miningModelsLoading = ref(false)
const miningModelsLoaded = ref(false)
const miningModelsError = ref('')
const miningArchive = ref([])
const miningArchiveLoading = ref(false)
let geneticTimer = null
let geneticStartTime = null
const candidateTraining = ref(false)
const candidateTrainingProgress = ref(null)
const candidateTrainingResult = ref(null)
let candidateTrainingTimer = null

const geneticCandidates = computed(() => geneticResult.value?.candidates || geneticResult.value?.factors || [])
const candidateTrainingText = computed(() => {
  if (candidateTraining.value) return candidateTrainingProgress.value?.message || '正在训练候选增强模型…'
  if (candidateTrainingResult.value) return '候选增强研究模型训练完成（未激活）'
  return ''
})
const compatibleMiningModels = computed(() => miningModels.value.filter(model => model.compatible))
const selectedMiningModel = computed(() => {
  if (gp.baseline_model_key === 'auto') return null
  return miningModels.value.find(model => miningModelKey(model) === gp.baseline_model_key) || null
})
const geneticStatusText = computed(() => ({
  idle: '待开始', submitting: '提交中', running: '运行中', stalled: '可能停滞',
  awaiting_promotion: '待人工晋级', completed: '已完成', failed: '失败', cancelled: '已取消',
}[geneticStatus.value] || geneticStatus.value))

const geneticStageText = computed(() => {
  const p = geneticProgress.value
  if (!p) return ''
  const names = {
    planning: '规划任务', snapshot: p.snapshot_mode === 'panel' ? '构建研究面板' : '预取快照', generate: '生成候选', train_filter: '训练期筛选',
    validation: '现有层级校验', walk_forward: '滚动样本外', deduplicate: '去冗余',
    incremental_backtest: '增量评估', report: '生成报告', awaiting_promotion: '等待人工晋级',
    waiting_resources: '等待空闲资源', mining_batch: '持续挖掘批次',
    completed: '已完成', failed: '失败', cancelled: '已取消', cancel_requested: '正在取消',
  }
  const child = p.stage === 'mining_batch' ? p.child_progress : null
  const name = child
    ? `持续第 ${p.batch_index} 批 · ${names[child.stage] || child.stage || '处理中'}`
    : (names[p.stage] || p.stage || '处理中')
  const current = child?.current ?? p.current
  const total = child?.total ?? p.total
  const suffix = current != null && total > 0 ? ` ${current}/${total}` : ''
  return `${name}${suffix}${p.last_date ? ' · ' + p.last_date : ''}`
})

const geneticPct = computed(() => {
  const raw = geneticProgress.value
  const p = raw?.stage === 'mining_batch' && raw.child_progress ? raw.child_progress : raw
  if (!p) return 0
  const base = { planning: 2, snapshot: 30, generate: 45, train_filter: 55, validation: 70, walk_forward: 85, deduplicate: 90, incremental_backtest: 95, report: 98, awaiting_promotion: 100, completed: 100, cancelled: 100, failed: 100 }
  const start = base[p.stage] ?? 0
  const next = { planning: 5, snapshot: 30, generate: 45, train_filter: 60, validation: 72, walk_forward: 85, deduplicate: 90, incremental_backtest: 97, report: 100 }[p.stage]
  if (next == null || p.current == null || p.total == null) return start
  return Math.min(next, start + Math.round((next - start) * p.current / Math.max(p.total, 1)))
})

const geneticElapsed = computed(() => {
  if (!geneticStartTime || !geneticMining.value) return ''
  const sec = Math.round((Date.now() - geneticStartTime) / 1000)
  if (sec < 60) return `${sec}秒`
  return `${Math.floor(sec / 60)}分${sec % 60}秒`
})

function miningModelKey(model) {
  return `${model.experiment_id}::${model.run_id}`
}

function miningModelLabel(model) {
  const state = model.compatible ? '可用' : `不可用：${(model.compatibility_reasons || []).join('；')}`
  const ic = model.ic == null ? '--' : `${(Number(model.ic) * 100).toFixed(2)}%`
  const coverage = `${(Number(model.baseline_coverage || 0) * 100).toFixed(0)}%`
  const modelId = `${model.experiment_id.slice(-6)}/${model.run_id.slice(0, 8)}`
  const horizon = model.label_horizon ? `h${model.label_horizon}` : 'h?'
  return `${modelId} · ${model.factor_set || '量化模型'} ${horizon} · 截止 ${model.observed_end || '--'} · 信息系数 ${ic} · 覆盖 ${coverage} · ${state}`
}

function resetMiningModel() {
  gp.baseline_model_key = 'auto'
  miningModels.value = []
  miningModelsLoaded.value = false
  miningModelsError.value = ''
}

async function loadMiningModels() {
  if (!gp.train_end || miningModelsLoading.value) return
  miningModelsLoading.value = true
  miningModelsError.value = ''
  try {
    const r = await api.get('/factor/mining/models', {
      params: { pool: gp.pool, baseline_set: gp.baseline_set, train_end: gp.train_end },
    })
    miningModels.value = r?.data?.models || []
    miningModelsLoaded.value = true
    if (gp.baseline_model_key !== 'auto' && !selectedMiningModel.value) gp.baseline_model_key = 'auto'
  } catch (e) {
    miningModels.value = []
    miningModelsLoaded.value = true
    miningModelsError.value = '模型列表加载失败: ' + (e.response?.data?.detail || e.message)
  } finally {
    miningModelsLoading.value = false
  }
}

async function loadMiningArchive() {
  if (miningArchiveLoading.value) return
  miningArchiveLoading.value = true
  try {
    const r = await api.get('/factor/mining/archive', { params: { limit: 100 } })
    miningArchive.value = r?.data?.candidates || []
  } catch (e) {
    geneticError.value = true
    geneticMsg.value = '全局档案加载失败: ' + (e.response?.data?.detail || e.message)
  } finally {
    miningArchiveLoading.value = false
  }
}

async function startGeneticMine() {
  const invalid = validateGeneticForm()
  if (invalid) { geneticError.value = true; geneticMsg.value = invalid; return }
  geneticStatus.value = 'submitting'
  geneticMining.value = true
  geneticMsg.value = ''
  geneticError.value = false
  geneticResult.value = null
  geneticProgress.value = null
  geneticResumeAvailable.value = false
  geneticStartTime = Date.now()
  try {
    const payload = { ...gp }
    payload.candidate_sources = gp.candidate_mode === 'hybrid' ? ['gp', 'llm'] : [gp.candidate_mode]
    if (gp.baseline_model_key !== 'auto') {
      const [experimentId, runId] = gp.baseline_model_key.split('::')
      payload.baseline_model_experiment_id = experimentId
      payload.baseline_model_run_id = runId
    }
    delete payload.candidate_mode
    delete payload.baseline_model_key
    const isCampaign = gp.execution_mode === 'continuous'
    delete payload.execution_mode
    const r = await api.post(isCampaign ? '/factor/mining/campaigns' : '/factor/mining/runs', payload)
    const jobId = isCampaign ? r?.data?.campaign_id : r?.data?.run_id
    if (!jobId) throw new Error(isCampaign ? '未返回 campaign_id' : '未返回 run_id')
    geneticRunId.value = jobId
    geneticTaskMode.value = isCampaign ? 'campaign' : 'single'
    geneticStatus.value = 'running'
    tracker.register('factor-genetic', jobId, { type: '因子挖矿', mode: geneticTaskMode.value })
    geneticMsg.value = '任务已提交，运行中…'
    pollGenetic(jobId, true)
  } catch (e) {
    geneticMining.value = false
    geneticStatus.value = 'failed'
    geneticError.value = true
    geneticMsg.value = '启动失败: ' + (e.response?.data?.detail || e.message)
  }
}

async function resumeGeneticMine() {
  if (!geneticRunId.value || !geneticResumeAvailable.value || geneticMining.value) return
  geneticMining.value = true
  geneticError.value = false
  geneticMsg.value = '正在校验检查点与数据指纹…'
  try {
    await api.post(`/factor/mining/runs/${geneticRunId.value}/resume`)
    geneticResumeAvailable.value = false
    geneticStatus.value = 'running'
    geneticTaskMode.value = 'single'
    tracker.register('factor-genetic', geneticRunId.value, { type: '因子挖矿', mode: 'single' })
    geneticMsg.value = '检查点校验通过，已恢复任务'
    pollGenetic(geneticRunId.value, true)
  } catch (e) {
    geneticMining.value = false
    geneticStatus.value = 'failed'
    geneticError.value = true
    geneticMsg.value = '恢复失败: ' + (e.response?.data?.detail || e.message)
    await refreshGenetic()
  }
}

function formatIc(value) {
  if (value == null || Number.isNaN(Number(value))) return '--'
  return `${(Number(value) * 100).toFixed(2)}%`
}

function candidateStatusText(status, inPool) {
  if (inPool || status === 'promoted') return '已入池'
  return ({ generated: '已生成', train_passed: '训练通过', validation_passed: '校验通过', strategy_passed: '策略通过', campaign_shortlisted: '持续挖掘入档', awaiting_promotion: '待晋级', rejected: '已拒绝' }[status] || status || '未入池')
}

function sourceText(source) {
  return ({ gp: '遗传编程', llm: '大语言模型' }[source] || source || '--')
}

function proposalSummary(candidate) {
  const proposal = candidate.source_metadata || candidate.detail?.proposal
  if (!proposal) return ''
  const hypothesis = proposal.hypothesis || '未提供假设'
  const model = proposal.model ? ` · ${proposal.model}` : ''
  return `大语言模型假设：${hypothesis}${model}`
}

function validationSummary(candidate) {
  const layers = candidate.validation_layers?.layers || []
  if (!layers.length) return '暂无层级结果'
  return layers.map(x => {
    const status = x.status || (x.pass ? 'passed' : 'failed')
    return `L${x.id}:${({ passed: '通过', failed: '未通过', skipped: '跳过' }[status] || status)}`
  }).join(' · ')
}

function candidateFailureSummary(candidate) {
  if (candidate.status !== 'rejected') return ''
  return `淘汰原因：${candidate.rejection_reason || candidate.strategy_metrics?.reason || '验证证据不足'}`
}

function abSummary(candidate) {
  const metrics = candidate.strategy_metrics
  if (!metrics || metrics.status !== 'evaluated') {
    const reason = metrics?.reason ? ` · ${metrics.reason}` : ''
    return `生产对照实验：${metrics?.status || '未执行'}${reason}`
  }
  const delta = metrics.delta || {}
  const ret = delta.total_return != null ? `${(delta.total_return * 100).toFixed(2)}%` : '--'
  const sharpe = delta.sharpe != null ? Number(delta.sharpe).toFixed(2) : '--'
  const lock = metrics.oos_lock?.status || '--'
  const adjusted = candidate.oos_metrics?.selection_adjusted_oos_ic
  const ic = adjusted != null ? ` · 校正样本外信息系数 ${(Number(adjusted) * 100).toFixed(2)}%` : ''
  return `生产对照实验：收益增量 ${ret} · 夏普比率增量 ${sharpe}${ic} · 样本外 ${lock}`
}

function detailScoreCount(candidate) {
  return candidate.detail?.scores?.length || 0
}

function detailBacktestCount(candidate) {
  return candidate.detail?.backtests?.length || 0
}

function detailWalkforwardScores(candidate) {
  return (candidate.detail?.scores || []).filter(item => item.stage === 'walk_forward')
}

function detailScoreSummary(item) {
  const metrics = item.metrics || {}
  const series = metrics.oos_ic_series || []
  return `样本外信息系数 ${formatIc(metrics.oos_ic)} · 每日信息系数 ${series.length}条`
}

function detailDailyIc(candidate) {
  return detailWalkforwardScores(candidate)
    .flatMap(item => item.metrics?.oos_ic_series || [])
    .slice(-10)
}

function gateSummary(candidate) {
  const gate = candidate.promotion_gate
  if (!gate) return '晋级门槛：未执行'
  if (gate.passed) return '晋级门槛：通过，等待人工确认'
  return `晋级门槛：未通过 · ${(gate.reasons || []).join('；') || '证据不足'}`
}

function validateGeneticForm() {
  const dates = [gp.train_start, gp.train_end, gp.valid_start, gp.valid_end].map(x => new Date(x).getTime())
  if (dates.some(Number.isNaN) || !(dates[0] < dates[1] && dates[1] < dates[2] && dates[2] < dates[3])) return '训练和验证日期必须严格递增'
  if (gp.candidate_mode !== 'llm' && (!Number.isInteger(gp.population) || gp.population < 2)) return '种群数量至少为 2'
  if (gp.candidate_mode !== 'llm' && (!Number.isInteger(gp.generations) || gp.generations < 1)) return '代数必须为正整数'
  if (gp.candidate_mode !== 'llm' && (!Number.isInteger(gp.max_depth) || gp.max_depth < 2 || gp.max_depth > 8)) return '最大深度必须在 2 到 8 之间'
  if (!Number.isInteger(gp.top_n) || gp.top_n < 1 || !Number.isInteger(gp.forward_days) || gp.forward_days < 1) return '候选数量和前瞻天数必须为正整数'
  if (!Number.isInteger(gp.step) || gp.step < 1 || !Number.isInteger(gp.seed) || gp.seed < 0) return '评估步长或随机种子无效'
  if (!gp.final_oos_start || !gp.final_oos_end) return '正式任务必须填写最终样本外起止日期'
  if (new Date(gp.final_oos_start).getTime() > new Date(gp.final_oos_end).getTime()) return '最终样本外日期范围无效'
  if (new Date(gp.final_oos_start).getTime() <= new Date(gp.valid_end).getTime()) return '最终样本外必须严格晚于验证结束日期'
  if (!Number.isInteger(gp.backtest_top_n) || gp.backtest_top_n < 1 || gp.overlay_weight < 0 || gp.overlay_weight > 0.5) return '生产对照实验参数无效'
  if (gp.baseline_model_key === 'auto' && miningModelsLoaded.value && !compatibleMiningModels.value.length) return '当前条件下没有可用的量化基线模型，请调整训练截止日或先训练历史模型'
  if (gp.baseline_model_key !== 'auto' && (!selectedMiningModel.value || !selectedMiningModel.value.compatible)) return '请选择满足当前时间边界和股票池约束的量化模型'
  if (gp.candidate_mode !== 'gp' && (!Number.isInteger(gp.llm_candidate_count) || gp.llm_candidate_count < 1 || gp.llm_candidate_count > 50)) return '大语言模型提议数必须在1到50之间'
  if (gp.candidate_mode !== 'gp' && (gp.llm_temperature < 0 || gp.llm_temperature > 1)) return '大语言模型生成随机度必须在0到1之间'
  if (gp.execution_mode === 'continuous' && (gp.max_batches < 0 || gp.max_hours < 0 || gp.min_free_memory_gb < 1 || gp.idle_cpu_percent < 10 || gp.idle_cpu_percent > 100)) return '持续挖掘资源或预算参数无效'
  return ''
}

async function applyGeneticTask(t, jobId) {
  geneticStatus.value = t.status || geneticStatus.value
  geneticProgress.value = t.progress || geneticProgress.value
  geneticResumeAvailable.value = t.resume_available === true
  if (t.status === 'failed' && geneticResumeAvailable.value) {
    geneticMsg.value = '任务在可恢复代际边界后失败；可显式恢复并重新校验输入指纹。'
  }
  if (t.status === 'stalled') {
    geneticMsg.value = '任务可能停滞，仍会继续监听后端心跳。'
    return
  }
  if (t.status === 'interrupted' || t.status === 'not_found') {
    if (t.status === 'interrupted' && geneticTaskMode.value !== 'campaign') {
      try {
        const r = await api.get(`/factor/mining/runs/${jobId}/candidates`)
        geneticResult.value = {
          candidates: r?.data?.candidates || [],
          promotion_conclusion: 'interrupted',
        }
      } catch (e) {}
    }
    clearInterval(geneticTimer)
    geneticMining.value = false
    if (t.status === 'interrupted' && geneticResumeAvailable.value) {
      tracker.update('factor-genetic', { status: 'interrupted', resume_available: true })
    } else {
      tracker.unregister('factor-genetic')
    }
    geneticError.value = t.status === 'not_found'
    geneticMsg.value = t.status === 'interrupted'
      ? (geneticResumeAvailable.value
        ? '后端任务已中断；存在代际边界检查点，点击恢复后会重新校验原数据和算法指纹。'
        : '后端任务已中断，查询不会自动重启；当前阶段没有可验证的恢复检查点，请重新提交任务。')
      : '找不到该任务记录，已停止轮询。'
    return
  }
  const terminal = ['completed', 'partial', 'awaiting_promotion', 'failed', 'cancelled']
  if (!terminal.includes(t.status)) return
  if (geneticTaskMode.value === 'campaign') {
    geneticResult.value = { candidates: t.candidates || [], promotion_conclusion: 'campaign_archived', campaign_runs: t.runs || [] }
  } else if (t.result?.report) geneticResult.value = t.result.report
  else if (t.result?.candidates || t.result?.factors) geneticResult.value = t.result
  else {
    try {
      const report = await api.get(`/factor/mining/runs/${jobId}/report`)
      geneticResult.value = report?.data?.report || null
    } catch (e) { /* 报告稍后可刷新 */ }
  }
  if (geneticResult.value && !geneticResult.value.candidates) {
    try {
      const r = await api.get(`/factor/mining/runs/${jobId}/candidates`)
      geneticResult.value.candidates = r?.data?.candidates || []
    } catch (e) { /* ignore */ }
  }
  clearInterval(geneticTimer)
  geneticMining.value = false
  if (t.status === 'failed' && geneticResumeAvailable.value) {
    tracker.update('factor-genetic', { status: 'failed', resume_available: true })
  } else {
    tracker.unregister('factor-genetic')
  }
  if (t.status === 'failed') {
    geneticError.value = true
    geneticMsg.value = '失败: ' + (t.error || '未知错误')
  } else if (t.status === 'partial') {
    geneticError.value = true
    geneticMsg.value = `任务部分完成：${t.error || t.progress?.message || '请检查已保存结果和失败项'}`
  } else if (t.status === 'cancelled') {
    geneticMsg.value = '任务已取消'
  } else {
    if (geneticTaskMode.value === 'campaign') {
      geneticMsg.value = `持续挖掘已结束：累计展示 ${geneticCandidates.value.length} 个唯一候选；冻结最终样本外尚未执行`
      return
    }
    const passed = geneticResult.value?.result_summary?.validation_passed_count || 0
    geneticMsg.value = passed
      ? `挖掘完成: ${geneticCandidates.value.length} 个候选，${passed} 个通过校验`
      : `挖掘完成: ${geneticCandidates.value.length} 个候选，无合格候选`
  }
}

function pollGenetic(jobId, immediate = false) {
  if (geneticTimer) clearInterval(geneticTimer)
  const fetchStatus = async () => {
    try {
      const path = geneticTaskMode.value === 'campaign' ? `/factor/mining/campaigns/${jobId}` : `/factor/mining/runs/${jobId}`
      const r = await api.get(path)
      const t = r?.data || {}
      await applyGeneticTask(t, jobId)
    } catch (e) { /* 网络抖动忽略 */ }
  }
  if (immediate) fetchStatus()
  geneticTimer = setInterval(fetchStatus, 3000)
}

async function cancelGeneticMine() {
  if (!geneticRunId.value || !confirm('确定取消当前因子挖矿任务吗？')) return
  try {
    const path = geneticTaskMode.value === 'campaign' ? `/factor/mining/campaigns/${geneticRunId.value}/cancel` : `/factor/mining/runs/${geneticRunId.value}/cancel`
    await api.post(path)
    geneticMsg.value = '已请求取消，等待当前步骤结束…'
  } catch (e) { geneticMsg.value = '取消失败: ' + (e.response?.data?.detail || e.message) }
}

async function refreshGenetic() {
  if (geneticRunId.value) pollGenetic(geneticRunId.value, true)
}

async function loadCandidateDetail(candidate) {
  const key = candidate.candidate_key || candidate.key
  const runId = candidate.run_id || geneticRunId.value
  if (!runId || !key || geneticDetailLoading[key]) return
  if (candidate.detail) {
    candidate.detail = null
    return
  }
  geneticDetailLoading[key] = true
  try {
    const r = await api.get(`/factor/mining/runs/${runId}/candidates/${key}`)
    candidate.detail = r?.data?.candidate || null
  } catch (e) {
    geneticMsg.value = '候选详情加载失败: ' + e.message
  } finally {
    delete geneticDetailLoading[key]
  }
}

async function promoteCandidate(candidate) {
  const target = gp.baseline_set === 'current_curated' ? '当前精选因子集' : gp.baseline_set
  if (!geneticRunId.value || !confirm(`确认将 ${candidate.candidate_key} 晋级到 ${target} 吗？\n\n${candidate.expr}`)) return
  try {
    const r = await api.post(`/factor/mining/runs/${geneticRunId.value}/promote`, {
      candidate_key: candidate.candidate_key, target_pool: gp.baseline_set, action: 'promote',
    })
    Object.assign(candidate, r?.data?.candidate || {}, { status: 'promoted' })
    geneticMsg.value = `${candidate.candidate_key} 已人工确认并写入 ${candidate.target_pool || target}`
    await loadPool()
  } catch (e) { geneticMsg.value = '晋级失败: ' + (e.response?.data?.detail || e.message) }
}

async function trainCandidate(candidate) {
  const runId = candidate.run_id || geneticRunId.value
  const key = candidate.candidate_key || candidate.key
  if (!runId || !key || candidateTraining.value) return
  if (!confirm(`使用 ${key} 与当前基线因子训练一个独立研究模型？\n该模型不会自动激活。`)) return
  candidateTraining.value = true
  candidateTrainingResult.value = null
  candidateTrainingProgress.value = { message: '正在提交研究训练…' }
  try {
    const r = await api.post(`/factor/mining/runs/${runId}/train`, {
      candidate_keys: [key], include_baseline: true,
      train_baseline_control: true, param_profile: 'conservative',
    })
    const trainingId = r?.data?.training_id
    if (!trainingId) throw new Error('未返回 training_id')
    if (candidateTrainingTimer) clearInterval(candidateTrainingTimer)
    const fetchTraining = async () => {
      const status = (await api.get(`/factor/mining/training/${trainingId}`))?.data || {}
      candidateTrainingProgress.value = status.progress || candidateTrainingProgress.value
      if (!['completed', 'failed', 'cancelled'].includes(status.status)) return
      clearInterval(candidateTrainingTimer)
      candidateTraining.value = false
      if (status.status === 'completed') {
        candidateTrainingResult.value = status.result || {}
        geneticMsg.value = `${key} 的候选增强研究模型已训练完成，尚未激活`
      } else {
        geneticError.value = true
        geneticMsg.value = '研究训练失败: ' + (status.error || '未知错误')
      }
    }
    await fetchTraining()
    if (candidateTraining.value) candidateTrainingTimer = setInterval(fetchTraining, 3000)
  } catch (e) {
    candidateTraining.value = false
    geneticError.value = true
    geneticMsg.value = '研究训练启动失败: ' + (e.response?.data?.detail || e.message)
  }
}

async function rejectCandidate(candidate) {
  const reason = prompt('请输入拒绝原因（会写入审计记录）', '验证证据不足')
  if (reason == null || !geneticRunId.value) return
  try {
    const r = await api.post(`/factor/mining/runs/${geneticRunId.value}/promote`, {
      candidate_key: candidate.candidate_key, target_pool: gp.baseline_set, action: 'reject', reason,
    })
    Object.assign(candidate, r?.data?.candidate || {}, { status: 'rejected' })
  } catch (e) { geneticMsg.value = '拒绝记录失败: ' + (e.response?.data?.detail || e.message) }
}

onMounted(() => {
  const geneticTask = tracker.getTask('factor-genetic')
  if (geneticTask && (
    ['running', 'stalled', 'cancel_requested'].includes(geneticTask.status)
    || (['interrupted', 'failed'].includes(geneticTask.status) && geneticTask.resume_available)
  )) {
    geneticRunId.value = geneticTask.jobId
    geneticTaskMode.value = geneticTask.mode || 'single'
    geneticMining.value = true
    geneticStatus.value = geneticTask.status
    geneticMsg.value = '正在恢复因子挖矿任务...'
    pollGenetic(geneticTask.jobId)
  }
})

onUnmounted(() => {
  if (geneticTimer) clearInterval(geneticTimer)
  if (candidateTrainingTimer) clearInterval(candidateTrainingTimer)
})
</script>

<style scoped>
.fm-root {
  max-width: var(--content-max-width);
  margin: 0 auto;
  padding: 0 var(--content-padding);
}

/* ═══ Tabs ═══ */
.fm-tabs {
  display: flex;
  gap: 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: var(--space-5);
  overflow-x: auto;
  scrollbar-width: thin;
}
.fm-tab {
  padding: 10px 20px;
  border: none;
  background: transparent;
  color: var(--text-muted);
  font-size: var(--fs-body);
  cursor: pointer;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
  transition: all var(--transition-fast);
  font-family: var(--font);
  white-space: nowrap;
  flex: 0 0 auto;
}
.fm-tab:hover { color: var(--text); }
.fm-tab.active { color: var(--primary); border-bottom-color: var(--primary); }
.fm-tab-badge {
  font-size: var(--fs-caption);
  background: var(--primary);
  color: #fff;
  padding: 1px 7px;
  border-radius: var(--radius-full);
  margin-left: 4px;
}

/* ═══ Section ═══ */
.fm-section {
  padding-bottom: var(--space-8);
}

/* ═══ Hint ═══ */
.fm-hint {
  font-size: var(--fs-sm);
  color: var(--text-muted);
  line-height: 1.6;
  margin-bottom: var(--space-4);
  padding: 12px 14px;
  background: var(--primary-muted);
  border-left: 3px solid var(--primary);
  border-radius: var(--radius-sm);
}

/* ═══ Form blocks ═══ */
.fm-form-block {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: var(--space-4);
  margin-bottom: var(--space-3);
}
.fm-form-title {
  font-size: var(--fs-caption);
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 10px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}

.fm-form-row {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-end;
  gap: 12px 16px;
}
.fm-form-row--grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px 16px;
}
.fm-form-row--end {
  align-items: center;
}

.fm-field {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: var(--fs-caption);
  color: var(--text-muted);
}
.fm-field select, .fm-field input[type=date], .fm-field input[type=number], .fm-field input[type=text], .fm-field textarea {
  padding: 5px 8px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  font-size: var(--fs-sm);
}
.fm-field select:focus, .fm-field input:focus, .fm-field textarea:focus {
  border-color: var(--primary);
}
.fm-field--inline {
  flex-direction: row;
  align-items: center;
  gap: 8px;
}
.fm-field-wide { min-width: 320px; flex: 1; }
.fm-field-wide textarea { resize: vertical; min-height: 34px; }
.fm-hint--inline { margin-top: 10px; margin-bottom: 0; }

.fm-field-label {
  font-size: var(--fs-sm);
  color: var(--text-muted);
  white-space: nowrap;
}

.fm-field-span {
  grid-column: 1 / -1;
}

.fm-input-narrow { width: 55px; }
.fm-input-search { width: 140px; padding: 3px 6px; font-size: var(--fs-caption); border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg); color: var(--text); }

.fm-check {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: var(--fs-sm);
  color: var(--text-secondary);
  cursor: pointer;
}
.fm-check input[type=checkbox] { width: auto; margin: 0; }

.fm-prefix-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; padding-top: 2px; }

.fm-checks-row { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 8px; }
.fm-checks-row .fm-check { font-size: var(--fs-body); }

.fm-sector-row { display: flex; align-items: center; gap: 8px; margin-top: 8px; flex-wrap: wrap; }

.fm-chip-row { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
.fm-chip-row--scroll { max-height: 200px; overflow-y: auto; }
.fm-chip {
  font-size: 10px;
  padding: 2px 6px;
  border-radius: var(--radius-full);
  cursor: pointer;
  border: 1px solid var(--border);
  color: var(--text-secondary);
  transition: all var(--transition-fast);
}
.fm-chip:hover { border-color: var(--primary); color: var(--primary); }
.fm-chip--sel { background: var(--primary-muted); border-color: var(--primary); color: var(--primary); }

.fm-banner {
  display: block;
  padding: 10px 14px;
  background: rgba(250, 176, 5, .08);
  border: 1px solid var(--warning);
  border-radius: var(--radius);
  margin-top: 8px;
  font-size: var(--fs-body);
}

/* ═══ Progress ═══ */
.fm-progress { margin-bottom: 14px; }
.fm-prog-head { display: flex; justify-content: space-between; align-items: center; padding: 8px 0 4px; }
.fm-prog-label { font-size: var(--fs-sm); color: var(--text-secondary); }
.fm-prog-time { color: var(--text-muted); margin-left: 8px; font-size: var(--fs-caption); }
.fm-prog-line { font-size: var(--fs-sm); color: var(--text-secondary); padding: 8px 0 4px; }
.fm-prog-bar { height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; }
.fm-prog-fill { height: 100%; background: var(--primary); border-radius: 4px; transition: width .3s ease; }
.fm-prog-pct { font-size: var(--fs-caption); color: var(--primary); font-weight: 600; }
.fm-prog-meta { color: var(--text-muted); font-size: var(--fs-caption); padding-top: 6px; }
.fm-actions-cell { white-space: nowrap; }
.fm-detail { color: var(--text-muted); font-size: var(--fs-caption); line-height: 1.6; padding-top: 4px; }

.fm-anchors { display: flex; flex-wrap: wrap; gap: 8px; padding-top: 10px; }
.fm-anchor {
  flex: 0 0 calc(25% - 6px);
  min-width: 160px;
  background: var(--bg);
  border-radius: var(--radius-sm);
  padding: 10px 12px;
  border: 1px solid var(--border);
}
.fm-anchor--completed {
  border-color: rgba(52, 211, 153, .3);
  background: rgba(52, 211, 153, .05);
}
.fm-anchor-head { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }
.fm-anchor-dot { font-size: 14px; flex-shrink: 0; color: var(--success); }
.fm-anchor-spin { display: inline-block; animation: fm-spin 1s linear infinite; }
@keyframes fm-spin { to { transform: rotate(360deg); } }
.fm-anchor-date { font-size: var(--fs-caption); color: var(--text); white-space: nowrap; }
.fm-anchor-passed { font-size: var(--fs-caption); color: var(--success); margin-left: auto; font-weight: 600; }
.fm-anchor-stage { font-size: var(--fs-caption); color: var(--text-muted); margin-bottom: 3px; }
.fm-anchor-bar { height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }
.fm-anchor-fill { height: 100%; background: var(--primary); border-radius: 2px; transition: width .3s ease; }

/* ═══ Result ═══ */
.fm-result { margin-top: 12px; }
.fm-result-title { font-size: var(--fs-base); color: var(--text); margin: 0 0 12px; }

/* ═══ Table ═══ */
.fm-table {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--fs-sm);
}
.fm-table th {
  text-align: left;
  padding: 8px 10px;
  border-bottom: 1px solid var(--border);
  color: var(--text-muted);
  font-weight: 600;
  font-size: var(--fs-caption);
  text-transform: uppercase;
  letter-spacing: 0.4px;
  position: sticky;
  top: 0;
  background: var(--card-bg);
  z-index: 1;
}
.fm-table td {
  padding: 8px 10px;
  border-bottom: 1px solid rgba(42, 42, 80, .4);
  color: var(--text-secondary);
}
.fm-table code {
  font-size: var(--fs-caption);
  background: var(--primary-muted);
  padding: 2px 6px;
  border-radius: 3px;
  color: var(--primary);
}
.fm-table tbody tr:hover td { background: var(--primary-muted); }
.fm-row--inactive td { opacity: .45; }
.fm-row--eliminated td { opacity: .45; text-decoration: line-through; }

.fm-cell-wrap {
  font-family: var(--font-mono);
  font-size: var(--fs-caption);
  max-width: 140px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.fm-cell-name { max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ═══ Toolbar ═══ */
.fm-toolbar {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 14px;
  flex-wrap: wrap;
}
.fm-toolbar-stat { font-size: var(--fs-sm); color: var(--text-muted); flex: 1; }
.fm-search {
  padding: 4px 8px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  font-size: var(--fs-caption);
  background: var(--bg);
  color: var(--text);
  width: 180px;
}
.fm-search:focus { border-color: var(--primary); }
.fm-select-sm {
  padding: 4px 8px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  font-size: var(--fs-caption);
  background: var(--bg);
  color: var(--text);
}

/* ═══ Pager ═══ */
.fm-pager {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 12px;
  font-size: var(--fs-sm);
  color: var(--text-muted);
}
.fm-pager button {
  padding: 4px 10px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-size: var(--fs-caption);
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
}
.fm-pager button:disabled { opacity: .4; cursor: default; }

/* ═══ Mini buttons ═══ */
.btn-mini {
  padding: 2px 8px;
  font-size: 10px;
  font-family: var(--font);
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  cursor: pointer;
  margin-right: 4px;
}
.btn-mini:hover { border-color: var(--primary); color: var(--primary); }
.btn-mini:disabled { opacity: .4; cursor: not-allowed; }
.btn-mini--danger { color: var(--danger); }
.btn-mini--danger:hover { border-color: var(--danger); color: var(--danger); }
.btn-mini--primary { background: var(--primary); color: var(--on-primary); border-color: var(--primary); }
.btn-mini--primary:hover { opacity: .85; }

/* ═══ Crowding dot ═══ */
.fm-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; }
.fm-dot--normal { background: var(--success); }
.fm-dot--warning { background: var(--warning); }
.fm-dot--crowded { background: var(--danger); }
.fm-dot--unknown { background: var(--text-muted); opacity: .3; }

/* ═══ Sort result ═══ */
.fm-sort-row td { padding: 0 !important; }
.fm-sort-card {
  padding: 10px 14px;
  margin: 4px 0;
  background: var(--bg);
  border-radius: var(--radius-sm);
  font-size: var(--fs-sm);
  color: var(--text-secondary);
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.fm-sort-card strong { color: var(--primary); }

/* ═══ Preview (training timeline) ═══ */
.fm-preview {
  margin-top: 8px;
  padding: 12px 16px;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  line-height: 1.8;
}
.fm-preview-title { color: var(--text-muted); margin-bottom: 6px; font-size: var(--fs-sm); }
.fm-preview-bar { font-family: var(--font-mono); font-size: var(--fs-sm); }
.fm-preview-tag { padding: 2px 8px; border-radius: 4px; font-size: var(--fs-sm); }
.fm-preview-tag--train { background: rgba(107, 159, 212, .15); color: var(--primary); }
.fm-preview-tag--valid { background: rgba(245, 158, 11, .15); color: var(--warning); }
.fm-preview-tag--test { background: rgba(34, 197, 94, .15); color: var(--success); }
.fm-preview-arr { color: var(--text-muted); margin: 0 4px; }

/* ═══ Metrics strip ═══ */
.fm-metrics-strip {
  display: flex;
  gap: 24px;
  flex-wrap: wrap;
  background: rgba(34, 197, 94, .06);
  border-radius: var(--radius);
  padding: 14px 18px;
  margin-bottom: 16px;
}
.fm-metric { text-align: center; min-width: 80px; }
.fm-metric-label { font-size: var(--fs-caption); color: var(--text-muted); margin-top: 2px; }

/* ═══ Filter result ═══ */
.fm-filter-result {
  background: rgba(34, 197, 94, .04);
  border: 1px solid var(--success);
  border-radius: var(--radius-sm);
  padding: 16px;
  margin-top: 12px;
}
.fm-detail-summary { cursor: pointer; font-weight: 600; font-size: var(--fs-base); color: var(--text); }
.fm-detail-summary:hover { color: var(--primary); }
.fm-failed-item { padding: 4px 0; font-size: var(--fs-sm); font-family: var(--font-mono); color: var(--text-muted); }

/* ═══ Importance table ═══ */
.fm-importance { margin-top: 14px; }

/* ═══ Rounds ═══ */
.fm-rounds { margin-top: 14px; }
.fm-model-card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 12px 14px;
  margin-bottom: 8px;
}
.fm-model-head { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.fm-model-name { font-size: var(--fs-base); font-weight: 600; color: var(--text); display: flex; align-items: center; gap: 8px; }
.fm-model-metrics { display: flex; gap: 18px; flex-wrap: wrap; }

.fm-mm { display: flex; flex-direction: column; align-items: center; gap: 2px; min-width: 55px; }
.fm-mm-val { font-size: var(--fs-base); font-weight: 600; color: var(--text); }
.fm-mm-label { font-size: 10px; color: var(--text-muted); }

/* ═══ Train result ═══ */
.fm-train-result { margin-top: 16px; }

/* ═══ Direction ═══ */
.fm-regime-bar {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 16px;
  padding: 12px 16px;
  background: var(--bg-secondary);
  border-radius: var(--radius);
  border: 1px solid var(--border);
  flex-wrap: wrap;
}
.fm-regime-tag {
  padding: 2px 10px;
  border-radius: 12px;
  font-size: var(--fs-body);
  font-weight: 600;
  color: #fff;
}

.fm-dir-grid { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 12px; }
.fm-dir-card {
  flex: 1;
  min-width: 260px;
  max-width: 340px;
  padding: 16px;
  background: var(--bg-secondary);
  border-radius: var(--radius);
  border: 1px solid var(--border);
  transition: border-color var(--transition-fast);
}
.fm-dir-card:hover { border-color: var(--border-light); }

.fm-dir-name { margin: 0 0 4px; font-size: var(--fs-base); color: var(--text); }
.fm-dir-desc { font-size: var(--fs-sm); color: var(--text-muted); margin: 0 0 8px; }
.fm-dir-cats { font-size: var(--fs-caption); color: var(--text-muted); margin-bottom: 8px; }
.fm-dir-stat { font-size: var(--fs-sm); margin-bottom: 4px; }
.fm-dir-actions { display: flex; gap: 6px; margin-top: 8px; }
.fm-dir-factors {
  margin-top: 6px;
  font-size: 10px;
  color: var(--text-muted);
  max-height: 100px;
  overflow-y: auto;
}

/* ═══ Curated set detail ═══ */
.fm-cat-grid { display: flex; flex-wrap: wrap; gap: 8px; }
.fm-cat-card {
  background: var(--bg-secondary);
  border-radius: var(--radius);
  padding: 10px 14px;
  min-width: 120px;
}
.fm-cat-name { font-size: var(--fs-caption); color: var(--primary); font-weight: 600; }
.fm-cat-keys { font-size: 10px; color: var(--text-muted); max-height: 60px; overflow-y: auto; }
</style>
