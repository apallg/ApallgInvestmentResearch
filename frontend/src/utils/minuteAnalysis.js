export function hasVerifiedMinuteProvenance(provenance) {
  return provenance?.quantity_status === 'verified' && provenance?.semantic_status === 'verified'
}

export function blockUnverifiedMinuteExecution(execution) {
  const reason = '分钟数据来源语义未核验，执行建议不可用'
  return {
    ...(execution || {}),
    action: 'wait',
    intent_action: 'wait',
    limit_price: null,
    status: 'blocked',
    can_execute: false,
    distance_pct: null,
    reason,
    blockers: [...new Set([...(execution?.blockers || []), reason])],
    trigger_text: `执行资格阻断：${reason}`,
  }
}
