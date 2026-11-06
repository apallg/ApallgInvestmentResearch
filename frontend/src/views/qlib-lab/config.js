/**
 * QlibLab 静态配置：代码前缀、参数定义。
 * 从 QlibLab.vue 顶部迁移到这里，供壳组件和各 Tab 复用。
 */
export const CODE_PREFIX_OPTIONS = [
  [['600', '601', '603', '605'], '上海主板'],
  [['000', '001', '002', '003'], '深圳主板'],
  [['300', '301'], '创业板'],
  [['688'], '科创板'],
  [['002'], '中小板'],
]

export const TRAIN_PARAM_DEFS = [
  ['learning_rate', 'learning_rate'],
  ['max_depth', 'max_depth'],
  ['num_leaves', 'num_leaves'],
  ['num_trees', 'num_trees'],
  ['subsample', 'subsample'],
  ['colsample_bytree', 'colsample'],
  ['reg_alpha', 'reg_alpha'],
  ['reg_lambda', 'reg_lambda'],
  ['min_child_samples', 'min_child'],
  ['label_horizon', '择时周期'],
  ['threshold', '阈值'],
  ['n_features', '特征数'],
]
