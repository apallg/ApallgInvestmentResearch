# -*- coding: utf-8 -*-
"""
t0_analyze_result.py — 聚宽 T+0 导出 CSV 离线分析
吃聚宽 record() 导出的日频 CSV（无需分钟级数据），输出：
  1) T 腿明细：轮数/胜率/累计T收益/盈亏比/平均盈亏/降本/T腿最大回撤(元)
  2) 出场归因：止盈/止损/尾盘/隔夜/尾盘止损/尾盘锁盈 的轮数与盈亏
  3) 组合与超额：策略/基准/超额收益曲线、各自最大回撤、超额回撤区间
  4) 过滤漏斗（策略导出后自动出现）：各过滤层拒绝次数
用法:
  py -3 t0_analyze_result.py <csv1> [csv2 ...]
  或 py -3 t0_analyze_result.py "C:/xxx/result_1*.csv"  （shell 展开）
  或 py -3 t0_analyze_result.py --dir <目录>             （扫描目录下全部 result_1*.csv）
"""
import csv
import glob
import os
import sys

GROUP_NAMES = {'1':'A','2':'B','3':'C','4':'D','5':'E','6':'G','7':'H','8':'I','9':'J','10':'K','11':'L','12':'M','13':'N'}

def load(path):
    for enc in ('utf-8-sig', 'gbk', 'gb18030'):
        try:
            with open(path, encoding=enc, newline='') as fh:
                return [dict(r) for r in csv.DictReader(fh)]
        except (UnicodeDecodeError, UnicodeError):
            continue
    return []

def num(d, k, default=0.0):
    try:
        v = d.get(k, '')
        return default if v in (None, '') else float(v)
    except (TypeError, ValueError):
        return default

def mdd_of(series):
    """series: 数值列表（时间顺序）。返回 (max_dd, peak_idx, trough_idx, peak_val_at_trough, trough_val)
    peak_idx 为谷底时刻的滚动峰值所在位置，保证 peak 在 trough 之前。"""
    peak = -1e300; peak_i = 0; max_dd = 0.0; trough_i = 0; trough_v = 0.0; peak_v = 0.0
    peak_at_trough = 0; peak_val_at_trough = 0.0
    for i, v in enumerate(series):
        if v > peak:
            peak = v; peak_i = i; peak_v = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd; trough_i = i; trough_v = v
            peak_at_trough = peak_i
            peak_val_at_trough = peak
    return max_dd, peak_at_trough, trough_i, peak_val_at_trough, trough_v

def equity_from_cumret(pcts):
    """JoinQuant 导出的是累计收益率(%), 净值 = 100 * (1 + r%)"""
    return [100.0 * (1.0 + p / 100.0) for p in pcts]

def analyze(path):
    rows = load(path)
    if not rows:
        return {'file': os.path.basename(path), 'error': 'cannot read'}
    last = rows[-1]
    cum = lambda k: num(last, k)
    days = len(rows)
    dates = [r['时间'][:10] for r in rows]
    gid = int(cum('t_experiment_group'))
    name = GROUP_NAMES.get(str(gid), str(gid))

    # T 腿
    rounds = int(cum('t_buy_first_rounds') + cum('t_sell_first_rounds'))
    t_cum = cum('t_cum_profit')
    t_marked = cum('t_marked_profit')
    t_series = [num(r, 't_marked_profit', num(r, 't_cum_profit')) for r in rows]
    t_mdd, tp_i, tt_i, tp_v, tt_v = mdd_of(t_series)
    day_t = [num(r, 't_day_profit') for r in rows]
    nz_days = [(dates[i], day_t[i]) for i in range(days) if day_t[i] != 0]
    win_days = sum(1 for v in day_t if v > 0)
    loss_days = sum(1 for v in day_t if v < 0)
    max_day_loss = min(day_t)
    max_day_win = max(day_t)

    # 出场归因（旧CSV只有profit，新CSV有rounds）
    reasons = ['止盈','移动止盈','止损','尾盘平仓','隔夜平仓','尾盘止损','尾盘锁盈']
    attr = {}
    for rname in reasons:
        pk, rk = 't_%s_profit' % {'止盈':'take','移动止盈':'trail','止损':'stop','尾盘平仓':'force','隔夜平仓':'overnight','尾盘止损':'as_cut','尾盘锁盈':'as_lock'}[rname], 't_%s_rounds' % {'止盈':'tp','移动止盈':'trail','止损':'sl','尾盘平仓':'force','隔夜平仓':'on','尾盘止损':'as_cut','尾盘锁盈':'as_lock'}[rname]
        attr[rname] = (round(cum(pk), 2), int(cum(rk)))

    # 组合/超额（JoinQuant 导出列）
    strat_pcts = [num(r, '策略收益') for r in rows]
    bench_pcts = [num(r, '基准收益') for r in rows]
    excess_pcts = [num(r, '超额收益(%)') for r in rows]
    strat_eq = equity_from_cumret(strat_pcts)
    bench_eq = equity_from_cumret(bench_pcts)
    s_mdd, s_pi, s_ti, s_pv, _ = mdd_of(strat_eq)
    b_mdd, _, _, b_pv, _ = mdd_of(bench_eq)
    e_mdd, e_pi, e_ti, _, _ = mdd_of(excess_pcts)
    # 每日超额（百分点）
    daily_excess = [strat_pcts[i] - bench_pcts[i] for i in range(days)]
    d_excess_mdd, d_pi, d_ti, _, _ = mdd_of(daily_excess)

    res = {
        'file': os.path.basename(path),
        'group': f"{name}({gid})",
        'period': f"{dates[0]}~{dates[-1]}",
        'days': days,
        'rounds': rounds,
        'win_rate': round(cum('t_win_rate'), 3),
        't_cum': round(t_cum, 2),
        't_marked': round(t_marked, 2),
        't_mdd_yuan': round(t_mdd, 2),
        't_mdd_range': f"{dates[tp_i]}~{dates[tt_i]}",
        'profit_factor': round(cum('t_profit_factor'), 2) if 't_profit_factor' in last else None,
        'avg_win': round(cum('t_avg_win'), 2) if 't_avg_win' in last else None,
        'avg_loss': round(cum('t_avg_loss'), 2) if 't_avg_loss' in last else None,
        'cost_red': round(cum('t_cost_reduction'), 4),
        'attr': attr,
        'scale_out': int(cum('t_scale_out_count')),
        'breaker_days': int(cum('t_breaker_days')),
        'vol_rejects': int(cum('t_volume_rejects')),
        'td_exits': int(cum('t_time_decay_exits')),
        'trail_exits': int(cum('t_trailing_exits')),
        'win_days': win_days, 'loss_days': loss_days,
        'max_day_win': round(max_day_win, 2), 'max_day_loss': round(max_day_loss, 2),
        'strat_ret': round(strat_pcts[-1], 2), 'bench_ret': round(bench_pcts[-1], 2),
        'excess_final': round(excess_pcts[-1], 2),
        'strat_mdd_pct': round(s_mdd / s_pv * 100.0, 2) if s_pv > 0 else 0.0,
        'bench_mdd_pct': round(b_mdd / b_pv * 100.0, 2) if b_pv > 0 else 0.0,
        'excess_peak': round(max(excess_pcts), 2),
        'excess_mdd_pp': round(e_mdd, 2),
        'excess_mdd_range': f"{dates[e_pi]}~{dates[e_ti]}",
        'daily_excess_mdd': round(d_excess_mdd, 3),
        'core_reduce_days': int(cum('t_core_reduce_days')) if 't_core_reduce_days' in last else None,
        'core_sell_shares': int(cum('t_core_sell_shares')) if 't_core_sell_shares' in last else None,
        # 过滤漏斗（策略导出后才有）
        'funnel': {k: int(cum('t_f_' + k)) for k in ['daily','bars','range','vol_low','vol_high','candidate','confirm','risk','rsi','location','bottom','market','pause','opening_path','opening_breakdown','volume','breaker'] if ('t_f_' + k) in last},
    }
    return res

def fmt_row(r):
    if 'error' in r:
        return f"{r['file']}: ERROR"
    lines = []
    lines.append(f"== {r['file']}  [{r['group']}]  {r['period']} ({r['days']}天) ==")
    lines.append(f"  T腿: 轮数={r['rounds']}  胜率={r['win_rate']}  累计T收益={r['t_cum']}元  盯市={r['t_marked']}元  降本={r['cost_red']}元/股")
    lines.append(f"  T腿最大回撤: {r['t_mdd_yuan']}元 ({r['t_mdd_range']})   单日最大T亏损={r['max_day_loss']}元  盈利日={r['win_days']}/亏损日={r['loss_days']}")
    if r['profit_factor'] is not None:
        lines.append(f"  盈亏比={r['profit_factor']}  平均盈利={r['avg_win']}元  平均亏损={r['avg_loss']}元")
    at = "  ".join(f"{k}:{v[1]}轮/{v[0]}元" for k, v in r['attr'].items())
    lines.append(f"  出场归因: {at}")
    lines.append(f"  组合: 策略收益={r['strat_ret']}%  基准={r['bench_ret']}%  累计超额={r['excess_final']}% (峰值{r['excess_peak']}%)")
    lines.append(f"  组合最大回撤: 策略={r['strat_mdd_pct']}%  基准={r['bench_mdd_pct']}%  超额回撤={r['excess_mdd_pp']}pp ({r['excess_mdd_range']})")
    if r['funnel']:
        funnel = "  ".join(f"{k}:{v}" for k, v in r['funnel'].items())
        lines.append(f"  过滤漏斗: {funnel}")
    if r.get('core_reduce_days') is not None:
        lines.append(f"  核心仓风控: 减仓天数={r['core_reduce_days']}  累计卖出={r['core_sell_shares']}股")
    return "\n".join(lines)

def main():
    args = sys.argv[1:]
    paths = []
    if args and args[0] == '--dir':
        d = args[1]
        paths = sorted(glob.glob(os.path.join(d, 'result_1*.csv')))
    else:
        for a in args:
            paths.extend(sorted(glob.glob(a)))
    if not paths:
        print(__doc__)
        return
    for p in paths:
        r = analyze(p)
        print(fmt_row(r))
        print()

if __name__ == '__main__':
    main()
