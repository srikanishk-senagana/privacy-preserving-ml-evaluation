# src/aggregate_and_plot.py
# Aggregate results/*.txt (Opacus + TFP) into CSVs and generate 7 plots.

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- resolve paths relative to this file (works from any cwd) ---
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
RESULT_DIR  = os.path.join(PROJECT_DIR, 'results')
FIG_DIR     = os.path.join(PROJECT_DIR, 'figs')

print("Using RESULT_DIR:", RESULT_DIR)
print("Using FIG_DIR   :", FIG_DIR)

if not os.path.isdir(RESULT_DIR):
    raise FileNotFoundError(
        f"Results folder not found at: {RESULT_DIR}\n"
        f"Create it and place your .txt files there "
        f"(e.g., threshold_*.txt, shadow_*.txt, metrics_*.txt)."
    )

os.makedirs(FIG_DIR, exist_ok=True)


# ---------- parsing helpers ----------
def parse_kv_file(path):
    d = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            k, v = line.strip().split(':', 1)
            try:
                d[k.strip()] = float(v.strip())
            except ValueError:
                pass
    return d

def parse_eps(tag):
    m = re.search(r'eps(\d+)', tag)
    return int(m.group(1)) if m else None

def lib_from_tag(tag):
    tag = tag.lower()
    if 'opacus' in tag:   return 'opacus'
    if 'tfp' in tag:      return 'tfp'
    if 'baseline' in tag: return 'baseline'
    return 'unknown'

# ---------- scan results directory ----------
files = [f for f in os.listdir(RESULT_DIR) if f.endswith('.txt')]
if not files:
    print("No .txt result files found in", RESULT_DIR)
    raise SystemExit

util_rows, thr_rows, sh_rows = [], [], []
baseline_util, baseline_thr, baseline_sh = None, None, None

for fname in files:
    path = os.path.join(RESULT_DIR, fname)

    # Utility
    if fname == 'baseline_metrics.txt':
        m = parse_kv_file(path)
        baseline_util = {'accuracy': m.get('accuracy'), 'f1': m.get('f1')}
        continue
    if fname.startswith('metrics_opacus_eps') or fname.startswith('metrics_tfp_eps'):
        m   = parse_kv_file(path)
        tag = fname.replace('metrics_','').replace('.txt','')  # e.g., opacus_eps5
        util_rows.append({
            'lib': lib_from_tag(tag),
            'eps': parse_eps(tag),
            'accuracy': m.get('accuracy'),
            'f1': m.get('f1'),
        })
        continue

    # Threshold
    if fname.startswith('threshold_'):
        m   = parse_kv_file(path)
        tag = fname.replace('threshold_','').replace('.txt','')
        rec = {
            'lib': lib_from_tag(tag),
            'eps': parse_eps(tag),
            'auc': m.get('auc'),
            'advantage': m.get('advantage'),
        }
        if rec['lib'] == 'baseline':
            baseline_thr = rec
        else:
            thr_rows.append(rec)
        continue

    # Shadow
    if fname.startswith('shadow_'):
        m   = parse_kv_file(path)
        tag = fname.replace('shadow_','').replace('.txt','')
        rec = {
            'lib': lib_from_tag(tag),
            'eps': parse_eps(tag),
            'attack_acc': m.get('accuracy'),
            'auc': m.get('auc'),
        }
        if rec['lib'] == 'baseline':
            baseline_sh = rec
        else:
            sh_rows.append(rec)
        continue

# ---------- dataframes ----------
util_df = pd.DataFrame(util_rows).dropna().sort_values(['lib','eps'])
thr_df  = pd.DataFrame(thr_rows).dropna().sort_values(['lib','eps'])
sh_df   = pd.DataFrame(sh_rows).dropna().sort_values(['lib','eps'])

# Save CSVs
util_csv = os.path.join(RESULT_DIR, 'utility_summary.csv')
thr_csv  = os.path.join(RESULT_DIR, 'threshold_summary.csv')
sh_csv   = os.path.join(RESULT_DIR, 'shadow_summary.csv')
util_df.to_csv(util_csv, index=False)
thr_df.to_csv(thr_csv, index=False)
sh_df.to_csv(sh_csv, index=False)

print("Wrote CSVs:")
print(" -", util_csv)
print(" -", thr_csv)
print(" -", sh_csv)
print("\nUtility:\n", util_df if not util_df.empty else "(none)")
print("\nThreshold MIA:\n", thr_df if not thr_df.empty else "(none)")
print("\nShadow MIA:\n", sh_df if not sh_df.empty else "(none)")

# ---------- plotting helpers ----------
def line_plot_eps(df, ycol, title, ylabel, outname, include_baseline=None):
    if df.empty or ycol not in df.columns:
        return
    plt.figure()
    for lib in ['opacus', 'tfp']:
        sub = df[df['lib']==lib].sort_values('eps')
        if sub.empty:
            continue
        plt.plot(sub['eps'], sub[ycol], marker='o', label=lib.capitalize())
    if include_baseline is not None:
        plt.axhline(include_baseline, linestyle='--', label='Baseline', alpha=0.7)
    plt.title(title)
    plt.xlabel('Privacy budget ε')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, outname + '.png'), dpi=300)
    plt.savefig(os.path.join(FIG_DIR, outname + '.pdf'))
    plt.close()

def grouped_bars(df, ycol, title, ylabel, outname):
    if df.empty or ycol not in df.columns:
        return
    eps_list = sorted([e for e in df['eps'].dropna().unique().tolist()])
    if not eps_list:
        return
    x = np.arange(len(eps_list))
    width = 0.35
    plt.figure()
    # align bars per ε
    sub_o = df[(df['lib']=='opacus') & (df['eps'].isin(eps_list))].sort_values('eps')
    sub_t = df[(df['lib']=='tfp')    & (df['eps'].isin(eps_list))].sort_values('eps')
    vals_o = sub_o[ycol].values if not sub_o.empty else np.zeros(len(eps_list))
    vals_t = sub_t[ycol].values if not sub_t.empty else np.zeros(len(eps_list))
    # pad if one side missing some eps
    if len(vals_o) != len(eps_list):
        tmp = np.full(len(eps_list), np.nan)
        for i, e in enumerate(eps_list):
            m = sub_o[sub_o['eps']==e]
            tmp[i] = m[ycol].values[0] if not m.empty else np.nan
        vals_o = tmp
    if len(vals_t) != len(eps_list):
        tmp = np.full(len(eps_list), np.nan)
        for i, e in enumerate(eps_list):
            m = sub_t[sub_t['eps']==e]
            tmp[i] = m[ycol].values[0] if not m.empty else np.nan
        vals_t = tmp
    plt.bar(x - width/2, vals_o, width, label='Opacus')
    plt.bar(x + width/2, vals_t, width, label='TFP')
    plt.title(title)
    plt.xlabel('Privacy budget ε')
    plt.ylabel(ylabel)
    plt.xticks(x, [str(e) for e in eps_list])
    plt.legend()
    plt.grid(True, axis='y', linestyle=':', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, outname + '.png'), dpi=300)
    plt.savefig(os.path.join(FIG_DIR, outname + '.pdf'))
    plt.close()

# ---------- plots (7 total) ----------
baseline_acc = baseline_util['accuracy'] if baseline_util and 'accuracy' in baseline_util else None

# 1) ε vs Test Accuracy (+ baseline line)
line_plot_eps(util_df, 'accuracy',
              'ε vs Test Accuracy (Opacus vs TFP)',
              'Test Accuracy',
              'epsilon_vs_accuracy',
              include_baseline=baseline_acc)

# 2) ε vs Threshold AUC
line_plot_eps(thr_df, 'auc',
              'ε vs Threshold MIA AUC (Opacus vs TFP)',
              'AUC',
              'epsilon_vs_auc_threshold')

# 3) ε vs Shadow Attack Accuracy
line_plot_eps(sh_df, 'attack_acc',
              'ε vs Shadow-Model Attack Accuracy (Opacus vs TFP)',
              'Attack Accuracy',
              'epsilon_vs_attackacc_shadow')

# 4) Bars: Threshold AUC by ε & library
grouped_bars(thr_df, 'auc',
             'Threshold MIA AUC by ε and Library',
             'AUC',
             'bars_threshold_auc')

# 5) ε vs Threshold Advantage
if 'advantage' in thr_df.columns:
    line_plot_eps(thr_df, 'advantage',
                  'ε vs Threshold MIA Advantage (Opacus vs TFP)',
                  'Advantage (Acc - 0.5)',
                  'epsilon_vs_adv_threshold')

# 6) Utility drop vs baseline
if baseline_acc is not None and not util_df.empty:
    util_drop = util_df.copy()
    util_drop['acc_drop'] = baseline_acc - util_drop['accuracy']
    line_plot_eps(util_drop, 'acc_drop',
                  'ε vs Utility Drop from Baseline',
                  'Accuracy drop (absolute)',
                  'epsilon_vs_accuracy_drop')

# 7) Bars: Shadow Attack Accuracy by ε & library
grouped_bars(sh_df, 'attack_acc',
             'Shadow MIA Attack Accuracy by ε and Library',
             'Attack Accuracy',
             'bars_shadow_attackacc')

print(f"\nSaved figures to: {FIG_DIR}")
