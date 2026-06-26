"""
multi_seed_compare.py
按 base architecture × hyperparams 分組，計算多 seed 的 mean ± std，
產出統計比較圖與摘要 CSV。

用法：python analysis/multi_seed_compare.py（從專案根目錄執行）

分組邏輯：把 SEED 變動視為「同一實驗的不同 seed」，但 MODES 與 DROPOUT 變動
視為「不同實驗」。

輸出：
    outputs/_comparison/multi_seed_summary.csv
    outputs/_comparison/multi_seed_plot.png
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import re
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字體
font_path = r"C:\Windows\Fonts\msjh.ttc"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUTS_DIR = os.environ.get('OUTPUT_ROOT', 'outputs')
COMPARE_DIR = os.path.join(OUTPUTS_DIR, '_comparison')
os.makedirs(COMPARE_DIR, exist_ok=True)


def get_group_key(cfg, fallback_name):
    """
    分組鍵：base_arch [+ _mMODES] [+ _dropDD]，但不含 seed。
    對 PR 1+2 舊 config（沒有 base_experiment_name）做 fallback：
    從 experiment_name 結尾遞迴剝掉 _sN / _mN / _dropN 後綴。
    """
    base = cfg.get('base_experiment_name')
    if base is None:
        base = fallback_name
        while True:
            new = re.sub(r'_(s\d+|m\d+|drop\d+)$', '', base)
            if new == base:
                break
            base = new

    parts = [base]
    if cfg.get('modes', 16) != 16:
        parts.append(f"m{cfg['modes']}")
    dropout = cfg.get('dropout', 0)
    if dropout and dropout > 0:
        parts.append(f"drop{int(round(dropout * 100))}")
    return '_'.join(parts)


################################################################
# 1. 掃描所有實驗
################################################################
experiment_dirs = sorted(glob.glob(os.path.join(OUTPUTS_DIR, '*')))
records = []
for d in experiment_dirs:
    if not os.path.isdir(d):
        continue
    name = os.path.basename(d)
    if name.startswith('_'):
        continue
    cfg_path = os.path.join(d, 'config.json')
    log_path = os.path.join(d, 'training_log.csv')
    if not (os.path.exists(cfg_path) and os.path.exists(log_path)):
        continue
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    df = pd.read_csv(log_path)
    if len(df) == 0:
        continue

    group_key      = get_group_key(cfg, name)
    best_test_mse  = cfg.get('best_test_mse',  df['test_mse'].min())
    final_test_mse = cfg.get('final_test_mse', df['test_mse'].iloc[-1])
    best_epoch     = cfg.get('best_epoch',     int(df.loc[df['test_mse'].idxmin(), 'epoch']))
    seed           = cfg.get('seed', 0)

    records.append({
        'arch':           name,
        'group_key':      group_key,
        'seed':           seed,
        'param_count':    cfg.get('param_count', 0),
        'best_test_mse':  best_test_mse,
        'final_test_mse': final_test_mse,
        'best_epoch':     best_epoch,
    })

if not records:
    raise RuntimeError(f"在 {OUTPUTS_DIR}/ 下沒有找到任何完成的實驗")

df = pd.DataFrame(records)
print(f"找到 {len(df)} 個實驗，分屬 {df['group_key'].nunique()} 個比較組")
print()
print(df[['arch', 'group_key', 'seed', 'best_test_mse', 'final_test_mse']]
        .to_string(index=False))

################################################################
# 2. 按 group_key 分組統計
################################################################
agg = df.groupby('group_key').agg(
    n_seeds              = ('seed', 'count'),
    seed_list            = ('seed', lambda s: ','.join(map(str, sorted(s)))),
    param_count          = ('param_count', 'mean'),
    best_test_mse_mean   = ('best_test_mse', 'mean'),
    best_test_mse_std    = ('best_test_mse', 'std'),
    best_test_mse_min    = ('best_test_mse', 'min'),
    final_test_mse_mean  = ('final_test_mse', 'mean'),
    final_test_mse_std   = ('final_test_mse', 'std'),
).reset_index()

# 單 seed 的 std 為 NaN → 顯示 0 並由 n_seeds 標示
agg['best_test_mse_std']  = agg['best_test_mse_std'].fillna(0)
agg['final_test_mse_std'] = agg['final_test_mse_std'].fillna(0)
agg = agg.sort_values('best_test_mse_mean').reset_index(drop=True)

csv_path = os.path.join(COMPARE_DIR, 'multi_seed_summary.csv')
agg.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n[輸出] {csv_path}")

################################################################
# 3. Console 摘要
################################################################
print("\n" + "=" * 100)
print("多 seed 統計（按 best_test_mse_mean 由低到高排序）")
print("=" * 100)
display_df = agg[['group_key', 'n_seeds', 'seed_list', 'param_count',
                  'best_test_mse_mean', 'best_test_mse_std',
                  'final_test_mse_mean', 'final_test_mse_std']].copy()
print(display_df.to_string(index=False, formatters={
    'param_count':         lambda v: f'{v/1e6:.2f}M',
    'best_test_mse_mean':  lambda v: f'{v:.4f}',
    'best_test_mse_std':   lambda v: f'±{v:.4f}',
    'final_test_mse_mean': lambda v: f'{v:.4f}',
    'final_test_mse_std':  lambda v: f'±{v:.4f}',
}))
print("=" * 100)

################################################################
# 4. Bar chart with error bars
################################################################
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
x_pos  = np.arange(len(agg))
colors = plt.cm.tab10(x_pos % 10)

for ax, mean_col, std_col, title in [
    (axes[0], 'best_test_mse_mean',  'best_test_mse_std',  'Best Test MSE (mean ± std)'),
    (axes[1], 'final_test_mse_mean', 'final_test_mse_std', 'Final Test MSE (mean ± std)'),
]:
    ax.bar(x_pos, agg[mean_col], yerr=agg[std_col],
           capsize=8, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agg['group_key'], rotation=25, ha='right', fontsize=9)
    ax.set_ylabel(title.split(' (')[0], fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.4, axis='y')

    # 在每個 bar 上方標 mean ± n_seeds
    for i, (mean, std, n) in enumerate(zip(agg[mean_col], agg[std_col], agg['n_seeds'])):
        ax.text(i, mean + std + 0.003,
                f'{mean:.3f}\n(n={n})',
                ha='center', va='bottom', fontsize=8)

plt.suptitle('Multi-Seed Architecture Comparison', fontsize=15, fontweight='bold')
plt.tight_layout()
plot_path = os.path.join(COMPARE_DIR, 'multi_seed_plot.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"[輸出] {plot_path}")

################################################################
# 5. 顯著性提示（哪些組已有 ≥2 seeds 可信地下結論）
################################################################
multi_seed_groups   = agg[agg['n_seeds'] >= 2]
single_seed_groups  = agg[agg['n_seeds'] == 1]

print(f"\n→ 已有多 seed 統計的組數：{len(multi_seed_groups)}")
if len(multi_seed_groups) > 0:
    print("  以下組可做顯著性比較：")
    for _, row in multi_seed_groups.iterrows():
        print(f"    - {row['group_key']:30s}  n={row['n_seeds']}  "
              f"best={row['best_test_mse_mean']:.4f}±{row['best_test_mse_std']:.4f}")

if len(single_seed_groups) > 0:
    print(f"\n→ 仍只有單 seed 的組數：{len(single_seed_groups)}（建議補跑 2-3 seeds）")
    for _, row in single_seed_groups.iterrows():
        print(f"    - {row['group_key']}")

print(f"\n所有結果已輸出至：{COMPARE_DIR}/")
