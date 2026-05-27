"""
compare_experiments.py
自動掃描 outputs/ 下所有實驗，產出橫向比較圖表與摘要 CSV。

用法：python analysis/compare_experiments.py（從專案根目錄執行）

輸出：
    outputs/_comparison/comparison_learning_curves.png  ← 學習曲線同框
    outputs/_comparison/comparison_final_metrics.png    ← 最終 MSE / 參數量 bar chart
    outputs/_comparison/comparison_summary.csv          ← 報告數字總表
"""
import os, sys
# 不論從哪裡執行，都切到 project root 讓相對路徑（outputs/, data/）正常運作
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --- 中文字體（沿用 fourier_2d.py 設定） ---
font_path = r"C:\Windows\Fonts\msjh.ttc"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUTS_DIR = 'outputs'
COMPARE_DIR = os.path.join(OUTPUTS_DIR, '_comparison')
os.makedirs(COMPARE_DIR, exist_ok=True)

################################################################
# 1. 掃描所有實驗
################################################################
experiment_dirs = sorted(glob.glob(os.path.join(OUTPUTS_DIR, '*')))
experiments = []
for d in experiment_dirs:
    if not os.path.isdir(d):
        continue
    name = os.path.basename(d)
    if name.startswith('_'):           # 跳過 _comparison 等非實驗資料夾
        continue
    log_path = os.path.join(d, 'training_log.csv')
    cfg_path = os.path.join(d, 'config.json')
    if not (os.path.exists(log_path) and os.path.exists(cfg_path)):
        print(f"[略過] {name}：缺少 training_log.csv 或 config.json")
        continue
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    df = pd.read_csv(log_path)
    if len(df) == 0:
        print(f"[略過] {name}：training_log.csv 為空")
        continue
    experiments.append({
        'arch': name,
        'display_name': cfg.get('display_name', name),
        'param_count':  cfg.get('param_count', 0),
        'df':           df,
        'cfg':          cfg,
    })

if not experiments:
    raise RuntimeError(f"在 {OUTPUTS_DIR}/ 下沒有找到任何完成的實驗")

print(f"找到 {len(experiments)} 組實驗：")
for e in experiments:
    print(f"  - {e['arch']:20s}  ({e['param_count']/1e6:.2f}M params, {len(e['df'])} epochs)")

################################################################
# 類別判定（給 bar chart 用顏色分群）
################################################################
def _categorize(name):
    if name in ['unet_2d', 'unetpp_2d', 'transunet_2d', '2d_fno'] \
       or name.startswith('unet_2d_') or name.startswith('unetpp_2d_') \
       or name.startswith('transunet_2d_'):
        return 'planar'
    if name.startswith('sphere_'):
        return 'pure_sphere'
    return 'fno_hybrid'

CAT_COLOR = {'planar': '#2ecc71', 'fno_hybrid': '#f39c12', 'pure_sphere': '#e74c3c'}
CAT_LABEL = {
    'planar':      'Pure 2D (lon pad)',
    'fno_hybrid':  'FNO Hybrid (SHT + planar)',
    'pure_sphere': 'Pure Spherical (SHT-only)',
}

################################################################
# 2. 圖 1：學習曲線同框（每個實驗獨立顏色，用 tab20 取代 tab10）
################################################################
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
n_exp = len(experiments)
# 用 tab20（20 色）；超過 20 個實驗時循環，但有 line style 區分
colormap = plt.cm.tab20(np.linspace(0, 1, max(n_exp, 20)))

for i, exp in enumerate(experiments):
    df = exp['df']
    color = colormap[i % len(colormap)]
    linestyle = '-' if i < 20 else '--'   # 超過 20 個換虛線
    label = f"{exp['arch']} ({exp['param_count']/1e6:.2f}M)"
    axes[0].plot(df['epoch'], df['train_mse'], label=label,
                 linewidth=1.5, color=color, linestyle=linestyle)
    axes[1].plot(df['epoch'], df['test_mse'],  label=label,
                 linewidth=1.5, color=color, linestyle=linestyle)

for ax, title in zip(axes, ['Train MSE', 'Test MSE']):
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE',   fontsize=12)
    ax.set_title(title,    fontsize=14)
    ax.legend(fontsize=7, ncol=2, loc='upper right', framealpha=0.85)
    ax.grid(True, alpha=0.4)
    ax.set_yscale('log')

plt.suptitle('Architecture Comparison — Learning Curves', fontsize=15, fontweight='bold')
plt.tight_layout()
out_path = os.path.join(COMPARE_DIR, 'comparison_learning_curves.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"[輸出] {out_path}")

################################################################
# 3. 圖 2：最終 metrics bar chart（左 final test MSE / 右 參數量）
#         加大圖寬、按 final test MSE 排序、類別著色
################################################################
# 按 final test MSE 升序排
sorted_exps = sorted(experiments, key=lambda e: e['df']['test_mse'].iloc[-1])
arch_names     = [e['arch'] for e in sorted_exps]
final_test_mse = [e['df']['test_mse'].iloc[-1]  for e in sorted_exps]
param_counts_M = [e['param_count'] / 1e6        for e in sorted_exps]
colors         = [CAT_COLOR[_categorize(n)] for n in arch_names]

fig, axes = plt.subplots(1, 2, figsize=(22, 7))   # 從 14×5 加大到 22×7

bars1 = axes[0].bar(arch_names, final_test_mse, color=colors, edgecolor='black', linewidth=0.5, alpha=0.9)
axes[0].set_ylabel('Final Test MSE', fontsize=12, fontweight='bold')
axes[0].set_title('Final Test MSE (sorted ascending)',  fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.4, axis='y', linestyle='--')
axes[0].tick_params(axis='x', rotation=45, labelsize=8)
for tick in axes[0].get_xticklabels():
    tick.set_ha('right')
for bar, val in zip(bars1, final_test_mse):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:.4f}', ha='center', va='bottom', fontsize=8, rotation=90)

bars2 = axes[1].bar(arch_names, param_counts_M, color=colors, edgecolor='black', linewidth=0.5, alpha=0.9)
axes[1].set_ylabel('Parameters (M)', fontsize=12, fontweight='bold')
axes[1].set_title('Model Size',      fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.4, axis='y', linestyle='--')
axes[1].tick_params(axis='x', rotation=45, labelsize=8)
for tick in axes[1].get_xticklabels():
    tick.set_ha('right')
for bar, val in zip(bars2, param_counts_M):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:.2f}M', ha='center', va='bottom', fontsize=9)

# 加類別圖例（同一張圖只需一份）
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=CAT_COLOR[c], edgecolor='black', label=CAT_LABEL[c])
                  for c in ['planar', 'fno_hybrid', 'pure_sphere']]
axes[0].legend(handles=legend_handles, loc='upper left', fontsize=10, framealpha=0.95)

plt.suptitle('Architecture Comparison — Final Metrics (sorted by Test MSE)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
out_path = os.path.join(COMPARE_DIR, 'comparison_final_metrics.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"[輸出] {out_path}")

################################################################
# 4. CSV 摘要（按 final_test_mse 排序，方便一眼看出贏家）
################################################################
summary_rows = []
for exp in experiments:
    df = exp['df']
    summary_rows.append({
        'arch':                 exp['arch'],
        'display_name':         exp['display_name'],
        'param_count':          exp['param_count'],
        'epochs_completed':     len(df),
        'final_train_mse':      df['train_mse'].iloc[-1],
        'final_test_mse':       df['test_mse'].iloc[-1],
        'best_test_mse':        df['test_mse'].min(),
        'best_epoch':           int(df.loc[df['test_mse'].idxmin(), 'epoch']),
        'total_train_time_sec': df['epoch_time_sec'].sum(),
        'avg_epoch_time_sec':   df['epoch_time_sec'].mean(),
    })

summary_df = pd.DataFrame(summary_rows).sort_values('final_test_mse').reset_index(drop=True)
summary_csv_path = os.path.join(COMPARE_DIR, 'comparison_summary.csv')
summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
print(f"[輸出] {summary_csv_path}")

################################################################
# 5. Console 摘要
################################################################
print("\n" + "=" * 90)
print("比較摘要（按 final_test_mse 由低到高排序，最低 = 表現最好）")
print("=" * 90)
display_cols = ['arch', 'param_count', 'final_test_mse', 'best_test_mse',
                'best_epoch', 'avg_epoch_time_sec']
print(summary_df[display_cols].to_string(index=False,
    formatters={
        'param_count':         lambda v: f'{v/1e6:.2f}M',
        'final_test_mse':      lambda v: f'{v:.4f}',
        'best_test_mse':       lambda v: f'{v:.4f}',
        'avg_epoch_time_sec':  lambda v: f'{v:.1f}s',
    }))
print("=" * 90)
print(f"\n所有比較結果已輸出至：{COMPARE_DIR}/")
