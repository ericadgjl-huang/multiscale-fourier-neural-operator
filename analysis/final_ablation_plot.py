"""
final_ablation_plot.py
產出最終報告用的 3×3 ablation 視覺化：
1. final_ablation_heatmap.png  — 架構×幾何 的 heatmap（報告封面圖）
2. final_ablation_bars.png     — 所有實驗排序 bar chart（含 std）
3. final_ablation_table.csv    — 3×3 表格匯出

用法：python analysis/final_ablation_plot.py（從專案根目錄執行）
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
from matplotlib.patches import Rectangle, Patch

# 中文字體
font_path = r"C:\Windows\Fonts\msjh.ttc"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUTS_DIR = 'outputs'
COMPARE_DIR = os.path.join(OUTPUTS_DIR, '_comparison')
os.makedirs(COMPARE_DIR, exist_ok=True)

################################################################
# 3×3 對照表的架構命名映射
################################################################
ABLATION_GRID = {
    'UNet':      {'planar': 'unet_2d',      'fno_hybrid': 'sufno',       'pure_sphere': 'sphere_unet'},
    'UNet++':    {'planar': 'unetpp_2d',    'fno_hybrid': 'sunetpp_fno', 'pure_sphere': 'sphere_unetpp'},
    'TransUNet': {'planar': 'transunet_2d', 'fno_hybrid': 'sutrans_fno', 'pure_sphere': 'sphere_transunet'},
}
GEOMETRY_LABELS = {
    'planar':      'Pure 2D\n(lon circular pad)',
    'fno_hybrid':  'FNO Hybrid\n(SHT ⊕ planar local)',
    'pure_sphere': 'Pure Spherical\n(SHT-only UNet)',
}

################################################################
# Helper: 把 _sN 後綴拿掉以分組
################################################################
def get_group(name):
    return re.sub(r'_s\d+$', '', name)


def categorize(name):
    """根據實驗名稱判斷屬於哪個 geometry 類別"""
    if name in ['unet_2d', 'unetpp_2d', 'transunet_2d', '2d_fno']:
        return 'planar'
    if name.startswith('sphere_'):
        return 'pure_sphere'
    return 'fno_hybrid'


CAT_COLOR = {
    'planar':      '#2ecc71',   # 綠
    'fno_hybrid':  '#f39c12',   # 橘
    'pure_sphere': '#e74c3c',   # 紅
}
CAT_LABEL = {
    'planar':      'Pure 2D (lon pad)',
    'fno_hybrid':  'FNO Hybrid (SHT ⊕ planar)',
    'pure_sphere': 'Pure Spherical (SHT-only)',
}

################################################################
# 載入並聚合所有實驗結果
################################################################
def is_canonical(cfg):
    """只保留 canonical 設定的實驗（modes 預設、無 dropout）— 排除 hyperparam search 變體"""
    if cfg.get('modes', 16) != 16:
        return False
    if cfg.get('dropout', 0) and cfg.get('dropout', 0) > 0:
        return False
    return True


def load_results():
    """掃描 outputs/，按 base_experiment_name 分組（只計入 canonical 變體），回傳 {group: ...}"""
    results = {}
    for d in sorted(glob.glob(os.path.join(OUTPUTS_DIR, '*'))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        if name.startswith('_'):
            continue
        cfg_path = os.path.join(d, 'config.json')
        log_path = os.path.join(d, 'training_log.csv')
        if not (os.path.exists(cfg_path) and os.path.exists(log_path)):
            continue
        with open(cfg_path, encoding='utf-8') as f:
            cfg = json.load(f)
        df = pd.read_csv(log_path)
        if len(df) == 0:
            continue

        # 跳過 hyperparam search 變體（dropout / modes 非預設值）— 這些不該併入 base seed 統計
        if not is_canonical(cfg):
            continue

        best = cfg.get('best_test_mse', df['test_mse'].min())

        # 用 base_experiment_name 優先（PR 3 後新增的欄位），fallback parse name
        group = cfg.get('base_experiment_name') or get_group(name)
        if group not in results:
            results[group] = {'vals': [], 'params': cfg.get('param_count', 0)}
        results[group]['vals'].append(best)
        results[group]['params'] = cfg.get('param_count', results[group]['params'])

    for k, v in results.items():
        vals = np.array(v['vals'])
        v['mean'] = vals.mean()
        v['std']  = vals.std(ddof=1) if len(vals) > 1 else 0.0
        v['n']    = len(vals)
    return results


results = load_results()

################################################################
# 圖 1：3×3 heatmap（報告封面圖）
################################################################
fig, ax = plt.subplots(figsize=(11, 7))

archs = list(ABLATION_GRID.keys())
geoms = list(GEOMETRY_LABELS.keys())

matrix = np.full((len(archs), len(geoms)), np.nan)
for i, arch in enumerate(archs):
    for j, geom in enumerate(geoms):
        exp_name = ABLATION_GRID[arch][geom]
        if exp_name in results:
            matrix[i, j] = results[exp_name]['mean']

vmin = np.nanmin(matrix) - 0.005
vmax = np.nanmax(matrix) + 0.005
im = ax.imshow(matrix, cmap='RdYlGn_r', vmin=vmin, vmax=vmax, aspect='auto')

# 標註每一格的數值
for i, arch in enumerate(archs):
    for j, geom in enumerate(geoms):
        exp_name = ABLATION_GRID[arch][geom]
        if exp_name not in results:
            continue
        r = results[exp_name]
        lines = [f"{r['mean']:.4f}"]
        if r['n'] > 1:
            lines.append(f"±{r['std']:.4f}")
        lines.append(f"n={r['n']} | {r['params']/1e6:.2f}M")
        text_color = 'white' if r['mean'] > (vmin + vmax) / 2 else 'black'
        ax.text(j, i, '\n'.join(lines), ha='center', va='center',
                color=text_color, fontsize=11, fontweight='bold')

# 高亮最佳值（金色框）
best_idx = np.unravel_index(np.nanargmin(matrix), matrix.shape)
ax.add_patch(Rectangle((best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
                       fill=False, edgecolor='gold', lw=5))

ax.set_xticks(range(len(geoms)))
ax.set_xticklabels([GEOMETRY_LABELS[g] for g in geoms], fontsize=12)
ax.set_yticks(range(len(archs)))
ax.set_yticklabels(archs, fontsize=13, fontweight='bold')
ax.set_xlabel('Geometry Handling', fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel('Architecture', fontsize=13, fontweight='bold', labelpad=10)
ax.set_title('Architecture × Geometry Ablation — Best Test MSE\n(10-Day Forecast on Mini-ERA5)',
             fontsize=14, fontweight='bold', pad=15)

cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label('Best Test MSE (lower = better)', fontsize=11)

plt.tight_layout()
out_heatmap = os.path.join(COMPARE_DIR, 'final_ablation_heatmap.png')
plt.savefig(out_heatmap, dpi=300, bbox_inches='tight')
plt.close()
print(f"[輸出] {out_heatmap}")

################################################################
# 圖 2：所有實驗排序 bar chart（含 std）
################################################################
fig, ax = plt.subplots(figsize=(15, 8))

sorted_items = sorted(results.items(), key=lambda kv: kv[1]['mean'])
groups = [k for k, _ in sorted_items]
means  = [v['mean'] for _, v in sorted_items]
stds   = [v['std']  for _, v in sorted_items]
ns     = [v['n']    for _, v in sorted_items]
prms   = [v['params'] / 1e6 for _, v in sorted_items]
colors = [CAT_COLOR[categorize(g)] for g in groups]

x_pos = np.arange(len(groups))
ax.bar(x_pos, means, yerr=stds, color=colors,
       edgecolor='black', linewidth=0.7, alpha=0.9, capsize=6)
for i, (m, s, n, p) in enumerate(zip(means, stds, ns, prms)):
    ax.text(i, m + s + 0.003, f"{m:.4f}\n(n={n}, {p:.1f}M)",
            ha='center', va='bottom', fontsize=9)

ax.set_xticks(x_pos)
ax.set_xticklabels(groups, rotation=30, ha='right', fontsize=10)
ax.set_ylabel('Best Test MSE (mean ± std)', fontsize=12, fontweight='bold')
ax.set_xlabel('Architecture (sorted by performance)', fontsize=12)
ax.set_title('All Architectures Ranked — Best Test MSE\n(Pure 2D < FNO Hybrid < Pure Spherical)',
             fontsize=14, fontweight='bold')

legend_elements = [Patch(facecolor=CAT_COLOR[c], edgecolor='black',
                          label=CAT_LABEL[c]) for c in ['planar', 'fno_hybrid', 'pure_sphere']]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_ylim(min(means) - 0.005, max(means) + max(stds) + 0.025)

plt.tight_layout()
out_bars = os.path.join(COMPARE_DIR, 'final_ablation_bars.png')
plt.savefig(out_bars, dpi=300, bbox_inches='tight')
plt.close()
print(f"[輸出] {out_bars}")

################################################################
# 表格匯出（給報告用）
################################################################
table_rows = []
for arch in archs:
    row = {'Architecture': arch}
    for geom in geoms:
        exp_name = ABLATION_GRID[arch][geom]
        col_name = GEOMETRY_LABELS[geom].replace('\n', ' ')
        if exp_name in results:
            r = results[exp_name]
            cell = f"{r['mean']:.4f}"
            if r['n'] > 1:
                cell += f" ± {r['std']:.4f}"
            cell += f" (n={r['n']})"
            row[col_name] = cell
        else:
            row[col_name] = 'N/A'
    table_rows.append(row)

table_df = pd.DataFrame(table_rows)
table_csv = os.path.join(COMPARE_DIR, 'final_ablation_table.csv')
table_df.to_csv(table_csv, index=False, encoding='utf-8-sig')
print(f"[輸出] {table_csv}")

################################################################
# Console 摘要
################################################################
print()
print('=' * 90)
print(' 3×3 Ablation 表（按 Best Test MSE）')
print('=' * 90)
print(table_df.to_string(index=False))
print()
print('=' * 90)
print(' 整體排名（前 5 名 / 後 3 名）')
print('=' * 90)
for i, (g, v) in enumerate(sorted_items):
    arrow = ' ←' if i == 0 else (' ' * 8)
    print(f"  {i+1:2d}. {g:<22s}  {v['mean']:.4f}  std=±{v['std']:.4f}  n={v['n']}  "
          f"params={v['params']/1e6:.2f}M  ({CAT_LABEL[categorize(g)]}){arrow}")
print('=' * 90)
print(f"\n所有結果已輸出至：{COMPARE_DIR}/")
