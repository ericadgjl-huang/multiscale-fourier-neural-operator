"""
crossover_plot.py
產出「Crossover 翻轉圖」— 全文最關鍵、簡報的高潮圖。

故事軸：固定低解析度 (33×64)，沿「變數豐富度 (state richness)」軸 4 → 6 → 96，
比較三種幾何處理 (Planar / FNO-Hybrid / Pure-Spherical) 的 best test MSE。

主結論：
  1. 純球面 (SHT-only UNet) 跨所有豐富度「永遠墊底」(~11–18% 比平面差) — 鐵結論。
  2. FNO-Hybrid 的球面頻譜路徑隨變數變多「從負擔翻成資產」，
     在 4/6 變數時輸給平面、到 96 變數時反超平面 → crossover (~40 變數)。
     ⚠ 6/96 變數目前 n=1，hybrid 對 planar 的領先尚未統計顯著 (observed trend)。

資料來源（控制變因、同一訓練流程）：
  * 4 變數：舊論文 PDF 的 n=3 結果（本機無 outputs，硬編於下方並標明來源）。
  * 6 變數：讀 outputs/      （舊命名 2d_unet…），每模型 config.json 的 best_test_mse。
  * 96 變數：讀 outputs96/   （新命名 unet_2d…），讀 _comparison/comparison_summary.csv。

用法：python analysis/crossover_plot.py（從專案根目錄執行）
輸出：outputs96/_comparison/ 下
  - crossover.png            （雙面板：絕對 MSE + 相對平面的超額誤差）
  - crossover_absolute.png   （單面板絕對 MSE，簡報用）
  - crossover_data.csv       （數字總表）
"""
import os, sys, json, glob
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

try:                                  # Windows console 預設 cp950，強制 utf-8 避免列印崩潰
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ── 中文字體（與專案其他圖一致）────────────────────────────────
font_path = r"C:\Windows\Fonts\msjh.ttc"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = os.path.join('outputs96', '_comparison')
os.makedirs(OUT_DIR, exist_ok=True)

# 幾何配色（沿用 final_ablation_plot.py）
COL = {'planar': '#2ecc71', 'fno_hybrid': '#f39c12', 'pure_sphere': '#e74c3c',
       'sfno': '#9b59b6'}

# ── 幾何分類 ───────────────────────────────────────────────────
def geom_of(name):
    """把實驗名稱歸到幾何桶。pure_sphere 與 sfno baseline 刻意分開。"""
    if name.startswith('sphere_'):
        return 'pure_sphere'          # 純 SHT 編解碼 (SHT-only UNet)
    if name == 'sfno':
        return 'sfno'                 # SFNO 基線 (SHT + 1x1 local)，非純球面
    if name in ('sufno', 'sunetpp_fno', 'sutrans_fno'):
        return 'fno_hybrid'           # SHT ⊕ conv 並行
    # 其餘皆視為平面 (lon circular pad)：unet_2d/2d_unet, unetpp_2d,
    # transunet_2d, 2d_fno（FFT 基線）, 2d_ufno（FFT+UNet，平面頻譜）
    return 'planar'


def best_per_geom(best_by_name):
    """從 {模型名: best_mse} 取每個幾何桶的最佳（最小）值。"""
    buckets = {}
    for name, mse in best_by_name.items():
        if mse is None or (isinstance(mse, float) and np.isnan(mse)):
            continue
        g = geom_of(name)
        if g not in buckets or mse < buckets[g][1]:
            buckets[g] = (name, mse)
    return buckets   # {geom: (winner_name, mse)}


# ── 4 變數：舊論文 PDF (n=3 mean)。本機無 outputs，硬編並標來源。──
# 來源：期末報告/論文 表2 + 基線。planar/hybrid n=3，pure_sphere n=1。
BEST_4VAR = {
    'unet_2d': 0.3751, 'unetpp_2d': 0.3803, 'transunet_2d': 0.3765,   # planar
    'sufno': 0.3969, 'sunetpp_fno': 0.3882, 'sutrans_fno': 0.4228,    # fno_hybrid
    'sphere_unet': 0.4224, 'sphere_unetpp': 0.4165,                   # pure_sphere
    'sphere_transunet': 0.4151,
    '2d_fno': 0.4017, 'sfno': 0.4034,                                 # baselines
}

# ── 跨 seed 聚合：掃描資料夾，依 base_experiment_name 分組取 best_test_mse 平均 ──
# 補了 seed 1/2 後（unet_2d_s1 …）本函式會自動把同架構多 seed 併成 mean±std，
# 不必改任何東西，重跑本腳本即得多 seed 版 crossover。
import re as _re
from collections import defaultdict as _dd

def load_agg(root):
    """回傳 (mean_dict, meta_dict)：
       mean_dict = {base_name: mean(best_test_mse)}
       meta_dict = {base_name: {'std':.., 'n':.., 'vals':[..]}}
       只收 canonical（modes=16、dropout=0）且已完成（有 best_test_mse）的 run。"""
    groups = _dd(list)
    for d in sorted(glob.glob(os.path.join(root, '*'))):
        if not os.path.isdir(d) or os.path.basename(d).startswith('_'):
            continue
        cfg_p = os.path.join(d, 'config.json')
        if not os.path.exists(cfg_p):
            continue
        with open(cfg_p, encoding='utf-8') as f:
            cfg = json.load(f)
        if cfg.get('modes', 16) != 16:                       # 排除 hyperparam search 變體
            continue
        if cfg.get('dropout', 0) and cfg.get('dropout', 0) > 0:
            continue
        # 只收「config 內有 best_test_mse」的已完成 run。
        # 不退回讀 training_log：否則會誤收像 outputs/unet_2d 這種放錯位置、
        # 未寫 best_test_mse 的殘留 96 變數資料夾，污染 6 變數統計。
        mse = cfg.get('best_test_mse')
        if mse is None:
            continue
        base = cfg.get('base_experiment_name') or _re.sub(r'_s\d+$', '', os.path.basename(d))
        groups[base].append(float(mse))
    mean_d, meta_d = {}, {}
    for k, v in groups.items():
        a = np.array(v)
        mean_d[k] = float(a.mean())
        meta_d[k] = {'std': float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                     'n': len(a), 'vals': sorted(v)}
    return mean_d, meta_d

BEST_6VAR,  META_6VAR  = load_agg('outputs')       # 6 變數
BEST_96VAR, META_96VAR = load_agg('outputs96')     # 96 變數（自動吃 seed 1/2）

# 4 變數 meta（來自論文 PDF）：planar/hybrid 為 n=3、pure_sphere 為 n=1
META_4VAR = {k: {'std': 0.0, 'n': (1 if k.startswith('sphere_') else 3), 'vals': [v]}
             for k, v in BEST_4VAR.items()}

# ── 組裝三個豐富度層級 ──────────────────────────────────────────
RICHNESS = [4, 6, 96]
RAW  = {4: BEST_4VAR,  6: BEST_6VAR,  96: BEST_96VAR}
META = {4: META_4VAR,  6: META_6VAR,  96: META_96VAR}
BUCKETS = {r: best_per_geom(RAW[r]) for r in RICHNESS}

def n_label(r):
    """該豐富度層級三條主線的 seed 數摘要（如 n=3 或 n=1-3）。"""
    ns = [META[r].get(BUCKETS[r][g][0], {'n': 1})['n'] for g in GEOMS if g in BUCKETS[r]]
    if not ns:
        return 'n=?'
    lo, hi = min(ns), max(ns)
    return f"n={lo}" if lo == hi else f"n={lo}-{hi}"

# 三條主線（每層取該幾何最佳模型）
GEOMS = ['planar', 'fno_hybrid', 'pure_sphere']
GEOM_LABEL = {
    'planar':      'Planar (2D + lon circular pad)',
    'fno_hybrid':  'FNO-Hybrid (SHT ⊕ conv)',
    'pure_sphere': 'Pure Spherical (SHT-only UNet)',
}

def series(geom):
    """回傳 (xs, ys, names) — 該幾何在各豐富度的最佳值（缺則跳過）。"""
    xs, ys, names = [], [], []
    for r in RICHNESS:
        if geom in BUCKETS[r]:
            nm, mse = BUCKETS[r][geom]
            xs.append(r); ys.append(mse); names.append(nm)
    return xs, ys, names

# ── 主表（給論文/console）───────────────────────────────────────
rows = []
for r in RICHNESS:
    row = {'richness(vars)': r, 'seeds': n_label(r)}
    for g in GEOMS:
        if g in BUCKETS[r]:
            nm, mse = BUCKETS[r][g]
            row[GEOM_LABEL[g]] = f"{mse:.4f} ({nm})"
        else:
            row[GEOM_LABEL[g]] = 'N/A'
    # 相對平面的超額誤差 %
    if 'planar' in BUCKETS[r]:
        base = BUCKETS[r]['planar'][1]
        for g in ('fno_hybrid', 'pure_sphere'):
            if g in BUCKETS[r]:
                gap = (BUCKETS[r][g][1] - base) / base * 100
                row[f'{g} vs planar %'] = f"{gap:+.1f}%"
    rows.append(row)
table = pd.DataFrame(rows)
table.to_csv(os.path.join(OUT_DIR, 'crossover_data.csv'),
             index=False, encoding='utf-8-sig')

# ── 幾何 2×2 對照（回應教授：spherical 是否為必須？）────────────
# 固定 local path，只換 spectral path：FFT(平面) vs SHT(球面)。
#   2d_ufno = FFT ⊕ UNet （平面）    sufno = SHT ⊕ UNet （球面）
#   2d_fno  = FFT ⊕ 1x1              sfno  = SHT ⊕ 1x1
# 若 2d_ufno ≈ sufno（同為 UNet-local，只差平面/球面頻譜），代表增益主要
# 來自 FNO+UNet 而非球面 → 佐證「spherical 非必須」。
CTRL = {'FFT (planar spectral)':    {'1x1': '2d_fno', 'UNet': '2d_ufno'},
        'SHT (spherical spectral)': {'1x1': 'sfno',   'UNet': 'sufno'}}

def fmt_cell(r, name):
    if name in RAW[r]:
        m = RAW[r][name]; md = META[r].get(name, {'std': 0.0, 'n': 1})
        s = f"{m:.4f}" + (f"±{md['std']:.4f}" if md['n'] > 1 else '')
        return s + f" (n={md['n']})"
    return 'N/A'

ctrl_rows = []
for r in RICHNESS:
    if r == 4:                       # 4 變數未跑 2d_ufno，略過此對照
        continue
    for spec, locs in CTRL.items():
        ctrl_rows.append({'richness': r, 'spectral': spec,
                          'local=1x1': fmt_cell(r, locs['1x1']),
                          'local=UNet': fmt_cell(r, locs['UNet'])})
ctrl_df = pd.DataFrame(ctrl_rows)
ctrl_df.to_csv(os.path.join(OUT_DIR, 'crossover_control_2x2.csv'),
               index=False, encoding='utf-8-sig')

# ── 估算 crossover 變數數（planar 與 hybrid 在 log-x 線性內插交點）─
def log_interp_cross():
    xp, yp, _ = series('planar')
    xh, yh, _ = series('fno_hybrid')
    common = sorted(set(xp) & set(xh))
    if len(common) < 2:
        return None
    px = {x: y for x, y in zip(xp, yp)}
    hx = {x: y for x, y in zip(xh, yh)}
    diff = [(np.log10(x), hx[x] - px[x]) for x in common]  # hybrid - planar
    for (x0, d0), (x1, d1) in zip(diff, diff[1:]):
        if d0 == 0:
            return 10 ** x0
        if d0 * d1 < 0:                       # 變號 → 內插交點
            t = d0 / (d0 - d1)
            return 10 ** (x0 + t * (x1 - x0))
    return None

x_cross = log_interp_cross()

################################################################
# 圖 A：雙面板 crossover.png
################################################################
fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.2))

# ---- 左：絕對 best test MSE，3 條線 ----
for g in GEOMS:
    xs, ys, names = series(g)
    style = dict(marker='o', ms=11, lw=3, color=COL[g], label=GEOM_LABEL[g],
                 zorder=3)
    if g == 'pure_sphere':
        # 6 變數沒有純球面 → 兩點以虛線連
        style.update(linestyle='--', alpha=0.9)
    axL.plot(xs, ys, **style)
    errs = [META[x].get(nm, {'std': 0})['std'] for x, nm in zip(xs, names)]
    if any(e > 0 for e in errs):                    # 有多 seed 才畫誤差棒
        axL.errorbar(xs, ys, yerr=errs, fmt='none', ecolor=COL[g],
                     elinewidth=1.5, capsize=5, zorder=4)
    for x, y, nm in zip(xs, ys, names):
        dy = 0.006 if g != 'planar' else -0.011
        axL.annotate(f"{y:.4f}", (x, y), textcoords='offset points',
                     xytext=(0, 10 if dy > 0 else -16), ha='center',
                     fontsize=9, color=COL[g], fontweight='bold')

# crossover 標記
if x_cross:
    axL.axvline(x_cross, color='gray', ls=':', lw=1.6, zorder=1)
    yloc = axL.get_ylim()
    axL.scatter([x_cross], [np.interp(np.log10(x_cross),
                [np.log10(v) for v in series('planar')[0]],
                series('planar')[1])], marker='*', s=420,
                color='#111', zorder=5)
    axL.annotate(f"crossover\n~ {x_cross:.0f} variables", (x_cross, 0.398),
                 ha='center', fontsize=11, fontweight='bold', color='#111',
                 bbox=dict(boxstyle='round,pad=0.3', fc='gold', ec='#111',
                           alpha=0.9))

axL.set_xscale('log')
axL.set_xticks(RICHNESS)
axL.set_xticklabels([f"{r}\n({n_label(r)})" for r in RICHNESS], fontsize=11)
axL.set_xlabel('Input richness — number of prognostic variables (log scale)',
               fontsize=12, fontweight='bold')
axL.set_ylabel('Best test MSE  (lower = better)', fontsize=12, fontweight='bold')
axL.set_title('(a) Absolute skill — FNO-Hybrid dives under Planar at high richness',
              fontsize=12, fontweight='bold')
axL.grid(True, alpha=0.3, ls='--')
axL.legend(fontsize=10, loc='upper left', framealpha=0.95)
axL.text(0.99, 0.02,
         'Absolute MSE not directly comparable across richness\n'
         '(different channel sets averaged) — read ordering, not slope.',
         transform=axL.transAxes, ha='right', va='bottom', fontsize=8,
         color='gray', style='italic')

# ---- 右：相對平面的超額誤差 %（真正的 crossover）----
axR.axhline(0, color=COL['planar'], lw=2.5, zorder=2)
axR.text(4, 0.4, 'Planar baseline (0%)', color=COL['planar'], fontsize=10,
         fontweight='bold', va='bottom')
# helps / hurts 區域
axR.axhspan(-100, 0, color=COL['planar'], alpha=0.06)
axR.axhspan(0, 100, color=COL['pure_sphere'], alpha=0.05)

for g in ('fno_hybrid', 'pure_sphere'):
    xs, ys, names = series(g)
    rel_x, rel_y = [], []
    for x, y in zip(xs, ys):
        if 'planar' in BUCKETS[x]:
            base = BUCKETS[x]['planar'][1]
            rel_x.append(x); rel_y.append((y - base) / base * 100)
    ls = '--' if g == 'pure_sphere' else '-'
    axR.plot(rel_x, rel_y, marker='o', ms=11, lw=3, color=COL[g], ls=ls,
             label=GEOM_LABEL[g], zorder=3)
    for x, yv in zip(rel_x, rel_y):
        axR.annotate(f"{yv:+.1f}%", (x, yv), textcoords='offset points',
                     xytext=(0, 11 if yv >= 0 else -16), ha='center',
                     fontsize=9.5, color=COL[g], fontweight='bold')

axR.set_xscale('log')
axR.set_xticks(RICHNESS)
axR.set_xticklabels([f"{r}\n({n_label(r)})" for r in RICHNESS], fontsize=11)
axR.set_ylim(-8, 24)
axR.set_xlabel('Input richness — number of prognostic variables (log scale)',
               fontsize=12, fontweight='bold')
axR.set_ylabel('Excess error vs. planar baseline  (%)', fontsize=12,
               fontweight='bold')
axR.set_title('(b) When Sphere hurts → helps — Hybrid crosses below planar',
              fontsize=12, fontweight='bold')
axR.grid(True, alpha=0.3, ls='--')
axR.legend(fontsize=10, loc='upper right', framealpha=0.95)
axR.annotate('Sphere HURTS  (worse than planar)', (4.2, 21),
             color=COL['pure_sphere'], fontsize=11, fontweight='bold', alpha=0.85)
axR.annotate('Sphere HELPS  (better than planar)', (4.2, -6),
             color=COL['planar'], fontsize=11, fontweight='bold', alpha=0.9)

fig.suptitle('When Sphere Hurts — and When It Starts to Help\n'
             'Input richness modulates the geometry trade-off '
             '(10-day rollout, mini-ERA5 33×64)',
             fontsize=15, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.93])
p_main = os.path.join(OUT_DIR, 'crossover.png')
fig.savefig(p_main, dpi=300, bbox_inches='tight')
plt.close(fig)

################################################################
# 圖 B：單面板絕對 MSE（簡報用 crossover_absolute.png）
################################################################
fig2, ax = plt.subplots(figsize=(9.5, 6.4))
for g in GEOMS:
    xs, ys, names = series(g)
    style = dict(marker='o', ms=13, lw=3.5, color=COL[g], label=GEOM_LABEL[g],
                 zorder=3)
    if g == 'pure_sphere':
        style.update(linestyle='--')
    ax.plot(xs, ys, **style)
    errs = [META[x].get(nm, {'std': 0})['std'] for x, nm in zip(xs, names)]
    if any(e > 0 for e in errs):
        ax.errorbar(xs, ys, yerr=errs, fmt='none', ecolor=COL[g],
                    elinewidth=1.5, capsize=5, zorder=4)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.4f}", (x, y), textcoords='offset points',
                    xytext=(0, 12 if g != 'planar' else -18), ha='center',
                    fontsize=10, color=COL[g], fontweight='bold')
if x_cross:
    ax.axvline(x_cross, color='gray', ls=':', lw=1.6)
    ax.annotate(f"★ crossover ~ {x_cross:.0f} vars",
                (x_cross, 0.392), ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.35', fc='gold', ec='#111'))
ax.set_xscale('log')
ax.set_xticks(RICHNESS)
ax.set_xticklabels([f"{r} vars\n({n_label(r)})" for r in RICHNESS], fontsize=12)
ax.set_xlabel('Input richness (number of prognostic variables, log scale)',
              fontsize=12.5, fontweight='bold')
ax.set_ylabel('Best test MSE (lower = better)', fontsize=12.5, fontweight='bold')
ax.set_title('When Sphere Hurts — and When It Starts to Help\n'
             'Planar wins at 4/6 vars; FNO-Hybrid overtakes at 96 vars; '
             'Pure-Spherical always worst',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, ls='--')
ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
ax.text(0.99, 0.02,
        'n=3 at 4 vars; n=1 at 6/96 vars (Hybrid→Planar lead = observed '
        'trend, not yet significant).\nPure-spherical not run at 6 vars '
        '(dashed). Absolute MSE not comparable across richness.',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=7.8,
        color='gray', style='italic')
fig2.tight_layout()
p_abs = os.path.join(OUT_DIR, 'crossover_absolute.png')
fig2.savefig(p_abs, dpi=300, bbox_inches='tight')
plt.close(fig2)

################################################################
# Console 摘要
################################################################
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)
print('=' * 100)
print(' Crossover 翻轉圖 — best test MSE by geometry × input richness')
print('=' * 100)
print(table.to_string(index=False))
print('-' * 100)
print(' 幾何 2×2 對照（固定 local，換 spectral：FFT 平面 vs SHT 球面）')
print(ctrl_df.to_string(index=False))
if '2d_ufno' in RAW[96] and 'sufno' in RAW[96]:
    a, b = RAW[96]['2d_ufno'], RAW[96]['sufno']         # 平面 vs 球面（皆 UNet-local）
    verdict = '球面較佳' if b < a else '平面較佳或相當'
    print(f" → 96 變數 UNet-local：平面 2d_ufno={a:.4f} vs 球面 sufno={b:.4f}；"
          f"球面相對平面 {(a - b) / a * 100:+.1f}% ({verdict})")
else:
    print(' → (2d_ufno 於 96 變數尚未跑完；補齊後本行會顯示平面 vs 球面對照)')
print('-' * 100)
if x_cross:
    print(f" 估算 Planar vs FNO-Hybrid crossover ~ {x_cross:.1f} variables "
          f"(log-x 內插，介於 6 與 96 之間)")
print(f" [輸出] {p_main}")
print(f" [輸出] {p_abs}")
print(f" [輸出] {os.path.join(OUT_DIR, 'crossover_data.csv')}")
print('=' * 100)
