"""
resource_comparison.py（96 變數 / 11 架構版）

從既有實驗結果聚合「模型大小 / 訓練時間 / 推論時間 / 峰值記憶體」對照表。
不需要重新訓練：
- Params (M)：從 config.json 直接讀（不再寫死）
- Avg epoch time / Total train time：從 training_log.csv 累計
- Peak GPU memory + Inference latency：用 models.build_model 以正確的 96 變數輸入通道數
  建模、載入 model_weights_best.pt，做 forward 量測

涵蓋 OUTPUT_ROOT 下「全部」出現的架構（11 個），而非寫死 5 個。
依名稱分類 geometry 上色：
- planar       ：名稱以 _2d 結尾或 2d_ 開頭（unet_2d / unetpp_2d / transunet_2d / 2d_fno）
- fno_hybrid   ：sufno / sunetpp_fno / sutrans_fno
- spherical    ：sphere_unet / sphere_unetpp / sphere_transunet / sfno（sfno 視為球面基線）

用法：
    set KMP_DUPLICATE_LIB_OK=TRUE
    set OUTPUT_ROOT=outputs96
    python analysis/resource_comparison.py   （從專案根目錄執行）

輸出：
    <OUTPUT_ROOT>/_comparison/resource_summary.csv
    <OUTPUT_ROOT>/_comparison/resource_comparison.png
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import json
import glob
import time

import numpy as np
import pandas as pd
import torch

# 重用 fourier_2d.py 的 triton 補丁（import 它即套用），再用 models.build_model。
import fourier_2d as f2d  # noqa: F401
from models import EXPERIMENTS, build_model

# ----------------------------------------------------------------------
# 繪圖後端說明（重要）：
# 本機 conda 環境（4000Ada_unet_cuda）的 matplotlib Agg C 渲染器在 savefig /
# fig.canvas.draw() 階段會「100% 觸發」原生 STATUS_STACK_BUFFER_OVERRUN
# （exit code 0xC0000409）崩潰——即使畫一張不含文字的小圖也一樣。
# 因此本檔的長條圖「不使用 matplotlib 畫圖」，改用 Pillow（PIL）直接合成 PNG。
# CSV 輸出不受影響（純文字）。
# ----------------------------------------------------------------------
from PIL import Image, ImageDraw, ImageFont

OUTPUTS_DIR = os.environ.get('OUTPUT_ROOT', 'outputs')
COMPARE_DIR = os.path.join(OUTPUTS_DIR, '_comparison')
os.makedirs(COMPARE_DIR, exist_ok=True)

CAT_COLOR = {'planar': '#2ecc71', 'fno_hybrid': '#3498db', 'spherical': '#e74c3c'}
CAT_LABEL = {
    'planar':     '平面 2D (planar)',
    'fno_hybrid': 'FNO 混合 (SHT ⊕ local)',
    'spherical':  '純球面 (pure SHT)',
}

# geometry 分類常數（清楚標示，方便編輯）
PLANAR_ARCHS    = {'unet_2d', 'unetpp_2d', 'transunet_2d', '2d_fno'}
FNO_HYBRID_ARCHS = {'sufno', 'sunetpp_fno', 'sutrans_fno'}
SPHERICAL_ARCHS = {'sphere_unet', 'sphere_unetpp', 'sphere_transunet', 'sfno'}


def categorize(name):
    if name in PLANAR_ARCHS:
        return 'planar'
    if name in FNO_HYBRID_ARCHS:
        return 'fno_hybrid'
    if name in SPHERICAL_ARCHS:
        return 'spherical'
    # 後備規則：依命名慣例推斷
    if name.endswith('_2d') or name.startswith('2d_'):
        return 'planar'
    if name.startswith('sphere_') or name == 'sfno':
        return 'spherical'
    return 'fno_hybrid'


def build_from_cfg(arch, cfg):
    """用 config.json 的超參數透過 models.build_model 建模（已 .to(device)）。失敗回傳 None。"""
    if arch not in EXPERIMENTS:
        print(f"  [略過 {arch}] 不在 models.EXPERIMENTS 內")
        return None
    try:
        sphere_modes = tuple(cfg.get('sphere_modes', [8, 4, 2]))
        return build_model(
            arch,
            num_channels=cfg.get('num_channels'),
            base_width=cfg.get('base_width', 32),
            modes=cfg.get('modes', 16),
            dropout=cfg.get('dropout', 0.0),
            nlat=cfg.get('nlat', 33),
            nlon=cfg.get('nlon', 64),
            sphere_modes=sphere_modes,
        ).to(device)
    except Exception as e:
        print(f"  [略過 {arch}] 模型建立失敗：{e}")
        return None


################################################################
# 1. 聚合既有 config + training_log（params / epoch time）— 涵蓋全部架構
################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
records = []
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
    arch = cfg.get('base_experiment_name', name)
    if arch not in EXPERIMENTS:
        print(f"  [略過 {name}] base_experiment_name='{arch}' 不在 models.EXPERIMENTS 內")
        continue
    records.append({
        'arch':            name,
        'group':           arch,
        'dir':             d,
        'cfg':             cfg,
        'param_count':     cfg.get('param_count', 0),
        'num_channels':    cfg.get('num_channels'),
        'nlat':            cfg.get('nlat', 33),
        'nlon':            cfg.get('nlon', 64),
        'avg_epoch_sec':   df['epoch_time_sec'].mean(),
        'total_train_sec': df['epoch_time_sec'].sum(),
        'epochs':          len(df),
        'best_test_mse':   cfg.get('best_test_mse', df['test_mse'].min()),
    })

if not records:
    raise RuntimeError(f"在 {OUTPUTS_DIR}/ 下找不到任何完成實驗（需 config.json + training_log.csv）")

df_all = pd.DataFrame(records)
print(f"找到 {len(df_all)} 個實驗：{', '.join(df_all['group'])}")

################################################################
# 2. 對每個模型量測 inference latency + peak GPU memory
################################################################
print("\n正在量測各模型的 inference latency 與 peak GPU memory...")
inference_results = {}   # arch (experiment name) -> dict

for _, row in df_all.iterrows():
    arch = row['group']
    cfg  = row['cfg']
    model = build_from_cfg(arch, cfg)
    if model is None:
        continue

    weights_path = os.path.join(row['dir'], 'model_weights_best.pt')
    if not os.path.exists(weights_path):
        weights_path = os.path.join(row['dir'], 'model_weights.pt')
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
        except Exception as e:
            print(f"  [警告 {arch}] 載入權重失敗（用 random init 量測）：{e}")
    model.eval()

    # 輸入 = num_channels 氣象 + 4 時間 = 模型 forward 期待的 channel-last 輸入
    # （模型內部會再 concat 2 個 grid 通道）。
    nch  = int(row['num_channels'])
    nlat = int(row['nlat'])
    nlon = int(row['nlon'])
    in_ch = nch + 4
    x = torch.randn(4, nlat, nlon, in_ch, device=device)

    try:
        with torch.no_grad():
            for _ in range(3):                      # warm up
                _ = model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        n_runs = 100
        with torch.no_grad():
            if device == 'cuda':
                torch.cuda.synchronize()
            t_start = time.perf_counter()
            for _ in range(n_runs):
                _ = model(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            t_end = time.perf_counter()
        inference_ms = (t_end - t_start) / n_runs * 1000
        peak_mem_MB = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                       if device == 'cuda' else float('nan'))
    except Exception as e:
        print(f"  [警告 {arch}] forward 量測失敗：{e}")
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
        continue

    inference_results[arch] = {
        'inference_ms_per_step': inference_ms,
        'peak_gpu_mem_MB':       peak_mem_MB,
    }
    print(f"  {arch:<18} | inference {inference_ms:6.2f} ms/step | peak mem {peak_mem_MB:7.1f} MB")

    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

df_all['inference_ms_per_step'] = df_all['group'].map(
    lambda g: inference_results.get(g, {}).get('inference_ms_per_step', np.nan))
df_all['peak_gpu_mem_MB'] = df_all['group'].map(
    lambda g: inference_results.get(g, {}).get('peak_gpu_mem_MB', np.nan))

################################################################
# 3. 輸出 CSV
################################################################
out_df = df_all.sort_values('best_test_mse').copy()
out_df['param_count_M']  = out_df['param_count'] / 1e6
out_df['avg_epoch_min']  = out_df['avg_epoch_sec'] / 60
out_df['total_train_hr'] = out_df['total_train_sec'] / 3600
out_df['category']       = out_df['group'].map(categorize)

export_cols = ['group', 'category', 'param_count_M', 'avg_epoch_min', 'total_train_hr',
               'inference_ms_per_step', 'peak_gpu_mem_MB', 'best_test_mse']
export_df = out_df[export_cols].copy()
for col, nd in [('param_count_M', 2), ('avg_epoch_min', 2), ('total_train_hr', 2),
                ('inference_ms_per_step', 2), ('peak_gpu_mem_MB', 1), ('best_test_mse', 4)]:
    export_df[col] = export_df[col].round(nd)

csv_path = os.path.join(COMPARE_DIR, 'resource_summary.csv')
export_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n[輸出] {csv_path}")
print("\n" + "=" * 110)
print(" Resource Comparison Table")
print("=" * 110)
print(export_df.to_string(index=False))
print("=" * 110)

################################################################
# 4. 4-panel bar chart（用 Pillow 合成，避開崩潰的 matplotlib Agg 渲染器）
################################################################
labels = out_df['group'].tolist()
colors = [CAT_COLOR[categorize(g)] for g in labels]

# PIL 用 Arial，無法描繪中文/特殊符號，圖內圖例改用 ASCII 標籤
CAT_LABEL_ASCII = {
    'planar':     'planar 2D',
    'fno_hybrid': 'FNO-hybrid (SHT + local)',
    'spherical':  'pure-spherical (SHT)',
}


def _font(sz, bold=False):
    path = r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"
    try:
        return ImageFont.truetype(path, sz)
    except Exception:
        return ImageFont.load_default()


def _text_w(draw, text, font):
    tb = draw.textbbox((0, 0), text, font=font)
    return tb[2] - tb[0]


def _draw_panel(draw, px, py, pw, ph, title, vals, labels, colors):
    """在 (px,py) 畫一個寬 pw 高 ph 的長條圖面板。"""
    # 標題
    f_title = _font(17, True)
    draw.text((px + (pw - _text_w(draw, title, f_title)) // 2, py), title, font=f_title, fill='black')

    plot_top = py + 34
    plot_bottom = py + ph - 60          # 留底部給 x 標籤
    plot_left = px + 55
    plot_right = px + pw - 30
    plot_h = plot_bottom - plot_top
    plot_w = plot_right - plot_left

    finite = [v for v in vals if not np.isnan(v)]
    vmax = max(finite) if finite else 1.0
    if vmax <= 0:
        vmax = 1.0
    vmax *= 1.18                         # 頂部留白給數值標籤

    # 座標軸
    draw.line([(plot_left, plot_top), (plot_left, plot_bottom)], fill='black', width=1)
    draw.line([(plot_left, plot_bottom), (plot_right, plot_bottom)], fill='black', width=1)
    # y 軸刻度（4 格）
    f_tick = _font(11)
    for k in range(5):
        yval = vmax * k / 4
        yy = plot_bottom - int(plot_h * k / 4)
        draw.line([(plot_left - 4, yy), (plot_left, yy)], fill='black', width=1)
        lab = f'{yval:.0f}' if vmax >= 10 else f'{yval:.1f}'
        draw.text((plot_left - 6 - _text_w(draw, lab, f_tick), yy - 6), lab, font=f_tick, fill='black')
        if k > 0:
            draw.line([(plot_left, yy), (plot_right, yy)], fill=(220, 220, 220), width=1)

    n = len(vals)
    slot = plot_w / max(n, 1)
    bar_w = slot * 0.62
    f_val = _font(10)
    f_xlab = _font(11)
    for i, (v, lab, col) in enumerate(zip(vals, labels, colors)):
        cx = plot_left + slot * (i + 0.5)
        x0 = cx - bar_w / 2
        x1 = cx + bar_w / 2
        if not np.isnan(v):
            bh = int(plot_h * (v / vmax))
            y0 = plot_bottom - bh
            draw.rectangle([x0, y0, x1, plot_bottom], fill=col, outline='black')
            vtxt = f'{v:.1f}' if v >= 10 else f'{v:.2f}'
            draw.text((cx - _text_w(draw, vtxt, f_val) / 2, y0 - 13), vtxt, font=f_val, fill='black')
        # x 標籤（旋轉成直書太麻煩；用小字水平、必要時截短）
        lab_s = lab if len(lab) <= 12 else lab[:11] + '.'
        lw = _text_w(draw, lab_s, f_xlab)
        # 為避免重疊，交錯兩層高度
        ytxt = plot_bottom + 4 + (18 if i % 2 else 2)
        draw.text((cx - lw / 2, ytxt), lab_s, font=f_xlab, fill='black')


N = len(labels)
PW, PH = 760, 380          # 每個面板尺寸
MARGIN = 30
GAP = 30
TOP = 70
LEGEND_H = 40
W = MARGIN * 2 + PW * 2 + GAP
H = TOP + PH * 2 + GAP + LEGEND_H + MARGIN

canvas = Image.new('RGB', (W, H), 'white')
draw = ImageDraw.Draw(canvas)

# 大標題
big_title = f'Resource Comparison Across {N} Architectures (sorted by best test MSE)'
f_big = _font(24, True)
draw.text(((W - _text_w(draw, big_title, f_big)) // 2, 16), big_title, font=f_big, fill='black')

# 圖例
lx = MARGIN
ly = 48
f_leg = _font(13, True)
for c in ['planar', 'fno_hybrid', 'spherical']:
    draw.rectangle([lx, ly, lx + 18, ly + 14], fill=CAT_COLOR[c], outline='black')
    txt = CAT_LABEL_ASCII[c]
    draw.text((lx + 24, ly), txt, font=f_leg, fill='black')
    lx += 24 + _text_w(draw, txt, f_leg) + 40

metrics = [
    ('param_count_M',         'Model Size (M params)',  0, 0),
    ('avg_epoch_min',         'Avg Epoch Time (min)',   0, 1),
    ('inference_ms_per_step', 'Inference Latency (ms)', 1, 0),
    ('peak_gpu_mem_MB',       'Peak GPU Memory (MB)',   1, 1),
]
for col, title, r, c in metrics:
    px = MARGIN + c * (PW + GAP)
    py = TOP + LEGEND_H + r * (PH + GAP)
    _draw_panel(draw, px, py, PW, PH, title,
                list(out_df[col].values), labels, colors)

out_png = os.path.join(COMPARE_DIR, 'resource_comparison.png')
canvas.save(out_png)
print(f"[輸出] {out_png}")
print(f"\n所有結果已輸出至：{COMPARE_DIR}/")
