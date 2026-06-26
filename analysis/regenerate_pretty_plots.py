"""
regenerate_pretty_plots.py（96 變數 / 11 架構版：跨架構對比圖）

說明：
- 每個模型「自己」的 pretty 圖（含經緯度軸）已經併進 fourier_2d.py 直接產出，
  不再需要事後重畫，所以本檔只保留「把多個訓練好的模型並排比較」這個
  fourier_2d.py 無法單獨完成的功能。
- 改用與 fourier_2d.py 完全相同的模型 API（models.build_model / EXPERIMENTS）
  與 96 變數資料載入流程，所以支援全部 11 個架構（含 UNet / UNet++ / TransUNet
  / sphere_* 家族），不再寫死 5 個 FNO2d 變體。

做法：
1. 用 fourier_2d.py 的 load_era5 流程讀真實經緯度與「自動偵測」的氣象通道數，
   再附加 4 個時間編碼通道、用「訓練集」統計量標準化（與 fourier_2d.py 一致）。
2. 從測試集取一個固定 batch（所有模型用同一筆 → 公平比較）。
3. 對每個 canonical 模型（seed=0, modes=16, 無 dropout）用其 config.json 的
   超參數透過 models.build_model 建模、載入 model_weights_best.pt、跑 40 步 rollout、
   取 Day 10（step index 39）通道 0 的預測。
4. 並排畫出 True vs 各模型的 Day 10 預測與誤差。

用法：
    set KMP_DUPLICATE_LIB_OK=TRUE
    set OUTPUT_ROOT=outputs96
    python analysis/regenerate_pretty_plots.py   （從專案根目錄執行）

輸出：
    <OUTPUT_ROOT>/_comparison/cross_arch_day10_comparison.png
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

# torch_harmonics 在某些 Windows 環境需要假 triton 補丁；fourier_2d.py 入口會處理。
# 為確保 sphere_* / sfno 模型可建立，這裡比照 fourier_2d.py：先 import 它（會套用補丁），
# 再使用 models.build_model / EXPERIMENTS（同一條 import 路徑，補丁已生效）。
import json
import glob

import numpy as np
import torch
import xarray as xr
import pandas as pd

# 重用 fourier_2d.py 的 triton 補丁 + 資料載入工具（import 它即套用補丁）。
import fourier_2d as f2d
from models import EXPERIMENTS, build_model

# ----------------------------------------------------------------------
# 繪圖後端說明（重要）：
# 本機 conda 環境（4000Ada_unet_cuda）的 matplotlib Agg C 渲染器（_backend_agg /
# _image / freetype）在 fig.canvas.draw() / savefig 階段會「100% 觸發」原生
# STATUS_STACK_BUFFER_OVERRUN（exit code 0xC0000409）崩潰——即使只畫一張小圖、
# 不含任何文字也一樣（已實測 plot / imshow / svg / pdf 後端皆崩）。
# 因此本檔「不使用 matplotlib 畫圖」，改用 Pillow（PIL）直接合成 PNG：
#   - 熱力圖：用 matplotlib colormap 把陣列上色成 RGBA，再交給 PIL 縮放、貼上。
#   - 文字：用 PIL ImageDraw + Arial TTF。
# 只借用 matplotlib 的 colormap（純 numpy 運算，不觸發 Agg），不觸發任何渲染。
# ----------------------------------------------------------------------
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps as _mpl_cmaps
from matplotlib.colors import Normalize

torch.manual_seed(0)
np.random.seed(0)

OUTPUTS_DIR   = os.environ.get('OUTPUT_ROOT', 'outputs')
COMPARE_DIR   = os.path.join(OUTPUTS_DIR, '_comparison')
os.makedirs(COMPARE_DIR, exist_ok=True)

ROLLOUT_STEPS = 40
BATCH_SIZE    = 4
TRAIN_SIZE    = 2920       # 與 fourier_2d.py 預設一致（config.json 也會覆寫）
DAY10_STEP    = 39         # 0-indexed：第 40 步 = Day 10
DATA_GLOB     = 'data/global_era5_96_factors_*.nc'

# ----------------------------------------------------------------------
# 想並排比較的架構（清楚標示的常數，方便編輯）。
# 每個 geometry 欄各挑一個代表：planar / FNO-hybrid / pure-spherical。
# 想多看幾個就把註解打開（例如 transunet_2d / sutrans_fno / sphere_transunet）。
# ----------------------------------------------------------------------
REP_ARCHS = [
    'unet_2d',          # planar
    'sufno',            # FNO-hybrid
    'sphere_unet',      # pure-spherical
    # 'transunet_2d',
    # 'sutrans_fno',
    # 'sphere_transunet',
]


################################################################
# 1. 載入 ERA5 96 變數資料 + 真實經緯度（與 fourier_2d.py 完全一致）
################################################################
print(f"讀取 ERA5（glob={DATA_GLOB}）+ 真實地理範圍...")
data_meteo, times, channel_names, geo_extent, nlat, nlon = f2d.load_era5(
    DATA_GLOB, var_whitelist='', verbose=True)
NUM_CHANNELS = data_meteo.shape[-1]
print(f"  自動偵測氣象通道數 = {NUM_CHANNELS}，網格 = {nlat}×{nlon}")
if geo_extent is not None:
    print(f"  經度 {geo_extent[0]:.1f}°~{geo_extent[1]:.1f}°, 緯度 {geo_extent[2]:.1f}°~{geo_extent[3]:.1f}°")

# 附加 4 個時間編碼通道（與 fourier_2d.py 相同）
time_ch = f2d.build_time_channels(times, nlat, nlon)
data = torch.cat([data_meteo, time_ch], dim=-1).float()       # (T, nlat, nlon, C_meteo+4)

# 標準化：僅用訓練集統計量（含時間通道，與 fourier_2d.py 一致）
x_mean    = data[:TRAIN_SIZE].mean(dim=(0, 1, 2))
x_std     = data[:TRAIN_SIZE].std(dim=(0, 1, 2))
data_norm = (data - x_mean) / (x_std + 1e-6)

# 固定 test batch（所有模型共用同一筆）
test_x = data_norm[TRAIN_SIZE : TRAIN_SIZE + BATCH_SIZE]               # (B, nlat, nlon, C)
test_y = torch.stack(
    [data_norm[TRAIN_SIZE + 1 + s : TRAIN_SIZE + 1 + s + BATCH_SIZE] for s in range(ROLLOUT_STEPS)],
    dim=1)                                                            # (B, rollout, nlat, nlon, C)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


################################################################
# 2. 找出 OUTPUT_ROOT 裡每個架構的 canonical 實驗（seed=0, modes=16, 無 dropout）
################################################################
found = {}   # arch -> (dir, cfg)
for d in sorted(glob.glob(os.path.join(OUTPUTS_DIR, '*'))):
    if not os.path.isdir(d):
        continue
    if os.path.basename(d).startswith('_'):
        continue
    cfg_path = os.path.join(d, 'config.json')
    if not os.path.exists(cfg_path):
        continue
    with open(cfg_path, encoding='utf-8') as f:
        cfg = json.load(f)
    if cfg.get('seed', 0) != 0 or cfg.get('modes', 16) != 16:
        continue
    if cfg.get('dropout', 0) and cfg.get('dropout', 0) > 0:
        continue
    arch = cfg.get('base_experiment_name', os.path.basename(d))
    if arch in EXPERIMENTS and arch not in found:
        found[arch] = (d, cfg)

print(f"\n在 {OUTPUTS_DIR}/ 找到 {len(found)} 個 canonical 實驗：{', '.join(sorted(found))}")
if not found:
    raise RuntimeError(f"在 {OUTPUTS_DIR}/ 下找不到任何 canonical 實驗（seed=0, modes=16, 無 dropout）")


################################################################
# 3. 對每個代表模型跑 rollout，取 Day 10 通道 0 預測
################################################################
gt_day10    = test_y[0, DAY10_STEP, :, :, 0].numpy()
preds_day10 = {}


def _build_from_cfg(arch, cfg):
    """用 config.json 的超參數透過 models.build_model 建模。"""
    sphere_modes = tuple(cfg.get('sphere_modes', [8, 4, 2]))
    return build_model(
        arch,
        num_channels=cfg.get('num_channels', NUM_CHANNELS),
        base_width=cfg.get('base_width', 32),
        modes=cfg.get('modes', 16),
        dropout=cfg.get('dropout', 0.0),
        nlat=cfg.get('nlat', nlat),
        nlon=cfg.get('nlon', nlon),
        sphere_modes=sphere_modes,
    ).to(device)


for arch in REP_ARCHS:
    if arch not in found:
        print(f"  [略過 {arch}] 沒有對應的 canonical 訓練結果")
        continue
    out_dir, cfg = found[arch]
    weights_path = os.path.join(out_dir, 'model_weights_best.pt')
    if not os.path.exists(weights_path):
        weights_path = os.path.join(out_dir, 'model_weights.pt')
    if not os.path.exists(weights_path):
        print(f"  [略過 {arch}] 找不到模型權重")
        continue

    try:
        model = _build_from_cfg(arch, cfg)
        model.load_state_dict(torch.load(weights_path, map_location=device))
    except Exception as e:
        print(f"  [略過 {arch}] 建模/載入權重失敗：{e}")
        continue
    model.eval()

    nch = cfg.get('num_channels', NUM_CHANNELS)
    with torch.no_grad():
        current = test_x.clone().to(device)
        y_in    = test_y.clone().to(device)
        for step in range(ROLLOUT_STEPS):
            pred = model(current)
            if step == DAY10_STEP:
                preds_day10[arch] = pred[0, :, :, 0].cpu().numpy()
            if step < ROLLOUT_STEPS - 1:
                # 氣象通道用模型預測；時間通道用 ground truth 補回（與 fourier_2d.py 一致）
                next_time = y_in[:, step, :, :, nch:]
                current = torch.cat([pred, next_time], dim=-1)
    print(f"  [完成] {arch} Day 10 預測")
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()


################################################################
# 4. 並排對比圖（用 Pillow 合成，避開崩潰的 matplotlib Agg 渲染器）
#    第一列 = True + 各模型預測；第二列 = 各模型誤差
################################################################
shown = [g for g in REP_ARCHS if g in preds_day10]
if not shown:
    raise RuntimeError("沒有任何模型成功產生 Day 10 預測，無法畫圖。")

ch0_name = channel_names[0] if channel_names else 'ch0'


def _font(sz, bold=False):
    path = r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"
    try:
        return ImageFont.truetype(path, sz)
    except Exception:
        return ImageFont.load_default()


def _heat_img(arr, cmap_name, vmin, vmax, out_w, out_h):
    """把 2D 陣列依 colormap 上色成 PIL RGB 影像並縮放到指定尺寸（不經 Agg）。"""
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgba = (_mpl_cmaps[cmap_name](norm(np.asarray(arr, dtype=np.float64))) * 255).astype('uint8')
    img = Image.fromarray(rgba, 'RGBA').convert('RGB')
    # 上下翻轉：imshow 預設原點在左上、緯度由北到南，這裡資料 row0 = 第一緯度，維持一致
    return img.resize((out_w, out_h), Image.NEAREST)


def _colorbar_img(cmap_name, vmin, vmax, h, w=18):
    grad = np.linspace(vmax, vmin, h).reshape(-1, 1)
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgba = (_mpl_cmaps[cmap_name](norm(grad)) * 255).astype('uint8')
    rgba = np.repeat(rgba, w, axis=1)
    return Image.fromarray(rgba, 'RGBA').convert('RGB')


def _centered(draw, x, y, w, text, font, fill='black'):
    tb = draw.textbbox((0, 0), text, font=font)
    tw = tb[2] - tb[0]
    draw.text((x + (w - tw) // 2, y), text, font=font, fill=fill)


# 版面參數
CELL_W, CELL_H = 340, 200
CBAR_W = 26
PAD = 22
TITLE_H = 30
TOP = 70
LABEL_W = 150          # 最左欄列標籤寬度
cols = len(shown) + 1  # 第 0 欄 = GT / 列標籤
rows = 2

cell_total_w = CELL_W + CBAR_W + 10
W = LABEL_W + cols * cell_total_w + (cols + 1) * PAD
H = TOP + rows * (TITLE_H + CELL_H + PAD)

canvas = Image.new('RGB', (W, H), 'white')
draw = ImageDraw.Draw(canvas)

# 大標題
title = f'Day 10 (T+40) Prediction [{ch0_name}] - Cross-Architecture Comparison'
_centered(draw, 0, 18, W, title, _font(26, True))

# 預測值範圍（GT + 各預測一起取範圍，色階一致）
all_pred_vals = [gt_day10] + [preds_day10[a] for a in shown]
pv_min = float(np.min([a.min() for a in all_pred_vals]))
pv_max = float(np.max([a.max() for a in all_pred_vals]))
ERR_MIN, ERR_MAX = -2.0, 2.0


def _draw_cell(r, c, arr, cmap_name, vmin, vmax, title_text):
    x = LABEL_W + PAD + c * (cell_total_w + PAD)
    y = TOP + r * (TITLE_H + CELL_H + PAD)
    _centered(draw, x, y, CELL_W, title_text, _font(15, True))
    canvas.paste(_heat_img(arr, cmap_name, vmin, vmax, CELL_W, CELL_H), (x, y + TITLE_H))
    cb = _colorbar_img(cmap_name, vmin, vmax, CELL_H, CBAR_W)
    canvas.paste(cb, (x + CELL_W + 8, y + TITLE_H))
    f = _font(11)
    draw.text((x + CELL_W + 8, y + TITLE_H - 14), f'{vmax:.1f}', font=f, fill='black')
    draw.text((x + CELL_W + 8, y + TITLE_H + CELL_H + 2), f'{vmin:.1f}', font=f, fill='black')


# 第一列：Ground Truth + 各模型預測
draw.text((PAD, TOP + TITLE_H + CELL_H // 2 - 20), 'Predictions', font=_font(16, True), fill='black')
_draw_cell(0, 0, gt_day10, 'jet', pv_min, pv_max, 'Ground Truth (Day 10)')
for j, arch in enumerate(shown):
    _draw_cell(0, j + 1, preds_day10[arch], 'jet', pv_min, pv_max, arch)

# 第二列：誤差（True - Pred）
draw.text((PAD, TOP + (TITLE_H + CELL_H + PAD) + TITLE_H + CELL_H // 2 - 20),
          'Errors\n(GT - Pred)', font=_font(16, True), fill='black')
for j, arch in enumerate(shown):
    err = gt_day10 - preds_day10[arch]
    _draw_cell(1, j + 1, err, 'coolwarm', ERR_MIN, ERR_MAX, f'Error: {arch}')

out_compare = os.path.join(COMPARE_DIR, 'cross_arch_day10_comparison.png')
canvas.save(out_compare)
print(f"\n[輸出] {out_compare}")
print(f"  並排比較的架構：{', '.join(shown)}")
print("========== 完成！ ==========")
