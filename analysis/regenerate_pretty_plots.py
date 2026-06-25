"""
regenerate_pretty_plots.py（重寫版：只負責「跨架構對比圖」）

說明：
- 每個模型「自己」的 pretty 圖（含經緯度軸）已經併進 fourier_2d.py 直接產出，
  不再需要事後重畫，所以本檔只保留「把多個訓練好的模型並排比較」這個
  fourier_2d.py 無法單獨完成的功能。
- 對應新的 5 個模型（全部是 fourier_2d.py 的 FNO2d 變體），改用 6 變數資料。

做法：
1. 從 NetCDF 讀真實經緯度範圍
2. 從測試集取一個固定 batch（所有模型用同一筆 → 公平比較）
3. 對每個 canonical 模型載入 model_weights_best.pt、跑 40 步 rollout、取 Day 10 預測
4. 並排畫出 True vs 各模型的 Day 10 溫度預測與誤差

用法：python analysis/regenerate_pretty_plots.py（從專案根目錄執行）

輸出：
    outputs/_comparison/cross_arch_day10_comparison.png
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import json
import glob

import numpy as np
import torch
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = r"C:\Windows\Fonts\msjh.ttc"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

torch.manual_seed(0)
np.random.seed(0)

OUTPUTS_DIR   = 'outputs'
COMPARE_DIR   = os.path.join(OUTPUTS_DIR, '_comparison')
os.makedirs(COMPARE_DIR, exist_ok=True)
ROLLOUT_STEPS = 40
BATCH_SIZE    = 4
TRAIN_SIZE    = 2920
NUM_CHANNELS  = 6          # 6 氣象變數（後面 4 個是時間特徵）
DAY10_STEP    = 39         # 0-indexed：第 40 步 = Day 10

# 想並排比較的 5 個模型（與 fourier_2d.py EXPERIMENTS 一致）
FNO_EXPERIMENTS = {
    '2d_fno':  {'local_type': '1x1',  'spectral_type': 'fft'},
    '2d_ufno': {'local_type': 'unet', 'spectral_type': 'fft'},
    'sfno':    {'local_type': '1x1',  'spectral_type': 'sht'},
    'sufno':   {'local_type': 'unet', 'spectral_type': 'sht'},
    '2d_unet': {'local_type': 'unet', 'spectral_type': ''},
}
REP_ARCHS = ['2d_fno', '2d_ufno', 'sfno', 'sufno', '2d_unet']  # 想要的顯示順序


################################################################
# 1. 載入 ERA5 6 變數資料 + 真實經緯度（與 fourier_2d.py 一致）
################################################################
print("讀取 ERA5 6 變數資料 + 真實地理範圍...")
ds = xr.open_mfdataset('data/global_era5_6_factors_*.nc', engine='h5netcdf', combine='by_coords')

try:
    lons = ds['longitude'].values
    lats = ds['latitude'].values
    geo_extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]
    print(f"  經度 {geo_extent[0]:.1f}°~{geo_extent[1]:.1f}°, 緯度 {geo_extent[2]:.1f}°~{geo_extent[3]:.1f}°")
except Exception as e:
    geo_extent = None
    print(f"  [警告] 讀取經緯度失敗，改用像素索引：{e}")

t2m   = torch.tensor(ds['t2m'].values)
msl   = torch.tensor(ds['msl'].values)
u10   = torch.tensor(ds['u10'].values)
v10   = torch.tensor(ds['v10'].values)
vimdf = torch.tensor(ds['vimdf'].values)
vitoe = torch.tensor(ds['vitoe'].values)

times    = ds['valid_time'].values
dt       = pd.to_datetime(times)
day_rad  = torch.tensor(dt.dayofyear.values, dtype=torch.float32) * (2 * np.pi / 365.25)
hour_rad = torch.tensor(dt.hour.values,      dtype=torch.float32) * (2 * np.pi / 24.0)
day_sin  = torch.sin(day_rad).view(-1, 1, 1).expand(-1, 33, 64)
day_cos  = torch.cos(day_rad).view(-1, 1, 1).expand(-1, 33, 64)
hour_sin = torch.sin(hour_rad).view(-1, 1, 1).expand(-1, 33, 64)
hour_cos = torch.cos(hour_rad).view(-1, 1, 1).expand(-1, 33, 64)

data = torch.stack([t2m, msl, u10, v10, vimdf, vitoe,
                    day_sin, day_cos, hour_sin, hour_cos], dim=-1)   # (T, 33, 64, 10)
data = torch.nan_to_num(data, nan=0.0).float()

x_mean    = data[:TRAIN_SIZE].mean(dim=(0, 1, 2))
x_std     = data[:TRAIN_SIZE].std(dim=(0, 1, 2))
data_norm = (data - x_mean) / (x_std + 1e-6)

# 固定 test batch（所有模型共用同一筆）
test_x = data_norm[TRAIN_SIZE : TRAIN_SIZE + BATCH_SIZE]               # (B, 33, 64, 10)
test_y = torch.stack(
    [data_norm[TRAIN_SIZE + 1 + s : TRAIN_SIZE + 1 + s + BATCH_SIZE] for s in range(ROLLOUT_STEPS)],
    dim=1)                                                            # (B, rollout, 33, 64, 10)


################################################################
# 2. 截斷 exec 載入 FNO2d class（不觸發訓練）
################################################################
_src = open('fourier_2d.py', encoding='utf-8').read()
_cutoff = _src.find('################################################################\n# ERA5RolloutDataset')
_ns = {}
exec(compile(_src[:_cutoff], 'fourier_2d.py', 'exec'), _ns)
FNO2d = _ns['FNO2d']


def build_model(group, cfg):
    ec = FNO_EXPERIMENTS[group]
    modes = cfg.get('modes', 16)
    width = cfg.get('width', 32)
    return FNO2d(modes, modes, width,
                 local_type=ec['local_type'],
                 spectral_type=ec['spectral_type'], dropout=0.0).cuda()


################################################################
# 3. 找出 outputs/ 裡屬於這 5 模型的 canonical 實驗（seed=0, modes=16, 無 dropout）
################################################################
found = {}   # group -> (dir, cfg)
for d in sorted(glob.glob(os.path.join(OUTPUTS_DIR, '*'))):
    if not os.path.isdir(d):
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
    group = cfg.get('base_experiment_name', os.path.basename(d))
    if group in FNO_EXPERIMENTS and group not in found:
        found[group] = (d, cfg)

print(f"\n找到 {len(found)} 個可比較的模型：{', '.join(found)}")
if not found:
    raise RuntimeError(f"在 {OUTPUTS_DIR}/ 下找不到這 5 個模型的 canonical 實驗")


################################################################
# 4. 對每個模型跑 rollout，取 Day 10 溫度預測
################################################################
gt_day10    = test_y[0, DAY10_STEP, :, :, 0].numpy()
preds_day10 = {}

for group in REP_ARCHS:
    if group not in found:
        print(f"  [略過 {group}] 沒有對應的訓練結果")
        continue
    out_dir, cfg = found[group]
    weights_path = os.path.join(out_dir, 'model_weights_best.pt')
    if not os.path.exists(weights_path):
        weights_path = os.path.join(out_dir, 'model_weights.pt')
    if not os.path.exists(weights_path):
        print(f"  [略過 {group}] 找不到模型權重")
        continue

    model = build_model(group, cfg)
    try:
        model.load_state_dict(torch.load(weights_path, map_location='cuda'))
    except Exception as e:
        print(f"  [略過 {group}] 載入權重失敗：{e}")
        del model; torch.cuda.empty_cache(); continue
    model.eval()

    with torch.no_grad():
        current = test_x.clone().cuda()
        y_in    = test_y.clone().cuda()
        for step in range(ROLLOUT_STEPS):
            pred = model(current)
            if step == DAY10_STEP:
                preds_day10[group] = pred[0, :, :, 0].cpu().numpy()
            if step < ROLLOUT_STEPS - 1:
                next_time = y_in[:, step, :, :, NUM_CHANNELS:]
                current = torch.cat([pred, next_time], dim=-1)
    print(f"  [完成] {group} Day 10 預測")
    del model
    torch.cuda.empty_cache()


################################################################
# 5. 並排對比圖：第一列 = True + 各模型預測；第二列 = 誤差
################################################################
shown = [g for g in REP_ARCHS if g in preds_day10]
n = len(shown)
imshow_kw = dict(extent=geo_extent, aspect='auto') if geo_extent is not None else {}

fig, axes = plt.subplots(2, n + 1, figsize=(5 * (n + 1), 9))

# 第一列：Ground Truth + 各模型預測
im = axes[0, 0].imshow(gt_day10, cmap='jet', **imshow_kw)
axes[0, 0].set_title('Ground Truth (Day 10)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Predictions\nLatitude (°)', fontsize=10)
fig.colorbar(im, ax=axes[0, 0])
for j, group in enumerate(shown):
    im = axes[0, j + 1].imshow(preds_day10[group], cmap='jet', **imshow_kw)
    axes[0, j + 1].set_title(group, fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=axes[0, j + 1])

# 第二列：誤差（True - Pred）
axes[1, 0].axis('off')
axes[1, 0].text(0.5, 0.5, 'Errors\n(Ground Truth - Pred)', ha='center', va='center',
                fontsize=12, fontweight='bold', transform=axes[1, 0].transAxes)
for j, group in enumerate(shown):
    err = gt_day10 - preds_day10[group]
    im = axes[1, j + 1].imshow(err, cmap='coolwarm', vmin=-2, vmax=2, **imshow_kw)
    axes[1, j + 1].set_title(f'Error: {group}', fontsize=11)
    if geo_extent is not None:
        axes[1, j + 1].set_xlabel('Longitude (°)', fontsize=10)
    fig.colorbar(im, ax=axes[1, j + 1])

plt.suptitle('Day 10 (T+40) Temperature Prediction — Cross-Model Comparison',
             fontsize=14, fontweight='bold')
plt.tight_layout()
out_compare = os.path.join(COMPARE_DIR, 'cross_arch_day10_comparison.png')
plt.savefig(out_compare, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n[輸出] {out_compare}")
print("========== 完成！ ==========")
