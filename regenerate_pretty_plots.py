"""
regenerate_pretty_plots.py
吸收學弟的經緯度視覺化改進，**不需重新訓練**。

做法：
1. 從 NetCDF 讀真實的經緯度範圍（geo_extent）
2. 對每個 canonical 實驗（SEED=0 / 無 dropout / 無 modes 變動），載入 model_weights_best.pt
3. 對固定的 test batch 跑 40 步 rollout 推論
4. 重新產出 weather_prediction_pretty.png：與原圖一樣但加上 Longitude/Latitude 軸標籤

輸出位置：
    outputs/<arch>/weather_prediction_pretty.png  ← 每個架構各一張
"""
import os
import json
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
ROLLOUT_STEPS = 40
BATCH_SIZE    = 4
TRAIN_SIZE    = 2920


################################################################
# 1. 載入 ERA5 資料 + 真實經緯度
################################################################
print("讀取 ERA5 資料 + 真實地理範圍...")
ds = xr.open_mfdataset('data/global_era5_mini_*.nc', engine='h5netcdf', combine='by_coords')

lons = ds['longitude'].values
lats = ds['latitude'].values
# matplotlib extent: [left, right, bottom, top]
# ERA5 經度通常 0~360°，緯度通常從 90 降到 -90（北極→南極）
# imshow 預設 origin='upper'（row 0 在上）→ 設定 bottom=lat_min, top=lat_max
geo_extent = [float(lons.min()), float(lons.max()),
              float(lats.min()), float(lats.max())]
print(f"  經度範圍：{geo_extent[0]:.1f}° ~ {geo_extent[1]:.1f}°")
print(f"  緯度範圍：{geo_extent[2]:.1f}° ~ {geo_extent[3]:.1f}°")

t2m = torch.tensor(ds['t2m'].values)
msl = torch.tensor(ds['msl'].values)
u10 = torch.tensor(ds['u10'].values)
v10 = torch.tensor(ds['v10'].values)

times    = ds['valid_time'].values
dt       = pd.to_datetime(times)
day_rad  = torch.tensor(dt.dayofyear.values, dtype=torch.float32) * (2 * np.pi / 365.25)
hour_rad = torch.tensor(dt.hour.values,      dtype=torch.float32) * (2 * np.pi / 24.0)
day_sin  = torch.sin(day_rad).view(-1, 1, 1).expand(-1, 33, 64)
day_cos  = torch.cos(day_rad).view(-1, 1, 1).expand(-1, 33, 64)
hour_sin = torch.sin(hour_rad).view(-1, 1, 1).expand(-1, 33, 64)
hour_cos = torch.cos(hour_rad).view(-1, 1, 1).expand(-1, 33, 64)

data = torch.stack([t2m, msl, u10, v10, day_sin, day_cos, hour_sin, hour_cos], dim=-1)
data = torch.nan_to_num(data, nan=0.0).float()

# 訓練集統計量做正規化（與訓練腳本一致）
x_mean    = data[:TRAIN_SIZE].mean(dim=(0, 1, 2))
x_std     = data[:TRAIN_SIZE].std(dim=(0, 1, 2))
data_norm = (data - x_mean) / (x_std + 1e-6)

# 從測試集挑一個固定 batch 來做可視化（所有架構用同一筆 → 公平比較）
test_start = TRAIN_SIZE   # 測試集第一筆
test_x = data_norm[test_start : test_start + BATCH_SIZE]   # (B, 33, 64, 8)
test_y_list = []
for step in range(ROLLOUT_STEPS):
    test_y_list.append(data_norm[test_start + 1 + step : test_start + 1 + step + BATCH_SIZE])
test_y = torch.stack(test_y_list, dim=1)   # (B, rollout, 33, 64, 8)


################################################################
# Hardcoded EXPERIMENTS（從 fourier_2d.py 複製，避免 exec cutoff 問題）
################################################################
FNO_EXPERIMENTS = {
    '2d_fno':      {'local_type': '1x1',           'spectral_type': 'fft'},
    'sfno':        {'local_type': '1x1',           'spectral_type': 'sht'},
    'sufno':       {'local_type': 'unet',          'spectral_type': 'sht'},
    'sunetpp_fno': {'local_type': 'advanced_unet', 'spectral_type': 'sht'},
    'sutrans_fno': {'local_type': 'transformer',   'spectral_type': 'sht'},
}


################################################################
# 2. Model factory（按 group 名稱建立對應的模型 class）
################################################################
def build_model_for_group(group):
    """從各個 baseline script 載入 model class，避免觸發訓練。"""
    try:
        if group in FNO_EXPERIMENTS:
            src = open('fourier_2d.py', encoding='utf-8').read()
            cutoff = src.find('################################################################\n# ERA5RolloutDataset')
            ns = {}
            exec(compile(src[:cutoff], 'fourier_2d.py', 'exec'), ns)
            ec = FNO_EXPERIMENTS[group]
            return ns['FNO2d'](16, 16, 32,
                                local_type=ec['local_type'],
                                spectral_type=ec['spectral_type'],
                                dropout=0.0).cuda()
        elif group == 'unet_2d':
            src = open('unet_baseline.py', encoding='utf-8').read()
            cutoff = src.find('################################################################\n# ERA5RolloutDataset')
            ns = {}
            exec(compile(src[:cutoff], 'unet_baseline.py', 'exec'), ns)
            return ns['UNet2DRollout']().cuda()
        elif group == 'unetpp_2d':
            src = open('unetpp_baseline.py', encoding='utf-8').read()
            cutoff = src.find('################################################################\n# ERA5RolloutDataset')
            ns = {}
            exec(compile(src[:cutoff], 'unetpp_baseline.py', 'exec'), ns)
            return ns['UNetPlusPlus2DRollout']().cuda()
        elif group == 'transunet_2d':
            src = open('transunet_baseline.py', encoding='utf-8').read()
            cutoff = src.find('################################################################\n# ERA5RolloutDataset')
            ns = {}
            exec(compile(src[:cutoff], 'transunet_baseline.py', 'exec'), ns)
            return ns['TransUNet2DRollout']().cuda()
        elif group == 'sphere_unet':
            src = open('sphere_unet_baseline.py', encoding='utf-8').read()
            cutoff = src.find('################################################################\n# ERA5RolloutDataset')
            ns = {}
            exec(compile(src[:cutoff], 'sphere_unet_baseline.py', 'exec'), ns)
            return ns['SphereUNet2DRollout']().cuda()
        elif group == 'sphere_unetpp':
            src = open('sphere_unetpp_baseline.py', encoding='utf-8').read()
            cutoff = src.find('################################################################\n# ERA5RolloutDataset')
            ns = {}
            exec(compile(src[:cutoff], 'sphere_unetpp_baseline.py', 'exec'), ns)
            return ns['SphereUNetPlusPlus2DRollout']().cuda()
        elif group == 'sphere_transunet':
            src = open('sphere_transunet_baseline.py', encoding='utf-8').read()
            cutoff = src.find('################################################################\n# ERA5RolloutDataset')
            ns = {}
            exec(compile(src[:cutoff], 'sphere_transunet_baseline.py', 'exec'), ns)
            return ns['SphereTransUNet2DRollout']().cuda()
    except Exception as e:
        print(f"    [錯誤] 建立 {group} 失敗：{e}")
        return None
    return None


################################################################
# 3. 找出所有 canonical 實驗（SEED=0, dropout=0, modes=16）
################################################################
canonical = []
for d in sorted(glob.glob(os.path.join(OUTPUTS_DIR, '*'))):
    if not os.path.isdir(d):
        continue
    name = os.path.basename(d)
    if name.startswith('_'):
        continue
    cfg_path = os.path.join(d, 'config.json')
    if not os.path.exists(cfg_path):
        continue
    with open(cfg_path, encoding='utf-8') as f:
        cfg = json.load(f)
    # 跳過非 canonical（seed!=0、有 dropout、modes 非預設）
    if cfg.get('seed', 0) != 0:
        continue
    if cfg.get('dropout', 0) and cfg.get('dropout', 0) > 0:
        continue
    if cfg.get('modes', 16) != 16:
        continue
    group = cfg.get('base_experiment_name', name)
    canonical.append((group, d, cfg))

print(f"\n找到 {len(canonical)} 個 canonical 實驗：")
for g, _, _ in canonical:
    print(f"  - {g}")


################################################################
# 4. 對每個架構：載入 best weights，跑 rollout，產出 pretty plot
################################################################
target_steps  = [3, 11, 27, 39]   # T+4, T+12, T+28, T+40 → Day 1, 3, 7, 10
target_labels = ['Day 1 (T+4)', 'Day 3 (T+12)', 'Day 7 (T+28)', 'Day 10 (T+40)']

print("\n開始產出 weather_prediction_pretty.png ...")
for group, out_dir, cfg in canonical:
    print(f"\n→ {group}")
    model = build_model_for_group(group)
    if model is None:
        print(f"    [略過] 模型建立失敗")
        continue

    weights_path = os.path.join(out_dir, 'model_weights_best.pt')
    if not os.path.exists(weights_path):
        weights_path = os.path.join(out_dir, 'model_weights.pt')
    if not os.path.exists(weights_path):
        print(f"    [略過] 找不到模型權重")
        del model
        torch.cuda.empty_cache()
        continue

    try:
        state = torch.load(weights_path, map_location='cuda')
        model.load_state_dict(state)
    except Exception as e:
        print(f"    [略過] 載入權重失敗：{e}")
        del model
        torch.cuda.empty_cache()
        continue
    model.eval()

    # 跑 40 步 rollout（用同一個 test batch，所有架構公平比較）
    x_in = test_x.clone().cuda()
    y_in = test_y.clone().cuda()
    all_preds = []
    with torch.no_grad():
        current = x_in
        for step in range(ROLLOUT_STEPS):
            pred = model(current)
            all_preds.append(pred.cpu())
            if step < ROLLOUT_STEPS - 1:
                next_time = y_in[:, step, :, :, 4:]
                current = torch.cat([pred, next_time], dim=-1)

    # 繪圖：4 個時效 × 3 欄（True / Pred / Error），全部加經緯度軸
    idx = 0
    fig, axes = plt.subplots(len(target_steps), 3, figsize=(16, len(target_steps) * 4))
    for row, (ts, label) in enumerate(zip(target_steps, target_labels)):
        gt   = test_y[idx, ts, :, :, 0].numpy()
        pred = all_preds[ts][idx, :, :, 0].numpy()
        err  = gt - pred

        im0 = axes[row, 0].imshow(gt,   cmap='jet',      extent=geo_extent, aspect='auto')
        axes[row, 0].set_title(f'True {label}', fontsize=11)
        axes[row, 0].set_ylabel('Latitude (°)', fontsize=10)
        fig.colorbar(im0, ax=axes[row, 0])

        im1 = axes[row, 1].imshow(pred, cmap='jet',      extent=geo_extent, aspect='auto')
        axes[row, 1].set_title(f'Pred {label}', fontsize=11)
        fig.colorbar(im1, ax=axes[row, 1])

        im2 = axes[row, 2].imshow(err,  cmap='coolwarm', extent=geo_extent, aspect='auto')
        axes[row, 2].set_title(f'Error {label}', fontsize=11)
        fig.colorbar(im2, ax=axes[row, 2])

    # 只在最底列加 Longitude 標籤（避免每張都重複）
    for ax in axes[-1, :]:
        ax.set_xlabel('Longitude (°)', fontsize=10)

    display_name = cfg.get('display_name', group)
    plt.suptitle(f'Temperature Prediction Error Maps — {display_name}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_png = os.path.join(out_dir, 'weather_prediction_pretty.png')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    [輸出] {out_png}")

    del model
    torch.cuda.empty_cache()


################################################################
# 5. 額外：跨架構對比圖（True vs 4 個代表性架構，Day 10 預測）
################################################################
COMPARE_DIR = os.path.join(OUTPUTS_DIR, '_comparison')
os.makedirs(COMPARE_DIR, exist_ok=True)

# 挑 4 個代表性架構：每類別各取最強的
REP_ARCHS = ['unet_2d',           # 平面冠軍
             'transunet_2d',      # 平面 + Transformer（參數最少）
             'sunetpp_fno',       # FNO 雜交家族最佳
             'sphere_unet']       # 純球面代表

# 從先前的迴圈拿到的 all_preds 已 del 了，需要重新對這 4 個跑
print("\n產出跨架構對比圖（True vs 4 代表架構，Day 10 預測）...")

# 找這 4 個的 output dir
rep_dirs = {}
for group, out_dir, cfg in canonical:
    if group in REP_ARCHS:
        rep_dirs[group] = (out_dir, cfg)

# 對每個代表架構跑 rollout，留 Day 10 的預測
day10_step = 39
preds_day10 = {}
gt_day10 = test_y[0, day10_step, :, :, 0].numpy()

for group in REP_ARCHS:
    if group not in rep_dirs:
        print(f"  [略過 {group}] 找不到 canonical 實驗")
        continue
    out_dir, cfg = rep_dirs[group]
    weights_path = os.path.join(out_dir, 'model_weights_best.pt')
    if not os.path.exists(weights_path):
        weights_path = os.path.join(out_dir, 'model_weights.pt')
    if not os.path.exists(weights_path):
        continue

    model = build_model_for_group(group)
    if model is None:
        continue
    try:
        model.load_state_dict(torch.load(weights_path, map_location='cuda'))
    except Exception:
        del model; torch.cuda.empty_cache(); continue
    model.eval()

    x_in = test_x.clone().cuda()
    y_in = test_y.clone().cuda()
    with torch.no_grad():
        current = x_in
        for step in range(ROLLOUT_STEPS):
            pred = model(current)
            if step == day10_step:
                preds_day10[group] = pred[0, :, :, 0].cpu().numpy()
            if step < ROLLOUT_STEPS - 1:
                next_time = y_in[:, step, :, :, 4:]
                current = torch.cat([pred, next_time], dim=-1)

    del model
    torch.cuda.empty_cache()

# 對比圖：第一列顯示預測，第二列顯示誤差
n_archs = len(preds_day10)
fig, axes = plt.subplots(2, n_archs + 1, figsize=(5 * (n_archs + 1), 9))

# 第一列：True 與各架構預測
axes[0, 0].imshow(gt_day10, cmap='jet', extent=geo_extent, aspect='auto')
axes[0, 0].set_title('Ground Truth (Day 10)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Predictions\nLatitude (°)', fontsize=10)
fig.colorbar(axes[0, 0].images[0], ax=axes[0, 0])

for j, group in enumerate([g for g in REP_ARCHS if g in preds_day10]):
    pred = preds_day10[group]
    im = axes[0, j + 1].imshow(pred, cmap='jet', extent=geo_extent, aspect='auto')
    axes[0, j + 1].set_title(f'{group}', fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=axes[0, j + 1])

# 第二列：誤差（True - Pred）
axes[1, 0].axis('off')
axes[1, 0].text(0.5, 0.5, 'Errors\n(Ground Truth - Pred)', ha='center', va='center',
                 fontsize=12, fontweight='bold', transform=axes[1, 0].transAxes)
for j, group in enumerate([g for g in REP_ARCHS if g in preds_day10]):
    err = gt_day10 - preds_day10[group]
    im = axes[1, j + 1].imshow(err, cmap='coolwarm', extent=geo_extent, aspect='auto',
                                  vmin=-2, vmax=2)
    axes[1, j + 1].set_title(f'Error: {group}', fontsize=11)
    axes[1, j + 1].set_xlabel('Longitude (°)', fontsize=10)
    fig.colorbar(im, ax=axes[1, j + 1])

axes[1, 0].set_xlabel('')  # First col is text label, no xlabel needed
plt.suptitle('Day 10 (T+40) Temperature Prediction — Cross-Architecture Comparison',
             fontsize=14, fontweight='bold')
plt.tight_layout()
out_compare = os.path.join(COMPARE_DIR, 'cross_arch_day10_comparison.png')
plt.savefig(out_compare, dpi=300, bbox_inches='tight')
plt.close()
print(f"  [輸出] {out_compare}")

print("\n========== 全部完成！ ==========")
print(f"每個架構的 pretty plot：outputs/<arch>/weather_prediction_pretty.png")
print(f"跨架構對比圖：{out_compare}")
