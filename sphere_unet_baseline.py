"""
sphere_unet_baseline.py
純球面 UNet baseline：UNet 結構 + 所有 conv 用 SphericalConv2d（SHT-based）取代。
不走 FNO 框架（沒有 spectral 與 local 並聯求和），是「真正純球面」的 UNet 對照組。

對標：unet_baseline.py (planar)
共用模組：sphere_blocks.py
"""
import os
import json
import csv

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

from timeit import default_timer
from utilities3 import *
from sphere_blocks import SHTDoubleConv, SHTDown, SHTUp

# Seeds 在實驗設定區段依 SEED 變數重設
torch.manual_seed(0)
np.random.seed(0)


################################################################
# SphereUNet2DRollout：3 層階層 + skip concat，全部 SHT-based
#
# 解析度路徑：33×64 → 16×32 → 8×16 → 16×32 → 33×64
# Modes:        8      4      2      4      8
# Channels:     32     64    128     64    32  (base=32)
################################################################
class SphereUNet2DRollout(nn.Module):
    def __init__(self, in_channels=10, out_channels=4, base_width=32,
                 nlat=33, nlon=64, modes=(8, 4, 2)):
        """
        in_channels  = 8 (氣象+時間) + 2 (grid) = 10
        out_channels = 4 (t2m, msl, u10, v10)
        nlat, nlon   = 33, 64（ERA5 mini）
        modes        = (level_0, level_1, level_2) 每層的 SHT modes
        """
        super().__init__()
        m0, m1, m2 = modes

        # Encoder
        self.inc   = SHTDoubleConv(in_channels, base_width,
                                    nlat=nlat, nlon=nlon, modes=m0)
        self.down1 = SHTDown(base_width, base_width * 2,
                              nlat=nlat // 2, nlon=nlon // 2, modes=m1)
        self.down2 = SHTDown(base_width * 2, base_width * 4,
                              nlat=nlat // 4, nlon=nlon // 4, modes=m2)

        # Decoder（in_channels = x_up + skip）
        self.up1 = SHTUp(base_width * 4 + base_width * 2, base_width * 2,
                         nlat=nlat // 2, nlon=nlon // 2, modes=m1)
        self.up2 = SHTUp(base_width * 2 + base_width, base_width,
                         nlat=nlat,      nlon=nlon,      modes=m0)

        # 1x1 投影到 4 個氣象通道（不需要 SHT）
        self.outc = nn.Conv2d(base_width, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, 33, 64, 8)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)              # (B, 33, 64, 10)
        x = x.permute(0, 3, 1, 2)                     # (B, 10, 33, 64)
        # 不需要 lon_pad — SHT 自然處理週期性

        x1 = self.inc(x)                              # (B, 32, 33, 64)
        x2 = self.down1(x1)                           # (B, 64, 16, 32)
        x3 = self.down2(x2)                           # (B, 128, 8, 16)

        x = self.up1(x3, x2)                          # (B, 64, 16, 32)
        x = self.up2(x, x1)                           # (B, 32, 33, 64)

        x = self.outc(x)                              # (B, 4, 33, 64)
        return x.permute(0, 2, 3, 1)                  # (B, 33, 64, 4)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


################################################################
# ERA5RolloutDataset
################################################################
class ERA5RolloutDataset(torch.utils.data.Dataset):
    def __init__(self, data, start_idx, count, rollout_steps):
        self.data = data
        self.start_idx = start_idx
        self.count = count
        self.rollout_steps = rollout_steps

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        i = self.start_idx + idx
        x = self.data[i]
        y = self.data[i + 1 : i + 1 + self.rollout_steps]
        return x, y


################################################################
# 實驗設定（PR 6：sphere_unet 純球面）
################################################################
base_experiment_name = 'sphere_unet'
display_name         = 'Sphere UNet (Pure SHT-based UNet, no FNO scaffold)'
SEED                 = 0   # ← 改成 1, 2 跑多 seed 驗證

suffix          = f'_s{SEED}' if SEED != 0 else ''
experiment_name = base_experiment_name + suffix
output_dir      = os.path.join('outputs', experiment_name)

if os.path.exists(os.path.join(output_dir, 'training_log.csv')):
    raise FileExistsError(
        f"\n[防覆蓋保護] {output_dir} 已有完整訓練紀錄！\n"
        f"如需重跑請改 SEED 或手動刪除該資料夾。"
    )

os.makedirs(output_dir, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

print(f"========================================")
print(f" 實驗：{experiment_name}  →  {display_name}")
print(f" 輸出資料夾：{output_dir}")
print(f" SEED={SEED}")
print(f"========================================")

base_width      = 32
batch_size      = 4
epochs          = 50
rollout_steps   = 40
TBPTT_K         = 8
step_loss_gamma = 0.95
lr              = 0.001
weight_decay    = 1e-4
clip_norm       = 1.0
modes_per_level = (8, 4, 2)   # (level 0, 1, 2)

################################################################
# 讀取 ERA5 資料
################################################################
print("正在讀取 ERA5 氣象資料...")
ds = xr.open_mfdataset('data/global_era5_mini_*.nc', engine='h5netcdf', combine='by_coords')
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

train_size = 2920
total_size = len(data)

x_mean    = data[:train_size].mean(dim=(0, 1, 2))
x_std     = data[:train_size].std(dim=(0, 1, 2))
data_norm = (data - x_mean) / (x_std + 1e-6)

train_dataset = ERA5RolloutDataset(data_norm, start_idx=0,
                                   count=train_size,
                                   rollout_steps=rollout_steps)
test_dataset  = ERA5RolloutDataset(data_norm, start_idx=train_size,
                                   count=total_size - train_size - rollout_steps,
                                   rollout_steps=rollout_steps)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

################################################################
# 訓練與評估
################################################################
model = SphereUNet2DRollout(
    in_channels=10, out_channels=4,
    base_width=base_width, nlat=33, nlon=64,
    modes=modes_per_level,
).cuda()
print(f"模型總參數數量: {count_params(model)}")

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

print(f"========================================")
print(f" 正在啟動訓練：{display_name}")
print(f" Base width：{base_width}")
print(f" SHT modes per level：{modes_per_level}")
print(f" 解析度：33×64 → 16×32 → 8×16 → ... → 33×64")
print(f" 預測步數：{rollout_steps} 步（{rollout_steps * 6 // 24} 天）")
print(f" T-BPTT 視窗：{TBPTT_K} 步")
print(f" 模型總參數：{count_params(model)}")
print(f"========================================")

config_snapshot = {
    'experiment_name':      experiment_name,
    'base_experiment_name': base_experiment_name,
    'display_name':         display_name,
    'arch_family':          'pure_spherical_unet',
    'seed':                 SEED,
    'base_width':           base_width,
    'modes_per_level':      list(modes_per_level),
    'batch_size':           batch_size,
    'epochs':               epochs,
    'rollout_steps':        rollout_steps,
    'TBPTT_K':              TBPTT_K,
    'step_loss_gamma':      step_loss_gamma,
    'lr':                   lr,
    'weight_decay':         weight_decay,
    'clip_norm':            clip_norm,
    'train_size':           train_size,
    'total_size':           total_size,
    'param_count':          count_params(model),
    'optimizer':            'Adam',
    'scheduler':            'CosineAnnealingLR',
}
config_path = os.path.join(output_dir, 'config.json')
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config_snapshot, f, indent=2, ensure_ascii=False)
print(f"超參數快照已寫入：{config_path}")

csv_path = os.path.join(output_dir, 'training_log.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    csv.writer(f).writerow(['epoch', 'train_mse', 'test_mse', 'lr', 'epoch_time_sec'])

step_weights = torch.tensor(
    [step_loss_gamma ** i for i in range(rollout_steps)], dtype=torch.float32
).cuda()

history_train_mse = []
history_test_mse  = []
best_test_mse     = float('inf')
best_epoch        = -1

for ep in range(epochs):
    model.train()
    t1        = default_timer()
    train_mse = 0.0

    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        current_input = x
        batch_mse     = 0.0

        for window_start in range(0, rollout_steps, TBPTT_K):
            window_end  = min(window_start + TBPTT_K, rollout_steps)
            optimizer.zero_grad()
            window_loss = torch.tensor(0.0, device=x.device)

            for step in range(window_start, window_end):
                pred_weather = model(current_input)
                true_weather = y[:, step, :, :, :4]
                step_loss    = step_weights[step] * F.mse_loss(pred_weather, true_weather)
                window_loss  = window_loss + step_loss
                batch_mse   += F.mse_loss(pred_weather, true_weather).item()

                if step < rollout_steps - 1:
                    next_time     = y[:, step, :, :, 4:]
                    current_input = torch.cat([pred_weather, next_time], dim=-1)

            window_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            current_input = current_input.detach()

        train_mse += batch_mse / rollout_steps

    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step()

    model.eval()
    test_mse = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            current_input = x
            for step in range(rollout_steps):
                pred_weather = model(current_input)
                true_weather = y[:, step, :, :, :4]
                test_mse    += F.mse_loss(pred_weather, true_weather).item()
                if step < rollout_steps - 1:
                    next_time     = y[:, step, :, :, 4:]
                    current_input = torch.cat([pred_weather, next_time], dim=-1)

    train_mse /= len(train_loader)
    test_mse  /= (len(test_loader) * rollout_steps)
    t2         = default_timer()
    epoch_time = t2 - t1

    is_best = test_mse < best_test_mse
    if is_best:
        best_test_mse = test_mse
        best_epoch    = ep
        torch.save(model.state_dict(), os.path.join(output_dir, 'model_weights_best.pt'))

    marker = "  ← new best" if is_best else ""
    print(f"Epoch {ep:02d} | 耗時: {epoch_time:.1f}s | LR: {current_lr:.2e} | "
          f"Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f}{marker}")

    history_train_mse.append(train_mse)
    history_test_mse.append(test_mse)

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([ep, train_mse, test_mse, current_lr, epoch_time])

torch.save(model.state_dict(), os.path.join(output_dir, 'model_weights.pt'))
print(f"\n最終模型權重已儲存：{os.path.join(output_dir, 'model_weights.pt')}")
print(f"最佳模型權重（epoch {best_epoch}, test_mse={best_test_mse:.4f}）："
      f"{os.path.join(output_dir, 'model_weights_best.pt')}")

config_snapshot['best_epoch']     = best_epoch
config_snapshot['best_test_mse']  = best_test_mse
config_snapshot['final_test_mse'] = history_test_mse[-1]
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config_snapshot, f, indent=2, ensure_ascii=False)

################################################################
# 視覺化（與其他 baseline 相同的 4 張圖）
################################################################
plt.figure(figsize=(10, 6))
plt.plot(history_train_mse, label='Train MSE', linewidth=2)
plt.plot(history_test_mse,  label='Test MSE',  linewidth=2)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('MSE Loss', fontsize=14)
plt.title(f'Learning Curve — {display_name} ({rollout_steps * 6 // 24}-Day Forecast)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'learning_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

print("正在計算分通道預報技巧分數...")
var_names_en = ['Temperature (t2m)', 'Pressure (msl)', 'U-Wind', 'V-Wind']
lead_hours   = np.arange(1, rollout_steps + 1) * 6

channel_step_rmse = np.zeros((4, rollout_steps))
n_skill_batches   = 0
model.eval()
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        current_input = x
        step_preds    = []
        for step in range(rollout_steps):
            pred_weather = model(current_input)
            step_preds.append(pred_weather.cpu())
            if step < rollout_steps - 1:
                next_time     = y[:, step, :, :, 4:]
                current_input = torch.cat([pred_weather, next_time], dim=-1)
        for step in range(rollout_steps):
            true = y[:, step, :, :, :4].cpu().numpy()
            pred = step_preds[step].numpy()
            for ch in range(4):
                channel_step_rmse[ch, step] += np.sqrt(
                    np.mean((pred[:, :, :, ch] - true[:, :, :, ch]) ** 2)
                )
        n_skill_batches += 1
        if n_skill_batches >= 10:
            break
channel_step_rmse /= n_skill_batches

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ch, ax in enumerate(axes):
    ax.plot(lead_hours, channel_step_rmse[ch], linewidth=2, color=f'C{ch}')
    ax.set_title(f'{var_names_en[ch]} RMSE vs Lead Time', fontsize=11)
    ax.set_xlabel('Forecast Lead Time (hours)', fontsize=10)
    ax.set_ylabel('RMSE (normalized)', fontsize=10)
    ax.axvline(x=120, color='orange', linestyle='--', alpha=0.8, label='Day 5')
    ax.axvline(x=240, color='red',    linestyle='--', alpha=0.8, label='Day 10')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
plt.suptitle(f'Forecast Skill — {display_name}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'forecast_skill.png'), dpi=300, bbox_inches='tight')
plt.close()

print("正在繪製多時效誤差熱點圖...")
target_steps  = [3, 11, 27, 39]
target_labels = ['Day 1 (T+4)', 'Day 3 (T+12)', 'Day 7 (T+28)', 'Day 10 (T+40)']
model.eval()
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        current_input = x
        all_preds     = []
        for step in range(rollout_steps):
            pred_weather = model(current_input)
            all_preds.append(pred_weather)
            if step < rollout_steps - 1:
                next_time     = y[:, step, :, :, 4:]
                current_input = torch.cat([pred_weather, next_time], dim=-1)
        break

idx = 0
fig, axes = plt.subplots(len(target_steps), 3, figsize=(15, len(target_steps) * 4))
for row, (ts, label) in enumerate(zip(target_steps, target_labels)):
    gt   = y[idx, ts, :, :, 0].cpu().numpy()
    pred = all_preds[ts][idx, :, :, 0].cpu().numpy()
    err  = gt - pred
    im0 = axes[row, 0].imshow(gt,   cmap='jet');      axes[row, 0].set_title(f'True {label}');   fig.colorbar(im0, ax=axes[row, 0])
    im1 = axes[row, 1].imshow(pred, cmap='jet');      axes[row, 1].set_title(f'Pred {label}');   fig.colorbar(im1, ax=axes[row, 1])
    im2 = axes[row, 2].imshow(err,  cmap='coolwarm'); axes[row, 2].set_title(f'Error {label}');  fig.colorbar(im2, ax=axes[row, 2])
plt.suptitle(f'Temperature Prediction Error Maps — {display_name}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'weather_prediction.png'), dpi=300, bbox_inches='tight')
plt.close()

print("正在繪製 3D 球體預測圖...")
temp_pred = all_preds[-1][idx, :, :, 0].cpu().numpy()
lon = np.linspace(0, 2 * np.pi, 64)
lat = np.linspace(0, np.pi, 33)
lon, lat = np.meshgrid(lon, lat)
X = np.sin(lat) * np.cos(lon)
Y = np.sin(lat) * np.sin(lon)
Z = np.cos(lat)
temp_norm = (temp_pred - temp_pred.min()) / (temp_pred.max() - temp_pred.min() + 1e-6)
colors    = plt.cm.jet(temp_norm)
fig = plt.figure(figsize=(10, 10))
ax  = fig.add_subplot(111, projection='3d')
ax.axis('off')
ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, antialiased=True, shade=False)
ax.set_title(f"Global Temp Prediction — Day {rollout_steps * 6 // 24} Forecast", fontsize=15, pad=20)
plt.savefig(os.path.join(output_dir, 'weather_prediction_3d.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n========== 全部完成！所有結果已儲存至 {output_dir}/ ==========")
