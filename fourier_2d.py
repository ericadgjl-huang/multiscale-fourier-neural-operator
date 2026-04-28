"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""
import torch_harmonics as th
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import xarray as xr   # <--- 補上這行

import matplotlib.pyplot as plt
import pandas as pd

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *

from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)

################################################################
# local path
################################################################
class LocalUNetBlock2d(nn.Module):
    def __init__(self, width):
        super(LocalUNetBlock2d, self).__init__()
        # 升級為 2D 卷積
        self.down = nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1)
        self.conv = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.final = nn.Conv2d(width, width, 1)

    def forward(self, x):
        res = x
        x_down = F.gelu(self.down(x))
        x_conv = F.gelu(self.conv(x_down))
        # 2D 上採樣，確保尺寸與輸入一致
        x_up = F.interpolate(x_conv, size=(res.shape[2], res.shape[3]), mode='bilinear', align_corners=True)
        return self.final(x_up) + res
################################################################
# SphericalConv2d
################################################################
class SphericalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, nlat=33, nlon=64):
        super(SphericalConv2d, self).__init__()
        """
        Spherical Harmonic Transform (SHT) Layer
        專為地球球面設計，取代傳統的 2D FFT。
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes # 球面調和函數的最高階數 (lmax, mmax)
        
        # 1. 宣告正向與反向的球面調和轉換 (SHT / ISHT)
        # ERA5 是標準的經緯度網格，所以我們使用 "equiangular"
        self.sht = th.RealSHT(nlat, nlon, lmax=modes, mmax=modes, grid="equiangular")
        self.isht = th.InverseRealSHT(nlat, nlon, lmax=modes, mmax=modes, grid="equiangular")
        
        # 2. 頻域上的可學習權重 (Complex Weights)
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, modes, dtype=torch.cfloat))

    def forward(self, x):
        # 正向 SHT：將空間氣象場轉為「球面頻譜」
        # x shape: (batch, in_channels, nlat, nlon)
        x_sht = self.sht(x) 
        
        # 在頻譜空間中進行矩陣相乘 (過濾與特徵提取)
        # out shape: (batch, out_channels, lmax, mmax)
        out_sht = torch.einsum("b i l m, i o l m -> b o l m", x_sht, self.weights)
        
        # 反向 ISHT：將處理好的頻譜轉回「空間氣象場」
        x = self.isht(out_sht)
        return x
    

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class AdvancedUNetBlock2d(nn.Module):
    def __init__(self, width):
        super(AdvancedUNetBlock2d, self).__init__()
        # --- Encoder (編碼器：提取深層特徵，通道數倍增) ---
        self.down1 = nn.Conv2d(width, width*2, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(width*2, width*2, kernel_size=3, padding=1)

        self.down2 = nn.Conv2d(width*2, width*2, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(width*2, width*2, kernel_size=3, padding=1)

        # --- Decoder (解碼器：結合淺層輪廓與深層語意) ---
        # 接收 width*2 (來自上採樣) + width*2 (來自 Encoder) = width*4
        self.up1 = nn.Conv2d(width*4, width*2, kernel_size=3, padding=1) 
        # 接收 width*2 (來自上採樣) + width (來自最原來的輸入) = width*3
        self.up2 = nn.Conv2d(width*3, width, kernel_size=3, padding=1)   

        self.final = nn.Conv2d(width, width, 1)

    def forward(self, x):
        res = x  # 最外層的殘差

        # --- 下採樣路徑 ---
        e1 = x 
        d1 = F.gelu(self.down1(e1))
        c1 = F.gelu(self.conv1(d1)) # 縮小 1/2

        d2 = F.gelu(self.down2(c1))
        c2 = F.gelu(self.conv2(d2)) # 縮小 1/4

        # --- 上採樣與特徵拼接 (Skip Connection) ---
        # 放大回 1/2，並與 c1 拼接
        u1 = F.interpolate(c2, size=(c1.shape[2], c1.shape[3]), mode='bilinear', align_corners=True)
        concat1 = torch.cat([u1, c1], dim=1)
        u1_conv = F.gelu(self.up1(concat1))

        # 放大回原尺寸，並與 e1 拼接
        u2 = F.interpolate(u1_conv, size=(e1.shape[2], e1.shape[3]), mode='bilinear', align_corners=True)
        concat2 = torch.cat([u2, e1], dim=1)
        u2_conv = F.gelu(self.up2(concat2))

        return self.final(u2_conv) + res

class ConvNeXtBlock2d(nn.Module):
    def __init__(self, width):
        super(ConvNeXtBlock2d, self).__init__()
        # 1. Depthwise Convolution (超大 7x7 卷積核，不縮小圖片，捕捉廣域氣象特徵)
        self.dwconv = nn.Conv2d(width, width, kernel_size=7, padding=3, groups=width)
        
        # 2. Layer Normalization (氣象資料各變數差異大，Norm 能幫助穩定)
        self.norm = nn.GroupNorm(1, width) 
        
        # 3. Pointwise Convolution (特徵維度放大 4 倍再壓縮，這是 Transformer 的精髓)
        self.pwconv1 = nn.Conv2d(width, 4 * width, 1) 
        self.pwconv2 = nn.Conv2d(4 * width, width, 1) 

    def forward(self, x):
        res = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = F.gelu(x)
        x = self.pwconv2(x)
        return x + res

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width, local_type='1x1'):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        # self.padding = 9 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(10, self.width) # input channel is 10: (a(x, y), x, y, x^2, y^2, xy, day_sin, day_cos, hour_sin, hour_cos)
        self.local_type = local_type # 儲存實驗開關

        # 換上全新的球面調和引擎！注意參數只傳入 modes1 即可
        self.conv0 = SphericalConv2d(self.width, self.width, self.modes1)
        self.conv1 = SphericalConv2d(self.width, self.width, self.modes1)
        self.conv2 = SphericalConv2d(self.width, self.width, self.modes1)
        self.conv3 = SphericalConv2d(self.width, self.width, self.modes1)
        self.w0 = self._get_local_path()
        self.w1 = self._get_local_path()
        self.w2 = self._get_local_path()
        self.w3 = self._get_local_path()

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 4)

    def _get_local_path(self):
        if self.local_type == '1x1':
            return nn.Conv2d(self.width, self.width, 1)
        elif self.local_type == 'unet':
            return LocalUNetBlock2d(self.width)
        # --- 新增下面這兩行 ---
        elif self.local_type == 'advanced_unet':
            return AdvancedUNetBlock2d(self.width)
        elif self.local_type == 'convnext':
            return ConvNeXtBlock2d(self.width)
        elif self.local_type == 'none':
            return None

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

################################################################
# 讀取 ERA5 氣象資料與設定 (取代原本的 configurations 與 read data)
################################################################
modes = 16    # 從 8 提升到 16 (因為空間變大了，最高其實可以設到 161//2=80，但 16 是一個兼具速度與精度的甜蜜點)
width = 32
batch_size = 4
epochs = 50

print("正在讀取 ERA5 氣象資料...")
# 請確保你的檔案名稱與路徑正確，如果不同請修改這裡
ds = xr.open_dataset('data/global_era5_mini_202301.nc', engine='h5netcdf') # 改成這顆迷你地球

# 提取資料並轉換成 PyTorch Tensor
t2m = torch.tensor(ds['t2m'].values)
msl = torch.tensor(ds['msl'].values)
u10 = torch.tensor(ds['u10'].values)
v10 = torch.tensor(ds['v10'].values)

# --- 幫模型裝上「時鐘」 (Time Embedding) ---
# 1. 抓出 NetCDF 裡面的時間戳記
times = ds['valid_time'].values
dt = pd.to_datetime(times)

# 2. 將 365 天與 24 小時轉換成圓周上的弧度
day_rad = torch.tensor(dt.dayofyear.values, dtype=torch.float32) * (2 * np.pi / 365.25)
hour_rad = torch.tensor(dt.hour.values, dtype=torch.float32) * (2 * np.pi / 24.0)

# 3. 算出 sin 和 cos，並將形狀擴張 (Broadcast) 到與氣象圖一樣的空間大小 (33, 64)
day_sin = torch.sin(day_rad).view(-1, 1, 1).expand(-1, 33, 64)
day_cos = torch.cos(day_rad).view(-1, 1, 1).expand(-1, 33, 64)
hour_sin = torch.sin(hour_rad).view(-1, 1, 1).expand(-1, 33, 64)
hour_cos = torch.cos(hour_rad).view(-1, 1, 1).expand(-1, 33, 64)

# 4. 把 4 個氣象變數 + 4 個時間變數，疊合成 8 個通道！
data = torch.stack([t2m, msl, u10, v10, day_sin, day_cos, hour_sin, hour_cos], dim=-1)
data = torch.nan_to_num(data, nan=0.0) # 填補可能的空值
# 建立預測序列：X 是現在 (0~122), Y 是未來 (1~123)
# x_data = data[:-1, :, :, :]
# y_data = data[1:, :, :, :]

# --- 動態多步滾動設定 ---
rollout_steps = 4  #我們先挑戰連續預測 4 步 (相當於未來 24 小時)
x_data = data[:-rollout_steps, :, :, :]

# 用 Python 的 List Comprehension 動態抓出未來每一步的答案
y_list = []
for i in range(1, rollout_steps + 1):
    if i == rollout_steps:
        y_list.append(data[i:, :, :, :])
    else:
        y_list.append(data[i : -rollout_steps + i, :, :, :])

# 將未來的答案堆疊在一起，維度變成 (batch, rollout_steps, lat, lon, channels)
y_data = torch.stack(y_list, dim=1) 

x_train, y_train = x_data[:100], y_data[:100]
x_test, y_test   = x_data[100:], y_data[100:]

# 切割訓練集 (前 100 筆) 與測試集 (後 23 筆)
x_train, y_train = x_data[:100], y_data[:100]
x_test, y_test   = x_data[100:], y_data[100:]

# 資料標準化 Normalization (氣象資料必做，否則數值差異太大無法收斂)
x_mean, x_std = x_train.mean(dim=(0, 1, 2), keepdim=True), x_train.std(dim=(0, 1, 2), keepdim=True)

# 統一使用 x_mean 與 x_std 進行正規化，因為 x 是 4D，y 是 5D，PyTorch 會自動從右邊對齊完美廣播！
x_train = (x_train - x_mean) / (x_std + 1e-6)
x_test  = (x_test - x_mean) / (x_std + 1e-6)
y_train = (y_train - x_mean) / (x_std + 1e-6)
y_test  = (y_test - x_mean) / (x_std + 1e-6)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

################################################################
# 訓練與評估 (取代原本的 training and evaluation)
################################################################
# 這裡切換為 unet，準備跑你的 U-FNO 創新組
model = FNO2d(modes, modes, width, local_type='advanced_unet')
print(f"模型總參數數量: {count_params(model)}")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# 自動判斷實驗名稱
if model.local_type == 'unet':
    model_name = "U-FNO (極簡版)"
elif model.local_type == 'advanced_unet':
    model_name = "Advanced U-FNO (深層拼接版)"
elif model.local_type == 'convnext':
    model_name = "ConvNeXt-FNO (大卷積核版)"
else:
    model_name = "FNO Baseline (1x1對照組)"

print(f"========================================")
print(f" 正在啟動訓練：{model_name}")
print(f" 局部路徑類型：{model.local_type}")
print(f" 模型總參數：{count_params(model)}")
print(f"========================================")

# --- 新增：準備紀錄學習曲線的陣列 ---
history_train_mse = []
history_test_mse = []

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    for x, y in train_loader:
        x, y = x, y
        optimizer.zero_grad()
        
        # --- 動態自迴歸滾動預測 ---
        current_input = x
        total_loss = 0
        
        for step in range(rollout_steps):
            # 1. 預測下一步的氣象 (輸出只有 4 個通道)
            pred_weather = model(current_input) 
            
            # 2. 只抽出標準答案的「前 4 個氣象變數」來算 Loss
            true_weather = y[:, step, :, :, :4]
            total_loss += F.mse_loss(pred_weather, true_weather) 
            
            # 3. 準備下一次滾動的輸入！(結合 預測天氣 + 未來時間)
            if step < rollout_steps - 1:
                # 拿出下一步的「真實時間」通道 (索引 4~7)
                next_time_features = y[:, step, :, :, 4:]
                # 將「預測的氣象」與「未來的時間」像三明治一樣疊起來，成為完整的 8 通道輸入
                current_input = torch.cat([pred_weather, next_time_features], dim=-1)
            
        # 總分加總，一起做反向傳播
        total_loss.backward()
        optimizer.step()
        
        train_mse += total_loss.item()

    scheduler.step()
    
    # --- 測試迴圈 ---
    model.eval()
    test_mse = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cpu(), y.cpu()
            
            current_input = x
            for step in range(rollout_steps):
                # 預測氣象 (4通道)
                pred_weather = model(current_input)
                # 拿答案的前 4 個氣象通道算誤差
                true_weather = y[:, step, :, :, :4]
                test_mse += F.mse_loss(pred_weather, true_weather).item()
                
                # 準備下一步的輸入
                if step < rollout_steps - 1:
                    next_time_features = y[:, step, :, :, 4:]
                    current_input = torch.cat([pred_weather, next_time_features], dim=-1)

    train_mse /= len(train_loader)
    test_mse /= len(test_loader)
    t2 = default_timer()
    print(f"Epoch: {ep} | 耗時: {t2-t1:.2f}s | Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f}")

    # --- 新增：把每一圈的分數存起來 ---
    history_train_mse.append(train_mse)
    history_test_mse.append(test_mse)
# ==========================================
# (原本的訓練迴圈到這裡結束，注意縮排要退回最外層！)
# ==========================================

# --- 新增：訓練結束後，繪製學習曲線 ---
plt.figure(figsize=(10, 6))
plt.plot(history_train_mse, label='Train MSE', linewidth=2)
plt.plot(history_test_mse, label='Test MSE', linewidth=2)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('MSE Loss', fontsize=14)
plt.title(f'Learning Curve - {model_name}', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("學習曲線繪製完成！請查看 learning_curve.png")
################################################################
# 5. 視覺化預測結果 (畫圖)
################################################################
import matplotlib.pyplot as plt

print("正在繪製預測結果...")
model.eval()
with torch.no_grad():
    # 抓取測試集的第一筆資料
    for x, y in test_loader:
        x, y = x.cpu(), y.cpu()
        
        # --- 動態滾動預測畫圖 ---
        current_input = x
        for step in range(rollout_steps):
            pred_weather = model(current_input)
            
            if step < rollout_steps - 1:
                next_time_features = y[:, step, :, :, 4:]
                current_input = torch.cat([pred_weather, next_time_features], dim=-1)
        
        final_pred = pred_weather # 儲存最後一步的結果
        break # 畫一筆就好

# 將 Tensor 轉回 CPU 上的 numpy 陣列以便畫圖
idx = 0
time_step = rollout_steps - 1 # 0代表第一步(T+1)

# 加上 [time_step] 降維，並只拿出前 4 個「氣象通道」的真實答案
ground_truth = y[idx, time_step, :, :, :4].cpu().numpy()

# 拿出最後一步的預測結果 
prediction = final_pred[idx].cpu().numpy()

# 氣象變數名稱對應 (通道 0:溫度, 1:氣壓, 2:U風, 3:V風)
# 氣象變數名稱對應 (通道 0:溫度, 1:氣壓, 2:U風, 3:V風)
# ==========================================
# (修改後的程式碼 - 替換原本同一個位置)
# ==========================================

# ... (中間這段程式碼保持不變，直到底下這幾行) ...

# 氣象變數名稱對應 (通道 0:溫度, 1:氣壓, 2:U風, 3:V風)
var_names = ['Temperature (t2m)', 'Pressure (msl)', 'U-Wind', 'V-Wind']

# --- 新增：計算誤差 (True - Pred) ---
error_map = ground_truth - prediction

# ------------------------------------------
# --- [新增/修改]: 計算地理座標範圍 (地理座標化) ---
# 1. 從原本的 Dataset ds 中抓出經緯度 array
lons = ds['longitude'].values # 範圍應該是 0 ~ 360
lats = ds['latitude'].values  # 範圍應該是 90 ~ -90

# 2. 定義繪圖的 extent [xmin, xmax, ymin, ymax]
# 注意：ERA5 的緯度通常是從 90 (北極) 降到 -90 (南極)，所以 ymin=-90, ymax=90。
g_extent = [lons.min(), lons.max(), lats.min(), lats.max()] # 例如 [0, 359.9, -90, 90]
# ------------------------------------------

# 改成 4 行 3 列 (True, Pred, Error)，把圖片加寬到 figsize=(15, 16)
# 在 for i in range(4): 迴圈正上方，直接定義這個範圍
# ERA5 數據標準範圍：經度 0~360, 緯度 90~-90
# --- 1. 先定義座標範圍 ---
g_extent = [0, 360, -90, 90] 

# --- 2. 繪圖區塊 ---
fig, axes = plt.subplots(4, 3, figsize=(15, 16))
for i in range(4):
    # 畫出真實答案
   # 1. 畫出真實答案 (Ground Truth)
    ax_gt = axes[i, 0]
    im_gt = ax_gt.imshow(ground_truth[:, :, i], cmap='jet', extent=g_extent, aspect='auto')
    ax_gt.set_title(f'True {var_names[i]} (T+{time_step+1})')
    ax_gt.set_ylabel('Latitude')
    fig.colorbar(im_gt, ax=ax_gt)

    # 2. 畫出模型預測 (Prediction)
    ax_pred = axes[i, 1]
    im_pred = ax_pred.imshow(prediction[:, :, i], cmap='jet', extent=g_extent, aspect='auto')
    ax_pred.set_title(f'Pred {var_names[i]} (T+{time_step+1})')
    ax_pred.set_xlabel('Longitude')
    fig.colorbar(im_pred, ax=ax_pred)
    
    # 3. 畫出誤差圖 (Error Map)
    ax_err = axes[i, 2]
    im_err = ax_err.imshow(error_map[:, :, i], cmap='coolwarm', extent=g_extent, aspect='auto')
    ax_err.set_title(f'Error (True - Pred) {var_names[i]}')
    fig.colorbar(im_err, ax=ax_err)

plt.tight_layout()
plt.savefig('weather_prediction.png')
print("繪圖完成！請查看專案資料夾下的 weather_prediction.png")

print("正在繪製 3D 球體預測圖...")

# 取出模型預測的溫度資料 (第 0 個通道)
temp_pred = prediction[:, :, 0]

# 1. 建立球面網格 (經度 0~2pi, 緯度 0~pi)
lon = np.linspace(0, 2 * np.pi, 64)
lat = np.linspace(0, np.pi, 33)
lon, lat = np.meshgrid(lon, lat)

# 2. 將球面座標 (lat, lon) 轉換為 3D 笛卡兒座標 (X, Y, Z)
X = np.sin(lat) * np.cos(lon)
Y = np.sin(lat) * np.sin(lon)
Z = np.cos(lat)

# 3. 將溫度資料標準化到 0~1，以便對應顏色表 (Colormap)
temp_norm = (temp_pred - temp_pred.min()) / (temp_pred.max() - temp_pred.min() + 1e-6)
# 生成顏色矩陣
colors = plt.cm.jet(temp_norm)

# 4. 繪製 3D 球體
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.axis('off')
surf = ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, antialiased=True, shade=False)
ax.set_title(f"Global Temp Prediction Step {rollout_steps}", fontsize=16, pad=20)
plt.savefig('weather_prediction_3d.png', dpi=300, bbox_inches='tight')
print("3D 繪圖完成！")