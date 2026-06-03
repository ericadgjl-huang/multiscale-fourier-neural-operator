"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""
# ======================================================================
# --- 終極版：Windows 系統專用 Triton 雙重攔截補丁 (解決 PyTorch .cuda() 衝突) ---
# ======================================================================
import sys
import importlib.util
from types import ModuleType

# 1. 攔截 PyTorch 內部的 importlib 探測，強制讓 PyTorch 知道 Windows 沒安裝 Triton，安全跳過核心註冊
orig_find_spec = importlib.util.find_spec
def hooked_find_spec(name, package=None):
    if name == 'triton' or name.startswith('triton.'):
        return None  # 告訴探測機制：沒這個東西
    return orig_find_spec(name, package)
importlib.util.find_spec = hooked_find_spec

# 2. 建立虛擬模組，供 torch_harmonics 頂層直接 import 時放行
class MockTriton(ModuleType):
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        if name in ('jit', 'autotune', 'heuristics', 'jit_mutator'):
            return lambda *args, **kwargs: (lambda f: f)
        return MockTriton(name)
    def __call__(self, *args, **kwargs):
        return MockTriton("mock")

sys.modules['triton'] = MockTriton('triton')
sys.modules['triton.language'] = MockTriton('triton.language')
# ======================================================================

import torch_harmonics as th
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import xarray as xr   # <--- 補上這行

import matplotlib.pyplot as plt
from matplotlib import font_manager
# --- 新增：強制載入 Windows 系統內的微軟正黑體 ---
font_path = r"C:\Windows\Fonts\msjh.ttc"
font_manager.fontManager.addfont(font_path)
# --- 新增：解決 Matplotlib 中文顯示為方塊的問題 ---
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 設定字體為微軟正黑體
plt.rcParams['axes.unicode_minus'] = False # 解決座標軸負號 (-) 變方塊的問題

import pandas as pd

import os
import json
import csv

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

################################################################
# Transformer building blocks（PR 5：sutrans_fno 用）
# 從 transunet_baseline.py 複製，避免 import 引發 transunet 整個訓練腳本執行
################################################################
class TransformerBottleneck(nn.Module):
    """
    把 (B, C, H, W) flatten 成 (B, H*W, C) tokens 餵給 transformer，
    再 reshape 回 (B, C, H, W)。pre-norm（norm_first=True）提升訓練穩定度。
    """
    def __init__(self, channels, max_tokens, n_layers=4, n_heads=4, ffn_mult=4, dropout=0.0):
        super().__init__()
        self.channels   = channels
        self.max_tokens = max_tokens

        self.pos_emb = nn.Parameter(torch.zeros(1, max_tokens, channels))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = channels,
            nhead           = n_heads,
            dim_feedforward = channels * ffn_mult,
            dropout         = dropout,
            activation      = 'gelu',
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm_out    = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        n_tokens   = H * W
        assert n_tokens <= self.max_tokens, \
            f"序列長度 {n_tokens} 超過位置編碼上限 {self.max_tokens}"

        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_emb[:, :n_tokens, :]
        x = self.transformer(x)
        x = self.norm_out(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


class TransformerLocalBlock(nn.Module):
    """
    FNO local path 的 transformer 版本：
    1. Down-sample 4× (stride-4 conv)：33×64 → 9×16 (144 tokens)
    2. Transformer self-attention at low resolution（attention 矩陣只有 144²）
    3. Bilinear upsample 回到原解析度
    記憶體量級與 TransUNet bottleneck 相同。
    """
    def __init__(self, width, n_layers=2, n_heads=4, dropout=0.0):
        super().__init__()
        self.down = nn.Conv2d(width, width, kernel_size=3, stride=4, padding=1)
        self.transformer = TransformerBottleneck(
            channels   = width,
            max_tokens = 200,
            n_layers   = n_layers,
            n_heads    = n_heads,
            dropout    = dropout,
        )

    def forward(self, x):
        H_in, W_in = x.shape[2], x.shape[3]
        z = self.down(x)
        z = self.transformer(z)
        z = F.interpolate(z, size=(H_in, W_in), mode='bilinear', align_corners=True)
        return z


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, local_type='1x1', spectral_type='sht', dropout=0.0):
        super(FNO2d, self).__init__()

        """
        4 層 spectral block + 4 層 local path 的混合結構。
        spectral_type: 'sht'（球面調和，SFNO 系列）或 'fft'（標準 2D-FNO）
        local_type:    '1x1' / 'unet' / 'advanced_unet' / 'convnext'
        dropout:       每個 spectral block GELU 後的 Dropout2d 機率（0 = 關閉，零開銷）
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(12, self.width)
        self.local_type    = local_type
        self.spectral_type = spectral_type
        self.dropout_p     = dropout

        # spectral path：FFT 或 SHT 二選一
        self.conv0 = self._get_spectral_path()
        self.conv1 = self._get_spectral_path()
        self.conv2 = self._get_spectral_path()
        self.conv3 = self._get_spectral_path()
        # local path：1x1 / U-Net / Advanced U-Net / ConvNeXt
        self.w0 = self._get_local_path()
        self.w1 = self._get_local_path()
        self.w2 = self._get_local_path()
        self.w3 = self._get_local_path()

        # Dropout（channel-wise，spatial）：dropout=0 為 Identity，零開銷
        self.dropout_layer = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 6)

    def _get_spectral_path(self):
        if self.spectral_type == 'sht':
            return SphericalConv2d(self.width, self.width, self.modes1)
        elif self.spectral_type == 'fft':
            return SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        else:
            raise ValueError(f"Unknown spectral_type: {self.spectral_type}")

    def _get_local_path(self):
        if self.local_type == '1x1':
            return nn.Conv2d(self.width, self.width, 1)
        elif self.local_type == 'unet':
            return LocalUNetBlock2d(self.width)
        elif self.local_type == 'advanced_unet':
            return AdvancedUNetBlock2d(self.width)
        elif self.local_type == 'convnext':
            return ConvNeXtBlock2d(self.width)
        elif self.local_type == 'transformer':
            return TransformerLocalBlock(self.width, n_layers=2, n_heads=4,
                                          dropout=self.dropout_p)
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
        x = self.dropout_layer(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.dropout_layer(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.dropout_layer(x)

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
# ERA5RolloutDataset：動態切片，避免預先建構龐大 y_data 張量
# 40 步版本若預先建構 y_data 會消耗 ~12 GB RAM，改用 Dataset 即時切片
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
        x = self.data[i]                                    # (33, 64, 8)
        y = self.data[i + 1 : i + 1 + self.rollout_steps]  # (rollout_steps, 33, 64, 8)
        return x, y

################################################################
# 實驗設定（PR 3：multi-seed + FNO hyperparam search 旋鈕）
################################################################
EXPERIMENTS = {
    '2d_fno':    {'local_type': '1x1',    'spectral_type': 'fft', 'display': '2D-FNO Baseline (FFT)'},
    '2d_ufno':   {'local_type': 'unet',   'spectral_type': 'fft', 'display': '2D-UFNO (FFT + U-Net)'}, # ←新增
    'sfno':      {'local_type': '1x1',    'spectral_type': 'sht', 'display': 'SFNO (Spherical Baseline)'},
    'sufno':     {'local_type': 'unet',   'spectral_type': 'sht', 'display': 'SUFNO (Spherical + U-Net)'},
    '2d_unet':   {'local_type': 'unet',   'spectral_type': '',    'display': '2D U-Net Only'},
}

# === 主要旋鈕（這 4 個變數決定要跑哪個實驗）===
base_experiment_name = '2d_fno'  # ← 從 EXPERIMENTS 挑一個架構
SEED                 = 0               # ← 改成 1, 2 跑多 seed 驗證
MODES                = 16              # ← FNO modes（搜尋時可改 24, 32）
DROPOUT              = 0.0             # ← FNO dropout（搜尋時可改 0.1, 0.2）

cfg = EXPERIMENTS[base_experiment_name]

# 自動產生 experiment_name 後綴（預設值不會加後綴 → 保持與舊 baseline 同名）
suffix_parts = []
if SEED != 0:
    suffix_parts.append(f's{SEED}')
if MODES != 16:
    suffix_parts.append(f'm{MODES}')
if DROPOUT > 0:
    suffix_parts.append(f'drop{int(round(DROPOUT*100))}')
suffix = ('_' + '_'.join(suffix_parts)) if suffix_parts else ''
experiment_name = base_experiment_name + suffix

output_dir = os.path.join('outputs', experiment_name)

# 防止意外覆蓋既有結果（如 outputs/sunetpp_fno/ 已是 PR 1+2 baseline）
if os.path.exists(os.path.join(output_dir, 'training_log.csv')):
    raise FileExistsError(
        f"\n[防覆蓋保護] 輸出資料夾 {output_dir} 已存在完整訓練紀錄！\n"
        f"如需重跑請先：\n"
        f"  1. 改 SEED / DROPOUT / MODES 變數產生新後綴，或\n"
        f"  2. 手動刪除 {output_dir} 整個資料夾"
    )

os.makedirs(output_dir, exist_ok=True)

# 設定 random seed（涵蓋 torch / numpy / cuda）
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

print(f"========================================")
print(f" 實驗：{experiment_name}  →  {cfg['display']}")
print(f" 輸出資料夾：{output_dir}")
print(f" SEED={SEED} | MODES={MODES} | DROPOUT={DROPOUT}")
print(f"========================================")

################################################################
# 讀取 ERA5 氣象資料與設定
################################################################
TRAIN_VARIABLES = ['t2m', 'msl', 'u10', 'v10', 'vimdf', 'vitoe']
NUM_CHANNELS    = len(TRAIN_VARIABLES) # 也就是 6

modes           = MODES   # 沿用 PR 3 旋鈕
width           = 32
batch_size      = 4
epochs          = 50
rollout_steps   = 40   # 10 天中期預測（6 小時/步 × 40 步 = 240 小時）
TBPTT_K         = 8    # Truncated BPTT 視窗：每 8 步截斷計算圖，防止 40 步梯度鏈導致記憶體爆炸
step_loss_gamma = 0.95 # 越遠的時步誤差權重遞減，避免遠期梯度淹沒近期學習訊號
lr              = 0.001
weight_decay    = 1e-4
clip_norm       = 1.0

print("正在讀取 ERA5 氣象資料...")
ds = xr.open_mfdataset('data/global_era5_6_factors_*.nc', engine='h5netcdf', combine='by_coords')

t2m = torch.tensor(ds['t2m'].values)
msl = torch.tensor(ds['msl'].values)
u10 = torch.tensor(ds['u10'].values)
v10 = torch.tensor(ds['v10'].values)
# ---- 新增下面兩行 (請根據上方 print 出來的名稱調整 key 值，通常為 vifd 與 vite) ----
vimdf = torch.tensor(ds['vimdf'].values)  # 垂直積分水分散度
vitoe = torch.tensor(ds['vitoe'].values)  # 總能量的垂直積分

times    = ds['valid_time'].values
dt       = pd.to_datetime(times)
day_rad  = torch.tensor(dt.dayofyear.values, dtype=torch.float32) * (2 * np.pi / 365.25)
hour_rad = torch.tensor(dt.hour.values,      dtype=torch.float32) * (2 * np.pi / 24.0)

day_sin  = torch.sin(day_rad).view(-1, 1, 1).expand(-1, 33, 64)
day_cos  = torch.cos(day_rad).view(-1, 1, 1).expand(-1, 33, 64)
hour_sin = torch.sin(hour_rad).view(-1, 1, 1).expand(-1, 33, 64)
hour_cos = torch.cos(hour_rad).view(-1, 1, 1).expand(-1, 33, 64)

# 修改後：把 6 個氣象變數堆疊在最前面，時間特徵接在後面
data = torch.stack([t2m, msl, u10, v10, vimdf, vitoe, day_sin, day_cos, hour_sin, hour_cos], dim=-1)
data = torch.nan_to_num(data, nan=0.0).float()

# 前兩年 (2021-2022) 訓練，最後一年 (2023) 測試
train_size = 2920
total_size = len(data)

# 標準化：僅用訓練集統計量，防止資料洩漏到測試集
x_mean    = data[:train_size].mean(dim=(0, 1, 2))  # (8,)
x_std     = data[:train_size].std(dim=(0, 1, 2))   # (8,)
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
model = FNO2d(modes, modes, width,
              local_type=cfg['local_type'],
              spectral_type=cfg['spectral_type'],
              dropout=DROPOUT).cuda()
print(f"模型總參數數量: {count_params(model)}")

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# CosineAnnealing 比 StepLR 更適合長步預測：學習率平滑衰減，避免後期震盪
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

model_name = cfg['display']

print(f"========================================")
print(f" 正在啟動訓練：{model_name}")
print(f" Spectral 引擎：{cfg['spectral_type'].upper()}")
print(f" Local 路徑：{cfg['local_type']}")
print(f" 預測步數：{rollout_steps} 步（{rollout_steps * 6 // 24} 天）")
print(f" T-BPTT 視窗：{TBPTT_K} 步")
print(f" 模型總參數：{count_params(model)}")
print(f"========================================")

# --- 把超參數快照寫進 config.json（之後可 reproduce） ---
config_snapshot = {
    'experiment_name':      experiment_name,
    'base_experiment_name': base_experiment_name,   # PR 3：multi-seed 分組用
    'display_name':         cfg['display'],
    'local_type':           cfg['local_type'],
    'spectral_type':        cfg['spectral_type'],
    'seed':                 SEED,                   # PR 3 旋鈕
    'modes':                modes,
    'dropout':              DROPOUT,                # PR 3 旋鈕
    'width':                width,
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

# --- 開啟 CSV 訓練紀錄（每 epoch 即時 append，意外中斷也能保留） ---
NUM_CHANNELS = 6
TRAIN_VARIABLES = ['t2m', 'msl', 'u10', 'v10', 'vimdf', 'vitoe']

# --- 開啟 CSV 訓練紀錄（每 epoch 即時 append，意外中斷也能保留） ---
csv_path = os.path.join(output_dir, 'training_log.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    header = ['epoch', 'train_mse', 'test_mse', 'lr', 'epoch_time_sec']
    for ch, var in enumerate(TRAIN_VARIABLES):
        header.append(f'train_mse_ch{ch}_{var}')
    for ch, var in enumerate(TRAIN_VARIABLES):
        header.append(f'test_mse_ch{ch}_{var}')
    csv.writer(f).writerow(header)


# 各時步損失加權係數（step 0 權重最高，越遠越輕）
step_weights = torch.tensor(
    [step_loss_gamma ** i for i in range(rollout_steps)], dtype=torch.float32
).cuda()

history_train_mse = []
history_test_mse  = []
best_test_mse     = float('inf')   # PR 3：追蹤最佳 epoch
best_epoch        = -1

for ep in range(epochs):
    model.train()
    t1        = default_timer()
    train_mse = 0.0
    
    train_mse_per_channel = np.zeros(NUM_CHANNELS)
    test_mse_per_channel = np.zeros(NUM_CHANNELS)

    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        current_input = x
        batch_mse     = 0.0

        # --- Truncated BPTT：每 TBPTT_K 步做一次梯度更新 ---
        for window_start in range(0, rollout_steps, TBPTT_K):
            window_end  = min(window_start + TBPTT_K, rollout_steps)
            optimizer.zero_grad()
            window_loss = torch.tensor(0.0, device=x.device)

            for step in range(window_start, window_end):
                pred_weather = model(current_input)
                true_weather = y[:, step, :, :, :NUM_CHANNELS]
    
                # 逐通道 MSE
                mse_per_channel = torch.mean((pred_weather - true_weather) ** 2, dim=(0, 1, 2))
    
                # 累加訓練集通道誤差
                train_mse_per_channel += mse_per_channel.detach().cpu().numpy()
    
                step_loss = step_weights[step] * torch.mean(mse_per_channel)
                window_loss = window_loss + step_loss
                batch_mse += torch.mean(mse_per_channel).item()
    
                # 修正此處原先為 15 個空格的縮排錯誤
                if step < rollout_steps - 1:
                    next_time = y[:, step, :, :, NUM_CHANNELS:]
                    current_input = torch.cat([pred_weather, next_time], dim=-1)

            window_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()

            # 在 T-BPTT 視窗邊界截斷計算圖
            current_input = current_input.detach()

        train_mse += batch_mse / rollout_steps

    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step()

    # --- 測試迴圈（無梯度，純前向推演）---
    model.eval()
    test_mse = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            current_input = x
            for step in range(rollout_steps):
                pred_weather = model(current_input)
                true_weather = y[:, step, :, :, :NUM_CHANNELS]  
                
                mse_per_channel = torch.mean((pred_weather - true_weather) ** 2, dim=(0, 1, 2))
                # 補上測試集逐通道誤差累加
                test_mse_per_channel += mse_per_channel.detach().cpu().numpy()
                test_mse += torch.mean(mse_per_channel).item()
                
                if step < rollout_steps - 1:
                    next_time     = y[:, step, :, :, NUM_CHANNELS:]  
                    current_input = torch.cat([pred_weather, next_time], dim=-1)

    train_mse /= len(train_loader)
    train_mse_per_channel /= (len(train_loader) * rollout_steps)

    test_mse /= (len(test_loader) * rollout_steps)
    test_mse_per_channel /= (len(test_loader) * rollout_steps)

    t2 = default_timer()
    epoch_time = t2 - t1

    # 輸出逐通道 MSE
    print(f"Epoch {ep:02d} | Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f}")
    print(f"  Per-channel train MSE: {[f'{x:.4f}' for x in train_mse_per_channel]}")
    print(f"  Per-channel test MSE:  {[f'{x:.4f}' for x in test_mse_per_channel]}")

    is_best = test_mse < best_test_mse
    if is_best:
        best_test_mse = test_mse
        best_epoch    = ep
        torch.save(model.state_dict(), os.path.join(output_dir, 'model_weights_best.pt'))

    marker = "  ← new best" if is_best else ""
    print(f"Epoch {ep:02d} | 耗時: {epoch_time:.1f}s | LR: {current_lr:.2e} | Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f}{marker}")

    history_train_mse.append(train_mse)
    history_test_mse.append(test_mse)

    # 將完整的逐通道數據寫入 CSV 紀錄中
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        row = [ep, train_mse, test_mse, current_lr, epoch_time]
        row.extend(train_mse_per_channel.tolist())
        row.extend(test_mse_per_channel.tolist())
        csv.writer(f).writerow(row)

# --- 訓練結束後儲存最終模型權重 ---
weights_path = os.path.join(output_dir, 'model_weights.pt')
torch.save(model.state_dict(), weights_path)
print(f"\n最終模型權重已儲存：{weights_path}")

# 把 best epoch 資訊補進 config.json
config_snapshot['best_epoch']        = best_epoch
config_snapshot['best_test_mse']     = best_test_mse
config_snapshot['final_test_mse']    = history_test_mse[-1]
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config_snapshot, f, indent=2, ensure_ascii=False)

# --- 學習曲線 ---
plt.figure(figsize=(10, 6))
plt.plot(history_train_mse, label='Train MSE', linewidth=2)
plt.plot(history_test_mse,  label='Test MSE',  linewidth=2)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('MSE Loss', fontsize=14)
plt.title(f'Learning Curve — {model_name} ({rollout_steps * 6 // 24}-Day Forecast)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'learning_curve.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"學習曲線繪製完成！請查看 {os.path.join(output_dir, 'learning_curve.png')}")

################################################################
# 視覺化 1：各預報時效分通道 RMSE（技巧分數圖）
################################################################
print("正在計算分通道預報技巧分數（RMSE vs Lead Time）...")

var_names_zh = ['溫度 t2m', '海平面氣壓 msl', 'U 風速', 'V 風速', '水分散度 vifd', '總能量 vite']
var_names_en = ['Temperature (t2m)', 'Pressure (msl)', 'U-Wind', 'V-Wind', 'Moisture Div (vifd)', 'Total Energy (vite)']
channel_step_rmse = np.zeros((6, rollout_steps))  

lead_hours = np.arange(1, rollout_steps + 1) * 6

n_skill_batches   = 0
model.eval()
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        current_input = x
        step_preds = []  # 修正：宣告儲存預測時間步的列表
        
        # 1. 執行 Rollout 並收集所有時間步的預測結果
        for step in range(rollout_steps):
            pred_weather = model(current_input)
            step_preds.append(pred_weather.detach().cpu())  # 修正：將預測結果存入
            
            if step < rollout_steps - 1:
                next_time = y[:, step, :, :, NUM_CHANNELS:]
                current_input = torch.cat([pred_weather, next_time], dim=-1)
                
        # 2. 計算各通道與時效的 RMSE
        for step in range(rollout_steps):
            true = y[:, step, :, :, :6].cpu().numpy()  
            pred = step_preds[step].numpy()
            for ch in range(6):  
                channel_step_rmse[ch, step] += np.sqrt(
                    np.mean((pred[:, :, :, ch] - true[:, :, :, ch]) ** 2)
                )
                
        n_skill_batches += 1
        if n_skill_batches >= 10:  # 前 10 批足夠評估趨勢
            break

channel_step_rmse /= n_skill_batches

fig, axes = plt.subplots(1, 6, figsize=(28, 5))
for ch, ax in enumerate(axes):
    ax.plot(lead_hours, channel_step_rmse[ch], linewidth=2, color=f'C{ch}')
    ax.set_title(f'{var_names_en[ch]} RMSE vs Lead Time', fontsize=11)
    ax.set_xlabel('Forecast Lead Time (hours)', fontsize=10)
    ax.set_ylabel('RMSE (normalized)', fontsize=10)
    ax.axvline(x=120, color='orange', linestyle='--', alpha=0.8, label='Day 5')
    ax.axvline(x=240, color='red',    linestyle='--', alpha=0.8, label='Day 10')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
plt.suptitle(f'Forecast Skill — {model_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'forecast_skill.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"技巧分數圖繪製完成！請查看 {os.path.join(output_dir, 'forecast_skill.png')}")

################################################################
# 視覺化 2：多時效誤差熱點圖（Day 1 / Day 3 / Day 7 / Day 10）
################################################################
print("正在繪製多時效誤差熱點圖...")

# 展示 4 個代表性時效（溫度通道）
target_steps  = [3, 11, 27, 39]   # 0-indexed：T+4/12/28/40 步
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
                next_time     = y[:, step, :, :, 6:]
                current_input = torch.cat([pred_weather, next_time], dim=-1)
        break  # 只畫第一批

idx = 0
fig, axes = plt.subplots(len(target_steps), 3, figsize=(15, len(target_steps) * 4))
for row, (ts, label) in enumerate(zip(target_steps, target_labels)):
    gt   = y[idx, ts, :, :, 0].cpu().numpy()           # 溫度通道 ground truth
    pred = all_preds[ts][idx, :, :, 0].cpu().numpy()   # 溫度通道預測
    err  = gt - pred

    im0 = axes[row, 0].imshow(gt,   cmap='jet');      axes[row, 0].set_title(f'True {label}');   fig.colorbar(im0, ax=axes[row, 0])
    im1 = axes[row, 1].imshow(pred, cmap='jet');      axes[row, 1].set_title(f'Pred {label}');   fig.colorbar(im1, ax=axes[row, 1])
    im2 = axes[row, 2].imshow(err,  cmap='coolwarm'); axes[row, 2].set_title(f'Error {label}');  fig.colorbar(im2, ax=axes[row, 2])

plt.suptitle(f'Temperature Prediction Error Maps — {model_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'weather_prediction.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"多時效誤差熱點圖繪製完成！請查看 {os.path.join(output_dir, 'weather_prediction.png')}")

################################################################
# 視覺化 3：3D 球體預測圖（Day 10 最終時效）
################################################################
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
ax.set_title(f"Global Temp Prediction — Day {rollout_steps * 6 // 24} Forecast", fontsize=16, pad=20)
plt.savefig(os.path.join(output_dir, 'weather_prediction_3d.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"3D 繪圖完成！請查看 {os.path.join(output_dir, 'weather_prediction_3d.png')}")
print(f"\n========== 全部完成！所有結果已儲存至 {output_dir}/ ==========")