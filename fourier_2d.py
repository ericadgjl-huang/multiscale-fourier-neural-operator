"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import xarray as xr   # <--- 補上這行

import matplotlib.pyplot as plt

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
        self.padding = 9 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(6, self.width) # input channel is 6: (a(x, y), x, y, x^2, y^2, xy)
        self.local_type = local_type # 儲存實驗開關

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
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
        elif self.local_type == 'none':
            return None
        else:
            raise ValueError("local_type must be '1x1', 'unet', or 'none'")

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

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

        x = x[..., :-self.padding, :-self.padding]
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
ds = xr.open_dataset('data/east_asia_era5_202301.nc', engine='h5netcdf')

# 提取資料並轉換成 PyTorch Tensor
t2m = torch.tensor(ds['t2m'].values)
msl = torch.tensor(ds['msl'].values)
u10 = torch.tensor(ds['u10'].values)
v10 = torch.tensor(ds['v10'].values)

# 將四個變數疊合成一個 tensor，形狀會是 (124, 21, 17, 4)
data = torch.stack([t2m, msl, u10, v10], dim=-1)
data = torch.nan_to_num(data, nan=0.0) # 填補可能的空值

# 建立預測序列：X 是現在 (0~122), Y 是未來 (1~123)
x_data = data[:-1, :, :, :]
y_data = data[1:, :, :, :]

# 切割訓練集 (前 100 筆) 與測試集 (後 23 筆)
x_train, y_train = x_data[:100], y_data[:100]
x_test, y_test   = x_data[100:], y_data[100:]

# 資料標準化 Normalization (氣象資料必做，否則數值差異太大無法收斂)
x_mean, x_std = x_train.mean(dim=(0, 1, 2), keepdim=True), x_train.std(dim=(0, 1, 2), keepdim=True)
y_mean, y_std = y_train.mean(dim=(0, 1, 2), keepdim=True), y_train.std(dim=(0, 1, 2), keepdim=True)

x_train = (x_train - x_mean) / (x_std + 1e-6)
x_test  = (x_test - x_mean) / (x_std + 1e-6)
y_train = (y_train - y_mean) / (y_std + 1e-6)
y_test  = (y_test - y_mean) / (y_std + 1e-6)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

################################################################
# 訓練與評估 (取代原本的 training and evaluation)
################################################################
# 這裡切換為 unet，準備跑你的 U-FNO 創新組
model = FNO2d(modes, modes, width, local_type='unet').cuda()
print(f"模型總參數數量: {count_params(model)}")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# 根據 local_type 自動判斷顯示名稱
model_name = "U-FNO (創新組)" if model.local_type == 'unet' else "FNO Baseline (對照組)"
if model.local_type == 'none': model_name = "Pure FNO (消融組)"

print(f"========================================")
print(f" 正在啟動訓練：{model_name}")
print(f" 局部路徑類型：{model.local_type}")
print(f" 模型總參數：{count_params(model)}")
print(f"========================================")

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        
        out = model(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        train_mse += loss.item()

    scheduler.step()
    
    model.eval()
    test_mse = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            test_mse += F.mse_loss(out, y).item()

    train_mse /= len(train_loader)
    test_mse /= len(test_loader)
    t2 = default_timer()
    print(f"Epoch: {ep} | 耗時: {t2-t1:.2f}s | Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f}")

################################################################
# 5. 視覺化預測結果 (畫圖)
################################################################
import matplotlib.pyplot as plt

print("正在繪製預測結果...")
model.eval()
with torch.no_grad():
    # 抓取測試集的第一筆資料
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        pred = model(x)
        break # 畫一筆就好

# 將 Tensor 轉回 CPU 上的 numpy 陣列以便畫圖
# 取 batch 中的第 0 筆資料
idx = 0
ground_truth = y[idx].cpu().numpy()
prediction = pred[idx].cpu().numpy()

# 氣象變數名稱對應 (通道 0:溫度, 1:氣壓, 2:U風, 3:V風)
var_names = ['Temperature (t2m)', 'Pressure (msl)', 'U-Wind', 'V-Wind']

fig, axes = plt.subplots(4, 2, figsize=(10, 16))
for i in range(4):
    # 畫出真實答案 (Ground Truth)
    ax_gt = axes[i, 0]
    im_gt = ax_gt.imshow(ground_truth[:, :, i], cmap='jet')
    ax_gt.set_title(f'True {var_names[i]}')
    fig.colorbar(im_gt, ax=ax_gt)

    # 畫出模型預測 (Prediction)
    ax_pred = axes[i, 1]
    im_pred = ax_pred.imshow(prediction[:, :, i], cmap='jet')
    ax_pred.set_title(f'Pred {var_names[i]}')
    fig.colorbar(im_pred, ax=ax_pred)

plt.tight_layout()
plt.savefig('weather_prediction.png')
print("繪圖完成！請查看專案資料夾下的 weather_prediction.png")