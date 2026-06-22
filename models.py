"""
models.py — 把原本散在多支腳本裡的「全部 11 個架構」整合到同一個模組。

來源整併：
  - FNO 家族（FNO2d）          ← 原 fourier_2d.py
  - 平面 UNet / UNet++ / TransUNet ← 原 unet_baseline.py / unetpp_baseline.py / transunet_baseline.py
  - 純球面 UNet / UNet++ / TransUNet ← 原 sphere_*_baseline.py（+ sphere_blocks.py）

設計重點：
  1. 所有模型都用「channel-last」介面：輸入 (B, nlat, nlon, C_meteo+C_time)，
     模型內部自己 concat 2 個 grid 座標通道，輸出 (B, nlat, nlon, out_channels)。
     → 訓練迴圈完全不必管是哪個架構。
  2. 所有模型的 in_channels / out_channels 都是參數，不再寫死 → 支援 4 / 6 / 96 變數。
     in_channels = C_meteo + C_time + 2(grid)；out_channels = C_meteo。
  3. SHT 的 modes 只跟「空間解析度 (nlat,nlon)」有關，跟變數數量無關，
     所以 96 變數時球面模型一樣是 33×64 網格、不必改 modes。

torch_harmonics 在某些 Windows 環境需要 triton 補丁，補丁邏輯放在 fourier_2d.py 入口，
這裡假設 import torch_harmonics 之前該補丁（若需要）已套用。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# torch_harmonics 在舊 Windows 環境（torch 2.2、無 triton）需要假 triton 補丁。
# 正常情況下 fourier_2d.py 入口已先 import 過（補丁已套用 / 已快取）；
# 但若有人直接 import models（如 analysis 腳本），這裡自備同樣的後備邏輯。
try:
    import torch_harmonics as th
except Exception:
    import sys as _sys
    import importlib.util as _ilu
    from types import ModuleType as _MT

    _orig_find_spec = _ilu.find_spec

    def _hooked(name, package=None):
        if name == 'triton' or name.startswith('triton.'):
            return None
        return _orig_find_spec(name, package)
    _ilu.find_spec = _hooked

    class _MockTriton(_MT):
        def __getattr__(self, name):
            if name.startswith('__'):
                raise AttributeError(name)
            if name in ('jit', 'autotune', 'heuristics', 'jit_mutator'):
                return lambda *a, **k: (lambda f: f)
            return _MockTriton(name)

        def __call__(self, *a, **k):
            return _MockTriton("mock")

    _sys.modules['triton'] = _MockTriton('triton')
    _sys.modules['triton.language'] = _MockTriton('triton.language')
    import torch_harmonics as th


# ======================================================================
# 共用工具：grid 座標通道（所有模型 forward 內部都會呼叫）
# ======================================================================
def make_grid(shape, device):
    """產生 (B, nlat, nlon, 2) 的正規化經緯度座標通道。"""
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).to(device)


# ======================================================================
# === FNO 家族 building blocks（原 fourier_2d.py）===
# ======================================================================
class LocalUNetBlock2d(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.down = nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1)
        self.conv = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.final = nn.Conv2d(width, width, 1)

    def forward(self, x):
        res = x
        x_down = F.gelu(self.down(x))
        x_conv = F.gelu(self.conv(x_down))
        x_up = F.interpolate(x_conv, size=(res.shape[2], res.shape[3]),
                             mode='bilinear', align_corners=True)
        return self.final(x_up) + res


class SphericalConv2d(nn.Module):
    """Forward SHT → 複數權重相乘 → Inverse SHT（球面頻譜卷積）。"""
    def __init__(self, in_channels, out_channels, modes, nlat=33, nlon=64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.sht = th.RealSHT(nlat, nlon, lmax=modes, mmax=modes, grid="equiangular")
        self.isht = th.InverseRealSHT(nlat, nlon, lmax=modes, mmax=modes, grid="equiangular")
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, modes, dtype=torch.cfloat))

    def forward(self, x):
        x_sht = self.sht(x)
        out_sht = torch.einsum("bilm,iolm->bolm", x_sht, self.weights)
        return self.isht(out_sht)


class SpectralConv2d(nn.Module):
    """標準 2D FFT spectral conv（2D-FNO 用）。"""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class AdvancedUNetBlock2d(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.down1 = nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(width * 2, width * 2, kernel_size=3, padding=1)
        self.down2 = nn.Conv2d(width * 2, width * 2, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(width * 2, width * 2, kernel_size=3, padding=1)
        self.up1 = nn.Conv2d(width * 4, width * 2, kernel_size=3, padding=1)
        self.up2 = nn.Conv2d(width * 3, width, kernel_size=3, padding=1)
        self.final = nn.Conv2d(width, width, 1)

    def forward(self, x):
        res = x
        e1 = x
        d1 = F.gelu(self.down1(e1))
        c1 = F.gelu(self.conv1(d1))
        d2 = F.gelu(self.down2(c1))
        c2 = F.gelu(self.conv2(d2))
        u1 = F.interpolate(c2, size=(c1.shape[2], c1.shape[3]), mode='bilinear', align_corners=True)
        u1_conv = F.gelu(self.up1(torch.cat([u1, c1], dim=1)))
        u2 = F.interpolate(u1_conv, size=(e1.shape[2], e1.shape[3]), mode='bilinear', align_corners=True)
        u2_conv = F.gelu(self.up2(torch.cat([u2, e1], dim=1)))
        return self.final(u2_conv) + res


class ConvNeXtBlock2d(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.dwconv = nn.Conv2d(width, width, kernel_size=7, padding=3, groups=width)
        self.norm = nn.GroupNorm(1, width)
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


# ======================================================================
# === Transformer building blocks（TransUNet / sutrans_fno 共用）===
# ======================================================================
class TransformerBottleneck(nn.Module):
    """(B,C,H,W) → tokens + pos_emb → TransformerEncoder → 還原 (B,C,H,W)。"""
    def __init__(self, channels, max_tokens, n_layers=4, n_heads=4, ffn_mult=4, dropout=0.0):
        super().__init__()
        self.channels = channels
        self.max_tokens = max_tokens
        self.pos_emb = nn.Parameter(torch.zeros(1, max_tokens, channels))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=n_heads, dim_feedforward=channels * ffn_mult,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm_out = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        n_tokens = H * W
        assert n_tokens <= self.max_tokens, \
            f"序列長度 {n_tokens} 超過位置編碼上限 {self.max_tokens}"
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_emb[:, :n_tokens, :]
        x = self.transformer(x)
        x = self.norm_out(x)
        return x.transpose(1, 2).reshape(B, C, H, W)


class TransformerLocalBlock(nn.Module):
    """FNO local path 的 transformer 版本（sutrans_fno 用）。"""
    def __init__(self, width, n_layers=2, n_heads=4, dropout=0.0):
        super().__init__()
        self.down = nn.Conv2d(width, width, kernel_size=3, stride=4, padding=1)
        self.transformer = TransformerBottleneck(
            channels=width, max_tokens=200, n_layers=n_layers, n_heads=n_heads, dropout=dropout)

    def forward(self, x):
        H_in, W_in = x.shape[2], x.shape[3]
        z = self.down(x)
        z = self.transformer(z)
        return F.interpolate(z, size=(H_in, W_in), mode='bilinear', align_corners=True)


# ======================================================================
# === FNO2d：FNO 家族通用骨幹（2d_fno / 2d_ufno / sfno / sufno / sunetpp_fno
#            / sutrans_fno / 2d_unet 都由它組出來）===
# ======================================================================
class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width,
                 in_channels, out_channels,
                 local_type='1x1', spectral_type='sht', dropout=0.0,
                 nlat=33, nlon=64):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.nlat = nlat
        self.nlon = nlon
        self.fc0 = nn.Linear(in_channels, self.width)
        self.local_type = local_type
        self.spectral_type = spectral_type
        self.dropout_p = dropout

        self.conv0 = self._get_spectral_path()
        self.conv1 = self._get_spectral_path()
        self.conv2 = self._get_spectral_path()
        self.conv3 = self._get_spectral_path()
        self.w0 = self._get_local_path()
        self.w1 = self._get_local_path()
        self.w2 = self._get_local_path()
        self.w3 = self._get_local_path()

        self.dropout_layer = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def _get_spectral_path(self):
        if self.spectral_type == 'sht':
            return SphericalConv2d(self.width, self.width, self.modes1, nlat=self.nlat, nlon=self.nlon)
        elif self.spectral_type == 'fft':
            return SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        elif self.spectral_type in ('', 'none'):
            return None
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
            return TransformerLocalBlock(self.width, n_layers=2, n_heads=4, dropout=self.dropout_p)
        elif self.local_type == 'none':
            return None

    def _mix(self, conv, w, x):
        if conv is not None and w is not None:
            return conv(x) + w(x)
        if conv is not None:
            return conv(x)
        return w(x)

    def forward(self, x):
        grid = make_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x = self._mix(self.conv0, self.w0, x); x = F.gelu(x); x = self.dropout_layer(x)
        x = self._mix(self.conv1, self.w1, x); x = F.gelu(x); x = self.dropout_layer(x)
        x = self._mix(self.conv2, self.w2, x); x = F.gelu(x); x = self.dropout_layer(x)
        x = self._mix(self.conv3, self.w3, x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


# ======================================================================
# === 平面 UNet / UNet++ / TransUNet building blocks ===
# ======================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """bilinear upsample + skip concat + DoubleConv（平面 UNet 用）。"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet2DRollout(nn.Module):
    """平面 UNet + 經度 circular padding（對應實驗名 unet_2d）。"""
    def __init__(self, in_channels=10, out_channels=4, base_width=32, lon_pad=4):
        super().__init__()
        self.lon_pad = lon_pad
        self.inc = DoubleConv(in_channels, base_width)
        self.down1 = Down(base_width, base_width * 2)
        self.down2 = Down(base_width * 2, base_width * 4)
        self.down3 = Down(base_width * 4, base_width * 8)
        self.up1 = Up(base_width * 8 + base_width * 4, base_width * 4, bilinear=True)
        self.up2 = Up(base_width * 4 + base_width * 2, base_width * 2, bilinear=True)
        self.up3 = Up(base_width * 2 + base_width, base_width, bilinear=True)
        self.outc = nn.Conv2d(base_width, out_channels, kernel_size=1)

    def forward(self, x):
        grid = make_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, (self.lon_pad, self.lon_pad, 0, 0), mode='circular')
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        x = x[..., self.lon_pad:-self.lon_pad]
        return x.permute(0, 2, 3, 1)


class UNetPlusPlus2DRollout(nn.Module):
    """平面 UNet++（nested skip）+ 經度 circular padding（對應 unetpp_2d）。"""
    def __init__(self, in_channels=10, out_channels=4, base=32, lon_pad=4):
        super().__init__()
        self.lon_pad = lon_pad
        ch = [base, base * 2, base * 4, base * 8]
        self.pool = nn.MaxPool2d(2)
        self.x00 = DoubleConv(in_channels, ch[0])
        self.x10 = DoubleConv(ch[0], ch[1])
        self.x20 = DoubleConv(ch[1], ch[2])
        self.x30 = DoubleConv(ch[2], ch[3])
        self.x01 = DoubleConv(ch[0] + ch[1], ch[0])
        self.x11 = DoubleConv(ch[1] + ch[2], ch[1])
        self.x21 = DoubleConv(ch[2] + ch[3], ch[2])
        self.x02 = DoubleConv(ch[0] * 2 + ch[1], ch[0])
        self.x12 = DoubleConv(ch[1] * 2 + ch[2], ch[1])
        self.x03 = DoubleConv(ch[0] * 3 + ch[1], ch[0])
        self.outc = nn.Conv2d(ch[0], out_channels, kernel_size=1)

    @staticmethod
    def _up_match(x_low, x_high):
        return F.interpolate(x_low, size=x_high.shape[-2:], mode='bilinear', align_corners=True)

    def forward(self, x):
        grid = make_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, (self.lon_pad, self.lon_pad, 0, 0), mode='circular')
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))
        x01 = self.x01(torch.cat([x00, self._up_match(x10, x00)], dim=1))
        x11 = self.x11(torch.cat([x10, self._up_match(x20, x10)], dim=1))
        x21 = self.x21(torch.cat([x20, self._up_match(x30, x20)], dim=1))
        x02 = self.x02(torch.cat([x00, x01, self._up_match(x11, x00)], dim=1))
        x12 = self.x12(torch.cat([x10, x11, self._up_match(x21, x10)], dim=1))
        x03 = self.x03(torch.cat([x00, x01, x02, self._up_match(x12, x00)], dim=1))
        out = self.outc(x03)
        out = out[..., self.lon_pad:-self.lon_pad]
        return out.permute(0, 2, 3, 1)


class TransUNetUp(nn.Module):
    """TransUNet 專用 Up（與平面 UNet 的 Up 略有不同，mid=in//2）。"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class TransUNet2DRollout(nn.Module):
    """平面 CNN encoder + Transformer bottleneck + CNN decoder（對應 transunet_2d）。"""
    def __init__(self, in_channels=10, out_channels=4, base_width=32, lon_pad=4,
                 trans_layers=4, trans_heads=4, trans_dropout=0.0):
        super().__init__()
        self.lon_pad = lon_pad
        self.inc = DoubleConv(in_channels, base_width)
        self.down1 = Down(base_width, base_width * 2)
        self.down2 = Down(base_width * 2, base_width * 4)
        self.bottleneck = TransformerBottleneck(
            channels=base_width * 4, max_tokens=200,
            n_layers=trans_layers, n_heads=trans_heads, dropout=trans_dropout)
        self.up1 = TransUNetUp(base_width * 4 + base_width * 2, base_width * 2)
        self.up2 = TransUNetUp(base_width * 2 + base_width, base_width)
        self.outc = nn.Conv2d(base_width, out_channels, kernel_size=1)

    def forward(self, x):
        grid = make_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, (self.lon_pad, self.lon_pad, 0, 0), mode='circular')
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = self.bottleneck(x3)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        x = x[..., self.lon_pad:-self.lon_pad]
        return x.permute(0, 2, 3, 1)


# ======================================================================
# === 純球面 building blocks（原 sphere_blocks.py）===
# ======================================================================
class SHTDoubleConv(nn.Module):
    """(SHT → BN → ReLU) × 2，對照平面 DoubleConv。"""
    def __init__(self, in_channels, out_channels, nlat, nlon, modes, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.sht1 = SphericalConv2d(in_channels, mid_channels, modes, nlat, nlon)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.sht2 = SphericalConv2d(mid_channels, out_channels, modes, nlat, nlon)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.sht1(x)), inplace=True)
        x = F.relu(self.bn2(self.sht2(x)), inplace=True)
        return x


class SHTDown(nn.Module):
    def __init__(self, in_channels, out_channels, nlat, nlon, modes):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = SHTDoubleConv(in_channels, out_channels, nlat, nlon, modes)

    def forward(self, x):
        return self.conv(self.maxpool(x))


class SHTUp(nn.Module):
    def __init__(self, in_channels, out_channels, nlat, nlon, modes):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = SHTDoubleConv(in_channels, out_channels, nlat, nlon, modes,
                                  mid_channels=in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class SphereUNet2DRollout(nn.Module):
    """純 SHT 編解碼 UNet（對應 sphere_unet）。"""
    def __init__(self, in_channels=10, out_channels=4, base_width=32,
                 nlat=33, nlon=64, modes=(8, 4, 2)):
        super().__init__()
        m0, m1, m2 = modes
        self.inc = SHTDoubleConv(in_channels, base_width, nlat=nlat, nlon=nlon, modes=m0)
        self.down1 = SHTDown(base_width, base_width * 2, nlat=nlat // 2, nlon=nlon // 2, modes=m1)
        self.down2 = SHTDown(base_width * 2, base_width * 4, nlat=nlat // 4, nlon=nlon // 4, modes=m2)
        self.up1 = SHTUp(base_width * 4 + base_width * 2, base_width * 2,
                         nlat=nlat // 2, nlon=nlon // 2, modes=m1)
        self.up2 = SHTUp(base_width * 2 + base_width, base_width, nlat=nlat, nlon=nlon, modes=m0)
        self.outc = nn.Conv2d(base_width, out_channels, kernel_size=1)

    def forward(self, x):
        grid = make_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 3, 1, 2)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        return x.permute(0, 2, 3, 1)


class SphereUNetPlusPlus2DRollout(nn.Module):
    """純 SHT 編解碼 UNet++（對應 sphere_unetpp）。"""
    def __init__(self, in_channels=10, out_channels=4, base=32,
                 nlat=33, nlon=64, modes=(8, 4, 2)):
        super().__init__()
        m0, m1, m2 = modes
        ch = [base, base * 2, base * 4]
        self.pool = nn.MaxPool2d(2)
        self.x00 = SHTDoubleConv(in_channels, ch[0], nlat=nlat, nlon=nlon, modes=m0)
        self.x10 = SHTDoubleConv(ch[0], ch[1], nlat=nlat // 2, nlon=nlon // 2, modes=m1)
        self.x20 = SHTDoubleConv(ch[1], ch[2], nlat=nlat // 4, nlon=nlon // 4, modes=m2)
        self.x01 = SHTDoubleConv(ch[0] + ch[1], ch[0], nlat=nlat, nlon=nlon, modes=m0)
        self.x11 = SHTDoubleConv(ch[1] + ch[2], ch[1], nlat=nlat // 2, nlon=nlon // 2, modes=m1)
        self.x02 = SHTDoubleConv(ch[0] * 2 + ch[1], ch[0], nlat=nlat, nlon=nlon, modes=m0)
        self.outc = nn.Conv2d(ch[0], out_channels, kernel_size=1)

    @staticmethod
    def _up_match(x_low, x_high):
        return F.interpolate(x_low, size=x_high.shape[-2:], mode='bilinear', align_corners=True)

    def forward(self, x):
        grid = make_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 3, 1, 2)
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x01 = self.x01(torch.cat([x00, self._up_match(x10, x00)], dim=1))
        x11 = self.x11(torch.cat([x10, self._up_match(x20, x10)], dim=1))
        x02 = self.x02(torch.cat([x00, x01, self._up_match(x11, x00)], dim=1))
        out = self.outc(x02)
        return out.permute(0, 2, 3, 1)


class SphereTransUNet2DRollout(nn.Module):
    """純 SHT 編解碼 + Transformer bottleneck（對應 sphere_transunet）。"""
    def __init__(self, in_channels=10, out_channels=4, base_width=32,
                 nlat=33, nlon=64, modes=(8, 4, 2),
                 trans_layers=4, trans_heads=4, trans_dropout=0.0):
        super().__init__()
        m0, m1, m2 = modes
        self.inc = SHTDoubleConv(in_channels, base_width, nlat=nlat, nlon=nlon, modes=m0)
        self.down1 = SHTDown(base_width, base_width * 2, nlat=nlat // 2, nlon=nlon // 2, modes=m1)
        self.down2 = SHTDown(base_width * 2, base_width * 4, nlat=nlat // 4, nlon=nlon // 4, modes=m2)
        self.bottleneck = TransformerBottleneck(
            channels=base_width * 4, max_tokens=200,
            n_layers=trans_layers, n_heads=trans_heads, dropout=trans_dropout)
        self.up1 = SHTUp(base_width * 4 + base_width * 2, base_width * 2,
                         nlat=nlat // 2, nlon=nlon // 2, modes=m1)
        self.up2 = SHTUp(base_width * 2 + base_width, base_width, nlat=nlat, nlon=nlon, modes=m0)
        self.outc = nn.Conv2d(base_width, out_channels, kernel_size=1)

    def forward(self, x):
        grid = make_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 3, 1, 2)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = self.bottleneck(x3)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        return x.permute(0, 2, 3, 1)


# ======================================================================
# === 實驗註冊表（單一真相來源）===
# 11 個主架構 + 既有基線；family 決定用哪個骨幹，kind/local/spectral 是細部設定。
# ======================================================================
EXPERIMENTS = {
    # ---- 平面家族（circular padding，paper 的左欄）----
    'unet_2d':      {'family': 'planar', 'kind': 'unet',      'display': 'Pure UNet 2D (CNN Baseline + lon circular pad)'},
    'unetpp_2d':    {'family': 'planar', 'kind': 'unetpp',    'display': 'Pure UNet++ 2D (Nested CNN + lon circular pad)'},
    'transunet_2d': {'family': 'planar', 'kind': 'transunet', 'display': 'TransUNet 2D (CNN enc + Transformer bottleneck + CNN dec)'},

    # ---- FNO 混合家族（SHT ⊕ planar，paper 的中欄）----
    'sufno':        {'family': 'fno', 'local_type': 'unet',          'spectral_type': 'sht', 'display': 'SUFNO (Spherical + U-Net)'},
    'sunetpp_fno':  {'family': 'fno', 'local_type': 'advanced_unet', 'spectral_type': 'sht', 'display': 'SU-Net++ FNO (Spherical + Advanced U-Net)'},
    'sutrans_fno':  {'family': 'fno', 'local_type': 'transformer',   'spectral_type': 'sht', 'display': 'SU-Trans FNO (Spherical + Transformer local)'},

    # ---- 純球面家族（SHT-only，paper 的右欄）----
    'sphere_unet':      {'family': 'sphere', 'kind': 'unet',      'display': 'Sphere UNet (Pure SHT-based UNet)'},
    'sphere_unetpp':    {'family': 'sphere', 'kind': 'unetpp',    'display': 'Sphere UNet++ (Pure SHT-based Nested UNet)'},
    'sphere_transunet': {'family': 'sphere', 'kind': 'transunet', 'display': 'Sphere TransUNet (SHT enc/dec + Transformer bottleneck)'},

    # ---- 經典基線 ----
    '2d_fno': {'family': 'fno', 'local_type': '1x1', 'spectral_type': 'fft', 'display': '2D-FNO Baseline (FFT)'},
    'sfno':   {'family': 'fno', 'local_type': '1x1', 'spectral_type': 'sht', 'display': 'SFNO (Spherical Baseline)'},

    # ---- 六變數延伸實驗用到的額外組合 ----
    '2d_ufno': {'family': 'fno', 'local_type': 'unet', 'spectral_type': 'fft', 'display': '2D-UFNO (FFT + U-Net)'},
    '2d_unet': {'family': 'fno', 'local_type': 'unet', 'spectral_type': '',    'display': '2D U-Net Only (FNO scaffold, local-only)'},
}


def build_model(name, *, num_channels, time_channels=4, grid_channels=2,
                base_width=32, modes=16, dropout=0.0,
                nlat=33, nlon=64, sphere_modes=(8, 4, 2), lon_pad=4):
    """
    依實驗名稱建立模型。

    in_channels  = num_channels (氣象變數) + time_channels (時間編碼) + grid_channels (座標)
    out_channels = num_channels（只預測氣象變數，時間/座標通道在 rollout 時由 ground truth 補上）
    """
    if name not in EXPERIMENTS:
        raise ValueError(f"未知的實驗名稱 '{name}'。可用：{', '.join(EXPERIMENTS.keys())}")
    cfg = EXPERIMENTS[name]
    in_ch = num_channels + time_channels + grid_channels
    out_ch = num_channels
    fam = cfg['family']

    if fam == 'fno':
        return FNO2d(modes, modes, base_width,
                     in_channels=in_ch, out_channels=out_ch,
                     local_type=cfg['local_type'], spectral_type=cfg['spectral_type'],
                     dropout=dropout, nlat=nlat, nlon=nlon)

    if fam == 'planar':
        kind = cfg['kind']
        if kind == 'unet':
            return UNet2DRollout(in_ch, out_ch, base_width, lon_pad)
        if kind == 'unetpp':
            return UNetPlusPlus2DRollout(in_ch, out_ch, base_width, lon_pad)
        if kind == 'transunet':
            return TransUNet2DRollout(in_ch, out_ch, base_width, lon_pad, trans_dropout=dropout)

    if fam == 'sphere':
        kind = cfg['kind']
        if kind == 'unet':
            return SphereUNet2DRollout(in_ch, out_ch, base_width, nlat, nlon, sphere_modes)
        if kind == 'unetpp':
            return SphereUNetPlusPlus2DRollout(in_ch, out_ch, base_width, nlat, nlon, sphere_modes)
        if kind == 'transunet':
            return SphereTransUNet2DRollout(in_ch, out_ch, base_width, nlat, nlon, sphere_modes,
                                            trans_dropout=dropout)

    raise ValueError(f"無法建立模型：{name}（family={fam}）")
