"""
sphere_blocks.py
共用的球面（SHT-based）building blocks，給 sphere_unet / sphere_unetpp / sphere_transunet 用。

設計原則：
- SHT 完全取代平面 Conv2d 3×3（不混搭 FFT/平面 conv，與 FNO 框架徹底分離）
- 階層式 encoder/decoder 透過 MaxPool2d / Bilinear upsample 實現
- Modes 隨解析度遞減（底層用較少 modes，因為 lmax ≤ nlat-1）
- 不需要 lon_pad：SHT 本身就把球面當週期處理
- BatchNorm + ReLU 跟 unet_baseline.py 保持一致，方便對照
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_harmonics as th


################################################################
# SphericalConv2d（從 fourier_2d.py 複製，避免 import 整段訓練腳本）
################################################################
class SphericalConv2d(nn.Module):
    """
    Forward SHT → complex 權重乘 → Inverse SHT。
    在球面上做「卷積」，數學上等同 FNO 的 spectral path 但在球面而非平面。
    """
    def __init__(self, in_channels, out_channels, modes, nlat, nlon):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes        = modes
        self.sht  = th.RealSHT(nlat, nlon, lmax=modes, mmax=modes, grid="equiangular")
        self.isht = th.InverseRealSHT(nlat, nlon, lmax=modes, mmax=modes, grid="equiangular")
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        x_sht = self.sht(x)
        out_sht = torch.einsum("bilm,iolm->bolm", x_sht, self.weights)
        return self.isht(out_sht)


################################################################
# SHTDoubleConv：取代 UNet 的 DoubleConv
################################################################
class SHTDoubleConv(nn.Module):
    """
    (SHT → BN → ReLU) × 2。
    跟 unet_baseline.py 的 DoubleConv 完全平行，只是內部 conv 換成 SHT。
    """
    def __init__(self, in_channels, out_channels, nlat, nlon, modes, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.sht1 = SphericalConv2d(in_channels, mid_channels, modes, nlat, nlon)
        self.bn1  = nn.BatchNorm2d(mid_channels)
        self.sht2 = SphericalConv2d(mid_channels, out_channels, modes, nlat, nlon)
        self.bn2  = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.sht1(x)), inplace=True)
        x = F.relu(self.bn2(self.sht2(x)), inplace=True)
        return x


class SHTDown(nn.Module):
    """MaxPool 2x2 + SHTDoubleConv。下採樣到半解析度。"""
    def __init__(self, in_channels, out_channels, nlat, nlon, modes):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv    = SHTDoubleConv(in_channels, out_channels, nlat, nlon, modes)

    def forward(self, x):
        return self.conv(self.maxpool(x))


class SHTUp(nn.Module):
    """
    Bilinear upsample + skip concat + SHTDoubleConv。
    奇數尺寸對齊用 F.pad 處理（從 unet_baseline.py 沿用）。
    SHT 在 padded 後的 (nlat, nlon) 上運作。
    """
    def __init__(self, in_channels, out_channels, nlat, nlon, modes):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = SHTDoubleConv(in_channels, out_channels, nlat, nlon, modes,
                                   mid_channels=in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


################################################################
# TransformerBottleneck（給 sphere_transunet 用，與 transunet_baseline.py 相同）
################################################################
class TransformerBottleneck(nn.Module):
    """
    (B, C, H, W) → flatten → tokens + pos_emb → transformer encoder → reshape 回 (B, C, H, W)
    Pre-norm 架構（norm_first=True）以提升訓練穩定度。
    """
    def __init__(self, channels, max_tokens, n_layers=4, n_heads=4, ffn_mult=4, dropout=0.0):
        super().__init__()
        self.channels   = channels
        self.max_tokens = max_tokens
        self.pos_emb    = nn.Parameter(torch.zeros(1, max_tokens, channels))
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
        return x.transpose(1, 2).reshape(B, C, H, W)
