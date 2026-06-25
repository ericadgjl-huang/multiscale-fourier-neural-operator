# Spherical Neural Operators vs Planar CNNs on Low-Resolution Weather Forecasting

A systematic 3×3 ablation study comparing **architecture (UNet / UNet++ / TransUNet)** × **geometry handling (planar with longitude circular padding / FNO-hybrid / pure spherical SHT)** for 10-day rollout forecasting on mini-ERA5 (33×64 grid).

**Key finding**: Pure planar CNNs with longitude circular padding statistically significantly **outperform** all spherical neural operator variants on low-resolution data. The "spherical is necessary" assumption inherited from large-scale weather DL papers does not hold at this scale.

## Result Summary (Best Test MSE)

| Architecture | Pure 2D (lon pad) | FNO Hybrid (SHT ⊕ planar) | Pure Spherical (SHT-only) |
|---|---|---|---|
| **UNet**       | **0.3751 ± 0.0010** (n=3) | 0.3969 ± 0.0077 (n=3) | 0.4224 (n=1) |
| **UNet++**     | 0.3803 ± 0.0044 (n=3)     | 0.3882 ± 0.0059 (n=3) | 0.4165 (n=1) |
| **TransUNet**  | 0.3765 ± 0.0031 (n=3)     | 0.4228 ± 0.0077 (n=3) | 0.4151 (n=1) |

→ **planar > FNO hybrid > pure spherical** holds consistently across all three architectures.

---

## 🆕 整合版（單一入口 + 任意變數數 + 多機多卡 DDP）

為了 96 變數的 TAAI 投稿實驗，本資料夾已把**全部 11 個架構 + 基線**整合進**同一支** `fourier_2d.py`，
並支援任意變數數（4 / 6 / 96，自動偵測 NetCDF 通道，含氣壓層展開）與 DDP 多機多卡訓練。

```bash
# 單卡，任一架構（6 變數現成資料）
python fourier_2d.py unet_2d
python fourier_2d.py sphere_unet --seed 1

# 96 變數：先下載 96 個 single-level 變數，再指定 --data-glob
python data_tools/download_global_96_factors.py
python fourier_2d.py unet_2d --data-glob "data/global_era5_96_factors_*.nc"

# 多機多卡 DDP（詳見 DISTRIBUTED.md / run_ddp.sh / run_ddp.ps1）
torchrun --nnodes=3 --nproc_per_node=2 --node_rank=0 \
         --master_addr=192.168.0.10 --master_port=29500 \
         fourier_2d.py unet_2d --data-glob "data/global_era5_96_factors_*.nc"
```

- 架構定義與工廠：`models.py`（`EXPERIMENTS` 註冊表列出所有可用名稱）。
- 多機多卡、後端選擇（Linux NCCL / Windows gloo）、96 變數工作流程：見 **`DISTRIBUTED.md`**。
- 原本散落的 `*_baseline.py`、`sphere_blocks.py` 已全部整併進 `models.py`，不再需要。

---

## Project Structure

```
fourier-neural-operator/
├── README.md                          (this file)
├── LICENSE                            (MIT)
├── .gitignore
│
├── fourier_2d.py                      ← 5 FNO variants (2d_fno, sfno, sufno, sunetpp_fno, sutrans_fno)
├── unet_baseline.py                   ← Pure 2D UNet
├── unetpp_baseline.py                 ← Pure 2D UNet++
├── transunet_baseline.py              ← Pure 2D TransUNet
├── sphere_unet_baseline.py            ← Pure spherical UNet
├── sphere_unetpp_baseline.py          ← Pure spherical UNet++
├── sphere_transunet_baseline.py       ← Pure spherical TransUNet
├── pca_net.py                         ← PCA+MLP traditional baseline
│
├── sphere_blocks.py                   ← Shared SHT building blocks
├── utilities3.py                      ← Shared utilities (from upstream FNO repo)
├── Adam.py                            ← Adam optimizer (from upstream)
│
├── analysis/                          ← Result analysis & visualization
│   ├── compare_experiments.py             scan outputs/ → learning curves + bar chart
│   ├── multi_seed_compare.py              group by (base, modes, dropout) → mean ± std
│   ├── final_ablation_plot.py             3×3 heatmap (paper Figure 1)
│   ├── resource_comparison.py             params / time / inference / memory (paper Table 2)
│   └── regenerate_pretty_plots.py         re-render plots with geographic axes
│
├── data_tools/                        ← Data download & inspection
│   ├── check_data.py                       inspect a single NetCDF
│   ├── download_global.py                  download 2021-2023 mini-ERA5
│   └── download_global_mini.py             download a single file
│
├── archive/                           ← Frozen v1 baseline scripts (Git history backup)
│
├── data/        (gitignored)          ← ERA5 NetCDF files go here
└── outputs/     (gitignored)          ← Training results
    ├── <arch>/                            per-experiment folder
    │   ├── config.json                    hyperparameter snapshot
    │   ├── training_log.csv               per-epoch metrics
    │   ├── model_weights.pt               final weights
    │   ├── model_weights_best.pt          best test_mse weights
    │   └── *.png                          4 visualization plots
    └── _comparison/                       analysis script outputs
        ├── final_ablation_heatmap.png     ★ Paper Figure 1
        ├── multi_seed_plot.png            multi-seed bars
        ├── resource_comparison.png        Table 2 visualization
        └── *.csv                          summary tables
```

---

## Quick Start

### 1. Setup environment
Requires Python 3.10+ with PyTorch, `torch_harmonics`, `xarray`, `h5netcdf`, `pandas`, `matplotlib`.

```bash
pip install torch torchvision torch_harmonics xarray h5netcdf pandas matplotlib
```

### 2. Download ERA5 mini data
Configure `~/.cdsapirc` with CDS credentials, then:

```bash
python data_tools/download_global.py
```

Files land in `data/global_era5_mini_*.nc`.

### 3. Train one architecture
Edit the top of any training script to choose the experiment, then run:

```bash
# FNO family — set base_experiment_name to one of:
# '2d_fno' / 'sfno' / 'sufno' / 'sunetpp_fno' / 'sutrans_fno'
python fourier_2d.py

# Or any pure-CNN baseline
python unet_baseline.py
python unetpp_baseline.py
python transunet_baseline.py

# Or pure-spherical UNet variants
python sphere_unet_baseline.py
python sphere_unetpp_baseline.py
python sphere_transunet_baseline.py
```

Each script writes to `outputs/<experiment_name>/`. The overwrite-protection check will raise an error if the folder already has training results — change `SEED`, `MODES`, or `DROPOUT` to get a new folder name.

### 4. Analyze results

```bash
python analysis/compare_experiments.py     # per-experiment view
python analysis/multi_seed_compare.py      # grouped multi-seed statistics
python analysis/final_ablation_plot.py     # 3×3 heatmap + bars
python analysis/resource_comparison.py     # params / time / memory / latency
python analysis/regenerate_pretty_plots.py # re-render plots with lat/lon axes
```

All analysis scripts auto-`chdir` to project root, so they work from anywhere.

---

## Experimental Setup

- **Dataset**: ERA5 mini, 2021-2023, 33×64 (~5.45° lat × 5.625° lon), 6-hourly
- **Variables**: 2m temperature, sea-level pressure, 10m U/V wind
- **Time embedding**: sin/cos of day-of-year and hour-of-day (4 extra input channels)
- **Train / Test split**: 2920 timesteps train (2021-2022) / remainder test (2023)
- **Task**: 10-day rollout forecasting (40 autoregressive steps, 6 h each)
- **Loss**: step-decay-weighted MSE (γ=0.95) with Truncated BPTT (K=8)
- **Optimizer**: Adam, lr=1e-3, weight_decay=1e-4, gradient clipping max_norm=1.0
- **Schedule**: CosineAnnealingLR, 50 epochs
- **Hardware**: single NVIDIA RTX 5070 (12 GB)

---

## Acknowledgments

The base FNO scaffold (`fourier_2d.py`, `utilities3.py`, `Adam.py`) is forked from [neuraloperator/Geo-FNO](https://github.com/neuraloperator/Geo-FNO) by Zongyi Li et al. Pure UNet building blocks adapted from the [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) implementation. Spherical Harmonic Transform operations use [torch_harmonics](https://github.com/NVIDIA/torch-harmonics) (Bonev et al., NVIDIA).
