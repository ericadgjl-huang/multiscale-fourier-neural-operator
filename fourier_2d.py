"""
fourier_2d.py — 「單一入口」整合版

一支腳本跑完所有 11 個架構（平面 / FNO 混合 / 純球面）+ 經典基線，
並支援：
  * 任意變數數量（4 / 6 / 96 …）：自動偵測 NetCDF 內的氣象通道（含氣壓層展開）
  * 多卡 / 多機 DDP 訓練（torchrun 啟動；單卡時自動退化成原本行為）

用法（單卡，跑單一架構）：
    python fourier_2d.py unet_2d
    python fourier_2d.py sphere_unet --seed 1
    python fourier_2d.py sufno --modes 16 --dropout 0.1

用法（96 變數 + 多機多卡 DDP，每台機器執行；詳見 walkthrough.md / DISTRIBUTED.md）：
    torchrun --nnodes=3 --nproc_per_node=2 --node_rank=0 \
             --master_addr=192.168.0.10 --master_port=29500 \
             fourier_2d.py unet_2d --data-glob "data/global_era5_96_factors_*.nc"

可用架構名稱見 models.EXPERIMENTS。
"""
# ======================================================================
# --- Windows 系統專用 Triton 雙重攔截補丁（解決 torch_harmonics import）---
# 新版 torch（2.10）不需要；舊環境（torch 2.2.0、Windows 無 triton）需要假 triton。
# ======================================================================
import os
import sys
import importlib.util
from types import ModuleType


def _install_fake_triton():
    orig_find_spec = importlib.util.find_spec

    def hooked_find_spec(name, package=None):
        if name == 'triton' or name.startswith('triton.'):
            return None
        return orig_find_spec(name, package)
    importlib.util.find_spec = hooked_find_spec

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


try:
    import torch_harmonics as th  # noqa: F401
    _TH_MSG = "[資訊] torch_harmonics 直接 import 成功，未啟用假 triton 補丁。"
except Exception as _th_err:
    _install_fake_triton()
    import torch_harmonics as th  # noqa: F401
    _TH_MSG = f"[資訊] torch_harmonics 需要假 triton 補丁，已套用後重試成功（原錯誤：{_th_err}）。"
# ======================================================================

import argparse
import json
import csv
from timeit import default_timer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import xarray as xr
import pandas as pd

import matplotlib
matplotlib.use('Agg')          # 無顯示環境（伺服器）也能存圖
import matplotlib.pyplot as plt
from matplotlib import font_manager

from utilities3 import count_params
from models import EXPERIMENTS, build_model


# ======================================================================
# 0. 命令列參數
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(description="When Sphere Hurts — 整合版單一入口訓練腳本")
    p.add_argument('arch', nargs='?', default='unet_2d',
                   help=f"架構名稱，可選：{', '.join(EXPERIMENTS.keys())}")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--modes', type=int, default=16, help='FNO modes（球面模型用 --sphere-modes）')
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=4, help='每張 GPU 的 batch size')
    p.add_argument('--base-width', type=int, default=32)
    p.add_argument('--rollout-steps', type=int, default=40)
    p.add_argument('--tbptt-k', type=int, default=8)
    p.add_argument('--train-size', type=int, default=2920, help='訓練集時間步數（其餘為測試集）')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--clip-norm', type=float, default=1.0)
    p.add_argument('--step-loss-gamma', type=float, default=0.95)
    p.add_argument('--sphere-modes', type=str, default='8,4,2',
                   help='純球面模型每層 SHT modes，逗號分隔')
    p.add_argument('--data-glob', type=str, default='data/global_era5_6_factors_*.nc',
                   help='ERA5 NetCDF glob；換 96 變數請指向對應檔案')
    p.add_argument('--vars', type=str, default='',
                   help='逗號分隔的變數白名單；留空＝自動偵測 NetCDF 內所有氣象變數')
    p.add_argument('--output-root', type=str, default='outputs')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--ddp-backend', type=str, default='auto',
                   choices=['auto', 'nccl', 'gloo'],
                   help='分散式後端；auto＝Linux+CUDA 用 nccl，否則 gloo')
    p.add_argument('--skill-plot-channels', type=int, default=4,
                   help='forecast skill 圖最多畫幾個通道（96 變數時避免畫爆）')
    return p.parse_args()


# ======================================================================
# 1. DDP 工具
# ======================================================================
def setup_distributed(backend_choice):
    """讀取 torchrun 注入的環境變數；回傳 (use_ddp, rank, local_rank, world_size, device)。"""
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    use_ddp = world_size > 1

    if use_ddp:
        if backend_choice == 'auto':
            backend = 'nccl' if (torch.cuda.is_available() and sys.platform != 'win32') else 'gloo'
        else:
            backend = backend_choice
        dist.init_process_group(backend=backend, init_method='env://')
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cpu')
        if rank == 0:
            print(f"[DDP] backend={backend} world_size={world_size}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return use_ddp, rank, local_rank, world_size, device


def is_main_process(rank):
    return rank == 0


def reduce_sum_(tensor_like, device, use_ddp):
    """把 numpy 陣列或 python 純量做跨 rank SUM；非 DDP 時原樣回傳。"""
    if not use_ddp:
        return tensor_like
    t = torch.as_tensor(np.asarray(tensor_like, dtype=np.float64), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    out = t.cpu().numpy()
    return out if out.ndim > 0 else float(out)


# ======================================================================
# 2. 資料載入（自動偵測變數 / 氣壓層展開 → 支援 4 / 6 / 96 變數）
# ======================================================================
def _name_of(ds, candidates):
    for c in candidates:
        if c in ds.dims or c in ds.coords:
            return c
    return None


def load_era5(data_glob, var_whitelist, verbose=True):
    """
    回傳：
      data_meteo : (T, nlat, nlon, C_meteo) float32 tensor
      times      : pandas DatetimeIndex（長度 T）
      channel_names : list[str]，長度 C_meteo
      geo_extent : [lon_min, lon_max, lat_min, lat_max] 或 None
      nlat, nlon
    """
    ds = xr.open_mfdataset(data_glob, engine='h5netcdf', combine='by_coords')

    lat_name = _name_of(ds, ['latitude', 'lat'])
    lon_name = _name_of(ds, ['longitude', 'lon'])
    time_name = _name_of(ds, ['valid_time', 'time'])
    if lat_name is None or lon_name is None or time_name is None:
        raise RuntimeError(f"無法辨識 lat/lon/time 維度，實際 dims={list(ds.dims)}")

    nlat = int(ds.sizes[lat_name])
    nlon = int(ds.sizes[lon_name])

    if var_whitelist:
        var_list = [v.strip() for v in var_whitelist.split(',') if v.strip()]
    else:
        # 保留 NetCDF 內原本的變數順序（下載順序），確保 channel 0 = 第一個下載變數
        var_list = list(ds.data_vars)

    # CDS 偶爾會帶 expver / number 這類非垂直維度（如混到 ERA5T 近即時資料）；
    # 這些不是氣壓層，取第 0 個壓平，避免被誤展開成假通道。
    SQUEEZE_DIMS = ('expver', 'number', 'realization')

    channels = []
    channel_names = []
    for v in var_list:
        da = ds[v]
        for sd in SQUEEZE_DIMS:
            if sd in da.dims:
                da = da.isel({sd: 0}, drop=True)
        other_dims = [d for d in da.dims if d not in (time_name, lat_name, lon_name)]
        if not other_dims:
            channels.append(da)
            channel_names.append(v)
        else:
            level_dim = other_dims[0]               # 氣壓層維度（pressure_level / level / isobaricInhPa）
            for lev in da[level_dim].values:
                channels.append(da.sel({level_dim: lev}))
                channel_names.append(f"{v}{int(lev)}")

    if verbose:
        print(f"偵測到 {len(channel_names)} 個氣象通道（{nlat}×{nlon} 網格）")

    # 逐通道轉 tensor 並 stack 到最後一維 → (T, nlat, nlon, C)
    arrs = [torch.as_tensor(np.asarray(c.values), dtype=torch.float32) for c in channels]
    data_meteo = torch.stack(arrs, dim=-1)
    data_meteo = torch.nan_to_num(data_meteo, nan=0.0)

    times = pd.to_datetime(ds[time_name].values)

    try:
        lons = ds[lon_name].values
        lats = ds[lat_name].values
        geo_extent = [float(lons.min()), float(lons.max()),
                      float(lats.min()), float(lats.max())]
    except Exception:
        geo_extent = None

    return data_meteo, times, channel_names, geo_extent, nlat, nlon


def build_time_channels(times, nlat, nlon):
    """day-of-year / hour-of-day 的 sin/cos，共 4 通道，broadcast 成 (T, nlat, nlon, 4)。"""
    day_rad = torch.tensor(times.dayofyear.values, dtype=torch.float32) * (2 * np.pi / 365.25)
    hour_rad = torch.tensor(times.hour.values, dtype=torch.float32) * (2 * np.pi / 24.0)
    feats = [torch.sin(day_rad), torch.cos(day_rad), torch.sin(hour_rad), torch.cos(hour_rad)]
    feats = [f.view(-1, 1, 1).expand(-1, nlat, nlon) for f in feats]
    return torch.stack(feats, dim=-1)


class ERA5RolloutDataset(torch.utils.data.Dataset):
    """動態切片：x=(nlat,nlon,C)；y=(rollout_steps, nlat, nlon, C)。"""
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
        y = self.data[i + 1:i + 1 + self.rollout_steps]
        return x, y


# ======================================================================
# 3. 主程式
# ======================================================================
def main():
    args = parse_args()
    use_ddp, rank, local_rank, world_size, device = setup_distributed(args.ddp_backend)
    main_proc = is_main_process(rank)

    if main_proc:
        print(_TH_MSG)

    if args.arch not in EXPERIMENTS:
        raise SystemExit(f"[參數錯誤] 未知架構 '{args.arch}'。可用：{', '.join(EXPERIMENTS.keys())}")
    cfg = EXPERIMENTS[args.arch]

    # --- 中文字型（找不到就略過，不影響訓練）---
    if main_proc:
        try:
            font_path = r"C:\Windows\Fonts\msjh.ttc"
            if os.path.exists(font_path):
                font_manager.fontManager.addfont(font_path)
                plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
        except Exception as e:
            print(f"[警告] 載入中文字型失敗：{e}")
        plt.rcParams['axes.unicode_minus'] = False

    # --- random seed ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- 輸出資料夾名稱（與舊 baseline 命名相容：seed/modes/dropout 預設值不加後綴）---
    suffix_parts = []
    if args.seed != 0:
        suffix_parts.append(f's{args.seed}')
    if args.modes != 16:
        suffix_parts.append(f'm{args.modes}')
    if args.dropout > 0:
        suffix_parts.append(f'drop{int(round(args.dropout * 100))}')
    suffix = ('_' + '_'.join(suffix_parts)) if suffix_parts else ''
    experiment_name = args.arch + suffix
    output_dir = os.path.join(args.output_root, experiment_name)

    # 防覆蓋（所有 rank 都看得到同一路徑 → 一起 raise，不會卡住 collective）
    if os.path.exists(os.path.join(output_dir, 'training_log.csv')):
        raise FileExistsError(
            f"\n[防覆蓋保護] {output_dir} 已有完整訓練紀錄！\n"
            f"請改 --seed / --modes / --dropout 產生新後綴，或手動刪除該資料夾。")

    if main_proc:
        os.makedirs(output_dir, exist_ok=True)

    # --- 超參數 ---
    sphere_modes = tuple(int(x) for x in args.sphere_modes.split(','))
    rollout_steps = args.rollout_steps
    TBPTT_K = args.tbptt_k

    # DDP 用 static_graph 包裝（因 TBPTT 一次 backward 重複用到參數）。
    # static_graph 要求每個 backward 的計算圖一致 → 每個 TBPTT 視窗的 forward 次數必須相同，
    # 也就是 rollout_steps 必須能被 tbptt_k 整除（預設 40 / 8 = 5，符合）。
    if use_ddp and rollout_steps % TBPTT_K != 0:
        raise SystemExit(
            f"[DDP 限制] rollout_steps({rollout_steps}) 必須能被 tbptt_k({TBPTT_K}) 整除，"
            f"static_graph 才能正確處理 TBPTT。請調整 --tbptt-k（例如 8 或 10）。")

    # --- 讀資料（每個 rank 各自讀；同機多卡會重複佔記憶體，但程式最簡單）---
    if main_proc:
        print(f"正在讀取 ERA5（glob={args.data_glob}）...")
    data_meteo, times, channel_names, geo_extent, nlat, nlon = load_era5(
        args.data_glob, args.vars, verbose=main_proc)
    NUM_CHANNELS = data_meteo.shape[-1]
    time_ch = build_time_channels(times, nlat, nlon)
    data = torch.cat([data_meteo, time_ch], dim=-1).float()   # (T, nlat, nlon, C_meteo+4)

    train_size = args.train_size
    total_size = len(data)
    if total_size <= train_size + rollout_steps:
        raise SystemExit(f"[資料錯誤] 總步數 {total_size} 不足以切出 train_size={train_size} + rollout={rollout_steps}")

    # 標準化：僅用訓練集統計量（含時間通道一起標準化，與原腳本一致）
    x_mean = data[:train_size].mean(dim=(0, 1, 2))
    x_std = data[:train_size].std(dim=(0, 1, 2))
    data_norm = (data - x_mean) / (x_std + 1e-6)

    train_dataset = ERA5RolloutDataset(data_norm, 0, train_size, rollout_steps)
    test_dataset = ERA5RolloutDataset(data_norm, train_size,
                                      total_size - train_size - rollout_steps, rollout_steps)

    if use_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler,
            num_workers=args.num_workers, drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, sampler=test_sampler,
            num_workers=args.num_workers)
    else:
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- 建模型 ---
    model = build_model(args.arch, num_channels=NUM_CHANNELS, base_width=args.base_width,
                        modes=args.modes, dropout=args.dropout,
                        nlat=nlat, nlon=nlon, sphere_modes=sphere_modes).to(device)
    param_count = count_params(model)

    if use_ddp:
        # 小 batch 的 BatchNorm 跨卡同步，統計量才正確
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_kwargs = dict(static_graph=True)   # TBPTT 一次 backward 重複用到參數 → 需 static_graph
        if torch.cuda.is_available():
            ddp_kwargs['device_ids'] = [local_rank]
        model = nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
    core_model = model.module if use_ddp else model

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    model_name = cfg['display']
    if main_proc:
        print("========================================")
        print(f" 實驗：{experiment_name}  →  {model_name}")
        print(f" 輸出：{output_dir}")
        print(f" 變數數：{NUM_CHANNELS} | 網格：{nlat}×{nlon} | 參數量：{param_count}")
        print(f" SEED={args.seed} MODES={args.modes} DROPOUT={args.dropout} "
              f"DDP={use_ddp} (world_size={world_size})")
        print("========================================")

    # --- config.json（rank0）---
    config_snapshot = {
        'experiment_name': experiment_name, 'base_experiment_name': args.arch,
        'display_name': model_name, 'family': cfg['family'],
        'num_channels': NUM_CHANNELS, 'channel_names': channel_names,
        'nlat': nlat, 'nlon': nlon,
        'seed': args.seed, 'modes': args.modes, 'dropout': args.dropout,
        'sphere_modes': list(sphere_modes), 'base_width': args.base_width,
        'batch_size_per_gpu': args.batch_size, 'world_size': world_size, 'ddp': use_ddp,
        'epochs': args.epochs, 'rollout_steps': rollout_steps, 'TBPTT_K': TBPTT_K,
        'step_loss_gamma': args.step_loss_gamma, 'lr': args.lr,
        'weight_decay': args.weight_decay, 'clip_norm': args.clip_norm,
        'train_size': train_size, 'total_size': total_size,
        'param_count': param_count, 'data_glob': args.data_glob,
        'optimizer': 'Adam', 'scheduler': 'CosineAnnealingLR',
    }
    config_path = os.path.join(output_dir, 'config.json')
    csv_path = os.path.join(output_dir, 'training_log.csv')
    if main_proc:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_snapshot, f, indent=2, ensure_ascii=False)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            header = ['epoch', 'train_mse', 'test_mse', 'lr', 'epoch_time_sec']
            header += [f'train_mse_ch{c}_{n}' for c, n in enumerate(channel_names)]
            header += [f'test_mse_ch{c}_{n}' for c, n in enumerate(channel_names)]
            csv.writer(f).writerow(header)

    step_weights = torch.tensor(
        [args.step_loss_gamma ** i for i in range(rollout_steps)], dtype=torch.float32).to(device)

    history_train_mse, history_test_mse = [], []
    best_test_mse, best_epoch = float('inf'), -1

    # ==================================================================
    # 訓練迴圈
    # ==================================================================
    for ep in range(args.epochs):
        if use_ddp:
            train_sampler.set_epoch(ep)
        model.train()
        t1 = default_timer()

        train_loss_accum = 0.0
        n_train_batches = 0
        n_skipped_steps = 0          # NaN 守門跳過的更新次數（穩定架構恆為 0）
        train_mse_pc = np.zeros(NUM_CHANNELS)

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            current_input = x
            batch_mse = 0.0

            for window_start in range(0, rollout_steps, TBPTT_K):
                window_end = min(window_start + TBPTT_K, rollout_steps)
                optimizer.zero_grad()
                window_loss = torch.tensor(0.0, device=device)

                for step in range(window_start, window_end):
                    pred = model(current_input)
                    true = y[:, step, :, :, :NUM_CHANNELS]
                    mse_pc = torch.mean((pred - true) ** 2, dim=(0, 1, 2))
                    train_mse_pc += mse_pc.detach().cpu().numpy()
                    window_loss = window_loss + step_weights[step] * torch.mean(mse_pc)
                    batch_mse += torch.mean(mse_pc).item()
                    if step < rollout_steps - 1:
                        next_time = y[:, step, :, :, NUM_CHANNELS:]
                        current_input = torch.cat([pred, next_time], dim=-1)

                window_loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
                # --- NaN 守門 ---
                # transformer bottleneck 在 rollout 早期偶爾會吐出爆炸（inf）梯度，
                # clip_grad_norm_ 會把 inf 洗成 nan（inf * (max_norm/inf) = inf*0 = nan）並寫進權重，
                # 一旦中毒就「全程 nan」。total_norm 非有限時跳過這次更新，
                # 壞梯度交給下一個 zero_grad() 清掉。對 FNO/UNet 等穩定架構恆不觸發＝no-op。
                if torch.isfinite(total_norm):
                    optimizer.step()
                else:
                    n_skipped_steps += 1
                current_input = current_input.detach()

            train_loss_accum += batch_mse / rollout_steps
            n_train_batches += 1

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # ---- 測試 ----
        model.eval()
        test_loss_accum = 0.0
        n_test_steps = 0          # batches * rollout_steps
        test_mse_pc = np.zeros(NUM_CHANNELS)
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                current_input = x
                for step in range(rollout_steps):
                    pred = model(current_input)
                    true = y[:, step, :, :, :NUM_CHANNELS]
                    mse_pc = torch.mean((pred - true) ** 2, dim=(0, 1, 2))
                    test_mse_pc += mse_pc.detach().cpu().numpy()
                    test_loss_accum += torch.mean(mse_pc).item()
                    n_test_steps += 1
                    if step < rollout_steps - 1:
                        next_time = y[:, step, :, :, NUM_CHANNELS:]
                        current_input = torch.cat([pred, next_time], dim=-1)

        # ---- 跨 rank 匯總 ----
        g_train_loss = reduce_sum_(train_loss_accum, device, use_ddp)
        g_train_batches = reduce_sum_(n_train_batches, device, use_ddp)
        g_train_pc = reduce_sum_(train_mse_pc, device, use_ddp)
        g_test_loss = reduce_sum_(test_loss_accum, device, use_ddp)
        g_test_steps = reduce_sum_(n_test_steps, device, use_ddp)
        g_test_pc = reduce_sum_(test_mse_pc, device, use_ddp)

        train_mse = g_train_loss / max(g_train_batches, 1)
        train_mse_pc_avg = g_train_pc / max(g_train_batches * rollout_steps, 1)
        test_mse = g_test_loss / max(g_test_steps, 1)
        test_mse_pc_avg = g_test_pc / max(g_test_steps, 1)

        epoch_time = default_timer() - t1
        history_train_mse.append(train_mse)
        history_test_mse.append(test_mse)

        is_best = test_mse < best_test_mse
        if is_best:
            best_test_mse, best_epoch = test_mse, ep
            if main_proc:
                torch.save(core_model.state_dict(), os.path.join(output_dir, 'model_weights_best.pt'))

        if main_proc:
            marker = "  ← new best" if is_best else ""
            skip_note = f" | skipped {n_skipped_steps}" if n_skipped_steps else ""
            print(f"Epoch {ep:02d} | {epoch_time:.1f}s | LR {current_lr:.2e} | "
                  f"Train {train_mse:.4f} | Test {test_mse:.4f}{skip_note}{marker}")
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                row = [ep, train_mse, test_mse, current_lr, epoch_time]
                row += list(np.asarray(train_mse_pc_avg).ravel())
                row += list(np.asarray(test_mse_pc_avg).ravel())
                csv.writer(f).writerow(row)

    # ==================================================================
    # 收尾（rank0 才存權重、畫圖）
    # ==================================================================
    if not main_proc:
        if use_ddp:
            dist.barrier()
            dist.destroy_process_group()
        return

    torch.save(core_model.state_dict(), os.path.join(output_dir, 'model_weights.pt'))
    config_snapshot['best_epoch'] = best_epoch
    config_snapshot['best_test_mse'] = best_test_mse
    config_snapshot['final_test_mse'] = history_test_mse[-1]
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_snapshot, f, indent=2, ensure_ascii=False)
    print(f"\n最終/最佳模型權重已存於 {output_dir}/")

    make_plots(core_model, test_loader, device, output_dir, model_name, rollout_steps,
               NUM_CHANNELS, channel_names, geo_extent, history_train_mse, history_test_mse,
               n_skill_channels=min(args.skill_plot_channels, NUM_CHANNELS))

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()
    print(f"\n========== 全部完成！結果在 {output_dir}/ ==========")


# ======================================================================
# 4. 視覺化（rank0）
# ======================================================================
def make_plots(model, test_loader, device, output_dir, model_name, rollout_steps,
               NUM_CHANNELS, channel_names, geo_extent, history_train_mse, history_test_mse,
               n_skill_channels=4):
    # --- 學習曲線 ---
    plt.figure(figsize=(10, 6))
    plt.plot(history_train_mse, label='Train MSE', linewidth=2)
    plt.plot(history_test_mse, label='Test MSE', linewidth=2)
    plt.xlabel('Epochs'); plt.ylabel('MSE Loss')
    plt.title(f'Learning Curve — {model_name} ({rollout_steps * 6 // 24}-Day Forecast)')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'learning_curve.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # --- forecast skill（只畫前 n_skill_channels 個通道）---
    nP = max(1, n_skill_channels)
    lead_hours = np.arange(1, rollout_steps + 1) * 6
    channel_step_rmse = np.zeros((nP, rollout_steps))
    n_batches = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            current_input = x
            step_preds = []
            for step in range(rollout_steps):
                pred = model(current_input)
                step_preds.append(pred.detach().cpu())
                if step < rollout_steps - 1:
                    current_input = torch.cat([pred, y[:, step, :, :, NUM_CHANNELS:]], dim=-1)
            for step in range(rollout_steps):
                true = y[:, step, :, :, :NUM_CHANNELS].cpu().numpy()
                pred = step_preds[step].numpy()
                for ch in range(nP):
                    channel_step_rmse[ch, step] += np.sqrt(np.mean((pred[..., ch] - true[..., ch]) ** 2))
            n_batches += 1
            if n_batches >= 10:
                break
    channel_step_rmse /= max(n_batches, 1)

    fig, axes = plt.subplots(1, nP, figsize=(5 * nP, 5), squeeze=False)
    for ch in range(nP):
        ax = axes[0, ch]
        ax.plot(lead_hours, channel_step_rmse[ch], linewidth=2, color=f'C{ch}')
        ax.set_title(f'{channel_names[ch]} RMSE vs Lead Time', fontsize=11)
        ax.set_xlabel('Forecast Lead Time (hours)'); ax.set_ylabel('RMSE (normalized)')
        ax.axvline(x=120, color='orange', linestyle='--', alpha=0.8, label='Day 5')
        ax.axvline(x=240, color='red', linestyle='--', alpha=0.8, label='Day 10')
        ax.legend(); ax.grid(True, alpha=0.4)
    plt.suptitle(f'Forecast Skill — {model_name}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'forecast_skill.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # --- 多時效誤差熱點圖（通道 0）---
    target_steps = [min(s, rollout_steps - 1) for s in (3, 11, 27, 39)]
    target_labels = ['Day 1 (T+4)', 'Day 3 (T+12)', 'Day 7 (T+28)', 'Day 10 (T+40)']
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            current_input = x
            all_preds = []
            for step in range(rollout_steps):
                pred = model(current_input)
                all_preds.append(pred)
                if step < rollout_steps - 1:
                    current_input = torch.cat([pred, y[:, step, :, :, NUM_CHANNELS:]], dim=-1)
            break

    idx = 0
    kw = dict(extent=geo_extent, aspect='auto') if geo_extent is not None else {}
    fig, axes = plt.subplots(len(target_steps), 3, figsize=(15, len(target_steps) * 4))
    for row, (ts, label) in enumerate(zip(target_steps, target_labels)):
        gt = y[idx, ts, :, :, 0].cpu().numpy()
        pr = all_preds[ts][idx, :, :, 0].cpu().numpy()
        err = gt - pr
        im0 = axes[row, 0].imshow(gt, cmap='jet', **kw); axes[row, 0].set_title(f'True {label}'); fig.colorbar(im0, ax=axes[row, 0])
        im1 = axes[row, 1].imshow(pr, cmap='jet', **kw); axes[row, 1].set_title(f'Pred {label}'); fig.colorbar(im1, ax=axes[row, 1])
        im2 = axes[row, 2].imshow(err, cmap='coolwarm', **kw); axes[row, 2].set_title(f'Error {label}'); fig.colorbar(im2, ax=axes[row, 2])
    plt.suptitle(f'{channel_names[0]} Prediction Error Maps — {model_name}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weather_prediction.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # --- 3D 球體圖（通道 0，Day 10）---
    temp_pred = all_preds[-1][idx, :, :, 0].cpu().numpy()
    lon = np.linspace(0, 2 * np.pi, temp_pred.shape[1])
    lat = np.linspace(0, np.pi, temp_pred.shape[0])
    lon, lat = np.meshgrid(lon, lat)
    X, Y, Z = np.sin(lat) * np.cos(lon), np.sin(lat) * np.sin(lon), np.cos(lat)
    tn = (temp_pred - temp_pred.min()) / (temp_pred.max() - temp_pred.min() + 1e-6)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d'); ax.axis('off')
    ax.plot_surface(X, Y, Z, facecolors=plt.cm.jet(tn), rstride=1, cstride=1, antialiased=True, shade=False)
    ax.set_title(f"Global Prediction — Day {rollout_steps * 6 // 24} Forecast")
    plt.savefig(os.path.join(output_dir, 'weather_prediction_3d.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("圖表（learning_curve / forecast_skill / weather_prediction / 3d）已輸出。")


if __name__ == '__main__':
    main()
