#!/usr/bin/env bash
# ======================================================================
# Linux 多機多卡 DDP 啟動範本（4000 Ada）
# 用法：每台機器把 NODE_RANK 改成自己的編號後執行 ./run_ddp.sh
# ======================================================================
set -e

# ---- 依你的環境修改這幾行 ----
ARCH="unet_2d"                                   # 架構名稱（見 DISTRIBUTED.md）
DATA_GLOB="data/global_era5_96_factors_*.nc"     # 96 變數資料
NNODES=3                                          # 機器總數
NPROC_PER_NODE=2                                  # 這台機器要用幾張 GPU
NODE_RANK=0                                        # ★ 每台不同：0 / 1 / 2 ...
MASTER_ADDR="192.168.0.10"                         # master（node 0）的 IP，所有機器填一樣
MASTER_PORT=29500
EXTRA_ARGS="--epochs 50 --batch-size 4"            # 其他參數（seed / modes / dropout 等）
# --------------------------------

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    fourier_2d.py "${ARCH}" \
    --data-glob "${DATA_GLOB}" \
    ${EXTRA_ARGS}

# 單機多卡（同一台多張 GPU）改用：
#   torchrun --standalone --nproc_per_node=2 fourier_2d.py "${ARCH}" --data-glob "${DATA_GLOB}"
