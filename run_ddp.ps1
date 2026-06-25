# ======================================================================
# Windows 多機多卡 DDP 啟動範本（4000 Ada）
# 用法：每台機器把 $NodeRank 改成自己的編號後，在 conda 環境中執行：
#   powershell -ExecutionPolicy Bypass -File .\run_ddp.ps1
# 注意：Windows 不支援 NCCL，會自動退回 gloo（較慢，需放行 master port）。
# ======================================================================

# ---- 依你的環境修改這幾行 ----
$Arch        = "unet_2d"                               # 架構名稱（見 DISTRIBUTED.md）
$DataGlob    = "data/global_era5_96_factors_*.nc"      # 96 變數資料
$NNodes      = 3                                        # 機器總數
$NprocPerNode= 2                                        # 這台機器要用幾張 GPU
$NodeRank    = 0                                        # ★ 每台不同：0 / 1 / 2 ...
$MasterAddr  = "192.168.0.10"                           # master（node 0）的 IP，所有機器填一樣
$MasterPort  = 29500
# --------------------------------

# 確保防火牆已放行 $MasterPort（master 機器執行一次即可）：
#   New-NetFirewallRule -DisplayName "torchrun DDP" -Direction Inbound -LocalPort 29500 -Protocol TCP -Action Allow

torchrun `
    --nnodes=$NNodes `
    --nproc_per_node=$NprocPerNode `
    --node_rank=$NodeRank `
    --master_addr=$MasterAddr `
    --master_port=$MasterPort `
    fourier_2d.py $Arch `
    --data-glob $DataGlob `
    --epochs 50 --batch-size 4

# 單機多卡（同一台多張 GPU）改用：
#   torchrun --standalone --nproc_per_node=2 fourier_2d.py $Arch --data-glob $DataGlob
