# 多機多卡（DDP）+ 96 變數 實驗 SOP

> 目標：用**多台 4000 Ada 機器、每台 1~數張 GPU、同網段**，以 PyTorch DDP
> **多卡一起訓練同一個模型**，跑 96 變數的 ERA5 預測，準備投稿 TAAI。
>
> 全部架構（11 個主架構 + 基線）都由**同一支** `fourier_2d.py` 跑出來，
> 只差命令列參數。模型 / 資料載入都會自動依「變數數量」調整，4 / 6 / 96 變數通用。

---

## 0. 整合後的檔案結構（重點）

```
multiscale-fourier-neural-operator-feat-10-day-rollout-/
├── fourier_2d.py        ← 唯一入口：所有架構 + 任意變數數 + DDP 都在這
├── models.py            ← 全部 11 個架構的模型定義 + build_model 工廠 + EXPERIMENTS 註冊表
├── data_tools/
│   ├── download_global_6_factors.py    ← 既有 6 變數（single-level）
│   └── download_global_96_factors.py   ← 新增 96 變數（96 個 single-level 變數）
├── run_ddp.sh           ← Linux torchrun 啟動範本
├── run_ddp.ps1          ← Windows torchrun 啟動範本
└── DISTRIBUTED.md       ← 本檔
```

可用架構名稱（`python fourier_2d.py <名稱>`）：
```
平面：   unet_2d  unetpp_2d  transunet_2d
FNO混合：sufno    sunetpp_fno  sutrans_fno
純球面： sphere_unet  sphere_unetpp  sphere_transunet
基線：   2d_fno  sfno  2d_ufno  2d_unet
```

---

## 1. 下載 96 變數資料（先在一台機器做，或各機器都做）

```bash
python data_tools/download_global_96_factors.py
```
- 產生 `data/global_era5_96_factors_2021.nc`（2022、2023）。
- 96 通道 = **96 個不同的 single-level 變數**（含你原本的 6 個 + 90 個我幫你挑的）；載入時自動偵測、用檔案內短名命名。
- ⚠ 變數名稱一字之差就會整包失敗；若被退回，按 CDS 網頁「Show API request code」取得正確名稱。下載量大、可能數小時，已存在的年份檔會自動跳過。

> **資料要放哪？** DDP 每個 process 會各自讀同一份資料。
> - 若多機**有共享網路磁碟**：把 `data/` 放共享磁碟，所有機器都讀得到最方便。
> - 若**沒有共享磁碟**：把三個 `.nc` 複製到每台機器的 `data/`（路徑要一致）。

---

## 2. 先在單卡確認能跑（強烈建議）

DDP 出問題很難 debug，**務必先單卡確認資料、架構、環境都 OK**：

```bash
# 單卡、單一架構、96 變數（先用少 epoch 測試流程）
python fourier_2d.py unet_2d --data-glob "data/global_era5_96_factors_*.nc" --epochs 2
```
跑得出 `outputs/unet_2d/training_log.csv` 就代表整條 pipeline 正常，再上 DDP。

---

## 3. 單機多卡（同一台機器多張 GPU）

```bash
torchrun --standalone --nproc_per_node=2 \
    fourier_2d.py unet_2d \
    --data-glob "data/global_era5_96_factors_*.nc"
```
- `--nproc_per_node` = 這台機器要用幾張 GPU。
- 不必設 master IP，`--standalone` 會自動處理。

---

## 4. 多機多卡（本次主要情境）

假設共 **3 台機器**，編號 node 0/1/2；**node 0 當 master**，IP 例如 `192.168.0.10`，
每台各 **2 張 GPU**（`--nproc_per_node` 依各機實際張數調整）。

> `--nnodes` = 機器總數；`--node_rank` = 這台是第幾號（0 起算，每台不同）；
> `--master_addr/--master_port` = master（node 0）的 IP 與 port，**所有機器填一樣**。

**node 0（master，192.168.0.10）：**
```bash
torchrun --nnodes=3 --nproc_per_node=2 --node_rank=0 \
    --master_addr=192.168.0.10 --master_port=29500 \
    fourier_2d.py unet_2d --data-glob "data/global_era5_96_factors_*.nc"
```

**node 1：**
```bash
torchrun --nnodes=3 --nproc_per_node=2 --node_rank=1 \
    --master_addr=192.168.0.10 --master_port=29500 \
    fourier_2d.py unet_2d --data-glob "data/global_era5_96_factors_*.nc"
```

**node 2：**（同上，`--node_rank=2`）

三台幾乎同時啟動；torchrun 會自動 rendezvous。全域 batch size = `--batch-size`(預設4) × 總 GPU 數。

啟動範本見 `run_ddp.sh`（Linux）/ `run_ddp.ps1`（Windows），把變數填好即可。

---

## 5. ⚠ 後端：Linux 用 NCCL、Windows 用 gloo（很重要）

`fourier_2d.py` 會自動選後端（`--ddp-backend auto`）：

| 平台 | 自動選用 | 說明 |
|---|---|---|
| **Linux + CUDA** | `nccl` | 多卡/多機 GPU 的標準高速後端，**建議用 Linux 跑** |
| **Windows** | `gloo` | Windows **不支援 NCCL**；gloo 可用但較慢，多機需確認防火牆放行 `master_port` |

- 若 4000 Ada 機器是 **Linux** → 直接用，速度最好。
- 若是 **Windows** → 仍可跑（自動退回 gloo），但請：
  1. 確認 master 機器的 `29500`（或你設的 port）在防火牆放行；
  2. 三台機器同網段、能互相 ping 到；
  3. 速度會比 Linux+NCCL 慢，這是 Windows 限制，非程式問題。
- 想強制指定：加 `--ddp-backend nccl` 或 `--ddp-backend gloo`。

> 💡 96 變數模型其實不大（~2–3M 參數、33×64 網格）。多機 DDP 的通訊成本有時會
> 蓋過運算加速。若發現多機沒有比單卡快多少，屬正常現象；此時「每台機器各跑一個
> 不同架構/seed」的平行掃描，總吞吐量反而更高（見第 7 節）。

---

## 6. DDP 實作重點（給要改程式的人）

`fourier_2d.py` 已處理好下列細節，一般不用動：
- **只有 rank 0** 會寫 `config.json` / `training_log.csv` / 存權重 / 畫圖；其餘 rank 只算數。
- **DistributedSampler**：train `shuffle=True, drop_last=True`（確保各 rank 步數一致，DDP 才不會卡）；每個 epoch `set_epoch(ep)`。
- **指標跨卡匯總**：train/test MSE 與逐通道 MSE 都用 `all_reduce(SUM)` 後再平均。
- **Truncated BPTT × DDP**：一次 backward 內參數被重複使用 → DDP 以 `static_graph=True` 包裝（必要，否則會報 "marked ready twice"）。
- **小 batch 的 BatchNorm** → 自動轉 `SyncBatchNorm`，跨卡同步統計量。
- **單卡 / 不經 torchrun 執行**：自動退化成原本單機行為，行為與整合前一致。

---

## 7.（備選）每台機器各跑一個實驗 —— 平行掃描

若只是要「把 11 個架構 × 多 seed 全部跑完」，不需要 DDP，
直接每台機器跑不同參數即可（吞吐量通常最高）：

```bash
# 機器 A
python fourier_2d.py unet_2d   --data-glob "data/global_era5_96_factors_*.nc"
# 機器 B
python fourier_2d.py sphere_unet --data-glob "data/global_era5_96_factors_*.nc"
# 機器 C
python fourier_2d.py sufno --seed 1 --data-glob "data/global_era5_96_factors_*.nc"
```
輸出資料夾會依 seed/modes/dropout 自動加後綴，不會互相覆蓋；
跑完把各機 `outputs/` 收集到一起即可做統計與畫圖。

---

## 8. 常見問題

- **`FileExistsError 防覆蓋保護`**：該 `outputs/<name>/` 已有結果。改 `--seed` 或手動刪除資料夾。
- **多機卡住不動**：通常是 master IP/port 不通或防火牆擋住；先確認 `ping` 與 port。
- **記憶體不足**：96 變數資料佔 RAM 較多；可調小 `--batch-size`，或減少 `PRESSURE_LEVELS` 通道數。
- **想改變數數量**：編輯 `data_tools/download_global_96_factors.py` 的 `VARIABLES`（通道數 = 變數個數），
  訓練程式會自動偵測新通道數，不必改 `fourier_2d.py`。
