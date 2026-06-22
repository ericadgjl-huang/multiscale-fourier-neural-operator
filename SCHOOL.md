# 在學校 4000 Ada 電腦執行 SOP

> 情境：在家用 5070 下載好 96 變數 `.nc`，整包（含 `data/`）放上 Google 雲端 →
> 在學校電腦下載雲端資料夾 → 跑 96 變數實驗。
>
> 全部架構由**同一支** `fourier_2d.py` 跑出來；多機多卡細節見 `DISTRIBUTED.md`。

---

## 0. 把雲端資料夾下載到學校電腦

從 Google 雲端下載整個資料夾後，確認結構大致如下（重點是 `data/` 裡有 96 變數檔）：

```
multiscale-fourier-neural-operator-feat-10-day-rollout-/
├── fourier_2d.py / models.py / utilities3.py / Adam.py
├── environment.yml
├── data/
│   ├── global_era5_96_factors_g00.nc   # 每批 2 變數 × 完整三年（共 48 個），loader 自動合併
│   ├── global_era5_96_factors_g01.nc
│   └── ... (g02 ~ g47)
├── data_tools/  ├── analysis/
├── run_ddp.sh / run_ddp.ps1
├── DISTRIBUTED.md / SCHOOL.md / README.md
```

> 若雲端資料夾**沒有**附 `.nc`（例如檔案太大沒上傳），就在學校照第 3 節重新下載。

---

## 1. 啟動環境

學校電腦若已建好 conda 環境（`4000Ada_unet_cuda`）：

```bash
conda activate 4000Ada_unet_cuda
```

若是全新電腦，用本資料夾的 `environment.yml` 建（**兩步驟**）：

```bash
conda env create -f environment.yml
conda activate 4000Ada_unet_cuda
# torch-harmonics 一定要單獨、加 --no-deps 裝（否則 Windows 會卡 triton）：
pip install torch-harmonics==0.6.5 --no-deps
```

確認關鍵套件都在：

```bash
python -c "import torch, xarray; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```
- 應印出 `cuda True`（在 4000 Ada 上）。
- `torch_harmonics` 不要單獨 `python -c "import torch_harmonics"` 測（Windows 會因缺 triton 報錯）；
  實際執行 `fourier_2d.py` 時有內建假 triton 補丁會自動處理，能跑就對了。
- 若要在學校重新下載資料：`cdsapi` 已在環境內，設定好 `~/.cdsapirc` 即可。

---

## 2. 確認資料

```bash
python data_tools/check_data.py        # 檢視 .nc 內容（變數、維度）
```
- `fourier_2d.py` 預設讀 `data/global_era5_6_factors_*.nc`，所以 96 變數一定要**手動指定** glob：
  `--data-glob "data/global_era5_96_factors_*.nc"`

---

## 3.（只有缺資料時）在學校重新下載 96 變數

```bash
python data_tools/download_global_96_factors.py
```
- 需要 `cdsapi` + `~/.cdsapirc`；下載量大、可能數小時。
- 產生 `data/global_era5_96_factors_2021/2022/2023.nc`。

---

## 4. 先單卡冒煙測試（強烈建議，2 分鐘內）

正式長跑前，先用少 epoch 確認資料/環境/架構都沒問題：

```bash
python fourier_2d.py unet_2d --data-glob "data/global_era5_96_factors_*.nc" --epochs 2
```
- 跑得出 `outputs/unet_2d/training_log.csv` 就 OK。
- 測完那個資料夾要砍掉（不然正式跑會被「防覆蓋保護」擋下）：`rm -rf outputs/unet_2d`

---

## 5. 正式訓練

### (a) 單卡，跑單一架構
```bash
python fourier_2d.py unet_2d      --data-glob "data/global_era5_96_factors_*.nc"
python fourier_2d.py sphere_unet  --data-glob "data/global_era5_96_factors_*.nc"
python fourier_2d.py sufno --seed 1 --data-glob "data/global_era5_96_factors_*.nc"
```
可用架構：`unet_2d unetpp_2d transunet_2d sufno sunetpp_fno sutrans_fno
sphere_unet sphere_unetpp sphere_transunet 2d_fno sfno 2d_ufno 2d_unet`

### (b) 單機多卡（一台多張 GPU，DDP 一起訓練同個模型）
```bash
torchrun --standalone --nproc_per_node=2 \
    fourier_2d.py unet_2d --data-glob "data/global_era5_96_factors_*.nc"
```

### (c) 多機多卡（多台 4000 Ada）
詳見 `DISTRIBUTED.md`；或用 `run_ddp.sh`（Linux）/`run_ddp.ps1`（Windows）範本，
每台改 `node_rank` 後執行。重點：所有機器填同一個 `master_addr/master_port`，同網段。

> Windows 不支援 NCCL，會自動退回 gloo（較慢、需放行 master port）；Linux 會自動用 NCCL。

---

## 6. 產物與收集

每個實驗輸出在 `outputs/<架構名>/`：
- `training_log.csv`（逐 epoch、逐通道 MSE）
- `config.json`（超參數 + best/final MSE）
- `model_weights_best.pt` / `model_weights.pt`
- `learning_curve.png` / `forecast_skill.png` / `weather_prediction.png` / `weather_prediction_3d.png`

跑完把各機 `outputs/` 收回同一處，再用 `analysis/` 的腳本做比較與畫圖。

---

## 7. 常見問題

| 症狀 | 處理 |
|---|---|
| `FileExistsError 防覆蓋保護` | 該 `outputs/<name>/` 已有結果；改 `--seed` 或 `rm -rf` 該資料夾 |
| `torch_harmonics` import 失敗 | 程式內建假 triton 補丁會自動處理；仍失敗就 `pip install torch-harmonics==0.6.5` |
| 多機卡住 | master IP/port 不通或防火牆擋住；先 `ping` master、確認 port |
| 記憶體不足 | 調小 `--batch-size`；或減少 `download_global_96_factors.py` 的變數數 |
| 變數順序/通道想固定 | 用 `--vars a,b,c,...` 指定白名單順序（留空＝自動偵測檔案內全部變數）|
