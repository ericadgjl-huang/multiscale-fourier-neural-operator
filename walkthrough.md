# 5 台電腦跑模型 SOP（從環境架設 → 跑模型 → 上傳結果到雲端）

> 目標：借 5 台學校電腦（NVIDIA RTX 4000 Ada），一台跑一個模型，全部跑完後把
> `outputs/` 的結果上傳到同一個雲端共享資料夾。
>
> 全部 5 個模型都由**同一支** `fourier_2d.py` 跑出來，只差一個命令列參數，不需要改程式碼。

---

## 0. 模型分配表（每台電腦負責一個）

| 電腦 | 類型 | 模型 | 執行指令 | 輸出資料夾 |
|---|---|---|---|---|
| 電腦 1 | 球體 | FNO (SFNO) | `python fourier_2d.py sfno` | `outputs/sfno/` |
| 電腦 2 | 球體 | UFNO (SUFNO) | `python fourier_2d.py sufno` | `outputs/sufno/` |
| 電腦 3 | 平面 2D | FNO | `python fourier_2d.py 2d_fno` | `outputs/2d_fno/` |
| 電腦 4 | 平面 2D | UFNO | `python fourier_2d.py 2d_ufno` | `outputs/2d_ufno/` |
| 電腦 5 | 平面 2D | UNet | `python fourier_2d.py 2d_unet` | `outputs/2d_unet/` |

> ⚠️ 模型名稱一定要打對（大小寫、底線都要對）。打錯會直接報「未知的實驗名稱」並列出可用選項。

---

## Part A：在你自己的電腦（RTX 5070）做一次

這部分只做一次，目的是 (1) 把改好的程式碼推上 GitHub、(2) 把資料放到雲端讓 5 台電腦下載。

### A1. 把程式碼推到 GitHub

這個資料夾目前**還不是 git repo**（是從 ZIP 解壓出來的），所以要先接上遠端再推。
在 PowerShell 中，於專案根目錄執行：

```powershell
# 1) 初始化並接上你的遠端 repo
git init
git remote add origin https://github.com/ericadgjl-huang/multiscale-fourier-neural-operator.git

# 2) 抓下遠端的目標分支
git fetch origin

# 3) 讓 HEAD 對齊遠端分支，但保留你本機改好的檔案
git reset --soft origin/feat-10-day-rollout翔

# 4) 設定提交身分（若這台電腦沒設過）
git config user.name  "你的名字"
git config user.email "ericadgjl@gmail.com"

# 5) 把你的修改打包成一個 commit
git add -A
git commit -m "Fix 2d_unet spectral path + add CLI arg + font fallback + walkthrough"

# 6) 推回原本那個分支
git push origin HEAD:feat-10-day-rollout翔
```

- 第一次 push 會跳出 GitHub 登入視窗（Git Credential Manager），用瀏覽器登入即可。
- 如果 `git reset --soft origin/feat-10-day-rollout翔` 找不到分支，先用 `git branch -r` 確認遠端分支的正確全名再貼上。
- 推完後到 GitHub 網頁確認 `fourier_2d.py` 的修改時間有更新。

### A2. 準備資料並上傳到雲端共享資料夾

訓練需要 3 個 ERA5 資料檔，這些檔案**不在 repo 裡**（被 `.gitignore` 排除），所以要另外發給每台電腦。

1. 在你的電腦找出這 3 個檔（你之前下載/訓練用的）：
   ```
   data/global_era5_6_factors_2021.nc
   data/global_era5_6_factors_2022.nc
   data/global_era5_6_factors_2023.nc
   ```
   （三個加起來大約 200~300 MB，很小，用雲端傳很快。）

2. 在 Google Drive 建一個**共享資料夾**，例如 `BigData_Final/`，裡面開兩個子資料夾：
   ```
   BigData_Final/
   ├── data/        ← 放這 3 個 .nc 檔
   └── results/     ← 之後每台電腦上傳結果到這
   ```
   把資料夾權限設成「知道連結的人可檢視」，並把連結記下來給 5 台電腦用。

3. 把 3 個 `.nc` 檔上傳到 `BigData_Final/data/`。

> 💡 如果你手邊沒有資料檔，可在任一台電腦設定好 `~/.cdsapirc`（CDS 帳號）後執行
> `python data_tools/download_global_6_factors.py` 下載。但 CDS 會排隊、可能要等很久，
> **強烈建議用上面「下載一次 → 雲端分發」的方式**，不要讓 5 台電腦各自去 CDS 排隊。

---

## Part B：在「每一台」學校電腦（4000 Ada）做

以下每台電腦都做一遍，差別只有 **B4 的那一行指令**（看你這台負責哪個模型）。

### B0. 前置確認
- 確認電腦有裝 **Anaconda / Miniconda** 和 **Git**。
  - 沒有 conda → 裝 Miniconda：https://docs.conda.io/en/latest/miniconda.html
  - 沒有 git → 裝 Git for Windows：https://git-scm.com/download/win
- 確認顯卡正常：開 PowerShell 打 `nvidia-smi`，要看得到 `RTX 4000 Ada` 和驅動版本。

### B1. 取得程式碼

```powershell
# 找個工作目錄，例如 D:\
cd D:\
git clone -b feat-10-day-rollout翔 https://github.com/ericadgjl-huang/multiscale-fourier-neural-operator.git
cd multiscale-fourier-neural-operator
```

> 若分支名稱含中文造成困擾，也可改用：`git clone <url>` 後再 `git checkout feat-10-day-rollout翔`。

### B2. 建立 conda 環境

`environment.yml` 已經幫你鎖好對應 4000 Ada 的版本（PyTorch 2.2.0 + CUDA 12.1 + torch-harmonics）。

```powershell
conda env create -f environment.yml
conda activate 4000Ada_unet_cuda
```

建完後**務必驗證 GPU 抓得到**：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

要看到類似：`2.2.0+cu121 True NVIDIA RTX 4000 Ada Generation`。
如果 `True` 變成 `False`，先別跑，回到 Part C 排查。

### B3. 取得資料

從 Part A2 的雲端 `BigData_Final/data/` 下載 3 個 `.nc` 檔，放進專案的 `data/` 資料夾：

```
multiscale-fourier-neural-operator/
└── data/
    ├── global_era5_6_factors_2021.nc
    ├── global_era5_6_factors_2022.nc
    └── global_era5_6_factors_2023.nc
```

> `data/` 資料夾若不存在，自己建一個（`mkdir data`）。檔名一定要對，程式是用
> `data/global_era5_6_factors_*.nc` 這個樣式去抓的。

### B4. 開始訓練（看這台負責哪個模型，挑「一行」執行）

建議用 `Tee-Object` 同時印在畫面上、又存一份 log，方便事後檢查：

```powershell
# 電腦 1：球體 FNO
python fourier_2d.py sfno      2>&1 | Tee-Object -FilePath run_sfno.log

# 電腦 2：球體 UFNO
python fourier_2d.py sufno     2>&1 | Tee-Object -FilePath run_sufno.log

# 電腦 3：平面 FNO
python fourier_2d.py 2d_fno    2>&1 | Tee-Object -FilePath run_2d_fno.log

# 電腦 4：平面 UFNO
python fourier_2d.py 2d_ufno   2>&1 | Tee-Object -FilePath run_2d_ufno.log

# 電腦 5：平面 UNet
python fourier_2d.py 2d_unet   2>&1 | Tee-Object -FilePath run_2d_unet.log
```

**開跑後先別離開**，盯著看到：
1. 印出「實驗：xxx」「模型總參數」等設定資訊（代表資料讀進來了、模型建好了）。
2. 出現第一行 `Epoch 00 | ... | Test MSE: ...`（代表完整一個 epoch 能跑完、沒有崩潰）。
3. `outputs/<模型>/training_log.csv` 開始有資料。

確認這三點後就可以放著讓它跑完 50 個 epoch（會花數小時，視模型而定，球體 SHT 模型通常較慢）。
跑的過程中**不要關掉這個 PowerShell 視窗**，也建議把電腦的「睡眠 / 自動關機」設成永不。

> 🔁 想重跑同一個模型？因為有防覆蓋保護，要先刪掉舊資料夾（例如 `Remove-Item -Recurse -Force outputs\sfno`）再跑。

### B5. 確認完成並上傳結果到雲端

跑完最後會印出：`========== 全部完成！所有結果已儲存至 outputs/<模型>/ ==========`

該模型資料夾裡會有：
```
outputs/<模型>/
├── config.json              超參數快照（含 best_epoch / best_test_mse）
├── training_log.csv         每個 epoch 的逐通道 MSE（含 6 個變數各自的欄位）
├── model_weights.pt         最終權重
├── model_weights_best.pt    test MSE 最佳的權重
├── learning_curve.png       學習曲線
├── forecast_skill.png       ★ 逐通道 RMSE vs 預報時效
├── weather_prediction.png   溫度多時效誤差熱點圖
└── weather_prediction_3d.png 3D 球體預測圖
```

把整個資料夾打包後上傳到雲端 `BigData_Final/results/`：

```powershell
# 以電腦 1 為例，壓成一個 zip
Compress-Archive -Path outputs\sfno -DestinationPath sfno_results.zip
```

然後用瀏覽器把 `sfno_results.zip`（其它電腦換成對應名稱）拖進 Google Drive 的
`BigData_Final/results/` 即可。同時把 `run_<模型>.log` 也一併上傳，方便事後查問題。

---

## Part C：常見問題排查

| 症狀 | 原因 / 解法 |
|---|---|
| `torch.cuda.is_available()` 是 `False` | 驅動太舊或裝到 CPU 版 torch。先 `nvidia-smi` 看驅動；必要時 `pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.0 torchvision==0.17.0` 重裝。 |
| `FileNotFoundError` 找不到 `.nc` | 資料沒放對位置或檔名不符。確認 `data/global_era5_6_factors_2021.nc` 等三個檔存在。 |
| `FileExistsError [防覆蓋保護]` | 該模型的 `outputs/<模型>/` 已有舊紀錄。刪掉資料夾或改 `SEED` 重跑。 |
| `未知的實驗名稱` | CLI 參數打錯。可用：`sfno / sufno / 2d_fno / 2d_ufno / 2d_unet`。 |
| 中文字型警告 | 無害，圖照常產生，只是標題中文可能變方塊。 |
| `CUDA out of memory` | 把 `fourier_2d.py` 裡的 `batch_size`（預設 4）改成 2 再跑。 |
| `torch_harmonics` import 錯誤（只發生在球體模型） | 確認在 `4000Ada_unet_cuda` 環境內，且 `pip show torch-harmonics` 是 0.6.5。 |

---

## 附錄：逐通道（per-channel）RMSE 在哪裡看

是的，每個變數（通道）都有自己的誤差，分兩個地方：

1. **`training_log.csv`** — 每個 epoch 都記錄 6 個變數**各自的 MSE**：
   - 欄位 `train_mse_ch0_t2m` … `train_mse_ch5_vitoe`
   - 欄位 `test_mse_ch0_t2m`  … `test_mse_ch5_vitoe`
   - （想要 RMSE 就對該欄開根號。）

2. **`forecast_skill.png`** — 6 張子圖，每個變數一張，畫的是**該通道的 RMSE 隨預報時效（lead time）變化**，
   並標出 Day 5 / Day 10 兩條參考線。這是真正逐通道的 RMSE 曲線。

> 註：所有 MSE / RMSE 都是在**標準化後**的尺度上計算（不是原始物理單位），
> 用來比較模型好壞沒問題；若要還原成攝氏、百帕等物理單位，需乘回各通道的標準差 `x_std`。
