# TAAI 投稿論文大綱 — 96 變數延伸（cowork 用：產 Word + PPT）

> 本檔給家裡電腦用 cowork 產出 Word 論文與 PPT 簡報。
> 定位：延續期末報告《When Sphere Hurts》(4 變數主實驗 + 6 變數延伸)，
> 升級成以「**變數豐富度 (state richness)**」為新軸的研究，主打 96 變數實驗發現的
> **cost–benefit 交叉 (crossover)**。
>
> 資料來源：`outputs/`（4/6 變數，舊）、`outputs96/`（96 變數，新）。
> 圖表已產於 `outputs96/_comparison/`（需手動上傳雲端，該資料夾被 .gitignore 擋住不進 git）。

---

## 0. 一句話主張 (One-liner)

低解析度 (33×64) 下，**純球面（SHT-only）神經算子始終拖累**（跨 4/6/96 變數，96 變數時比最佳架構差約 **19–21%**，n=3 下穩健）。
把「全域頻譜 + UNet 局部路徑」（FNO 混合）加到 UNet 骨幹後，隨**變數變多**從負擔翻成資產——
96 變數時最佳 hybrid（sunetpp_fno 0.3915）勝過純卷積平面基線約 **2.5%**（n=3 顯著）。
但關鍵對照 **2d_ufno**（平面 FFT+UNet）顯示：這份增益主要來自 **FNO+UNet 結構、而非球面座標**——
平面 FFT+UNet（0.3951）與球面 SHT+UNet（0.3980）幾乎打平（平面甚至略勝），
故本文主張「幾何取捨被變數豐富度調節，但**球面本身並非必須**；真正該避免的是純球面」。

---

## 1. 核心數據

### 1.1 96 變數 3×3 消融（best test MSE，越低越好；planar/hybrid **n=3**、sphere **n=1**）

| 架構家族 | Planar (2D + lon pad) | FNO Hybrid (SHT ⊕ conv) | Pure Spherical |
|---|---|---|---|
| **UNet**      | unet_2d 0.4037 ± 0.0004      | sufno **0.3980 ± 0.0005** ✅       | sphere_unet 0.4767 |
| **UNet++**    | unetpp_2d 0.4017 ± 0.0006    | sunetpp_fno **0.3915 ± 0.0005** 🏆 | sphere_unetpp 0.4780 |
| **TransUNet** | transunet_2d 0.4305 ± 0.0256 | sutrans_fno 0.4358 ± 0.0034        | sphere_transunet 0.5269 |

基線：2d_fno 0.4029 ± 0.0013、sfno 0.4043 ± 0.0007。（皆 n=3；sphere_* 仍 n=1）

### 1.2 排序翻轉（全文最關鍵的對照）

| 變數數 | 排序 | 最佳架構 |
|---|---|---|
| 4 變數（舊論文，n=3） | **平面 < FNO 混合 < 純球面** | 平面 unet_2d (0.3751) |
| 96 變數（本文，n=3） | **FNO 混合 < 平面 < 純球面** | FNO 混合 sunetpp_fno (0.3915) |

- 家族內 hybrid vs 自家平面版（n=3）：**UNet +1.4%**、**UNet++ +2.5%**（hybrid 較佳、顯著）；
  **TransUNet −1.2%**（平面 transunet_2d 0.4305 反而略勝 hybrid sutrans_fno 0.4358，但 transunet_2d 變異大 ±0.0256 → 未定論）。
  → 翻轉為 **2/3 家族一致**，非全部；TransUNet 屬例外。
- ⚠ 平面對照 2d_ufno（FFT+UNet，0.3951 ± 0.0005）幾乎追平最佳 hybrid（sunetpp_fno 0.3915，差 0.9%）；
  以「最佳平面 vs 最佳 hybrid」計，hybrid 領先僅約 **0.9%**（見 §6 控制實驗 → 球面非必須）。
- 純球面（SHT-only）**永遠墊底**：96 變數下比最佳架構差約 **19–21%**（0.477–0.527 vs ~0.39–0.40），n=3 下穩健。

### 1.3 資源效率（96 變數）

> 各欄來源（硬體須誠實標註）：
> - **參數**：config.json 直接讀，與硬體無關。
> - **epoch / total train time**：training_log.csv 累計，反映**訓練硬體 = 多台 RTX 4000 Ada**。
> - **推論延遲 / 峰值記憶體**：`resource_comparison.py` 分析時**於單台 RTX 5070 桌機量測**（一次執行、全架構同機，故絕對值為裝置相依、**相對排序有效**）。
> - **bestMSE 欄已改用 n=3 平均**（來自 `multi_seed_summary.csv`），故排序略有變動。
> - 平面對照 **2d_ufno**（FFT+UNet，bestMSE 0.3951、4.29M）尚未納入本表（`resource_comparison.py` 未重跑）。

```
架構           幾何       參數   epoch(min) 推論(ms) bestMSE(n=3)
sunetpp_fno   hybrid    3.04M   8.77      5.33    0.3915  🏆最準
sufno         hybrid    2.20M   7.17      4.12    0.3980
unetpp_2d     planar    2.27M   4.36      2.17    0.4017
2d_fno        planar    4.22M   4.00      2.04    0.4029
unet_2d       planar    2.37M   3.64      1.76    0.4037  ⚡最快
sfno          spherical 2.12M   5.52      3.45    0.4043
transunet_2d  planar    1.42M   5.98      2.61    0.4305  (±0.0256 高變異)
sutrans_fno   hybrid    2.28M  12.78      6.33    0.4358
sphere_unet   spherical 2.52M  11.61      7.56    0.4767
sphere_unetpp spherical 2.65M  13.56      8.89    0.4780
sphere_transunet spherical 3.34M 14.49    8.87    0.5269  墊底
```

三方權衡：**平面 = 效率最佳（且 2d_ufno 準度直逼最佳 hybrid）；FNO 混合 = 準度最佳；純球面 = 又慢又差被全面輾壓。**

---

## 2. 建議標題（擇一精修）

- EN: **"When Sphere Hurts — and When It Starts to Help: State Richness Modulates the Geometry Trade-off in Low-Resolution Weather Forecasting"**
- ZH: **《球面何時拖累、何時翻身：變數豐富度如何調節低解析度天氣預測的幾何取捨》**

---

## 3. 論文章節大綱（英文撰寫、IEEE 雙欄、約 6–8 頁）

### Title / Abstract
- Hook：低解析度純球面始終拖累；FNO 混合優勢隨變數數翻轉，96 變數時最佳 hybrid 小幅勝過純卷積平面，
  但平面 FFT+UNet 對照顯示這份增益源於 FNO+UNet 結構、球面並非必須。
- 三點貢獻：
  1. 把 architecture × geometry ablation 從 4 → 6 → 96 變數延伸，引入「變數豐富度」新軸。
  2. 發現 FNO-hybrid vs planar 的 **crossover**（n=3；UNet/UNet++ 一致，TransUNet 例外；
     並以 2d_ufno 對照證明增益來自 FNO+UNet 而非球面座標）。
  3. 純球面（SHT-only）劣勢跨變數數穩健（~20%）。
- Keywords: neural operator, spherical harmonic transform, weather forecasting, ablation study, ERA5, channel scaling, U-Net.

### 1. Introduction
- 既有文獻假設「球面感知必要」建立於高解析度；本團隊前作已在低解析度質疑之。
- 新問題：**固定低解析度下，增加 prognostic 變數 (4→96) 會否改變幾何取捨？**
- 預告 crossover 主結論 + 3×3 熱力圖。

### 2. Related Work
- 2.1 Neural Operators：FNO (FFT)、SFNO (SHT)；spectral path ∥ local path 結構。
- 2.2 DL 天氣預測：FourCastNet / Pangu / GraphCast（皆高解析度 + 球面感知）。
- 2.3 U-Net 家族：UNet / UNet++ / TransUNet。
- 2.4（新增）多變數 / channel scaling 對神經算子的影響——點出此軸少被系統性探討。

### 3. Method
- 3.1 問題定義：33×64 網格、10 天 rollout（40 步，每步 6h）；輸入 = **96 氣象通道 + 4 時間編碼 + 2 網格座標**。
- 3.2 架構 × 幾何 3×3 設計矩陣（11 個模型，沿用設計表）。
- 3.3 幾何處理光譜：Planar+lon pad（半步）/ FNO Hybrid（全套）/ Pure Spherical（純 SHT）；保留「論證護欄」段落。
- 3.4（新增）**變數豐富度軸**：4 / 6 / 96 變數三個 regime 的定義與資料來源（說明 96 factor = 多層位 × 變數）。
- 3.5 訓練與評估：Truncated BPTT (K=8)、step-weighted MSE (γ=0.95)、Adam (lr=1e-3, wd=1e-4, clip=1.0)、CosineAnnealingLR、50 epoch、batch=4。

### 4. Experimental Setup
- 4.1 資料：96 變數 ERA5，2021–23、6h 一筆；train 2021–22 / test 2023；標準化僅用訓練集統計量。
- 4.2 硬體：訓練用**多台 NVIDIA RTX 4000 Ada 分散訓練**（呼應本次多機流程）；
  §1.3 的**推論延遲與峰值記憶體則於單台 RTX 5070 桌機量測**（全架構同機一次跑完，供相對比較；絕對值為裝置相依，論文表格須註明此點）。
- 4.3 超參數：base width 32；FNO modes 16；sphere modes (8,4,2)；Transformer bottleneck 4 層 / 4 head / FFN×4。

### 5. Results
- 5.1 **核心：96 變數 3×3 熱力圖**（FNO-hybrid 欄含全域最佳）→ `final_ablation_heatmap.png`
- 5.2 **Crossover 分析**：4-var vs 96-var 排序翻轉對照 → `crossover.png` / `crossover_absolute.png`（已產）。
- 5.3 跨架構一致性：UNet / UNet++ 家族 hybrid 贏平面（顯著），TransUNet 例外（高變異）→ `comparison_final_metrics.png`
- 5.4 純球面的穩健劣勢（~20%，跨 4/6/96 變數）。
- 5.5 **控制實驗（球面是否必須）**：固定 local、換 spectral（FFT vs SHT）的 2×2 →
  平面 FFT+UNet ≈ 球面 SHT+UNet（平面略勝）→ `crossover_control_2x2.csv`。
- 5.6 資源效率：球面「又慢又差」是否仍成立？三方權衡 → `resource_comparison.png`
- 5.7 質化：Day-10 預測並排圖 → `cross_arch_day10_comparison.png`

- 6.1 為何 spectral+UNet 在高變數翻身：多個高相關變數下，全域頻譜混合能抓跨通道結構，平行 conv 路徑又保住 locality；純卷積平面基線缺全域路徑，純球面又缺 conv 路徑（被 truncation / locality loss 拖累）。
- 6.2 **球面是否必須（核心控制）**：固定 UNet local、只換頻譜路徑，平面 FFT+UNet（2d_ufno 0.3951）與球面 SHT+UNet（sufno 0.3980）幾乎打平、平面甚至略勝；固定 1×1 local 亦同（2d_fno 0.4029 vs sfno 0.4043）。→ 增益來自 **FNO+UNet 結構、非球面座標**；球面在此解析度非但非必須，反而略拖累。
- 6.3 為何純球面（SHT-only）不變：劣勢是解析度 (lmax≤16) 綁定，加變數救不了。
- 6.4 為何 TransUNet 家族雙雙掉隊：Transformer bottleneck 不利於高通道數（4 變數時它很能打）；且其平面版 seed 間變異大，家族內翻轉未定論。
- 6.5 對實務者：低解析度**多變數**原型 → 首選 spectral+UNet（sunetpp_fno 最準，或用更省的平面 **2d_ufno**，準度直逼、無需 SHT 依賴）；變數很少 → 平面 CNN 足矣；任何情況都先別上純球面。

### 7. Limitations & Future Work
- 平面 / hybrid 已補到 **n=3**：家族內 hybrid vs planar 在 UNet / UNet++ 已達統計顯著（std ~0.0005）；但「最佳平面（含 2d_ufno）vs 最佳 hybrid」僅差 0.9%，**crossover 幅度小、須審慎陳述**。
- **純球面仍為 n=1**，惟其 ~20% 劣勢遠大於 seed 雜訊，結論穩健；未來仍可補 seed 收尾。
- TransUNet 家族平面版變異大（±0.0256）→ 該家族翻轉未定論，值得補更多 seed 或排查不穩定來源。
- 高解析度是否再次翻轉；其他資料集 (IFS, MERRA-2)；季節尺度；與 graph-based 方法比較。

---

## 4. 論述誠實性（務必遵守）

- 平面 / hybrid 為 **n=3**（sphere 仍 n=1）。hybrid vs planar 家族內差異在 UNet / UNet++ 已統計顯著，可如實聲稱。
- **不可再宣稱「三家族一致翻轉」** —— TransUNet 家族平面反而略勝（且高變異），須如實寫「2/3 家族」。
- 最強、最穩的錘：**純球面（SHT-only）差約 20%**（跨 4/6/96 變數穩健）。
- 誠實面對 2d_ufno 對照：平面 FFT+UNet ≈ 球面 SHT+UNet（平面略勝）→ 增益來自 **FNO+UNet 結構、非球面**；「hybrid 反超平面」以最佳-對-最佳計僅約 0.9%，宜定位為「小幅、方向大致一致（除 TransUNet）」。
- crossover 已具 n=3 統計基礎，可從「observed trend」升格為「supported result」，惟幅度須誠實標明。

---

## 5. 圖表清單（皆已產於 `outputs96/_comparison/`）

| 論文用途 | 檔名 | 狀態 |
|---|---|---|
| 主圖：3×3 熱力圖 | `final_ablation_heatmap.png` | ✅ 已產 |
| 排序長條 + 表 | `final_ablation_bars.png` / `final_ablation_table.csv` | ✅ |
| 最終 MSE / 參數量 | `comparison_final_metrics.png` / `comparison_summary.csv` | ✅ |
| 學習曲線同框 | `comparison_learning_curves.png` | ✅ |
| Day-10 預測並排 | `cross_arch_day10_comparison.png` | ✅ |
| 資源效率四面板 | `resource_comparison.png` / `resource_summary.csv` | ✅ |
| **Crossover 翻轉圖（4 vs 96）** | `crossover.png` / `crossover_absolute.png`（`crossover_plot.py`） | ✅ 已產 |

---

## 6. cowork 產出建議

1. **Word 論文**：依 §3 章節順序撰寫；數字直接引用 §1 的表格；圖插入 §5 清單檔案。
2. **PPT 簡報**：沿用期末報告風格，主軸改為「crossover」故事——
   - 開場：舊結論（4 變數平面贏）→ 提問（變數變多會怎樣？）→ 96 變數 hybrid 小幅勝出
     → 但 2d_ufno 對照證明「球面非必須」→ 純球面永遠輸（~20%）。
3. crossover 翻轉圖（`crossover.png` / `crossover_absolute.png`）已產，是簡報高潮，直接採用。
