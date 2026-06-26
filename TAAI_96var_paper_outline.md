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

低解析度 (33×64) 下，**純球面神經算子始終拖累**（跨 4/6/96 變數，準度差約 20%）；
但 **FNO 混合**的球面頻譜路徑會隨**變數變多**從負擔翻成資產——在 96 變數時**反超平面 CNN**。
幾何的取捨不是固定的，而是被「變數豐富度」調節的。

---

## 1. 核心數據

### 1.1 96 變數 3×3 消融（best test MSE，越低越好，目前 **n=1**）

| 架構家族 | Planar (2D + lon pad) | FNO Hybrid (SHT ⊕ conv) | Pure Spherical |
|---|---|---|---|
| **UNet**      | unet_2d **0.4033**   | sufno **0.3981** ✅       | sphere_unet 0.4767 |
| **UNet++**    | unetpp_2d 0.4022     | sunetpp_fno **0.3912** 🏆 | sphere_unetpp 0.4780 |
| **TransUNet** | transunet_2d 0.4418  | sutrans_fno **0.4328** ✅ | sphere_transunet 0.5269 |

基線：2d_fno 0.4035、sfno 0.4044。

### 1.2 排序翻轉（全文最關鍵的對照）

| 變數數 | 排序 | 最佳架構 |
|---|---|---|
| 4 變數（舊論文，n=3） | **平面 < FNO 混合 < 純球面** | 平面 unet_2d (0.3751) |
| 96 變數（本文，n=1） | **FNO 混合 < 平面 < 純球面** | FNO 混合 sunetpp_fno (0.3912) |

- 三個家族「**全部一致**」翻轉：FNO 混合各贏自家平面版約 1.3% / 2.7% / 2.0%（UNet / UNet++ / TransUNet）。
- 純球面**永遠墊底**：96 變數下比同族最佳差約 **20%**（0.477–0.527 vs ~0.40）。

### 1.3 資源效率（96 變數，來自 `resource_summary.csv`）

```
架構           幾何       參數   epoch(min) 推論(ms) bestMSE
sunetpp_fno   hybrid    3.04M   8.77      5.33    0.3912  🏆最準
sufno         hybrid    2.20M   7.17      4.12    0.3981
unetpp_2d     planar    2.27M   4.36      2.17    0.4022
unet_2d       planar    2.37M   3.64      1.76    0.4033  ⚡最快
2d_fno        planar    4.22M   4.00      2.04    0.4035
sfno          spherical 2.12M   5.52      3.45    0.4044
sutrans_fno   hybrid    2.28M  12.78      6.33    0.4328
transunet_2d  planar    1.42M   5.98      2.61    0.4418
sphere_unet   spherical 2.52M  11.61      7.56    0.4767
sphere_unetpp spherical 2.65M  13.56      8.89    0.4780
sphere_transunet spherical 3.34M 14.49    8.87    0.5269  墊底
```

三方權衡：**平面 = 效率最佳；FNO 混合 = 準度最佳；純球面 = 又慢又差被全面輾壓。**

---

## 2. 建議標題（擇一精修）

- EN: **"When Sphere Hurts — and When It Starts to Help: State Richness Modulates the Geometry Trade-off in Low-Resolution Weather Forecasting"**
- ZH: **《球面何時拖累、何時翻身：變數豐富度如何調節低解析度天氣預測的幾何取捨》**

---

## 3. 論文章節大綱（英文撰寫、IEEE 雙欄、約 6–8 頁）

### Title / Abstract
- Hook：低解析度純球面始終拖累，但 FNO 混合優勢隨變數數翻轉，96 變數時超越平面 CNN。
- 三點貢獻：
  1. 把 architecture × geometry ablation 從 4 → 6 → 96 變數延伸，引入「變數豐富度」新軸。
  2. 發現 FNO-hybrid vs planar 的 **crossover**（三家族一致同向）。
  3. 純球面劣勢跨變數數穩健（~20%）。
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
- 4.2 硬體：**多台 NVIDIA RTX 4000 Ada 分散訓練**（呼應本次多機流程）。
- 4.3 超參數：base width 32；FNO modes 16；sphere modes (8,4,2)；Transformer bottleneck 4 層 / 4 head / FFN×4。

### 5. Results
- 5.1 **核心：96 變數 3×3 熱力圖**（FNO-hybrid 欄最深）→ `final_ablation_heatmap.png`
- 5.2 **Crossover 分析**：4-var vs 96-var 排序翻轉對照（建議新做一張「ordering flip」圖，見 §6）。
- 5.3 跨架構一致性：三家族 hybrid 都贏平面 → `comparison_final_metrics.png`
- 5.4 純球面的穩健劣勢（~20%，跨 4/6/96 變數）。
- 5.5 資源效率：球面「又慢又差」是否仍成立？三方權衡 → `resource_comparison.png`
- 5.6 質化：Day-10 預測並排圖 → `cross_arch_day10_comparison.png`

### 6. Discussion
- 6.1 為何 hybrid 在高變數翻身：多個高相關變數下，SHT 全域頻譜混合能抓跨通道結構，平行 conv 路徑又保住 locality；純球面缺 conv 路徑，仍被 truncation / locality loss 拖累。
- 6.2 為何純球面不變：劣勢是解析度 (lmax≤16) 綁定，加變數救不了。
- 6.3 為何 TransUNet 在 96 變數掉隊：Transformer bottleneck 不利於高通道數（4 變數時它很能打）。
- 6.4 對實務者：低解析度**多變數**原型 → 首選 FNO-hybrid（尤其 UNet++）；變數很少 → 平面 CNN 足矣；任何情況都先別上純球面。

### 7. Limitations & Future Work
- ⚠️ 96 變數目前 **n=1**：hybrid 對 planar 的領先（1–3%）尚未統計顯著；**純球面的 ~20% gap 才是鐵結論**。
- 補多種子（至少把便宜的平面/hybrid 各跑 3 seeds）使 crossover 升格為主結論。
- 高解析度是否再次翻轉；其他資料集 (IFS, MERRA-2)；季節尺度；與 graph-based 方法比較。

---

## 4. 論述誠實性（務必遵守）

- 96 變數是**單一種子**。hybrid 贏 planar 僅 1–3%，n=1 下**不可聲稱統計顯著**。
- 可用「**三家族一致同向翻轉**」當佐證（系統性 > 雜訊）。
- 主錘打「**純球面差約 20%**（穩健、跨變數數）」。
- crossover 定位為「observed trend，需多種子確認」；若時間允許補 3 seeds 即可升格主結論。

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
| **Crossover 翻轉圖（4 vs 96）** | （尚未產，建議補做 `crossover_plot.py`） | ⬜ 待補 |

---

## 6. cowork 產出建議

1. **Word 論文**：依 §3 章節順序撰寫；數字直接引用 §1 的表格；圖插入 §5 清單檔案。
2. **PPT 簡報**：沿用期末報告風格，主軸改為「crossover」故事——
   - 開場：舊結論（4 變數平面贏）→ 提問（變數變多會怎樣？）→ 翻轉（96 變數 hybrid 贏）→ 但純球面永遠輸。
3. 缺的那張 crossover 翻轉圖是簡報高潮，建議優先補。
