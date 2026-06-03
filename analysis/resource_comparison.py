"""
resource_comparison.py（重寫版：對應新的 5 模型，全部是 fourier_2d.py 的 FNO2d 變體）

從既有實驗結果聚合「模型大小 / 訓練時間 / 推論時間 / 峰值記憶體」對照表。
不需要重新訓練：
- Params (M)：從 config.json 直接讀
- Avg epoch time / Total train time：從 training_log.csv 累計
- Peak GPU memory + Inference latency：載入 model_weights_best.pt 做 forward 量測

支援的 5 個模型（與 fourier_2d.py 的 EXPERIMENTS 一致）：
    2d_fno / 2d_ufno / sfno / sufno / 2d_unet

用法：python analysis/resource_comparison.py（從專案根目錄執行）

輸出：
    outputs/_comparison/resource_summary.csv
    outputs/_comparison/resource_comparison.png
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import json
import glob
import time

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

# 中文字體（找不到就跳過）
font_path = r"C:\Windows\Fonts\msjh.ttc"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUTS_DIR = 'outputs'
COMPARE_DIR = os.path.join(OUTPUTS_DIR, '_comparison')
os.makedirs(COMPARE_DIR, exist_ok=True)

# fourier_2d.py 的 5 個實驗設定（與該檔 EXPERIMENTS 對齊）
FNO_EXPERIMENTS = {
    '2d_fno':  {'local_type': '1x1',  'spectral_type': 'fft'},
    '2d_ufno': {'local_type': 'unet', 'spectral_type': 'fft'},
    'sfno':    {'local_type': '1x1',  'spectral_type': 'sht'},
    'sufno':   {'local_type': 'unet', 'spectral_type': 'sht'},
    '2d_unet': {'local_type': 'unet', 'spectral_type': ''},
}

CAT_COLOR = {'planar': '#2ecc71', 'sphere': '#e74c3c'}
CAT_LABEL = {'planar': '平面 2D (FFT / 純 CNN)', 'sphere': '球體 (SHT)'}


def categorize(name):
    """球體 = sfno / sufno；其餘（2d_*）為平面。"""
    return 'sphere' if name in ('sfno', 'sufno') else 'planar'


################################################################
# 用「截斷 exec」的方式只載入 fourier_2d.py 的 class 定義（不觸發訓練）
################################################################
_FNO2D_CLASS = None
def _load_fno2d_class():
    global _FNO2D_CLASS
    if _FNO2D_CLASS is None:
        src = open('fourier_2d.py', encoding='utf-8').read()
        cutoff = src.find('################################################################\n# ERA5RolloutDataset')
        ns = {}
        exec(compile(src[:cutoff], 'fourier_2d.py', 'exec'), ns)
        _FNO2D_CLASS = ns['FNO2d']
    return _FNO2D_CLASS


def build_model(group, cfg):
    """根據 group 名稱建立對應的 FNO2d（已 .cuda()）。失敗回傳 None。"""
    if group not in FNO_EXPERIMENTS:
        print(f"  [略過 {group}] 不在已知的 5 個模型內")
        return None
    try:
        FNO2d = _load_fno2d_class()
        ec = FNO_EXPERIMENTS[group]
        modes = cfg.get('modes', 16)
        width = cfg.get('width', 32)
        return FNO2d(modes, modes, width,
                     local_type=ec['local_type'],
                     spectral_type=ec['spectral_type'],
                     dropout=0.0).cuda()
    except Exception as e:
        print(f"  [略過 {group}] 模型建立失敗：{e}")
        return None


################################################################
# 1. 聚合既有 config + training_log（params / epoch time）
################################################################
records = []
for d in sorted(glob.glob(os.path.join(OUTPUTS_DIR, '*'))):
    if not os.path.isdir(d):
        continue
    name = os.path.basename(d)
    if name.startswith('_'):
        continue
    cfg_path = os.path.join(d, 'config.json')
    log_path = os.path.join(d, 'training_log.csv')
    if not (os.path.exists(cfg_path) and os.path.exists(log_path)):
        continue
    with open(cfg_path, encoding='utf-8') as f:
        cfg = json.load(f)
    df = pd.read_csv(log_path)
    if len(df) == 0:
        continue
    group = cfg.get('base_experiment_name', name)
    if group not in FNO_EXPERIMENTS:
        continue   # 只處理新的 5 模型
    records.append({
        'arch':            name,
        'group':           group,
        'dir':             d,
        'cfg':             cfg,
        'param_count':     cfg.get('param_count', 0),
        'avg_epoch_sec':   df['epoch_time_sec'].mean(),
        'total_train_sec': df['epoch_time_sec'].sum(),
        'epochs':          len(df),
        'best_test_mse':   cfg.get('best_test_mse', df['test_mse'].min()),
    })

if not records:
    raise RuntimeError(
        f"在 {OUTPUTS_DIR}/ 下找不到任何屬於 5 模型 "
        f"({', '.join(FNO_EXPERIMENTS)}) 的完成實驗"
    )

df_all = pd.DataFrame(records)
print(f"找到 {len(df_all)} 個實驗：{', '.join(df_all['group'])}")

################################################################
# 2. 對每個模型量測 inference latency + peak GPU memory
################################################################
print("\n正在量測各模型的 inference latency 與 peak GPU memory...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
inference_results = {}

for _, row in df_all.iterrows():
    group = row['group']
    cfg   = row['cfg']
    model = build_model(group, cfg)
    if model is None:
        continue

    weights_path = os.path.join(row['dir'], 'model_weights_best.pt')
    if not os.path.exists(weights_path):
        weights_path = os.path.join(row['dir'], 'model_weights.pt')
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
        except Exception as e:
            print(f"  [警告 {group}] 載入權重失敗（用 random init 量測）：{e}")
    model.eval()

    # 輸入 = 6 氣象 + 4 時間 = 10 通道（FNO2d 內部會再補 2 個 grid 通道）
    x = torch.randn(4, 33, 64, 10, device=device)
    with torch.no_grad():
        for _ in range(3):                      # warm up
            _ = model(x)
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    n_runs = 100
    with torch.no_grad():
        if device == 'cuda':
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        for _ in range(n_runs):
            _ = model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        t_end = time.perf_counter()
    inference_ms = (t_end - t_start) / n_runs * 1000

    peak_mem_MB = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == 'cuda' else float('nan')

    inference_results[group] = {
        'inference_ms_per_step': inference_ms,
        'peak_gpu_mem_MB':       peak_mem_MB,
    }
    print(f"  {group:<10} | inference {inference_ms:6.2f} ms/step | peak mem {peak_mem_MB:7.1f} MB")

    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

df_all['inference_ms_per_step'] = df_all['group'].map(lambda g: inference_results.get(g, {}).get('inference_ms_per_step', np.nan))
df_all['peak_gpu_mem_MB']       = df_all['group'].map(lambda g: inference_results.get(g, {}).get('peak_gpu_mem_MB', np.nan))

################################################################
# 3. 輸出 CSV
################################################################
out_df = df_all.sort_values('best_test_mse').copy()
out_df['param_count_M']  = out_df['param_count'] / 1e6
out_df['avg_epoch_min']  = out_df['avg_epoch_sec'] / 60
out_df['total_train_hr'] = out_df['total_train_sec'] / 3600
out_df['category']       = out_df['group'].map(categorize)

export_cols = ['group', 'category', 'param_count_M', 'avg_epoch_min', 'total_train_hr',
               'inference_ms_per_step', 'peak_gpu_mem_MB', 'best_test_mse']
export_df = out_df[export_cols].copy()
for col, nd in [('param_count_M', 2), ('avg_epoch_min', 2), ('total_train_hr', 2),
                ('inference_ms_per_step', 2), ('peak_gpu_mem_MB', 1), ('best_test_mse', 4)]:
    export_df[col] = export_df[col].round(nd)

csv_path = os.path.join(COMPARE_DIR, 'resource_summary.csv')
export_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n[輸出] {csv_path}")
print("\n" + "=" * 100)
print(" Resource Comparison Table")
print("=" * 100)
print(export_df.to_string(index=False))
print("=" * 100)

################################################################
# 4. 4-panel bar chart
################################################################
labels = out_df['group'].tolist()
colors = [CAT_COLOR[categorize(g)] for g in labels]
x_pos  = np.arange(len(labels))

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
metrics = [
    ('param_count_M',         'Model Size (M params)',  axes[0, 0]),
    ('avg_epoch_min',         'Avg Epoch Time (min)',   axes[0, 1]),
    ('inference_ms_per_step', 'Inference Latency (ms)', axes[1, 0]),
    ('peak_gpu_mem_MB',       'Peak GPU Memory (MB)',   axes[1, 1]),
]
for col, title, ax in metrics:
    vals = out_df[col].values
    ax.bar(x_pos, vals, color=colors, edgecolor='black', linewidth=0.5, alpha=0.9)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
    ax.grid(True, alpha=0.4, axis='y', linestyle='--')
    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(i, v, f'{v:.1f}' if v >= 10 else f'{v:.2f}',
                    ha='center', va='bottom', fontsize=8)

legend_handles = [Patch(facecolor=CAT_COLOR[c], edgecolor='black', label=CAT_LABEL[c])
                  for c in ['planar', 'sphere']]
axes[0, 0].legend(handles=legend_handles, loc='upper left', fontsize=10, framealpha=0.95)

plt.suptitle('Resource Comparison Across 5 Models (sorted by best test MSE)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
out_png = os.path.join(COMPARE_DIR, 'resource_comparison.png')
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.close()
print(f"[輸出] {out_png}")
print(f"\n所有結果已輸出至：{COMPARE_DIR}/")
