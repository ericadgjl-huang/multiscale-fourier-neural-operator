"""
resource_comparison.py
教授建議 #2：從既有實驗結果聚合「模型大小 / 訓練時間 / 推論時間 / 峰值記憶體」對照表。

不需要重新訓練——
- Params (M)：從 config.json 直接讀
- Avg epoch time / Total train time：從 training_log.csv 累計
- Peak GPU memory + Inference latency：載入 model_weights_best.pt 做一次 forward pass 量測

用法：python analysis/resource_comparison.py（從專案根目錄執行）

輸出：
    outputs/_comparison/resource_summary.csv    — paper Table 2 直接可用
    outputs/_comparison/resource_comparison.png — 4 panel bar chart
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import re
import json
import glob
import time
import importlib.util

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

# 中文字體
font_path = r"C:\Windows\Fonts\msjh.ttc"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUTS_DIR = 'outputs'
COMPARE_DIR = os.path.join(OUTPUTS_DIR, '_comparison')
os.makedirs(COMPARE_DIR, exist_ok=True)


################################################################
# 類別判定
################################################################
def categorize(name):
    if name in ['unet_2d', 'unetpp_2d', 'transunet_2d', '2d_fno']:
        return 'planar'
    if name.startswith('sphere_'):
        return 'pure_sphere'
    return 'fno_hybrid'

CAT_COLOR = {'planar': '#2ecc71', 'fno_hybrid': '#f39c12', 'pure_sphere': '#e74c3c'}
CAT_LABEL = {
    'planar':      'Pure 2D (lon pad)',
    'fno_hybrid':  'FNO Hybrid (SHT + planar)',
    'pure_sphere': 'Pure Spherical (SHT-only)',
}


################################################################
# 1. 聚合既有 config + training_log 資料
################################################################
def get_group_key(cfg, fallback):
    """跟 multi_seed_compare.py 一致：按 (base_arch, modes, dropout) 分組，忽略 seed"""
    base = cfg.get('base_experiment_name')
    if base is None:
        base = re.sub(r'_(s\d+|m\d+|drop\d+)$', '', fallback)
        while re.search(r'_(s\d+|m\d+|drop\d+)$', base):
            base = re.sub(r'_(s\d+|m\d+|drop\d+)$', '', base)
    parts = [base]
    if cfg.get('modes', 16) != 16:
        parts.append(f"m{cfg['modes']}")
    if cfg.get('dropout', 0) and cfg.get('dropout', 0) > 0:
        parts.append(f"drop{int(round(cfg['dropout']*100))}")
    return '_'.join(parts)


def is_canonical(cfg):
    """過濾出 modes 預設 + 無 dropout 的 canonical 實驗（給 3×3 表）"""
    if cfg.get('modes', 16) != 16:
        return False
    if cfg.get('dropout', 0) and cfg.get('dropout', 0) > 0:
        return False
    return True


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
    if not is_canonical(cfg):
        continue   # 跳過 hyperparam search 變體

    group = get_group_key(cfg, name)
    records.append({
        'arch':            name,
        'group':           group,
        'param_count':     cfg.get('param_count', 0),
        'avg_epoch_sec':   df['epoch_time_sec'].mean(),
        'total_train_sec': df['epoch_time_sec'].sum(),
        'epochs':          len(df),
        'best_test_mse':   cfg.get('best_test_mse', df['test_mse'].min()),
    })

df_all = pd.DataFrame(records)

# 按 group 聚合（平均 epoch time）
agg = df_all.groupby('group').agg(
    n_seeds          = ('arch', 'count'),
    param_count      = ('param_count', 'first'),       # 同 group 參數量一致
    avg_epoch_sec    = ('avg_epoch_sec', 'mean'),
    total_train_sec  = ('total_train_sec', 'mean'),
    best_test_mse    = ('best_test_mse', 'mean'),
).reset_index()

print(f"找到 {len(df_all)} 個實驗 → 聚合為 {len(agg)} 個 canonical groups")


################################################################
# 2. 對每個 group 量測 inference latency + peak GPU memory
#    （載入 model_weights_best.pt，跑 100 次 forward 取平均）
################################################################
# 每個 group 找一個代表性的 outputs 資料夾來載入模型
def find_canonical_dir(group_name):
    """找該 group 的 seed=0（canonical）資料夾"""
    candidate = os.path.join(OUTPUTS_DIR, group_name)
    if os.path.isdir(candidate):
        return candidate
    # fallback：找任何屬於該 group 的資料夾
    for d in glob.glob(os.path.join(OUTPUTS_DIR, group_name + '*')):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, 'model_weights_best.pt')):
            return d
    return None


# 模型工廠：根據 group 名稱與 config 載入對應的 class
# 為了避免 import 整段腳本觸發訓練，這裡用「執行截斷的程式碼」方式只載入 class 定義
def build_model_for_group(group, cfg):
    """
    根據 group 名稱回傳對應的 model instance（已 to('cuda')）。
    若無法建立則回傳 None。
    """
    # Hardcoded FNO EXPERIMENTS dict (cutoff 截在 ERA5RolloutDataset 之前，所以 EXPERIMENTS 還沒被定義)
    FNO_EXP = {
        '2d_fno':      {'local_type': '1x1',           'spectral_type': 'fft'},
        'sfno':        {'local_type': '1x1',           'spectral_type': 'sht'},
        'sufno':       {'local_type': 'unet',          'spectral_type': 'sht'},
        'sunetpp_fno': {'local_type': 'advanced_unet', 'spectral_type': 'sht'},
        'sutrans_fno': {'local_type': 'transformer',   'spectral_type': 'sht'},
    }
    try:
        if group in FNO_EXP:
            # FNO2d 從 fourier_2d.py 載入 — 截斷在 ERA5RolloutDataset 之前
            src = open('fourier_2d.py', encoding='utf-8').read()
            cutoff = src.find('################################################################\n# ERA5RolloutDataset')
            ns = {}
            exec(compile(src[:cutoff], 'fourier_2d.py', 'exec'), ns)
            FNO2d = ns['FNO2d']
            ec = FNO_EXP[group]
            return FNO2d(16, 16, 32, local_type=ec['local_type'],
                         spectral_type=ec['spectral_type'], dropout=0.0).cuda()
        elif group == 'unet_2d':
            src = open('unet_baseline.py', encoding='utf-8').read()
            cutoff = src.find('################################################################\n# ERA5RolloutDataset')
            ns = {}
            exec(compile(src[:cutoff], 'unet_baseline.py', 'exec'), ns)
            return ns['UNet2DRollout']().cuda()
        elif group == 'unetpp_2d':
            src = open('unetpp_baseline.py', encoding='utf-8').read()
            cutoff = src.find('################################################################\n# ERA5RolloutDataset')
            ns = {}
            exec(compile(src[:cutoff], 'unetpp_baseline.py', 'exec'), ns)
            return ns['UNetPlusPlus2DRollout']().cuda()
        elif group == 'transunet_2d':
            src = open('transunet_baseline.py', encoding='utf-8').read()
            cutoff = src.find('################################################################\n# ERA5RolloutDataset')
            ns = {}
            exec(compile(src[:cutoff], 'transunet_baseline.py', 'exec'), ns)
            return ns['TransUNet2DRollout']().cuda()
        elif group == 'sphere_unet':
            src = open('sphere_unet_baseline.py', encoding='utf-8').read()
            cutoff = src.find('################################################################\n# ERA5RolloutDataset')
            ns = {}
            exec(compile(src[:cutoff], 'sphere_unet_baseline.py', 'exec'), ns)
            return ns['SphereUNet2DRollout']().cuda()
        elif group == 'sphere_unetpp':
            src = open('sphere_unetpp_baseline.py', encoding='utf-8').read()
            cutoff = src.find('################################################################\n# ERA5RolloutDataset')
            ns = {}
            exec(compile(src[:cutoff], 'sphere_unetpp_baseline.py', 'exec'), ns)
            return ns['SphereUNetPlusPlus2DRollout']().cuda()
        elif group == 'sphere_transunet':
            src = open('sphere_transunet_baseline.py', encoding='utf-8').read()
            cutoff = src.find('################################################################\n# ERA5RolloutDataset')
            ns = {}
            exec(compile(src[:cutoff], 'sphere_transunet_baseline.py', 'exec'), ns)
            return ns['SphereTransUNet2DRollout']().cuda()
    except Exception as e:
        print(f"  [略過 {group}] 模型建立失敗：{e}")
        return None
    return None


# 量測 inference latency 與 peak memory
print("\n正在量測各架構的 inference latency 與 peak GPU memory...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
inference_results = {}

for _, row in agg.iterrows():
    group = row['group']
    src_dir = find_canonical_dir(group)
    if src_dir is None:
        print(f"  [略過 {group}] 找不到 model_weights")
        continue

    # 載入對應 config
    with open(os.path.join(src_dir, 'config.json'), encoding='utf-8') as f:
        cfg = json.load(f)

    model = build_model_for_group(group, cfg)
    if model is None:
        continue

    # 載入 best weights（如果有的話）
    weights_path = os.path.join(src_dir, 'model_weights_best.pt')
    if not os.path.exists(weights_path):
        weights_path = os.path.join(src_dir, 'model_weights.pt')
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
        except Exception as e:
            print(f"  [警告 {group}] 載入權重失敗（用 random init 量測）：{e}")

    model.eval()

    # Warm up
    x = torch.randn(4, 33, 64, 8, device=device)
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # 量測 inference latency（forward 100 次取平均）
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
    inference_ms = (t_end - t_start) / n_runs * 1000  # ms per single forward

    # Peak memory
    if device == 'cuda':
        peak_mem_MB = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_mem_MB = float('nan')

    inference_results[group] = {
        'inference_ms_per_step': inference_ms,
        'peak_gpu_mem_MB': peak_mem_MB,
    }
    print(f"  {group:<22} | inference {inference_ms:6.2f}ms/step | peak mem {peak_mem_MB:7.1f}MB")

    del model
    if device == 'cuda':
        torch.cuda.empty_cache()


# 合併 inference 結果到 agg
agg['inference_ms_per_step'] = agg['group'].map(lambda g: inference_results.get(g, {}).get('inference_ms_per_step', np.nan))
agg['peak_gpu_mem_MB']       = agg['group'].map(lambda g: inference_results.get(g, {}).get('peak_gpu_mem_MB', np.nan))


################################################################
# 3. 輸出 CSV（paper Table 2 直接可用）
################################################################
out_df = agg.sort_values('best_test_mse').copy()
out_df['param_count_M']     = out_df['param_count'] / 1e6
out_df['avg_epoch_min']     = out_df['avg_epoch_sec'] / 60
out_df['total_train_hr']    = out_df['total_train_sec'] / 3600
out_df['category']          = out_df['group'].map(categorize)

export_cols = ['group', 'category', 'n_seeds', 'param_count_M',
               'avg_epoch_min', 'total_train_hr',
               'inference_ms_per_step', 'peak_gpu_mem_MB', 'best_test_mse']
export_df = out_df[export_cols].copy()

# Format columns
export_df['param_count_M']         = export_df['param_count_M'].round(2)
export_df['avg_epoch_min']         = export_df['avg_epoch_min'].round(2)
export_df['total_train_hr']        = export_df['total_train_hr'].round(2)
export_df['inference_ms_per_step'] = export_df['inference_ms_per_step'].round(2)
export_df['peak_gpu_mem_MB']       = export_df['peak_gpu_mem_MB'].round(1)
export_df['best_test_mse']         = export_df['best_test_mse'].round(4)

csv_path = os.path.join(COMPARE_DIR, 'resource_summary.csv')
export_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n[輸出] {csv_path}")

# Console 摘要
print("\n" + "=" * 110)
print(" Resource Comparison Table（paper Table 2 直接可用）")
print("=" * 110)
print(export_df.to_string(index=False))
print("=" * 110)


################################################################
# 4. 4-panel bar chart 視覺化
################################################################
sorted_agg = out_df  # 已按 best_test_mse 升序
labels  = sorted_agg['group'].tolist()
colors  = [CAT_COLOR[categorize(g)] for g in labels]
x_pos   = np.arange(len(labels))

fig, axes = plt.subplots(2, 2, figsize=(18, 11))
metrics = [
    ('param_count_M',         'Model Size (M params)',    axes[0, 0]),
    ('avg_epoch_min',         'Avg Epoch Time (min)',     axes[0, 1]),
    ('inference_ms_per_step', 'Inference Latency (ms)',   axes[1, 0]),
    ('peak_gpu_mem_MB',       'Peak GPU Memory (MB)',     axes[1, 1]),
]
for col, title, ax in metrics:
    vals = sorted_agg[col].values
    ax.bar(x_pos, vals, color=colors, edgecolor='black', linewidth=0.5, alpha=0.9)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.grid(True, alpha=0.4, axis='y', linestyle='--')
    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(i, v, f'{v:.1f}' if v >= 10 else f'{v:.2f}',
                    ha='center', va='bottom', fontsize=8)

# 類別 legend
legend_handles = [Patch(facecolor=CAT_COLOR[c], edgecolor='black', label=CAT_LABEL[c])
                  for c in ['planar', 'fno_hybrid', 'pure_sphere']]
axes[0, 0].legend(handles=legend_handles, loc='upper left', fontsize=10, framealpha=0.95)

plt.suptitle('Resource Comparison Across Architectures (sorted by best test MSE)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
out_png = os.path.join(COMPARE_DIR, 'resource_comparison.png')
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.close()
print(f"[輸出] {out_png}")

print(f"\n所有結果已輸出至：{COMPARE_DIR}/")
