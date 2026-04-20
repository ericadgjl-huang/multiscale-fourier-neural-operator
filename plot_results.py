import matplotlib.pyplot as plt
import numpy as np

# 這是你辛苦跑出來的四大模型數據
models = ['PCA-Net\n(Traditional)', 'Pure FNO\n(1x1 Baseline)', 'U-FNO\n(Simple)', 'Advanced\nU-FNO']
mse_scores = [0.1521, 0.0859, 0.0844, 0.0839]

# 設定圖表樣式
plt.figure(figsize=(10, 6))
bars = plt.bar(models, mse_scores, color=['#cccccc', '#88bbd6', '#4488aa', '#114466'], width=0.6)

# 在柱狀圖上方標示數值
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.002, f'{yval:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 標題與標籤
plt.title('East Asia Weather Prediction: Model Performance Comparison', fontsize=16, pad=20)
plt.ylabel('Test Mean Squared Error (MSE) ↓ lower is better', fontsize=12)
plt.ylim(0, 0.18)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('model_comparison_chart.png', dpi=300)
print("繪圖完成！請查看 model_comparison_chart.png")