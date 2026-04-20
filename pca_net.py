import numpy as np
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import time

# 1. 讀取與處理東亞氣象資料 (與 fourier_2d 保持絕對一致的基準)
print("正在讀取 ERA5 東亞氣象資料...")
ds = xr.open_dataset('data/east_asia_era5_202301.nc', engine='h5netcdf')

t2m = ds['t2m'].values
msl = ds['msl'].values
u10 = ds['u10'].values
v10 = ds['v10'].values

data = np.stack([t2m, msl, u10, v10], axis=-1)
data = np.nan_to_num(data, nan=0.0)

x_data = data[:-1]
y_data = data[1:]

x_train, y_train = x_data[:100], y_data[:100]
x_test, y_test   = x_data[100:], y_data[100:]

# 標準化 (Normalization)
x_mean, x_std = x_train.mean(axis=(0, 1, 2), keepdims=True), x_train.std(axis=(0, 1, 2), keepdims=True)
y_mean, y_std = y_train.mean(axis=(0, 1, 2), keepdims=True), y_train.std(axis=(0, 1, 2), keepdims=True)

x_train = (x_train - x_mean) / (x_std + 1e-6)
x_test  = (x_test - x_mean) / (x_std + 1e-6)
y_train = (y_train - y_mean) / (y_std + 1e-6)
y_test  = (y_test - y_mean) / (y_std + 1e-6)

# 將 2D 網格攤平成 1D 向量 (樣本數, 161 * 201 * 4)
N_train, N_test = x_train.shape[0], x_test.shape[0]
D = 161 * 201 * 4

x_train_flat = x_train.reshape(N_train, D)
x_test_flat  = x_test.reshape(N_test, D)
y_train_flat = y_train.reshape(N_train, D)
y_test_flat  = y_test.reshape(N_test, D)

# 2. 進行 PCA / EOF 降維
# 將 129,444 個特徵，濃縮成 50 個最具代表性的主成分
pca_components = 50 
print(f"正在進行 PCA 降維 (將空間特徵濃縮為 {pca_components} 個主成分)...")
t1 = time.time()

pca_X = PCA(n_components=pca_components)
pca_Y = PCA(n_components=pca_components)

x_train_pca = pca_X.fit_transform(x_train_flat)
x_test_pca  = pca_X.transform(x_test_flat)
y_train_pca = pca_Y.fit_transform(y_train_flat)
y_test_pca  = pca_Y.transform(y_test_flat)

# 3. 建立並訓練 MLP (預測未來的係數)
print("正在訓練 MLP (Multi-Layer Perceptron)...")
mlp = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=500, random_state=0)
mlp.fit(x_train_pca, y_train_pca)

# 4. 測試與還原
print("正在還原預測並計算誤差...")
# 先預測出未來的 PCA 係數
pred_y_pca = mlp.predict(x_test_pca)

# 把預測出的少數係數，還原回原本龐大的 161x201x4 氣象網格
pred_y_flat = pca_Y.inverse_transform(pred_y_pca)

# 計算最終的 Test MSE
test_mse = mean_squared_error(y_test_flat, pred_y_flat)
t2 = time.time()

explained_variance = np.sum(pca_X.explained_variance_ratio_) * 100
print(f"這 {pca_components} 個主成分保留了原始氣象資料 {explained_variance:.2f}% 的資訊量！")
print(f"========================================")
print(f" 實驗模型：傳統 PCA-Net (FPCA + MLP)")
print(f" 保留主成分：{pca_components}")
print(f" 總耗時：{t2-t1:.2f} 秒")
print(f" Test MSE: {test_mse:.4f}")
print(f"========================================")