"""check_data.py — 簡單的 NetCDF 結構檢查工具。
用法：python data_tools/check_data.py（從專案根目錄執行）"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import xarray as xr

# 1. 讀取資料夾內的氣象檔
file_path = "data/east_asia_era5_202301.nc"
ds = xr.open_dataset(file_path, engine='h5netcdf')

# 2. 印出這顆「資料方塊」的整體結構
print("=== ERA5 資料集結構 ===")
print(ds)

# 3. 偷看一下資料的維度大小 (時間, 緯度, 經度)
print("\n=== 維度大小 ===")
print(ds.dims)