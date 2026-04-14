import xarray as xr

# 1. 讀取你剛剛改名並放進 data 資料夾的氣象檔
# (如果你沒有改名，請把這裡替換成你原本的亂碼檔名)
file_path = "data/east_asia_era5_202301.nc"
ds = xr.open_dataset(file_path, engine='h5netcdf')

# 2. 印出這顆「資料方塊」的整體結構
print("=== ERA5 資料集結構 ===")
print(ds)

# 3. 偷看一下資料的維度大小 (時間, 緯度, 經度)
print("\n=== 維度大小 ===")
print(ds.dims)