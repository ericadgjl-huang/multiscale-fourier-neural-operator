"""verify_nc.py — 逐一打開 data/ 下的 96 變數 .nc，列出所有壞掉（開不起來）的檔。

用法（從專案根目錄執行）：
    python verify_nc.py

壞檔處理：把列出來的壞檔刪掉，再重跑
    python data_tools/download_global_96_factors.py
（下載腳本會自動跳過已存在的好檔，只補下被刪掉的壞檔。）
"""
import glob
import os

import xarray as xr

PATTERN = "data/global_era5_96_factors_*.nc"

files = sorted(glob.glob(PATTERN))
print(f"找到 {len(files)} 個檔，開始逐一檢查...\n")

bad = []
for f in files:
    size_mb = os.path.getsize(f) / 1e6
    try:
        # 真的讀一下每個變數的維度，確保不是只看 header 就過關
        with xr.open_dataset(f, engine="h5netcdf") as ds:
            _ = {k: ds[k].shape for k in ds.data_vars}
        print(f"  OK    {os.path.basename(f):42s} {size_mb:8.1f} MB")
    except Exception as e:
        print(f"  壞檔  {os.path.basename(f):42s} {size_mb:8.1f} MB  <-- {type(e).__name__}: {e}")
        bad.append(f)

print("\n" + "=" * 64)
if bad:
    print(f"共 {len(bad)} 個壞檔，需要重新下載。")
    print("\n[1] 先刪掉這些壞檔（複製貼上即可）：")
    for b in bad:
        # Windows 用 del；路徑用雙引號包起來
        print(f'    del "{os.path.abspath(b)}"')
    print("\n[2] 再重跑下載腳本（好檔會自動跳過，只補下被刪掉的）：")
    print("    python data_tools/download_global_96_factors.py")
else:
    print("全部 OK，沒有壞檔，可以直接訓練。")