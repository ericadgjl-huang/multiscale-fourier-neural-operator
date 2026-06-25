"""fix_zip_nc.py — 修復「副檔名是 .nc、內容其實是 ZIP」的批次檔。

背景：新版 CDS 對某些變數組合（瞬時量 + 累積/最大量走不同 stream）會回傳 ZIP，
裡面含 2 個 .nc（例：data_stream-oper_stepType-instant.nc + ...-accum.nc）。
download 腳本把整包 ZIP 直接存成 g??.nc，h5py 打不開（檔頭是 PK 不是 HDF）。

本腳本：掃出所有「PK 開頭」的 g??.nc → 解壓 → 把內部 nc 用座標合併成單一 Dataset
        → 覆蓋寫回成正常 NetCDF（HDF5）。其餘正常檔不動。

用法（從專案根目錄執行）：
    python fix_zip_nc.py
"""
import glob
import io
import os
import zipfile

import xarray as xr

PATTERN = "data/global_era5_96_factors_*.nc"
ZIP_MAGIC = b"PK\x03\x04"


def is_zip(path):
    with open(path, "rb") as fh:
        return fh.read(4) == ZIP_MAGIC


files = sorted(glob.glob(PATTERN))
zip_files = [f for f in files if is_zip(f)]

print(f"共 {len(files)} 個檔，其中 {len(zip_files)} 個是偽裝成 .nc 的 ZIP，開始修復...\n")

fixed, failed = [], []
for f in zip_files:
    name = os.path.basename(f)
    try:
        # 1) 把 ZIP 內每個 .nc 讀進記憶體、用 xarray 開啟
        inner_datasets = []
        with zipfile.ZipFile(f) as z:
            members = [m for m in z.namelist() if m.endswith(".nc")]
            for m in members:
                raw = z.read(m)
                # h5netcdf 可直接吃 bytes-like；用 BytesIO 包起來
                ds = xr.open_dataset(io.BytesIO(raw), engine="h5netcdf")
                inner_datasets.append(ds.load())  # load 進記憶體，才能關閉來源
        print(f"  {name}: 內含 {len(members)} 個 nc -> {members}")

        # 2) 依座標合併（各 nc 只帶不同的 data_vars，座標相同）
        merged = xr.merge(inner_datasets, compat="override", combine_attrs="override")

        # 3) 先寫到暫存檔，確認開得起來，再覆蓋原檔
        tmp = f + ".tmp.nc"
        merged.to_netcdf(tmp, engine="h5netcdf")
        merged.close()
        for ds in inner_datasets:
            ds.close()

        # 驗證暫存檔
        with xr.open_dataset(tmp, engine="h5netcdf") as chk:
            _ = {k: chk[k].shape for k in chk.data_vars}
            vars_in = list(chk.data_vars)

        os.replace(tmp, f)  # 原子覆蓋
        print(f"     -> 修好，合併後變數: {vars_in}\n")
        fixed.append(name)
    except Exception as e:
        print(f"     -> 失敗: {type(e).__name__}: {e}\n")
        failed.append(name)
        tmp = f + ".tmp.nc"
        if os.path.exists(tmp):
            os.remove(tmp)

print("=" * 64)
print(f"修好 {len(fixed)} 個；失敗 {len(failed)} 個。")
if failed:
    print("失敗清單:", failed)
print("\n接著跑驗證確認 48 個全 OK： python verify_nc.py")
