"""download_global_mini.py — 下載單一檔案的全球 mini-ERA5 資料。
用法：python data_tools/download_global_mini.py（從專案根目錄執行）
需要先設定 ~/.cdsapirc 帳號"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import cdsapi

c = cdsapi.Client()

print("開始下載全球迷你版 ERA5 資料...")

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind', 
            '2m_temperature', 'mean_sea_level_pressure',
        ],
        'year': '2023',
        'month': '01',
        'day': [
            '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', 
            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
            '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31',
        ],
        'time': [
            '00:00', '06:00', '12:00', '18:00',
        ],
        # 關鍵 1：全球範圍
        'area': [90, -180, -90, 180],
        
        # 關鍵 2：網頁上找不到的魔法參數！將解析度大幅降低
        'grid': ['5.625', '5.625'], 
    },
    'data/global_era5_mini_202301.nc')  # 儲存路徑

print("下載完成！")