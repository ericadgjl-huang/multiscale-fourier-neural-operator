"""download_global.py — 從 CDS 下載 2021-2023 三年全球 mini-ERA5 資料。
用法：python data_tools/download_global.py（從專案根目錄執行）
需要先設定 ~/.cdsapirc 帳號"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import cdsapi

c = cdsapi.Client()

# 我們把要下載的年份寫成一個列表
years = ['2021', '2022', '2023']

print("開始分批向 CDS 伺服器請求 2021~2023 年的全球迷你版 ERA5 資料...")

for year in years:
    # 動態產生檔名，例如 global_era5_mini_2021.nc
    filename = f'data/global_era5_mini_{year}.nc'
    print(f"\n---> 正在排隊下載 {year} 年的資料，準備存入 {filename} ...")
    
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    '10m_u_component_of_wind', '10m_v_component_of_wind', 
                    '2m_temperature', 'mean_sea_level_pressure',
                ],
                'year': year,  # 關鍵：每次迴圈只請求指定的這一年
                'month': [
                    '01', '02', '03', '04', '05', '06',
                    '07', '08', '09', '10', '11', '12',
                ],
                'day': [
                    '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', 
                    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
                    '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31',
                ],
                'time': [
                    '00:00', '06:00', '12:00', '18:00',
                ],
                'area': [90, -180, -90, 180],
                'grid': ['5.625', '5.625'], 
            },
            filename)
        print(f"✅ {year} 年資料下載完成！")
    except Exception as e:
        print(f"❌ {year} 年下載失敗，錯誤訊息：{e}")

print("\n🎉 三年份大資料全部處理完畢！")