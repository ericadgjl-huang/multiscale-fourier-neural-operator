"""download_global_96_factors.py — 從 CDS 下載 2021-2023 三年「96 個單層變數」全球 mini-ERA5。

96 通道 = 96 個**不同的 single-level 變數**（不是氣壓層展開）。
全部取自 `reanalysis-era5-single-levels`（即 ERA5變數.pdf 那個資料集），
每個變數在 33×64 網格上就是 1 個通道，所以 96 變數 = 96 通道。

fourier_2d.py 的資料載入會自動偵測 NetCDF 內所有變數、自動命名（用檔案內的短名，如
t2m, msl, sp, tp, ...），所以這裡只要把檔案抓下來放進 data/ 即可，不必改訓練程式。

下面 VARIABLES 已含你原本的 6 個（10u/10v/2t/msl/vimdf/vitoe），其餘 90 個是我幫你
從各類別挑的、對天氣有意義的變數，盡量涵蓋：近地面狀態、溫度、雲、輻射/熱通量、
平均通量率、降水、降雪、蒸發/逕流、土壤、植被、垂直積分/總量、不穩定度與邊界層。

  ✦ 想改數量：直接增刪 VARIABLES（通道數 = 變數個數）。訓練程式會自動跟著變，不必改。

⚠ 兩個務必注意：
  1) CDS 的「變數 API 名稱」一字之差就會整包請求失敗。若某個名稱被 CDS 退回，
     它會明確告訴你是哪一個 → 到 CDS 網頁勾選那些變數、按「Show API request code」，
     會給你**100% 正確**的 variable 清單可直接貼上。
     （fourier_2d.py 是自動偵測通道數，所以就算最後是 94 或 95 個也照跑、不會壞。）
  2) 96 變數、6-hourly、三年、global → 檔案不小（單年可能數百 MB），下載可能要數小時，
     CDS 佇列也較久。已存在的年份檔會自動跳過，可分年份重跑。

用法（從專案根目錄執行）：
    python data_tools/download_global_96_factors.py
需先設定好 ~/.cdsapirc（與既有 6 變數下載相同）。
（備註：若哪天想改用「少數變數 × 多氣壓層」的版本，改 dataset 為
 'reanalysis-era5-pressure-levels' 並加 'pressure_level' 即可，loader 一樣能自動展開。）
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import cdsapi

# ======================================================================
# 96 個 single-level 變數（通道數 = len(VARIABLES)）
# 分組只是方便閱讀，CDS 不在意順序。
# ======================================================================
VARIABLES = [
    # --- 近地面狀態 (10) ---
    '10m_u_component_of_wind',                  # 你原本就有
    '10m_v_component_of_wind',                  # 你原本就有
    '2m_temperature',                           # 你原本就有
    'mean_sea_level_pressure',                  # 你原本就有
    '2m_dewpoint_temperature',
    'surface_pressure',
    '100m_u_component_of_wind',
    '100m_v_component_of_wind',
    '10m_wind_gust_since_previous_post_processing',
    'instantaneous_10m_wind_gust',

    # --- 溫度 (4) ---
    # （已拿掉 ice_temperature_layer_1~4：海冰才有、其餘區域幾乎全是 0，最浪費通道）
    'skin_temperature',
    'sea_surface_temperature',
    'maximum_2m_temperature_since_previous_post_processing',
    'minimum_2m_temperature_since_previous_post_processing',

    # --- 雲 (7) ---
    'total_cloud_cover',
    'high_cloud_cover',
    'medium_cloud_cover',
    'low_cloud_cover',
    'total_column_cloud_ice_water',
    'total_column_cloud_liquid_water',
    'cloud_base_height',

    # --- 輻射 / 熱通量（累積量）(9) ---
    'surface_solar_radiation_downwards',
    'surface_thermal_radiation_downwards',
    'surface_net_solar_radiation',
    'surface_net_thermal_radiation',
    'top_net_solar_radiation',
    'top_net_thermal_radiation',
    'surface_latent_heat_flux',
    'surface_sensible_heat_flux',
    'toa_incident_solar_radiation',

    # --- 平均通量率 (9) ---
    'mean_surface_latent_heat_flux',
    'mean_surface_sensible_heat_flux',
    'mean_surface_net_short_wave_radiation_flux',
    'mean_surface_net_long_wave_radiation_flux',
    'mean_surface_downward_short_wave_radiation_flux',
    'mean_surface_downward_long_wave_radiation_flux',
    'mean_total_precipitation_rate',
    'mean_evaporation_rate',
    'mean_runoff_rate',

    # --- 降水 / 降雨 (8) ---
    'total_precipitation',
    'convective_precipitation',
    'large_scale_precipitation',
    'total_column_rain_water',
    'convective_rain_rate',
    'large_scale_rain_rate',
    'precipitation_type',
    'instantaneous_large_scale_surface_precipitation_fraction',

    # --- 降雪 (9) ---
    'snowfall',
    'snow_depth',
    'snow_density',
    'snow_albedo',
    'snowmelt',
    'convective_snowfall',
    'large_scale_snowfall',
    'temperature_of_snow_layer',
    'snow_evaporation',

    # --- 蒸發 / 逕流 (5) ---
    'evaporation',
    'potential_evaporation',
    'runoff',
    'sub_surface_runoff',
    'surface_runoff',

    # --- 土壤 (8) ---
    'soil_temperature_level_1',
    'soil_temperature_level_2',
    'soil_temperature_level_3',
    'soil_temperature_level_4',
    'volumetric_soil_water_layer_1',
    'volumetric_soil_water_layer_2',
    'volumetric_soil_water_layer_3',
    'volumetric_soil_water_layer_4',

    # --- 植被 (4) ---
    'leaf_area_index_high_vegetation',
    'leaf_area_index_low_vegetation',
    'high_vegetation_cover',
    'low_vegetation_cover',

    # --- 垂直積分 / 總量 (16)（全球連續，無陸海遮罩）---
    'vertical_integral_of_divergence_of_moisture_flux',   # 你原本就有 (vimdf)
    'vertical_integral_of_total_energy',                  # 你原本就有 (vitoe)
    'vertically_integrated_moisture_divergence',
    'vertical_integral_of_temperature',
    'vertical_integral_of_kinetic_energy',
    'vertical_integral_of_thermal_energy',
    'vertical_integral_of_potential_and_internal_energy',
    'vertical_integral_of_eastward_water_vapour_flux',
    'vertical_integral_of_northward_water_vapour_flux',
    'vertical_integral_of_eastward_heat_flux',            # ← 替換 ice_temperature 的全球連續變數
    'vertical_integral_of_northward_heat_flux',           # ←
    'vertical_integral_of_mass_of_atmosphere',            # ←
    'total_column_water_vapour',
    'total_column_water',
    'total_column_supercooled_liquid_water',              # ←
    'total_column_ozone',

    # --- 不穩定度 / 邊界層 / 其他 (7) ---
    'convective_available_potential_energy',
    'convective_inhibition',
    'k_index',
    'total_totals_index',
    'boundary_layer_height',
    'friction_velocity',
    'forecast_albedo',
]

YEARS = ['2021', '2022', '2023']

# 與 download_global_6_factors.py 完全相同的時間取樣 / 網格 / 範圍，
# 確保 33×64 網格與 train_size=2920 的切分一致。
MONTHS = [f'{m:02d}' for m in range(1, 13)]
DAYS = [f'{d:02d}' for d in range(1, 32)]
TIMES = ['00:00', '06:00', '12:00', '18:00']
AREA = [90, -180, -90, 180]
GRID = ['5.625', '5.625']

# ======================================================================
# 切批策略：CDS 的 cost 主要被「變數數量」撐爆，不是時間長度
#   → 按「變數」分批，時間維持完整三年。
#   參考你成功過的 6 變數 × 1 年（cost ≈ 6×12×31×4 ≈ 8,928）→ 一個請求只要把
#   「變數數 × 年數 × 12 × 31 × 4」壓到那個量級以下即可。
#   預設每批 2 變數 × 3 年 = 2×3×12×31×4 ≈ 8,928（與成功過的同等級）。
#   若這樣還是 cost limit：把 VARS_PER_REQUEST 改成 1（請求量再砍半）。
# ======================================================================
VARS_PER_REQUEST = 2

groups = [VARIABLES[i:i + VARS_PER_REQUEST] for i in range(0, len(VARIABLES), VARS_PER_REQUEST)]
print(f"準備下載 {len(VARIABLES)} 個 single-level 變數（= {len(VARIABLES)} 通道）")
print(f"按變數分批：每批 {VARS_PER_REQUEST} 變數 × 完整三年，共 {len(groups)} 個檔。"
      f" loader 會用 glob 自動把各檔的變數合併。")
if len(VARIABLES) != 96:
    print(f"⚠ 目前變數數為 {len(VARIABLES)}，不是 96（你可能增刪過）。訓練程式會自動偵測，照跑無妨。")

c = cdsapi.Client()

ok = fail = 0
for gi, group in enumerate(groups):
    # 每批一檔：global_era5_96_factors_g00.nc ...（glob *_*.nc 會全部抓到）
    filename = f'data/global_era5_96_factors_g{gi:02d}.nc'
    if os.path.exists(filename):
        print(f"---> 第 {gi:02d} 批 {group}：已存在，跳過。")
        ok += 1
        continue

    print(f"\n---> 正在排隊下載 第 {gi:02d} 批（{group}，完整三年），存入 {filename} ...")
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': group,           # ← 一次只要少數變數，壓低 cost
                'year': YEARS,               # 完整三年（時間不切，合併最單純）
                'month': MONTHS,
                'day': DAYS,
                'time': TIMES,
                'area': AREA,
                'grid': GRID,
            },
            filename)
        print(f"✅ 第 {gi:02d} 批完成！")
        ok += 1
    except Exception as e:
        print(f"❌ 第 {gi:02d} 批失敗：{e}")
        print("   若仍是 cost limit：把上面的 VARS_PER_REQUEST 改成 1 再重跑（已完成的批會自動跳過）。")
        print("   若是某個 variable 名稱無效：用 CDS 網頁『Show API request code』取得正確名稱替換。")
        fail += 1

print(f"\n🎉 處理完畢：成功/已存在 {ok} 批、失敗 {fail} 批。")
print("   （失敗的直接重跑本腳本即可，已完成的批會自動跳過。）")
print('   訓練時指定：python fourier_2d.py <arch> --data-glob "data/global_era5_96_factors_*.nc"')
