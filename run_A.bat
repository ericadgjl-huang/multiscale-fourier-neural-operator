@echo off
set KMP_DUPLICATE_LIB_OK=TRUE

python fourier_2d.py unet_2d --data-glob "data/global_era5_96_factors_*.nc" --output-root outputs96
if errorlevel 1 exit /b 1

python fourier_2d.py sufno --data-glob "data/global_era5_96_factors_*.nc" --output-root outputs96
if errorlevel 1 exit /b 1

python fourier_2d.py sphere_unet --data-glob "data/global_era5_96_factors_*.nc" --output-root outputs96
if errorlevel 1 exit /b 1

python fourier_2d.py 2d_fno --data-glob "data/global_era5_96_factors_*.nc" --output-root outputs96
if errorlevel 1 exit /b 1

echo.
echo All models completed successfully.
pause
