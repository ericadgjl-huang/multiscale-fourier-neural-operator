@echo off
REM ================================================================
REM  run_A_seeds.bat  --  96-var multi-seed + planar-FNO control (Machine A)
REM
REM  (1) 96-var: fill in seeds for significance (seed 0 already in outputs96/,
REM      this batch adds seed 1/2 -> n=3).
REM  (2) New planar "FNO+UNet" control 2d_ufno (FFT+UNet, planar twin of sufno)
REM      to test whether spherical is necessary: if 2d_ufno ~= sufno the gain
REM      comes from FNO+UNet, not the sphere.
REM
REM  Pure spherical (sphere_*) is NOT reseeded: its ~20% deficit is already
REM  a firm conclusion and it is the slowest (up to ~14 min/epoch).
REM  Finished folders are auto-skipped, so it is safe to stop and rerun
REM  (resume-friendly). Estimated total ~32 h (RTX 4000 Ada, batch=4).
REM ================================================================
setlocal
set KMP_DUPLICATE_LIB_OK=TRUE
set DG=data/global_era5_96_factors_*.nc
set OUT=outputs96

REM ---- work list for this machine (heaviest first) ----
call :run sutrans_fno 1
call :run sufno 1
call :run transunet_2d 2
call :run 2d_ufno 1
call :run unetpp_2d 2
call :run unet_2d 1

echo.
echo ============================================
echo  [Machine A] all jobs finished.
echo ============================================
pause
exit /b 0

REM ---- subroutine: call :run ARCH SEED ----
:run
set ARCH=%1
set SEED=%2
if "%SEED%"=="0" (set "DIR=%OUT%\%ARCH%") else (set "DIR=%OUT%\%ARCH%_s%SEED%")
if exist "%DIR%\training_log.csv" (
    echo [SKIP] %ARCH% seed=%SEED%  ^(already done^)
    exit /b 0
)
echo.
echo [RUN ] %ARCH% seed=%SEED%  -^>  %DIR%
python fourier_2d.py %ARCH% --seed %SEED% --data-glob "%DG%" --output-root %OUT%
if errorlevel 1 echo [WARN] %ARCH% seed=%SEED% returned an error ^(continuing^)
exit /b 0
