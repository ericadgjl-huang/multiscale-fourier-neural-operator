@echo off
REM ================================================================
REM  run_C_seeds.bat  --  96-var multi-seed + planar-FNO control (Machine C)
REM
REM  (1) 96-var: fill in seeds for significance (seed 0 already in outputs96/,
REM      this batch adds seed 1/2 -> n=3).
REM  (2) New planar "FNO+UNet" control 2d_ufno (this machine owns seed 2).
REM
REM  Pure spherical (sphere_*) is NOT reseeded (~20% deficit is firm, slowest).
REM  Finished folders are auto-skipped, safe to stop and rerun.
REM  This machine also runs sunetpp_fno x2 seeds (most accurate arch);
REM  estimated total ~35 h.
REM ================================================================
setlocal
set KMP_DUPLICATE_LIB_OK=TRUE
set DG=data/global_era5_96_factors_*.nc
set OUT=outputs96

REM ---- work list for this machine (heaviest first) ----
call :run sunetpp_fno 1
call :run sunetpp_fno 2
call :run transunet_2d 1
call :run sfno 2
call :run 2d_ufno 2
call :run 2d_fno 1
call :run unet_2d 2

echo.
echo ============================================
echo  [Machine C] all jobs finished.
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
