@echo off
REM ================================================================
REM  run_B_seeds.bat  —  96 變數 multi-seed + 平面 FNO 對照 (機器 B)
REM
REM  教授要求：
REM   (1) 96 變數補跑 seeds，取得統計顯著性（seed 0 已在 outputs96/，本批補 seed 1/2 -> n=3）
REM   (2) 新增「平面 FNO+UNet」對照組 2d_ufno（本機負責 seed 0，即基準 seed）
REM
REM  純球面 (sphere_*) 不補 seed（~20% 劣勢已是鐵結論、且最慢）。
REM  已完成的資料夾會自動略過，可安全中斷後重跑。
REM  預估本機總時長 ~32 小時（RTX 4000 Ada，batch=4）。
REM ================================================================
setlocal
set KMP_DUPLICATE_LIB_OK=TRUE
set DG=data/global_era5_96_factors_*.nc
set OUT=outputs96

REM ---- 本機工作清單（由重到輕）----
call :run sutrans_fno 2
call :run sufno 2
call :run sfno 1
call :run 2d_ufno 0
call :run unetpp_2d 1
call :run 2d_fno 2

echo.
echo ============================================
echo  [機器 B] 全部工作結束。
echo ============================================
pause
exit /b 0

REM ---- 子程序：call :run ARCH SEED ----
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
