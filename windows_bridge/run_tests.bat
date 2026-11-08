@echo off
setlocal

echo.
echo ============================================================
echo CS 1.6 RL Bot - Windows Bridge Tests
echo ============================================================
echo.

net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] NOT running as Administrator
    echo Please run as administrator
    pause
    exit /b 1
)

echo [OK] Running as Administrator
echo.

echo Choose test to run:
echo 1. Test Memory Reader
echo 2. Test Game Controller
echo 3. Test Bridge Connection
echo 4. Run Bridge Server
echo.
set /p choice=Enter choice (1-4): 

if "%choice%"=="1" goto opt1
if "%choice%"=="2" goto opt2
if "%choice%"=="3" goto opt3
if "%choice%"=="4" goto opt4
echo Invalid choice
goto end

:opt1
python test_memory.py
goto end

:opt2
python test_controller.py
goto end

:opt3
python test_bridge.py
goto end

:opt4
python bridge_server.py --host 0.0.0.0 --port 5555
goto end

:end
echo.
pause
