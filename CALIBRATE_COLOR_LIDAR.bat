@echo off
setlocal enabledelayedexpansion
title CS 1.6 Optical Color Lidar - 10-Round Lighting Calibrator
echo ============================================================
echo   Counter-Strike 1.6 - Optical Color Lidar Calibration Suite
echo   10-Round Empirical HSV Calibration (Lighting / Shadows)
echo ============================================================
echo.

set "PY_EXE="
if exist "%~dp0venv\Scripts\python.exe" set "PY_EXE=%~dp0venv\Scripts\python.exe"
if not defined PY_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not defined PY_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PY_EXE if exist "%ProgramFiles%\Python313\python.exe" set "PY_EXE=%ProgramFiles%\Python313\python.exe"
if not defined PY_EXE if exist "%ProgramFiles%\Python312\python.exe" set "PY_EXE=%ProgramFiles%\Python312\python.exe"
if not defined PY_EXE if exist "%ProgramFiles%\Python311\python.exe" set "PY_EXE=%ProgramFiles%\Python311\python.exe"
if not defined PY_EXE set "PY_EXE=python"

cd /d "%~dp0"
echo [Config] Working Directory: %CD%
echo [Config] Using Python: %PY_EXE%
echo.
echo [!] Ensure CS 1.6 is RUNNING with the bot on aim_ai_sandbox or $2000$!
echo [*] The calibrator will track 10 rounds of gameplay, sampling:
echo     - Green Walkable Corridors (Сюда МОЖНО) in sun and dark shadows
echo     - Red Obstacle Walls (Сюда НЕЛЬЗЯ) in all lighting angles
echo.
echo Starting 10-Round Color Lidar Calibrator...
echo.

"%PY_EXE%" calibrate_color_lidar.py

:done
echo.
echo Calibration finished. Check benchmarks\lidar_hsv_profile.json and LIDAR_CALIBRATION_REPORT.md!
pause
