@echo off
setlocal enabledelayedexpansion
title CS 1.6 - Master Self-Learning Arena Launcher

cd /d "%~dp0"

set "PY="
if exist "%~dp0venv\Scripts\python.exe" set "PY=%~dp0venv\Scripts\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PY if exist "%ProgramFiles%\Python313\python.exe" set "PY=%ProgramFiles%\Python313\python.exe"
if not defined PY if exist "%ProgramFiles%\Python312\python.exe" set "PY=%ProgramFiles%\Python312\python.exe"
if not defined PY if exist "%ProgramFiles%\Python311\python.exe" set "PY=%ProgramFiles%\Python311\python.exe"
if not defined PY set "PY=python"

echo ===============================================================================
echo   Starting CS 1.6 Master AI Arena Orchestrator...
echo   Working Directory: %CD%
echo   Using Python: %PY%
echo ===============================================================================
echo.

"%PY%" --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] ERROR: Python executable was not found.
    echo Please install Python 3.10-3.13 from https://python.org or run install_requirements.bat.
    pause
    exit /b 1
)

"%PY%" -c "import torch, onnxruntime, cv2, mss, sounddevice, numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Installing required packages...
    "%PY%" -m pip install -r requirements.txt
)

"%PY%" run_arena_master.py
if %errorlevel% neq 0 (
    echo.
    echo [!] Process exited with error code %errorlevel%.
    pause
)
