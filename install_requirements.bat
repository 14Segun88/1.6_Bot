@echo off
setlocal enabledelayedexpansion
title CS 1.6 AI Bot - Python Package Installer
echo ============================================================
echo   CS 1.6 AI Bot - Dependency Installer
echo ============================================================
echo.

set "PY="
if exist "%~dp0venv\Scripts\python.exe" set "PY=%~dp0venv\Scripts\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PY if exist "%ProgramFiles%\Python313\python.exe" set "PY=%ProgramFiles%\Python313\python.exe"
if not defined PY if exist "%ProgramFiles%\Python312\python.exe" set "PY=%ProgramFiles%\Python312\python.exe"
if not defined PY if exist "%ProgramFiles%\Python311\python.exe" set "PY=%ProgramFiles%\Python311\python.exe"
if not defined PY set "PY=python"

echo [*] Detected Python: %PY%
"%PY%" --version 2>nul
if %errorlevel% neq 0 (
    echo [!] ERROR: Python was not found!
    echo Please install Python 3.10 - 3.13 from https://python.org and check "Add Python to PATH".
    pause
    exit /b 1
)

cd /d "%~dp0"

echo.
echo [*] Upgrading pip...
"%PY%" -m pip install --upgrade pip

echo.
echo [*] Installing required packages (torch, onnxruntime, opencv-python, mss, pynput, sounddevice, numpy, Pillow)...
"%PY%" -m pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo.
    echo ============================================================
    echo   [SUCCESS] All dependencies are successfully installed!
    echo   You can now launch: START_ALL_IN_ONE_ARENA.bat
    echo ============================================================
) else (
    echo.
    echo [!] Some packages failed to install. Please check error messages above.
)

pause
