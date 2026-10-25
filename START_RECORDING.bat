@echo off
setlocal enabledelayedexpansion
title CS 1.6 Multimodal AI Recorder (Vision + Sound)
echo ============================================================
echo   Counter-Strike 1.6 - AI Dataset Recorder (Vision + Sound)
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

:check_deps
echo [1/2] Checking dependencies (opencv-python, sounddevice, etc.)...
"%PY_EXE%" -c "import cv2, sounddevice, numpy, mss, pynput" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing recorder dependencies...
    "%PY_EXE%" -m pip install opencv-python sounddevice numpy mss pynput
)

echo.
echo [2/2] Starting Multimodal Recorder...
echo.
echo ============================================================
echo  1. Launch Counter-Strike 1.6 (map $2000$)
echo  2. Press [F9] in game to START recording
echo  3. Play normally with sound for 45-60 minutes
echo  4. Press [F9] to PAUSE or [Ctrl+C] here to FINISH
echo ============================================================
echo.

"%PY_EXE%" recorder.py

:done
echo.
echo Recorder finished or stopped.
pause
