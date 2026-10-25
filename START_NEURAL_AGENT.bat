@echo off
setlocal enabledelayedexpansion
title CS 1.6 Multimodal Neural Agent Runner (Vision + Hearing + Online Policy)
echo ============================================================
echo   Counter-Strike 1.6 - Neural AI Agent (Vision + Online Policy)
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

"%PY_EXE%" -c "import onnxruntime, cv2, mss, sounddevice, numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies...
    "%PY_EXE%" -m pip install onnxruntime opencv-python mss sounddevice numpy
)
echo Starting Multimodal Neural Agent on $2000$...
echo.
"%PY_EXE%" play_agent.py

:done
echo.
pause
