@echo off
setlocal enabledelayedexpansion
title CS 1.6 AI Bot Runner
echo ======================================================
echo    Counter-Strike 1.6 AI Bot Runner
echo ======================================================
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

"%PY_EXE%" -c "import onnxruntime, mss, PIL, numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Installing required packages...
    "%PY_EXE%" -m pip install onnxruntime mss Pillow numpy
)

echo Starting AI Bot...
echo.
"%PY_EXE%" run_bot.py

:done
echo.
pause
