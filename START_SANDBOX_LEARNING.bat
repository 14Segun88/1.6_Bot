@echo off
setlocal enabledelayedexpansion
title [CS 1.6] AI Self-Learning Arena - aim_ai_sandbox
cls
echo ===============================================================================
echo   CS 1.6 - AI SELF-LEARNING ARENA: aim_ai_sandbox
echo   Pillars: 1. Navigation (3 Routes + Color Lidar)
echo            2. Perception (YOLOv8 + Crash-Test Dummy Bullseye Headshot)
echo            3. Combat (Counter-Strafe + Burst Fire + Reinforcement Learning)
echo ===============================================================================
echo.

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

"%PY%" --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] ERROR: Python executable was not found.
    echo Please install Python 3.10-3.13 or run install_requirements.bat.
    pause
    exit /b 1
)

echo [*] Launching Master Arena Orchestrator on aim_ai_sandbox...
"%PY%" run_arena_master.py
if %errorlevel% neq 0 (
    echo.
    echo [!] Process exited with error code %errorlevel%.
    pause
)
