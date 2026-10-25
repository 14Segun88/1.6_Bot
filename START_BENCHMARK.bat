@echo off
setlocal enabledelayedexpansion
title CS 1.6 AI Match Benchmark & Real-Time Evaluator
echo ============================================================
echo   Counter-Strike 1.6 - AI Objective Benchmark System
echo   Real Empirical Scoring (K/D, Survival Time, Visuals)
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

echo Starting Benchmark Monitor...
echo.
"%PY_EXE%" eval_benchmark.py

:done
echo.
pause
