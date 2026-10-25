@echo off
setlocal enabledelayedexpansion
title CS 1.6 AI Arena Arbitrator (10 Rounds Top-1 Goal)
echo ============================================================
echo   Counter-Strike 1.6 - AI Arena Evaluator
echo   Goal: 10 Consecutive Rounds as Top-1 Fragger on $2000$
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

echo Starting Arena Match Arbitrator...
echo.
"%PY_EXE%" arena_arbitrator.py

:done
echo.
pause
