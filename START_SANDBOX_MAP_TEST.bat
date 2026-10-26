@echo off
title [CS 1.6] AI Sandbox Map Test
cls
echo ===============================================================
echo     CS 1.6 - AI TRAINING SANDBOX MAP TEST
echo     Map: aim_ai_sandbox (Green Floor / Red Walls)
echo     Dummies: 11 Frozen Crash-Test Dummies with Bullseye Head
echo ===============================================================
echo.
echo   Closing any existing CS 1.6 processes...
taskkill /F /IM hl.exe >nul 2>&1
taskkill /F /IM cstrike.exe >nul 2>&1
timeout /t 1 /nobreak >nul
echo.
echo   Loading aim_ai_sandbox...
echo   Weapon racks: AK47, M4A1, AWP, MP5, Shotgun, Armor, Grenades (HE, Flash, Smoke)
echo.
cd /d "D:\CS 1.6 GoldClient"
start hl.exe -game cstrike -nomaster -novid -steam -noforcemparms -condebug +maxplayers 16 +mp_autoteambalance 0 +mp_limitteams 0 +mp_round_infinite 0 +mp_round_restart_delay 2 +mp_freezetime 0 +humans_join_team CT +bot_join_team T +bot_quota_mode normal +bot_quota 11 +bot_stop 1 +bot_freeze 1 +bot_zombie 1 +bot_difficulty 0 +bot_knives_only 1 +map aim_ai_sandbox +cl_minmodels 1 +exec maps/aim_ai_sandbox.cfg
echo Launched! Check your CS 1.6 window.
pause
