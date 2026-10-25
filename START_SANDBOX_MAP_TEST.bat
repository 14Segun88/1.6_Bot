@echo off
title [CS 1.6] AI Sandbox Map Test
cls
echo ===============================================================
echo     CS 1.6 - AI TRAINING SANDBOX MAP TEST
echo     Map: aim_ai_sandbox (Green Floor / Red Walls)
echo ===============================================================
echo.
echo   Loading map with 11 standing dummy targets (bot_stop 1)...
echo   Walk along green corridors, inspect red barriers and target dots!
echo.
cd /d "D:\CS 1.6 GoldClient"
start hl.exe -game cstrike -nomaster -novid +map aim_ai_sandbox +bot_join_team T +bot_quota 11 +bot_stop 1 +bot_difficulty 0 +cl_minmodels 1 +mp_freezetime 0
echo Launched! Check your CS 1.6 window.
pause
