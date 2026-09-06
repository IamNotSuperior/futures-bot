@echo off
REM Start the research bot. Double-click this file, or run it from a terminal.
REM The bot reads DISCORD_TOKEN from .env and syncs its slash commands on start.
title Research Bot

REM %~dp0 is this file's own folder, so the bat works from any working directory.
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo ERROR: venv\Scripts\python.exe not found in "%CD%".
    echo Create the virtualenv first, then run this file again.
    echo.
    pause
    exit /b 1
)

if not exist ".env" (
    echo ERROR: .env not found in "%CD%". The bot needs DISCORD_TOKEN.
    echo.
    pause
    exit /b 1
)

echo Starting the research bot from "%CD%"
echo Press Ctrl+C to stop it.
echo.

venv\Scripts\python.exe bots\research.py

echo.
echo Bot exited with code %ERRORLEVEL%.
pause
