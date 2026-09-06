@echo off
REM Start the desk bot in SHADOW mode. Double-click, or run from a terminal.
REM
REM Shadow mode places no orders. The only strategy it runs is orb2, which
REM research/hypotheses.md entry 4 REJECTED, and every ticket it posts says so.
REM This exercises the signal -> guard -> ticket -> journal path; it is not
REM trading and it is not evidence. See HANDOFF.md section 6.
REM
REM Bars arrive from a TradingView alert running pine\bar_feed.pine, POSTed to
REM http://127.0.0.1:8787/bar. To test without TradingView, use replay:
REM
REM     venv\Scripts\python.exe bots\desk.py --replay --start 2026-08-24 ^
REM                                          --end 2026-08-28 --speed 60
title Desk Bot (SHADOW - no orders)

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
    echo ERROR: .env not found in "%CD%".
    echo The desk needs DISCORD_TOKEN and DESK_TICKET_CHANNEL_ID.
    echo See .env.example for the full list.
    echo.
    pause
    exit /b 1
)

echo ============================================================
echo   DESK BOT - SHADOW MODE
echo   No orders are placed. orb2 is a REJECTED strategy and is
echo   running only to exercise the plumbing.
echo ============================================================
echo.
echo Starting from "%CD%"
echo Webhook will listen on http://127.0.0.1:8787/bar
echo Press Ctrl+C to stop it.
echo.

venv\Scripts\python.exe bots\desk.py %*

echo.
echo Desk exited with code %ERRORLEVEL%.
pause
