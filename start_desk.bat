@echo off
REM Start the Cloudflare Tunnel and the desk bot together, SHADOW mode.
REM Double-click, or run from a terminal. Ctrl+C stops the desk; the tunnel
REM window is closed automatically on the way out.
REM
REM Shadow mode places no orders. The only strategy it runs is orb2, which
REM research/hypotheses.md entry 4 REJECTED, and every ticket it posts says so.
REM This exercises the signal -> guard -> ticket -> journal path; it is not
REM trading and it is not evidence. See docs/HANDOFF.md section 6.
REM
REM The tunnel puts the /bar endpoint on the public internet, so
REM DESK_WEBHOOK_TOKEN must be set in .env - bots\tunnel.py refuses to start
REM without one. Set DESK_TUNNEL_HOSTNAME for a stable named-tunnel hostname;
REM leave it empty for a quick tunnel whose URL changes on every restart.
REM
REM To run without exposing anything, skip this file:
REM     venv\Scripts\python.exe bots\desk.py
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
    echo The desk needs DISCORD_TOKEN, DESK_TICKET_CHANNEL_ID and
    echo DESK_WEBHOOK_TOKEN. See .env.example for the full list.
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

REM --- the tunnel, in its own window -------------------------------------
REM Started first so the public hostname is printed before bars arrive. Its
REM window stays open with the live cloudflared log; journal\tunnel.log has
REM the same output if the window is closed.
echo Starting the Cloudflare Tunnel...
start "Desk Tunnel" venv\Scripts\python.exe bots\tunnel.py

REM Give the tunnel a moment to print its URL before the desk's own logging
REM starts competing for the console.
timeout /t 8 /nobreak >nul

echo.
echo Starting the desk bot...
echo Webhook listens on http://127.0.0.1:8787/bar
echo Press Ctrl+C to stop.
echo.

venv\Scripts\python.exe bots\desk.py %*
set DESK_EXIT=%ERRORLEVEL%

REM --- tear the tunnel down ----------------------------------------------
REM The desk has exited, so the tunnel now points at a closed port. Leaving it
REM up would keep a public hostname alive in front of nothing, and on a quick
REM tunnel the next run gets a different hostname anyway.
echo.
echo Desk exited with code %DESK_EXIT%. Closing the tunnel...
taskkill /FI "WINDOWTITLE eq Desk Tunnel*" /T /F >nul 2>&1
taskkill /IM cloudflared.exe /F >nul 2>&1

echo Done.
pause
