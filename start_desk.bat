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
REM Started first so the public hostname is printed before bars arrive.
REM
REM `cmd /k` keeps that window open after tunnel.py exits. Without it a failed
REM cloudflared start closes the window instantly and the error is unreadable -
REM and on success the window is where the ready-to-paste TradingView URL
REM lives, so it has to survive. cloudflared's ongoing log goes to
REM journal\tunnel.log rather than the console, so the URL stays on screen
REM instead of scrolling away behind connection chatter.
echo Starting the Cloudflare Tunnel...
start "Desk Tunnel - WEBHOOK URL IS HERE" cmd /k venv\Scripts\python.exe bots\tunnel.py

REM Give the tunnel time to negotiate and print its URL before the desk's own
REM logging starts competing for attention.
timeout /t 10 /nobreak >nul

echo.
echo ------------------------------------------------------------
echo   The TradingView webhook URL - hostname AND token, ready to
echo   paste - is printed in the "Desk Tunnel" window that just
echo   opened. Copy it from there.
echo.
echo   On a quick tunnel that URL changes every launch, so re-copy
echo   it into the alert each time you restart.
echo ------------------------------------------------------------
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
