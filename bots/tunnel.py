"""Cloudflare Tunnel launcher for the desk webhook.

Exposes ``127.0.0.1:<port>/bar`` so a TradingView alert can reach it.
``start_desk.bat`` runs this alongside ``desk.py``; it is also runnable alone
to check a tunnel comes up before wiring an alert to it.

Two modes, chosen by whether ``DESK_TUNNEL_HOSTNAME`` is set in ``.env``
-----------------------------------------------------------------------

**Named tunnel** (``DESK_TUNNEL_HOSTNAME=desk.example.com``) - a **stable**
hostname that survives restarts. Requires a Cloudflare account and a domain
whose DNS is on Cloudflare, set up once with ``cloudflared tunnel login``,
``tunnel create`` and ``tunnel route dns``. This is the only mode that gives a
URL worth pasting into a TradingView alert permanently.

**Quick tunnel** (hostname unset) - no account, works immediately, and
allocates a **random ``*.trycloudflare.com`` hostname that changes on every
restart**. Cloudflare documents it as unsuitable for production. It is useful
for a one-off end-to-end check; it is not useful for an alert you intend to
leave running, because every desk restart silently orphans the alert. This
module prints that warning every time rather than once, because the failure is
quiet: TradingView keeps POSTing to a dead hostname and nothing surfaces.

Security
--------
The tunnel makes the webhook reachable from the public internet, so
``DESK_WEBHOOK_TOKEN`` stops being optional. :func:`preflight` refuses to start
without one. TradingView cannot send custom headers, so the token has to travel
as a ``?token=`` query parameter - which means it appears in Cloudflare's
request logs and in TradingView's stored alert config. That is a real
trade-off and the reason the token is worth rotating if either is exposed.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

log = logging.getLogger("desk.tunnel")

#: cloudflared prints the assigned quick-tunnel hostname to stderr, inside a
#: box drawn with box-drawing characters. This matches the URL itself.
QUICK_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

DEFAULT_PORT = 8787
TUNNEL_LOG = PROJECT_ROOT / "journal" / "tunnel.log"


class TunnelError(RuntimeError):
    """The tunnel could not be started. The message says why."""


def cloudflared_path() -> str:
    """Locate the binary, including the winget install location.

    winget installs to ``C:\\Program Files (x86)\\cloudflared`` and the PATH
    change does not reach an already-running shell, so a plain ``which`` finds
    nothing in the session that just installed it.
    """
    found = shutil.which("cloudflared")
    if found:
        return found
    for candidate in (
        Path(r"C:\Program Files (x86)\cloudflared\cloudflared.exe"),
        Path(r"C:\Program Files\cloudflared\cloudflared.exe"),
    ):
        if candidate.exists():
            return str(candidate)
    raise TunnelError(
        "cloudflared not found. Install it with:\n"
        "    winget install --id Cloudflare.cloudflared"
    )


def preflight(token: str | None) -> None:
    """Refuse to expose an unauthenticated endpoint.

    Without a token every check in ``build_webhook_app`` is skipped and anyone
    who finds the hostname can inject bars. In shadow mode that only corrupts
    test output, which is exactly why it is worth refusing now: the same path
    is the one a real strategy would run through later, and a habit formed
    here is the habit that persists.
    """
    if not (token or "").strip():
        raise TunnelError(
            "DESK_WEBHOOK_TOKEN is not set.\n"
            "\n"
            "A tunnel puts the /bar endpoint on the public internet, and\n"
            "without a token the endpoint accepts bars from anyone. Set one:\n"
            "\n"
            "    python -c \"import secrets; print(secrets.token_urlsafe(32))\"\n"
            "\n"
            "then put it in .env as DESK_WEBHOOK_TOKEN=<value> and append\n"
            "?token=<value> to the TradingView webhook URL."
        )


def build_command(port: int, hostname: str | None, binary: str) -> list[str]:
    """The cloudflared argv for the chosen mode."""
    url = f"http://127.0.0.1:{port}"
    if hostname:
        # A named tunnel is configured out-of-band (`cloudflared tunnel login`,
        # `create`, `route dns`). Running it by hostname uses that config.
        return [binary, "tunnel", "run", "--url", url, hostname]
    return [binary, "tunnel", "--url", url]


def _pump(stream, hostname: str | None, on_url) -> None:
    """Mirror cloudflared output to the log, watching for the quick URL."""
    TUNNEL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with TUNNEL_LOG.open("a", encoding="utf-8") as handle:
        for raw in iter(stream.readline, ""):
            line = raw.rstrip()
            handle.write(line + "\n")
            handle.flush()
            if hostname is None:
                match = QUICK_URL.search(line)
                if match:
                    on_url(match.group(0))


def start(port: int = DEFAULT_PORT, hostname: str | None = None,
          token: str | None = None, wait: bool = True) -> subprocess.Popen:
    """Launch cloudflared. Returns the process."""
    preflight(token)
    binary = cloudflared_path()
    command = build_command(port, hostname, binary)
    log.info("starting: %s", " ".join(command))

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    found: list[str] = []
    thread = threading.Thread(
        target=_pump, args=(process.stdout, hostname, found.append), daemon=True)
    thread.start()

    if hostname:
        _banner(f"https://{hostname}", port, token, stable=True)
    elif wait:
        thread.join(timeout=30)
        if found:
            _banner(found[0], port, token, stable=False)
        else:
            print("  Tunnel started but no URL seen yet - check "
                  f"{TUNNEL_LOG}", flush=True)
    return process


def _banner(base: str, port: int, token: str | None, stable: bool) -> None:
    bar = "=" * 72
    masked = f"{token[:4]}...{token[-4:]}" if token and len(token) > 8 else "<token>"
    print(f"\n{bar}")
    print("  DESK WEBHOOK IS NOW PUBLIC")
    print(bar)
    print(f"  Local     http://127.0.0.1:{port}/bar")
    print(f"  Public    {base}/bar")
    print()
    print("  TradingView webhook URL (token from .env, shown masked here):")
    print(f"      {base}/bar?token={masked}")
    if not stable:
        print()
        print("  *** THIS HOSTNAME IS TEMPORARY ***")
        print("  A quick tunnel gets a new random hostname on every restart.")
        print("  The TradingView alert will keep POSTing to the old one and")
        print("  fail silently. Use a named tunnel for anything left running.")
    print(bar + "\n", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="tunnel.py",
        description="Expose the desk webhook through a Cloudflare Tunnel.")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--hostname", default=None,
                    help="named-tunnel hostname; omit for a quick tunnel")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")
    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:  # noqa: BLE001
        pass

    hostname = args.hostname or (os.environ.get("DESK_TUNNEL_HOSTNAME") or "").strip()
    token = os.environ.get("DESK_WEBHOOK_TOKEN")
    try:
        process = start(args.port, hostname or None, token)
    except TunnelError as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 1
    try:
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
