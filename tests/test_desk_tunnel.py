"""The tunnel launcher, and the refusal that guards it.

The property that matters most here is negative: a tunnel must not start
without ``DESK_WEBHOOK_TOKEN``. Exposing ``/bar`` publicly with authentication
disabled means anyone who finds the hostname can inject bars, and in shadow
mode the damage is quiet - corrupted test output rather than a lost trade -
which is exactly the kind of failure that survives to the run where it does
matter.
"""

from __future__ import annotations

import pytest

import tunnel


# -- the refusal ------------------------------------------------------------

@pytest.mark.parametrize("token", [None, "", "   ", "\t\n"])
def test_preflight_refuses_without_a_token(token):
    with pytest.raises(tunnel.TunnelError, match="DESK_WEBHOOK_TOKEN"):
        tunnel.preflight(token)


def test_preflight_message_says_how_to_fix_it():
    with pytest.raises(tunnel.TunnelError) as exc:
        tunnel.preflight(None)
    text = str(exc.value)
    assert "token_urlsafe" in text          # how to generate one
    assert "?token=" in text                # how TradingView sends it
    assert "public internet" in text        # why it matters


def test_preflight_accepts_a_real_token():
    tunnel.preflight("a-token-of-reasonable-length")


def test_start_refuses_before_launching_anything(monkeypatch):
    """The refusal must come before the subprocess, not after."""
    launched = []
    monkeypatch.setattr(tunnel.subprocess, "Popen",
                        lambda *a, **k: launched.append(a))
    with pytest.raises(tunnel.TunnelError):
        tunnel.start(port=8787, hostname=None, token=None)
    assert launched == [], "cloudflared was launched despite a missing token"


# -- command construction ---------------------------------------------------

def test_quick_tunnel_command_has_no_hostname():
    argv = tunnel.build_command(8787, None, "cloudflared")
    assert argv == ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8787"]
    assert "run" not in argv


def test_named_tunnel_command_runs_the_hostname():
    argv = tunnel.build_command(9000, "desk.example.com", "cloudflared")
    assert argv[:3] == ["cloudflared", "tunnel", "run"]
    assert "desk.example.com" in argv
    assert "http://127.0.0.1:9000" in argv


def test_tunnel_binds_to_loopback_not_all_interfaces():
    """The tunnel reaches the desk over loopback; the desk never binds 0.0.0.0.

    If the desk bound a public interface, the token would be bypassable by
    anyone on the same network segment regardless of what the tunnel does.
    """
    for hostname in (None, "desk.example.com"):
        argv = tunnel.build_command(8787, hostname, "cloudflared")
        target = [a for a in argv if a.startswith("http://")][0]
        assert target.startswith("http://127.0.0.1:")
        assert "0.0.0.0" not in target


# -- quick-tunnel URL parsing ----------------------------------------------

@pytest.mark.parametrize("line,expected", [
    ("|  https://random-words-here.trycloudflare.com   |",
     "https://random-words-here.trycloudflare.com"),
    ("INF +----------------------------------------+", None),
    ("2026-09-06T03:00:00Z INF Registered tunnel connection", None),
    ("https://abc-def-123.trycloudflare.com", "https://abc-def-123.trycloudflare.com"),
])
def test_quick_url_regex(line, expected):
    match = tunnel.QUICK_URL.search(line)
    assert (match.group(0) if match else None) == expected


def test_quick_url_does_not_match_other_hosts():
    """A named-tunnel hostname must not be mistaken for a quick-tunnel one."""
    assert tunnel.QUICK_URL.search("https://desk.example.com/bar") is None
    assert tunnel.QUICK_URL.search("https://trycloudflare.com") is None


# -- the binary -------------------------------------------------------------

def test_cloudflared_is_locatable():
    """winget installs outside PATH for an already-running shell."""
    assert tunnel.cloudflared_path().lower().endswith("cloudflared.exe")


def test_missing_binary_message_says_how_to_install(monkeypatch):
    monkeypatch.setattr(tunnel.shutil, "which", lambda _: None)
    monkeypatch.setattr(tunnel.Path, "exists", lambda _: False)
    with pytest.raises(tunnel.TunnelError, match="winget install"):
        tunnel.cloudflared_path()


# -- the banner -------------------------------------------------------------

def test_banner_prints_a_complete_ready_to_paste_url(capsys):
    """The token is printed in full, on purpose.

    The banner exists to be copied into a TradingView alert. A masked URL
    forces a detour through .env on every launch, which on a quick tunnel is
    every launch. It goes to a local console on the machine that already holds
    the token - not a transcript, a commit, or a network destination.
    """
    tunnel._banner("https://abc.trycloudflare.com", 8787,
                   "supersecrettokenvalue123", stable=False)
    out = capsys.readouterr().out
    assert "https://abc.trycloudflare.com/bar?token=supersecrettokenvalue123" in out
    assert "..." not in out.split("PASTE THIS")[1].split("Alert settings")[0]


def test_banner_includes_the_alert_settings_that_matter(capsys):
    tunnel._banner("https://desk.example.com", 8787, "tok" * 4, stable=True)
    out = capsys.readouterr().out
    assert "Once Per Bar Close" in out
    assert "alert() function call" in out
    assert "leave empty" in out


def test_banner_without_a_token_emits_no_query_string(capsys):
    """Unreachable via start() - preflight refuses - but the URL must not lie."""
    tunnel._banner("https://desk.example.com", 8787, None, stable=True)
    out = capsys.readouterr().out
    assert "?token=" not in out
    assert "https://desk.example.com/bar" in out


def test_banner_never_reaches_the_tunnel_log(tmp_path, monkeypatch, capsys):
    """The token must not be duplicated onto disk.

    ``_pump`` writes cloudflared's own output to journal/tunnel.log; the
    banner is printed to stdout only. If that ever changes, the token gains a
    second on-disk home beyond .env.
    """
    log = tmp_path / "tunnel.log"
    monkeypatch.setattr(tunnel, "TUNNEL_LOG", log)
    tunnel._banner("https://abc.trycloudflare.com", 8787, "s3cr3t-value", stable=False)
    capsys.readouterr()
    assert not log.exists() or "s3cr3t-value" not in log.read_text(encoding="utf-8")


def test_quick_tunnel_banner_warns_the_hostname_is_temporary(capsys):
    tunnel._banner("https://abc.trycloudflare.com", 8787, "tok" * 4, stable=False)
    out = capsys.readouterr().out
    assert "TEMPORARY" in out
    assert "every restart" in out


def test_named_tunnel_banner_does_not_warn(capsys):
    tunnel._banner("https://desk.example.com", 8787, "tok" * 4, stable=True)
    assert "TEMPORARY" not in capsys.readouterr().out
