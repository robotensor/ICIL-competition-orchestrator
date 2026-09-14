"""`icil-orchestrator admin serve`: what stops it starting, and the console script answering."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sysconfig
import threading
import urllib.error
import urllib.request
from pathlib import Path

from icil_orchestrator import admin
from icil_orchestrator.cli import build_parser, main

TOKEN = "tok-cli-51d2e0a4c9e7-never-in-a-log"


def test_admin_serve_defaults_are_the_intake_defaults():
    args = build_parser().parse_args(["admin", "serve", "--store", "s"])
    assert (args.host, args.port, args.token_env, args.queue) == (
        admin.DEFAULT_HOST,
        admin.DEFAULT_PORT,
        admin.DEFAULT_TOKEN_ENV,
        "queue",
    )


def test_admin_serve_does_not_start_without_a_token_or_a_store(tmp_path, capsys, monkeypatch):
    store = tmp_path / "store"
    assert main(["store", "init", str(store), "--key", str(tmp_path / "key")]) == 0
    capsys.readouterr()
    base = ["admin", "serve", "--store", str(store), "--queue", str(tmp_path / "queue")]

    monkeypatch.delenv("ICIL_ADMIN_TOKEN", raising=False)
    assert main([*base, "--port", "0"]) == 2
    assert "ICIL_ADMIN_TOKEN is not set" in capsys.readouterr().err
    monkeypatch.setenv("ICIL_OTHER_TOKEN", "  ")
    assert main([*base, "--port", "0", "--token-env", "ICIL_OTHER_TOKEN"]) == 2
    assert "ICIL_OTHER_TOKEN is not set" in capsys.readouterr().err
    # A token that can be guessed is no token.
    monkeypatch.setenv("ICIL_ADMIN_TOKEN", "dev-token")
    assert main([*base, "--port", "0"]) == 2
    err = capsys.readouterr().err
    assert "ICIL_ADMIN_TOKEN" in err and "32" in err and "dev-token" not in err

    monkeypatch.setenv("ICIL_ADMIN_TOKEN", TOKEN)
    not_a_store = ["admin", "serve", "--store", str(tmp_path / "nothing"), "--port", "0"]
    assert main(not_a_store) == 2
    err = capsys.readouterr().err
    assert "is not a store" in err and TOKEN not in err


def test_the_console_script_serves_health_on_loopback(tmp_path):
    store = tmp_path / "store"
    assert main(["store", "init", str(store), "--key", str(tmp_path / "key")]) == 0
    script = Path(sysconfig.get_path("scripts")) / "icil-orchestrator"
    command = [str(script), "admin", "serve", "--store", str(store), "--port", "0"]
    proc = subprocess.Popen(
        [*command, "--queue", str(tmp_path / "queue"), "--token-env", "ICIL_TEST_ADMIN_TOKEN"],
        stderr=subprocess.PIPE,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "ICIL_TEST_ADMIN_TOKEN": TOKEN},
    )
    lines: list[str] = []
    listening = threading.Event()

    def read() -> None:
        for line in proc.stderr:
            lines.append(line)
            if "listening on" in line:
                listening.set()
        listening.set()

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        assert listening.wait(60) and proc.poll() is None, "".join(lines)
        (url,) = re.findall(r"listening on (http://127\.0\.0\.1:\d+)", "".join(lines))

        def health(token: str | None) -> tuple[int, dict]:
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            request = urllib.request.Request(f"{url}/admin/health", headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    return response.status, json.loads(response.read())
            except urllib.error.HTTPError as exc:
                return exc.code, json.loads(exc.read())

        status, body = health(TOKEN)
        assert status == 200 and body["tracks"] == ["franka_1arm"]
        assert health(None)[0] == 401 and health("wrong")[0] == 401
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        reader.join(timeout=10)
    log = "".join(lines)
    assert "GET /admin/health 200" in log and "GET /admin/health 401" in log
    assert TOKEN not in log
