"""Backend reload isolation from frontend development (spec #39).

The backend dev server must watch only backend-relevant paths so that
frontend installation, compilation, and generated-file changes do not
restart the backend process. A long Live run keeps its request ID and
progress history while the frontend is developing concurrently.

These tests verify:
1. The documented and containerised dev commands scope uvicorn's --reload
   to backend/src only (static configuration checks).
2. At runtime, uvicorn with --reload-dir backend/src does not reload when
   files outside that directory change (integration check proving stability).
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "backend" / "Dockerfile"
README = REPO_ROOT / "README.md"


def _assert_scoped_reload(content: str, context: str, *, extract_uvicorn_cmd=None):
    """Assert that a uvicorn command in the content uses --reload-dir backend/src.

    The extract_uvicorn_cmd callable, if provided, narrows the content to the
    specific uvicorn command line before asserting. This prevents false passes
    from unrelated mentions of the flags elsewhere in the file.
    """
    if extract_uvicorn_cmd:
        content = extract_uvicorn_cmd(content)
        assert content is not None, f"{context}: could not find uvicorn command"

    assert "--reload" in content, f"{context}: must enable uvicorn reload"
    assert "--reload-dir" in content, f"{context}: must scope reload with --reload-dir"
    assert "backend/src" in content, f"{context}: --reload-dir must target backend/src"


def _extract_dockerfile_uvicorn_cmd(content: str) -> str | None:
    """Extract the CMD line containing uvicorn from a Dockerfile."""
    for line in content.splitlines():
        if "uvicorn" in line and line.strip().startswith("CMD"):
            return line
    return None


def _extract_readme_uvicorn_cmd(content: str) -> str | None:
    """Extract the uvicorn command from the README's code block.

    Looks for a line containing 'uvicorn' and '--reload' within a bash code block.
    """
    in_bash_block = False
    for line in content.splitlines():
        if line.strip().startswith("```bash"):
            in_bash_block = True
        elif line.strip().startswith("```"):
            in_bash_block = False
        elif in_bash_block and "uvicorn" in line and "--reload" in line:
            return line
    return None


def test_dockerfile_scopes_reload_to_backend_source():
    """The backend Dockerfile uses --reload-dir to watch only backend/src,
    preventing frontend or other non-backend files from triggering reloads."""
    content = DOCKERFILE.read_text()
    _assert_scoped_reload(content, "Dockerfile", extract_uvicorn_cmd=_extract_dockerfile_uvicorn_cmd)


def test_readme_documents_scoped_reload_for_local_dev():
    """The README's local dev instructions include --reload-dir so developers
    running from the repo root don't accidentally watch frontend files."""
    content = README.read_text()
    _assert_scoped_reload(content, "README", extract_uvicorn_cmd=_extract_readme_uvicorn_cmd)


@pytest.mark.integration
def test_uvicorn_does_not_reload_on_frontend_changes(tmp_path):
    """Runtime proof: uvicorn with --reload-dir backend/src does not reload
    when a file outside that directory changes.

    This is the automated integration check that proves the cross-stack
    development setup remains stable: a long Live run keeps its request ID
    and progress history while the frontend is developing concurrently,
    because the backend process never restarts.
    """
    port = 18765
    log_file = tmp_path / "uvicorn.log"
    env = {
        "REGULA_MODE": "demo",
        "QUERY_LOG_PATH": str(tmp_path / "queries.jsonl"),
    }

    with open(log_file, "w") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.src.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--reload",
                "--reload-dir",
                "backend/src",
            ],
            cwd=REPO_ROOT,
            env={**os.environ, **env},
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    try:
        _wait_for_server(f"http://127.0.0.1:{port}/health", timeout=10)

        frontend_dir = REPO_ROOT / "frontend"
        frontend_dir.mkdir(exist_ok=True)
        probe_file = frontend_dir / ".reload-probe-test"
        probe_file.write_text("probe")
        time.sleep(0.5)
        probe_file.unlink(missing_ok=True)

        time.sleep(2)

        proc.terminate()
        proc.wait(timeout=5)

    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    output = log_file.read_text()
    assert "Reloading" not in output, (
        "uvicorn reloaded after frontend file change; "
        "--reload-dir backend/src is not isolating the watch scope"
    )


def _wait_for_server(url: str, timeout: float) -> None:
    """Poll a URL until it responds or timeout expires."""
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass
        time.sleep(0.2)
    raise TimeoutError(f"Server at {url} did not start within {timeout}s")
