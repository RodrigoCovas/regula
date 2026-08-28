"""Backend reload isolation from frontend development (spec #39).

The backend dev server must watch only backend-relevant paths so that
frontend installation, compilation, and generated-file changes do not
restart the backend process. A long Live run keeps its request ID and
progress history while the frontend is developing concurrently.

These tests verify the documented and containerised dev commands scope
uvicorn's --reload to backend/src only.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "backend" / "Dockerfile"
README = REPO_ROOT / "README.md"


def test_dockerfile_scopes_reload_to_backend_source():
    """The backend Dockerfile uses --reload-dir to watch only backend/src,
    preventing frontend or other non-backend files from triggering reloads."""
    content = DOCKERFILE.read_text()
    assert "--reload" in content, "Dockerfile must enable uvicorn reload"
    assert "--reload-dir" in content, (
        "Dockerfile must scope reload watching with --reload-dir"
    )
    assert "backend/src" in content, (
        "--reload-dir must target backend/src"
    )


def test_readme_documents_scoped_reload_for_local_dev():
    """The README's local dev instructions include --reload-dir so developers
    running from the repo root don't accidentally watch frontend files."""
    content = README.read_text()
    assert "--reload" in content, "README must document uvicorn reload"
    assert "--reload-dir" in content, (
        "README must document --reload-dir for local dev"
    )
    assert "backend/src" in content, (
        "README --reload-dir must target backend/src"
    )
