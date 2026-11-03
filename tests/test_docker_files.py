"""The Docker image's definition: what users rely on when they run the web UI from it. Building
and serving it is CI's docker job; this pins the properties without needing Docker."""
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "docker" / "Dockerfile").read_text()


def test_it_runs_as_an_unprivileged_user():
    assert "\nUSER pyroorb\n" in DOCKERFILE


def test_everything_that_persists_is_in_the_data_volume():
    assert "VOLUME /data" in DOCKERFILE
    for var in ("PYROORB_EPHEM_DIR=/data/", "PYROORB_DEBIAS_DIR=/data/",
                "PYROORB_CACHE_DIR=/data/", "LOCAL_PERSISTENT_FOLDER=/data/"):
        assert var in DOCKERFILE, var


def test_the_final_image_carries_no_compiler():
    """Only the build stage starts from the full python image (which has gcc); the image users
    pull starts from the slim one and installs wheels from the build stage."""
    stages = [line.split()[1] for line in DOCKERFILE.splitlines() if line.startswith("FROM ")]
    assert stages[-1] == "python:3.12-slim" and len(stages) == 2


def test_the_build_context_cannot_carry_secrets_or_data():
    """docker/Dockerfile.dockerignore excludes everything, then lets in only what the build needs: .env, the
    ephemerides and local state can never reach an image."""
    lines = [ln.strip() for ln in (ROOT / "docker" / "Dockerfile.dockerignore").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    assert lines[0] == "*"
    allowed = {ln[1:].rstrip("/") for ln in lines if ln.startswith("!")}
    assert allowed == {"pyproject.toml", "README.md", "LICENSE.md", "src", "docker-entrypoint.sh"}


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh")
def test_the_entrypoint_parses_and_serves_on_all_interfaces():
    script = ROOT / "docker-entrypoint.sh"
    assert subprocess.run(["sh", "-n", str(script)]).returncode == 0
    assert "exec pyroorb-web --host 0.0.0.0" in script.read_text()


def test_assist_is_built_against_the_rebound_the_image_runs():
    """ASSIST compiles against REBOUND's headers and runs against REBOUND's library: the build
    stage installs REBOUND, builds ASSIST without isolation against it, and pins the image's
    REBOUND to that same version (no isolated build env picking another 4.x)."""
    build = DOCKERFILE[:DOCKERFILE.index("FROM python:3.12-slim")]
    assert "--no-build-isolation" in build and '"assist>=1.2,<1.3"' in build
    assert "print('rebound==' + rebound.__version__)" in build
    assert "--constraint /tmp/constraints.txt --only-binary assist" in build
    assert build.index('pip install --no-cache-dir "rebound>=4.4.11,<5"') < \
        build.index("--no-build-isolation")
