# PyRoOrb web UI in a container: no Python, no compiler on the host.
#
#   docker run -d -p 8000:8000 -v pyroorb-data:/data ghcr.io/petervanwylen/pyroorb
#   -> http://localhost:8000
#
# The reference tables (DE440, the debias tables; with PYROORB_ASTEROIDS=1 also the 616 MB
# asteroid ephemeris) download into the /data volume on the first start and are reused by every
# later start and every newer image. See docs/INSTALL.md ("Docker").
#
# Build from the repository root (the context is the root):
#   docker build -f docker/Dockerfile -t pyroorb .
# Its ignore file is docker/Dockerfile.dockerignore (BuildKit reads it next to the Dockerfile).
# Neither file sits at the root, on purpose: Railway builds a root Dockerfile in preference to
# railway.json's NIXPACKS builder, and a root .dockerignore filters the files Nixpacks
# generates (.nixpacks/) out of its build context. The deployment needs the Nixpacks install
# (requirements.txt: .[itf] + the deployment extras), which this image does not carry.

# --- build: compile ASSIST (it has no pre-built wheels) and PyRoOrb into wheels -------------
# The full python image already carries gcc and the C headers; only the wheels leave this stage.
FROM python:3.12 AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE.md ./
COPY src ./src
# ASSIST compiles against REBOUND's C headers and runs against REBOUND's library, so both must
# be the SAME release: install REBOUND, build ASSIST without build isolation (against exactly
# that REBOUND), then resolve PyRoOrb's wheels with REBOUND pinned to it and ASSIST taken only
# from the wheel just built.
RUN pip install --no-cache-dir "rebound>=4.4.11,<5" numpy setuptools wheel \
    && python -c "import rebound; print('rebound==' + rebound.__version__)" > /tmp/constraints.txt \
    && pip wheel --no-cache-dir --no-build-isolation --no-deps --wheel-dir /wheels "assist>=1.2,<1.3" \
    && pip wheel --no-cache-dir --constraint /tmp/constraints.txt --only-binary assist \
       --find-links /wheels --wheel-dir /wheels .

# --- run: the wheels only, no compiler ------------------------------------------------------
FROM python:3.12-slim
RUN useradd --create-home --uid 1000 pyroorb \
    && mkdir -p /data && chown pyroorb:pyroorb /data
# installed straight from the build stage (a bind mount), so the wheels never become a layer
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    pip install --no-cache-dir --no-index --find-links /wheels pyroorb
COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# everything that persists lives in the /data volume
ENV PYROORB_EPHEM_DIR=/data/ephem \
    PYROORB_DEBIAS_DIR=/data/debias \
    PYROORB_CACHE_DIR=/data/cache \
    LOCAL_PERSISTENT_FOLDER=/data/local \
    PYTHONUNBUFFERED=1 \
    PORT=8000
VOLUME /data
USER pyroorb
WORKDIR /home/pyroorb
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/healthz', timeout=4)"
ENTRYPOINT ["docker-entrypoint.sh"]
