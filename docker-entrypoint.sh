#!/bin/sh
# Container start-up: fetch the reference tables (DE440, debias; the asteroid-perturber kernel ships in the package)
# into the /data volume (skipped when they are
# already there), then serve the web UI.
#   PYROORB_ASTEROIDS=1   also fetch the 616 MB asteroid ephemeris (the 16 massive perturbers)
#   PYROORB_FETCH_DATA=0  skip the fetch entirely
#   PORT                  the port inside the container (default 8000)
# Any arguments are passed to pyroorb-web (e.g. --port 9000).
set -e
if [ "${PYROORB_FETCH_DATA:-1}" != "0" ]; then
  if [ "${PYROORB_ASTEROIDS:-0}" = "1" ]; then
    pyroorb-fetch-data --asteroids
  else
    pyroorb-fetch-data
  fi
fi
exec pyroorb-web --host 0.0.0.0 "$@"
