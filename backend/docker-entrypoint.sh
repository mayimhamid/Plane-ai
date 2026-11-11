#!/bin/sh
# Optionally apply migrations, then replace this shell with the server process
# (`exec`) so it receives SIGTERM directly from Docker.
set -eu

if [ "${PLANE_AI_RUN_MIGRATIONS:-false}" = "true" ]; then
    echo "docker-entrypoint: applying database migrations"
    alembic upgrade head
fi

exec "$@"
