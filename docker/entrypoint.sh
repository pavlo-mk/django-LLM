#!/bin/sh
# Container entrypoint: prepare the app, then exec the given command (CMD).
set -e

echo "Running migrations..."
python manage.py migrate --noinput

echo "Collecting static files..."
python manage.py collectstatic --noinput

exec "$@"
