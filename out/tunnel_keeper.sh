#!/bin/bash
# URL-keeper for the ephemeral trycloudflare quick tunnel.
# Runs as the tunnel service: starts cloudflared, watches for the assigned
# URL, and on every change (re)writes the server's CSRF origins + restarts
# OpenBench so the public URL keeps working without manual edits.
# The current URL is always readable at /home/ubuntu/current_bench_url.
set -u
URL_FILE=/home/ubuntu/current_bench_url
LOG=/var/log/openbench-tunnel.log
LAST=""

cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8000 >>"$LOG" 2>&1 &
CFPID=$!
trap 'kill '"$CFPID"' 2>/dev/null; exit 0' TERM INT

while kill -0 "$CFPID" 2>/dev/null; do
  sleep 10
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" | tail -1)
  if [ -n "$URL" ] && [ "$URL" != "$LAST" ]; then
    echo "$(date -u +%FT%TZ) new tunnel URL: $URL" >>"$LOG"
    echo "$URL" > "$URL_FILE"
    chmod 644 "$URL_FILE"
    sed -i "s|^OPENBENCH_CSRF_ORIGINS=.*|OPENBENCH_CSRF_ORIGINS=$URL|" /opt/openbench/.env
    systemctl restart openbench
    LAST="$URL"
  fi
done
