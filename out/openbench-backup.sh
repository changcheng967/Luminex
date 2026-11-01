#!/bin/bash
# Daily OpenBench backup: sqlite DB (14-day retention) + Media (nets, PGNs)
set -u
DEST=/opt/openbench-backups
mkdir -p "$DEST"
STAMP=$(date +%F)
sqlite3 /opt/openbench/db.sqlite3 ".backup $DEST/db-$STAMP.sqlite3"
cp -f /opt/openbench/Media/* "$DEST/" 2>/dev/null || true
ls -t "$DEST"/db-*.sqlite3 2>/dev/null | tail -n +15 | xargs -r rm -f
