#!/bin/bash
# Chain-waiter: block until the seed runs finish, then run the diversity
# screen on the same GPU. Started detached; safe to re-run (idempotent
# guards inside session2_diversity.sh).
WS=/workspace/luminex
cd $WS
while ! grep -q SESSION2_SEED_DONE logs/session2.log 2>/dev/null; do
  sleep 60
done
exec bash session2_diversity.sh
