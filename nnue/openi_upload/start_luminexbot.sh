#!/bin/bash
# One-command bot startup for a fresh mscope pod.
# Run on the pod:  bash /mnt/workspace/start_luminexbot.sh
# (after the pod restarts, also re-run the reverse tunnel from your laptop
#  if you want me to be able to reach the box; the bot itself needs no tunnel)
set -u
WS=/mnt/workspace

# 1. engine binary (rebuilt only if missing — the NAS copy usually survives)
if [ ! -x $WS/lichex-engine ]; then
    echo "[bot] engine binary missing — rebuilding (needs g++-13; install via"
    echo "[bot]   add-apt-repository -y ppa:ubuntu-toolchain-r/test && apt-get update && apt-get install -y g++-13"
    cd $WS/Luminex && git fetch -q origin && git reset -q --hard origin/main
    cmake -S . -B build-bot -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-13 > /dev/null 2>&1
    cmake --build build-bot -j14 --target luminex > /dev/null 2>&1
    cp build-bot/luminex $WS/lichex-engine && chmod +x $WS/lichex-engine
fi
[ -f $WS/luminex_sf1_i8.nnue ] || cp $WS/r1/r1sf_dcu_pilot_i8.nnue $WS/luminex_sf1_i8.nnue

# 2. lichess-bot (clone + deps survive on NAS; pip deps need reinstall per pod)
cd $WS/lichess-bot
[ -d lib ] || git clone -q --depth 1 https://github.com/lichess-bot-devs/lichess-bot tmpclone && rsync -a tmpclone/ ./ 2>/dev/null; rm -rf tmpclone
python3 -c "import requests, websockets, chess, rich" 2>/dev/null || python3 -m pip install -q -r requirements.txt

# 3. stop any previous instance, start fresh
pkill -f lichess-bot.py 2>/dev/null; sleep 1
nohup python3 lichess-bot.py > $WS/bot.log 2>&1 &
sleep 6
if pgrep -f lichess-bot.py > /dev/null; then
    echo "[bot] luminexbot is UP — awaiting challenges. Log: $WS/bot.log"
    grep -E "Welcome" $WS/bot.log | head -1
else
    echo "[bot] FAILED to start — last log lines:"; tail -5 $WS/bot.log
    exit 1
fi
