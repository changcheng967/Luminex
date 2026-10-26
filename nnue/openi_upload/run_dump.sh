#!/bin/bash
# Restart-safe dump runner: raw text appends to a part file on NAS, compressed
# once at the end. After a pod restart, re-run this script — it reads the
# progress log, skips already-emitted entries, and continues. The single-writer
# guard (mkdir lock) prevents the two-racing-processes bug that triplicated
# the first attempt. Usage: bash run_dump.sh <binpack> <out_prefix>
set -u
BIN=${1:-/mnt/workspace/r1/test80-full.binpack}
OUT=${2:-/mnt/workspace/r1/sf_train_full_q}
RAW=$OUT.raw
PROG=$BIN.progress

mkdir -p "$BIN.lock" 2>/dev/null || { echo "LOCK HELD — another dump is running"; exit 1; }
trap 'rmdir "$BIN.lock" 2>/dev/null' EXIT

SKIP=0
if [ -f "$PROG" ] && [ -f "$RAW" ]; then
    SKIP=$(grep -oE "scanned=[0-9]+" "$PROG" | tail -1 | cut -d= -f2)
    # the last line may be mid-write; resume from the second-to-last checkpoint
    N=$(grep -c "scanned=" "$PROG")
    if [ "$N" -gt 1 ]; then
        SKIP=$(grep -oE "scanned=[0-9]+" "$PROG" | tail -2 | head -1 | cut -d= -f2)
    fi
    echo "[resume] skipping $SKIP entries already emitted"
fi

gcc_ok=$(which g++-13 >/dev/null 2>&1 && echo 1 || echo 0)
if [ "$gcc_ok" = "0" ]; then
    apt-get install -y -qq software-properties-common >/dev/null 2>&1
    add-apt-repository -y ppa:ubuntu-toolchain-r/test >/dev/null 2>&1
    apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq g++-13 >/dev/null 2>&1
fi

DUMP=/tmp/sf_dump_r
[ -x $DUMP ] || g++-13 -O2 -std=c++20 \
    -I /mnt/workspace/r1/nnue-pytorch/data_loader/cpp \
    -o $DUMP /mnt/workspace/Luminex/nnue/openi_upload/sf_binpack_dump.cpp -lzstd

[ -f /mnt/workspace/Luminex/nnue/openi_upload/sf_binpack_dump.cpp ] || {
    mkdir -p /mnt/workspace && cd /mnt/workspace
    [ -d Luminex ] || for i in 1 2 3; do git clone -q https://github.com/changcheng967/Luminex && break; sleep 5; done
}

$DUMP "$BIN" 0 "$SKIP" >> "$RAW" 2>> "$PROG"
RC=$?
if [ $RC -ne 0 ]; then echo "[dump] exited $RC — rerun this script to resume"; exit $RC; fi

echo "[dump] complete — compressing $(du -h "$RAW" | cut -f1)"
xz -T0 -6 --block-size=64MiB -c "$RAW" > "$OUT.tsv.xz" && rm -f "$RAW" "$PROG"
echo "ALL_DONE $OUT.tsv.xz $(stat -c%s "$OUT.tsv.xz") bytes"
