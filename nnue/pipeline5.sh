#!/bin/bash
# pipeline5 FIX: overlapped download+pack. Bug was wrong offset for next wave
# (tail -n +$((WAVE*BATCH+1)) on shrinking pending.txt). Fixed: always lines 41-80
# of CURRENT pending.txt (same baseline as wave.txt = lines 1-40).
cd /teamspace/studios/this_studio/data
mkdir -p frames3 work
BIN=/teamspace/studios/this_studio/Luminex/build
LOG=pipeline5.log
BATCH=40
CONNS=96
URL="https://storage.lczero.org/files/training_data/test91/"

[ -s work/alltars.txt ] || curl -s "$URL" | grep -o 'training-run2-test91-[0-9]*-[0-9]*\.tar' | sort -u > work/alltars.txt
TOTAL=$(wc -l < work/alltars.txt); touch work/done_tars.txt
echo "pipeline5 start: $TOTAL tars batch=$BATCH conns=$CONNS overlapped $(date)" >> $LOG
next_frame() { ls frames3/frame_*.xz 2>/dev/null | wc -l; }

# Rebuild pending from scratch (ignore old buggy done list)
sort -r work/alltars.txt | grep -vxF -f work/done_tars.txt > work/pending.txt

DL_DIR=/tmp/dl_cur; NEXT_DIR=/tmp/dl_next
rm -rf $DL_DIR $NEXT_DIR; mkdir -p $DL_DIR $NEXT_DIR

download_wave() {
    local dir=$1; local list=$2
    rm -f $dir/*.tar
    local C=0
    while read -r tar; do
        curl -sL --max-time 300 -o "$dir/$tar" "$URL/$tar" &
        C=$((C+1)); [ $((C % CONNS)) -eq 0 ] && wait
    done < "$list"
    wait
}

# Download first wave (lines 1-40 of pending)
head -n $BATCH work/pending.txt > work/wave.txt
download_wave $DL_DIR work/wave.txt
echo "wave 1 pre-downloaded $(date)" >> $LOG

WAVE=0
while [ "$(wc -l < work/pending.txt)" -gt 0 ]; do
    WAVE=$((WAVE+1))

    # Current wave = lines 1-40 of current pending.txt
    head -n $BATCH work/pending.txt > work/wave.txt
    NT=$(wc -l < work/wave.txt)
    [ "$NT" -eq 0 ] && break

    # NEXT wave = lines 41-80 of SAME pending.txt (FIXED: same baseline, not WAVE*BATCH)
    tail -n +$((BATCH + 1)) work/pending.txt | head -n $BATCH > work/next_wave.txt
    NEXT_NT=$(wc -l < work/next_wave.txt)

    # Start downloading NEXT wave in background
    if [ "$NEXT_NT" -gt 0 ]; then
        download_wave $NEXT_DIR work/next_wave.txt &
        DL_PID=$!
    fi

    # Pack CURRENT wave (foreground)
    RAWS=()
    for tar in $(cat work/wave.txt); do
        f=$DL_DIR/$tar
        sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
        if [ "$sz" -le 10240 ]; then
            echo "$tar" >> work/done_tars.txt  # stub or failed download
            rm -f "$f"
        else
            RAWS+=("$f")
        fi
    done

    if [ ${#RAWS[@]} -gt 0 ]; then
        n=$(printf "%04d" $(next_frame))
        $BIN/lc0pack --threads 4 -o work/pack_$n.raw "${RAWS[@]}" >> $LOG 2>&1
        xz -3 -T0 -c work/pack_$n.raw > frames3/frame_$n.xz && rm -f work/pack_$n.raw
        echo "frame_$n: $(du -sh frames3/frame_$n.xz | cut -f1), total $(du -sh frames3 | cut -f1), $(date)" >> $LOG
    fi

    # Mark current wave done
    cat work/wave.txt >> work/done_tars.txt
    sort -r work/alltars.txt | grep -vxF -f work/done_tars.txt > work/pending.txt

    # Wait for next download to finish, then swap directories
    if [ "$NEXT_NT" -gt 0 ]; then
        wait $DL_PID
    fi
    rm -rf $DL_DIR
    mv $NEXT_DIR $DL_DIR
    mkdir -p $NEXT_DIR

    REMAIN=$(wc -l < work/pending.txt)
    echo "wave $WAVE done: $REMAIN remaining, frames=$(ls frames3 | wc -l) $(date)" >> $LOG
done

echo "COMPLETE: $(ls frames3 | wc -l) frames, $(du -sh frames3 | cut -f1), $(date)" >> $LOG
