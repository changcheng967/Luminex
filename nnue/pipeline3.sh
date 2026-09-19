#!/bin/bash
# test91 (Leela, latest) -> Luminex gamepack frames. RAW-move format, restart-safe.
# Per tar: download -> lc0pack -> keep small raw in work/raws/. Every 200 tars:
# merge batch -> xz -6 -> frames3/frame_NNNN.xz. Disk stays < ~2GB of work + frames.
cd /teamspace/studios/this_studio/data
mkdir -p frames3 work/raws
BUDGET=$((17*1024*1024*1024))   # frames cap (final dataset well under 20GB quota)
IDX_URL="https://storage.lczero.org/files/training_data/test91/"
BIN=/teamspace/studios/this_studio/Luminex/build
LOG=pipeline3.log

[ -s work/alltars.txt ] || curl -s "$IDX_URL" | grep -o 'training-run2-test91-[0-9]*-[0-9]*\.tar' | sort -u > work/alltars.txt
touch work/done_tars.txt
total_frames() { du -sb frames3 2>/dev/null | cut -f1; }
next_frame() { ls frames3/frame_*.xz 2>/dev/null | wc -l; }

merge_batch() {
  local raws=(work/raws/*.raw)
  [ ${#raws[@]} -eq 0 ] && return 0
  local n=$(printf "%04d" "$(next_frame)")
  $BIN/luminex-merge -o "work/batch_$n.raw" "${raws[@]}" >> $LOG 2>&1 \
    && xz -6 -T0 -c "work/batch_$n.raw" > "frames3/frame_$n.xz" \
    && rm -f "work/batch_$n.raw" work/raws/*.raw \
    && echo "frame_$n.xz: $(du -sh frames3/frame_$n.xz | cut -f1), total $(du -sh frames3 | cut -f1), $(date)" >> $LOG
}

sort -r work/alltars.txt | while read -r tar; do
  grep -qx "$tar" work/done_tars.txt && continue
  if [ "$(total_frames)" -ge "$BUDGET" ]; then echo "BUDGET REACHED $(date)" >> $LOG; break; fi
  curl -sL --max-time 600 -o work/cur.tar "$IDX_URL/$tar" || { echo "DLFAIL $tar" >> $LOG; continue; }
  sz=$(stat -c%s work/cur.tar 2>/dev/null || echo 0)
  if [ "$sz" -le 10240 ]; then echo "$tar" >> work/done_tars.txt; rm -f work/cur.tar; continue; fi
  base=$(echo "$tar" | tr '/' '_')
  if $BIN/lc0pack --threads 3 -o "work/raws/$base.raw" work/cur.tar >> $LOG 2>&1; then
    echo "$tar" >> work/done_tars.txt
  else
    echo "PACKFAIL $tar" >> $LOG
  fi
  rm -f work/cur.tar
  n=$(ls work/raws/*.raw 2>/dev/null | wc -l)
  [ "$n" -ge 200 ] && merge_batch
  # disk safety: work stays small (200 raws ~ 230MB + one tar)
done
merge_batch
echo "CYCLE COMPLETE frames=$(ls frames3 | wc -l) bytes=$(total_frames) done_tars=$(wc -l < work/done_tars.txt) $(date)" >> $LOG
