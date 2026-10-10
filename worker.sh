#!/bin/zsh
# worker.sh VIDEO : analyse + cut one video, append one line to the batch log
f="$1"; DIR=${0:A:h}; LOG=${LOG:-$DIR/batch.log}
b=$(basename "$f"); b=${b%.MP4}; b=${b%.mp4}
s=$(date +%s)
cd "$DIR"
if uv run python -m dronesift "$f" --mode copy > "$(dirname "$f")/${b}_cuts.log" 2>&1; then st=ok; else st=FAIL; fi
n=$(grep -c '^  ' "$(dirname "$f")/${b}_cuts.log")
echo "$(date '+%H:%M:%S') $st $b segments=$n $(( $(date +%s) - s ))s" >> "$LOG"
