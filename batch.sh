#!/bin/zsh
# Analyse + cut (copy mode) every MP4 in the given directories, 2 at a time. Usage: ./batch.sh DIR...
setopt nullglob
DIR=${0:A:h}                     # this script's directory (project root)
LOG=${LOG:-$DIR/batch.log}
: > "$LOG"
list=()
for d in "$@"; do for f in "$d"/*.MP4 "$d"/*.mp4; do list+=("$f"); done; done
echo "$(date '+%H:%M:%S') start: ${#list[@]} videos" >> "$LOG"
printf '%s\n' "${list[@]}" | xargs -P 2 -I{} "$DIR/worker.sh" {} 2>> "$LOG"
echo "$(date '+%H:%M:%S') done" >> "$LOG"
