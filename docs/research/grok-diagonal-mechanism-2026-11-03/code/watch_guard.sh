#!/usr/bin/env bash
# Watch a probe_guard log. Prints one line per event (for a Monitor):
#   SHED / GAVE UP / FAILED / ERROR lines as they appear (given-up jobs are also appended to RETRY file);
#   PROGRESS every 5 finished jobs; STALLED when no job finished for STALL_S seconds (repeated every
#   STALL_S while stalled, with the guard's last line); RESUMED when progress returns; DONE at the end.
# Usage: watch_guard.sh LOG RETRY_FILE [STALL_S]
LOG=$1
RETRY=$2
STALL_S=${3:-300}
until [ -f "$LOG" ]; do sleep 1; done
pos=0
last_progress=$(date +%s)
last_stall_note=0
stalled=0
done_n=0
# FROM_END=1: re-arming an expired watch; skip events already reported, keep the finished count
if [ -n "$FROM_END" ]; then pos=$(wc -c < "$LOG"); done_n=$(grep -c 'finished (' "$LOG"); fi
while true; do
  size=$(wc -c < "$LOG")
  if [ "$size" -gt "$pos" ]; then
    new=$(tail -c +$((pos + 1)) "$LOG" | head -c $((size - pos)))
    pos=$size
    while IFS= read -r line; do
      [ -z "$line" ] && continue
      case "$line" in
        *"finished ("*)
          done_n=$((done_n + 1))
          last_progress=$(date +%s)
          if [ $stalled -eq 1 ]; then echo "RESUMED: $line"; stalled=0; fi
          [ $((done_n % 5)) -eq 0 ] && echo "PROGRESS: $line"
          ;;
        *"GIVING UP"*|*"SET ASIDE"*)
          echo "GAVE UP (will retry later): $line"
          echo "$line" | sed -n 's/.*plan job \([^ ]*\) was shed.*/\1/p' >> "$RETRY"
          ;;
        *"was shed"*|*"shedding"*) echo "SHED: $line" ;;
        *"FINISHED WITH FAILURES"*|*"Traceback"*|*"Error"*|*"exited (code "[1-9-]*) echo "FAILED: $line" ;;
        *"ALL_DONE"*) echo "DONE: $line"; exit 0 ;;
      esac
    done <<< "$new"
  fi
  now=$(date +%s)
  if [ $((now - last_progress)) -ge "$STALL_S" ] && [ $((now - last_stall_note)) -ge "$STALL_S" ]; then
    stalled=1
    last_stall_note=$now
    echo "STALLED: no job finished for $(((now - last_progress) / 60)) min; guard says: $(tail -1 "$LOG")"
  fi
  sleep 10
done
