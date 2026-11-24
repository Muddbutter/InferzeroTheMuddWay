#!/usr/bin/env bash
# The watchdog for GPU test runs on Linux, as bench/gpu_guard.ps1 on Windows (docs/gpu-failsafe.md 5):
#   bench/gpu_guard.sh --timeout 60 -- ./bin/attn_bench --edge
# Refuses while the GPU hang marker (~/.inferzero/gpu_hang.txt, ARC_DATA_DIR when set) is from this boot or another
# GPU job runs; keeps the machine from idling or sleeping (systemd-inhibit, when there); runs the command with
# ARC_GPU_WAIT_S a little below the timeout (unless set), its output to the console and to --log; past the timeout
# kills its process group, and if it is still there 10 s later writes the marker (kind=hang, killed=no).
# Exit code: the command's; 124 when killed at the timeout; 3 for a refusal or a stuck process.
set -u
TIMEOUT=60 WAIT_S="" LOG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --timeout) TIMEOUT=$2; shift 2;;
    --wait-s) WAIT_S=$2; shift 2;;
    --log) LOG=$2; shift 2;;
    --) shift; break;;
    *) break;;
  esac
done
if [ $# -eq 0 ]; then echo "usage: gpu_guard.sh [--timeout 60] [--wait-s S] [--log file] -- program [arguments]"; exit 1; fi

DATA=${ARC_DATA_DIR:-$HOME/.inferzero}
mkdir -p "$DATA"
MARKER="$DATA/gpu_hang.txt" LOCK="$DATA/gpu_guard.lock"
BOOT=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null)
marker_this_boot() { [ -f "$MARKER" ] && [ -n "$BOOT" ] && grep -qx "boot=$BOOT" "$MARKER"; }
show_marker() {
  echo "The GPU hang marker ($MARKER):" >&2
  sed 's/^/  /' "$MARKER" >&2
  echo "Restart the computer before using the GPU again (or runz --clear-hang)." >&2
}

if marker_this_boot; then echo "gpu_guard: a GPU hang since this computer started: no run." >&2; show_marker; exit 3; fi
for n in runz serverz attn_bench xw_peak gpu_bench ref_forward; do
  if pgrep -x "$n" >/dev/null 2>&1; then echo "gpu_guard: another GPU job runs ($n): no run." >&2; exit 3; fi
done
if [ -f "$LOCK" ]; then
  lp=$(cat "$LOCK" 2>/dev/null)
  if [ -n "$lp" ] && kill -0 "$lp" 2>/dev/null; then echo "gpu_guard: another gpu_guard runs (process $lp): no run." >&2; exit 3; fi
  rm -f "$LOCK"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

[ -n "$LOG" ] || LOG="${TMPDIR:-/tmp}/gpu_guard_$(date +%Y%m%d_%H%M%S).log"
if [ -z "${ARC_GPU_WAIT_S:-}" ]; then
  if [ -n "$WAIT_S" ]; then export ARC_GPU_WAIT_S=$WAIT_S
  else w=$((TIMEOUT - 10)); [ $w -lt 5 ] && w=5; export ARC_GPU_WAIT_S=$w; fi
fi
echo "gpu_guard: $*  (timeout $TIMEOUT s, ARC_GPU_WAIT_S $ARC_GPU_WAIT_S, log $LOG)"
inhibit=()
command -v systemd-inhibit >/dev/null 2>&1 && inhibit=(systemd-inhibit --what=idle:sleep --who=gpu_guard --why="a GPU test run")
# (a process group of its own, so the kill reaches the whole tree)
setsid "${inhibit[@]}" "$@" > >(tee "$LOG") 2> >(tee "$LOG.err" >&2) &
pid=$!
trap 'kill -TERM -- -$pid 2>/dev/null' INT TERM
t=0
alive() { kill -0 "$pid" 2>/dev/null || kill -0 -- -"$pid" 2>/dev/null; }  # (either: before setsid has run, no group yet)
# (the group, not $pid: with systemd-inhibit, $pid is it and the GPU program its child, which can outlive it stuck
#  in the driver)
while alive; do
  sleep 0.5
  t=$((t + 1))
  if [ $t -gt $((TIMEOUT * 2)) ]; then
    echo "gpu_guard: $1 still running after $TIMEOUT s: killing it" >&2
    kill -KILL -- -"$pid" 2>/dev/null
    for _ in $(seq 20); do alive || break; sleep 0.5; done
    if alive; then
      tmp="$MARKER.tmp"
      printf 'time=%s\nprogram=%s\nkernel=(unknown: killed by gpu_guard)\nkind=hang\ngpu=\nboot=%s\nkilled=no\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(basename "$1")" "$BOOT" > "$tmp" && mv -f "$tmp" "$MARKER"
      echo "gpu_guard: $1 did not end 10 s after the kill (stuck in the driver): marker written." >&2
      show_marker
      exit 3
    fi
    echo "gpu_guard: killed." >&2
    exit 124
  fi
done
wait "$pid"; rc=$?
if { [ $rc -eq 3 ] || [ $rc -eq 4 ]; } && marker_this_boot; then show_marker; fi
echo "gpu_guard: exit code $rc"
exit $rc
