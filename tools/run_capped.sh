#!/bin/sh
# Run a command under a hard memory cap, to check the RAM claims for real.
#
#   tools/run_capped.sh 2G python3 -m cv3cpu bench model.safetensors
#
# Uses cgroup v1 (memory.limit_in_bytes) or cgroup v2 (memory.max), whichever
# the machine exposes, and counts page cache against the limit -- so the
# memory-mapped weights have to fit too.
set -e
LIMIT="$1"; shift
case "$LIMIT" in
  *G) BYTES=$(( ${LIMIT%G} * 1024 * 1024 * 1024 ));;
  *M) BYTES=$(( ${LIMIT%M} * 1024 * 1024 ));;
  *)  BYTES="$LIMIT";;
esac
NAME="cv3cap$$"
if [ -f /sys/fs/cgroup/cgroup.controllers ]; then
  DIR=/sys/fs/cgroup/$NAME
  mkdir -p "$DIR"
  echo "$BYTES" > "$DIR/memory.max"
  echo 0 > "$DIR/memory.swap.max" 2>/dev/null || true
else
  DIR=/sys/fs/cgroup/memory/$NAME
  mkdir -p "$DIR"
  echo "$BYTES" > "$DIR/memory.limit_in_bytes"
fi
cleanup() { rmdir "$DIR" 2>/dev/null || true; }
trap cleanup EXIT
echo $$ > "$DIR/cgroup.procs" 2>/dev/null || echo $$ > "$DIR/tasks"
echo "running under a ${LIMIT} cap:" "$@" >&2
exec "$@"
