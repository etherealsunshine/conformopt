#!/usr/bin/env bash
# Wait for the existing qfit-unet workspace to become Ready.
# If the workspace has actually disappeared, resume it with the saved/PVC-backed
# configuration. A merely Pending pod is never replaced.

set -u

NAMESPACE="${ACTL_NAMESPACE:-diffuse}"
PROJECT="${ACTL_PROJECT:-qfit-unet}"
PROFILE="${ACTL_PROFILE:-single}"
INTERVAL="${ACTL_INTERVAL_SECONDS:-15}"
LOG="${ACTL_WATCH_LOG:-${TMPDIR:-/tmp}/qfit-unet-pod-up.log}"

printf 'Watching %s/%s; profile=%s; interval=%ss\n' \
  "$NAMESPACE" "$PROJECT" "$PROFILE" "$INTERVAL"
printf 'Watcher log: %s\n' "$LOG"

while true; do
  status="$(actl pod status -n "$NAMESPACE" "$PROJECT" 2>&1 || true)"

  if printf '%s\n' "$status" | grep -q 'phase:.*Running' &&
     printf '%s\n' "$status" | grep -q 'ready:.*true'; then
    printf '%s\n' "$status"
    echo 'qfit-unet is Ready.'
    exit 0
  fi

  if printf '%s\n' "$status" | grep -q 'phase:.*Pending'; then
    now="$(date '+%Y-%m-%d %H:%M:%S')"
    reason="$(printf '%s\n' "$status" | sed -n 's/^  scheduling: //p' | head -n 1)"
    printf '[%s] still Pending: %s\n' "$now" "${reason:-scheduler has not assigned a node}" | tee -a "$LOG"
    sleep "$INTERVAL"
    continue
  fi

  if printf '%s\n' "$status" | grep -q 'replicas=0\|not found\|no such workspace\|Unable to find'; then
    now="$(date '+%Y-%m-%d %H:%M:%S')"
    printf '[%s] workspace is absent/scaled to zero; resuming the existing PVC-backed workspace\n' "$now" | tee -a "$LOG"
    nohup actl pod up -n "$NAMESPACE" "$PROJECT" \
      --no-tty --profile "$PROFILE" --yes \
      >"$LOG" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "${LOG}.pid"
    echo "Started pod-up controller PID $(cat "${LOG}.pid"); it will keep syncing until the pod is Ready."
    exit 0
  fi

  now="$(date '+%Y-%m-%d %H:%M:%S')"
  printf '[%s] status not classifiable; leaving workspace untouched and retrying\n' "$now" | tee -a "$LOG"
  sleep "$INTERVAL"
done
