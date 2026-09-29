#!/bin/sh
# timeout.sh SECONDS cmd args...  Kill cmd if it runs longer than SECONDS.
# Uses coreutils timeout/gtimeout when present, else perl's alarm (macOS).
secs=$1; shift
if command -v timeout >/dev/null 2>&1; then exec timeout "$secs" "$@"; fi
if command -v gtimeout >/dev/null 2>&1; then exec gtimeout "$secs" "$@"; fi
exec perl -e 'alarm shift; exec @ARGV or die "exec failed: $!"' "$secs" "$@"
