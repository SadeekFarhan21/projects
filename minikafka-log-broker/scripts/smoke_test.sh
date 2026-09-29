#!/usr/bin/env bash
# End-to-end smoke test of the real binaries: start mk-broker as a separate
# process, produce and consume with mk-cli, restart the broker, and check that
# both the data and the group's committed offsets survived.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=${BIN:-build/release}
PORT=${PORT:-19093}
DATA=$(mktemp -d "${TMPDIR:-/tmp}/minikafka-smoke.XXXXXX")
B="--broker 127.0.0.1:$PORT"
CLI="$BIN/mk-cli"

start_broker() {
  "$BIN/mk-broker" --data-dir "$DATA" --port "$PORT" >>"$DATA/broker.log" 2>&1 &
  BP=$!
  for _ in $(seq 1 100); do  # wait until it accepts connections
    if ! "$CLI" $B offsets __probe 0 2>&1 | grep -q "Connection refused"; then return; fi
    sleep 0.05
  done
  echo "broker did not start"; exit 1
}
stop_broker() { kill -INT "$BP"; wait "$BP" || true; }
trap 'kill $BP 2>/dev/null || true; rm -rf "$DATA"' EXIT

start_broker
"$CLI" $B create-topic greetings 2
printf 'hello\nworld\nfoo\n' | "$CLI" $B produce greetings
echo "--- consume with group demo"
"$CLI" $B consume greetings demo 10
echo "--- same group again (expect nothing: offsets were committed)"
"$CLI" $B consume greetings demo 10

echo "--- restart broker"
stop_broker
start_broker
printf 'after-restart\n' | "$CLI" $B produce greetings k
echo "--- group demo resumes and sees only the new message"
"$CLI" $B consume greetings demo 10
echo "--- a new group sees everything"
"$CLI" $B consume greetings other 10
"$CLI" $B offsets greetings 0
"$CLI" $B offsets greetings 1
stop_broker
echo "smoke test OK"
