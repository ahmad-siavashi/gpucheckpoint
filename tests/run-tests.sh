#!/bin/bash
# Test suite for gpucheckpoint.
#
# Modes:
#   make test                              functional tests (needs NVIDIA driver)
#   OFFICIAL=/path/to/official/cuda-checkpoint make test
#                                          additionally compares the
#                                          happy-path lifecycle output
#                                          (states, tids, exit codes) against
#                                          the official NVIDIA binary
#
# Auto-skips (exit 0) when no NVIDIA driver or CUDA header is available.

set -u
cd "$(dirname "$0")/.."

MINE=./gpucheckpoint
OFFICIAL=${OFFICIAL:-}
PASS=0
FAIL=0

ok()   { PASS=$((PASS + 1)); echo "PASS  $1"; }
bad()  { FAIL=$((FAIL + 1)); echo "FAIL  $1"; }

if [ ! -x "$MINE" ]; then
    echo "error: $MINE not built, run 'make' first" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "no working NVIDIA driver visible, skipping all tests"
    exit 0
fi

CUDA_INC=""
for d in /usr/local/cuda-*/include /usr/include; do
    if [ -f "$d/cuda.h" ]; then
        CUDA_INC=$d
        break
    fi
done
if [ -z "$CUDA_INC" ]; then
    echo "no cuda.h found, skipping GPU tests"
    exit 0
fi
CUDA_LIB=${CUDA_INC%/include}/lib64
CC=${CC:-$(command -v gcc || command -v cc)}

BUILD=$(mktemp -d)
trap 'pkill -x cc-counter 2>/dev/null; rm -rf "$BUILD"' EXIT

$CC -O2 -Wall -o "$BUILD/counter" tests/counter.c \
    -I"$CUDA_INC" -lcuda -L"$CUDA_LIB" -Wl,-rpath,"$CUDA_LIB"
chmod a+x "$BUILD/counter"

# Run under a unique name so pkill never hits unrelated processes.
cp "$BUILD/counter" "$BUILD/cc-counter"
setsid nohup "$BUILD/cc-counter" > "$BUILD/counter.log" 2>&1 < /dev/null &
P=""
for _ in $(seq 1 50); do
    P=$(pgrep -x cc-counter | head -1)
    [ -n "$P" ] && grep -q "gpu value" "$BUILD/counter.log" 2>/dev/null && break
    sleep 0.1
done
if [ -z "$P" ] || ! grep -q "gpu value" "$BUILD/counter.log" 2>&1; then
    echo "could not start test counter, skipping GPU tests"
    pkill -x cc-counter 2>/dev/null
    exit 0
fi
echo "test counter pid: $P"

# ---------------------------------------------------------------- lifecycle
# Byte-identical probe: same operations in each tool's own syntax. Only
# queries and silent actions, so both tools print the same 20 lines.
probe_run() {
    local style=$1 bin=$2 p=$3
    case $style in
        new)
            "$bin" state -p "$p";          echo "[$?]"
            "$bin" restore-tid -p "$p";    echo "[$?]"
            "$bin" lock -p "$p";           echo "[$?]"
            "$bin" state -p "$p";          echo "[$?]"
            "$bin" checkpoint -p "$p";     echo "[$?]"
            "$bin" state -p "$p";          echo "[$?]"
            "$bin" restore -p "$p";        echo "[$?]"
            "$bin" state -p "$p";          echo "[$?]"
            "$bin" unlock -p "$p";         echo "[$?]"
            "$bin" state -p "$p";          echo "[$?]"
            ;;
        old)
            "$bin" --get-state -p "$p";          echo "[$?]"
            "$bin" --get-restore-tid -p "$p";    echo "[$?]"
            "$bin" --action lock -p "$p";        echo "[$?]"
            "$bin" --get-state -p "$p";          echo "[$?]"
            "$bin" --action checkpoint -p "$p";  echo "[$?]"
            "$bin" --get-state -p "$p";          echo "[$?]"
            "$bin" --action restore -p "$p";     echo "[$?]"
            "$bin" --get-state -p "$p";          echo "[$?]"
            "$bin" --action unlock -p "$p";      echo "[$?]"
            "$bin" --get-state -p "$p";          echo "[$?]"
            ;;
    esac
}

# Full sequence in this tool's syntax, including the new composite
# commands and the wrong-state error paths.
seq_run() {
    local p=$1
    "$MINE" state -p "$p";            echo "[$?]"
    "$MINE" restore-tid -p "$p";      echo "[$?]"
    "$MINE" lock -p "$p";             echo "[$?]"
    "$MINE" state -p "$p";            echo "[$?]"
    "$MINE" checkpoint -p "$p";       echo "[$?]"
    "$MINE" state -p "$p";            echo "[$?]"
    "$MINE" restore -p "$p";          echo "[$?]"
    "$MINE" state -p "$p";            echo "[$?]"
    "$MINE" unlock -p "$p";           echo "[$?]"
    "$MINE" state -p "$p";            echo "[$?]"
    "$MINE" save -p "$p";             echo "[$?]"
    "$MINE" state -p "$p";            echo "[$?]"
    "$MINE" resume -p "$p";           echo "[$?]"
    "$MINE" state -p "$p";            echo "[$?]"
    "$MINE" toggle -p "$p";           echo "[$?]"
    "$MINE" state -p "$p";            echo "[$?]"
    "$MINE" toggle -p "$p";           echo "[$?]"
    "$MINE" state -p "$p";            echo "[$?]"
    "$MINE" checkpoint -p "$p";       echo "[$?]"
    "$MINE" unlock -p "$p";           echo "[$?]"
    "$MINE" restore -p "$p";          echo "[$?]"
    "$MINE" state -p 1;               echo "[$?]"
    "$MINE" lock -p 1 -t 50;          echo "[$?]"
}

if [ -n "$OFFICIAL" ]; then
    probe_run new "$MINE" "$P" > "$BUILD/mine.probe" 2>&1
    probe_run old "$OFFICIAL" "$P" > "$BUILD/real.probe" 2>&1
    if diff -q "$BUILD/real.probe" "$BUILD/mine.probe" >/dev/null; then
        ok "happy path matches official (byte-identical)"
    else
        bad "happy path differs from official"
        diff "$BUILD/real.probe" "$BUILD/mine.probe" | head -20
    fi
fi

seq_run "$P" > "$BUILD/mine.lifecycle" 2>&1
EXPECT_SEQ="running locked checkpointed locked running checkpointed checkpointed running running checkpointed checkpointed running running"
SEQ=$(grep -E '^(running|locked|checkpointed|failed)$' "$BUILD/mine.lifecycle" | paste -sd ' ')
TIDS=$(grep -cE '^[0-9]+$' "$BUILD/mine.lifecycle")
R0=$(grep -c '^\[0\]$' "$BUILD/mine.lifecycle")
R1=$(grep -c '^\[1\]$' "$BUILD/mine.lifecycle")
R2=$(grep -c '^\[2\]$' "$BUILD/mine.lifecycle")
if [ "$SEQ" == "$EXPECT_SEQ" ] && [ "$TIDS" -eq 1 ] &&
   [ "$R0" -eq 18 ] && [ "$R1" -eq 2 ] && [ "$R2" -eq 3 ] &&
   grep -q "run 'lock' first" "$BUILD/mine.lifecycle" &&
   grep -q "is already running" "$BUILD/mine.lifecycle" &&
   grep -q "run 'save' first" "$BUILD/mine.lifecycle" &&
   grep -q "failed to get state of process 1" "$BUILD/mine.lifecycle" &&
   grep -q "failed to lock process 1" "$BUILD/mine.lifecycle"; then
    ok "lifecycle state machine"
else
    bad "lifecycle state machine"
    echo "  states: $SEQ"
    echo "  tids=$TIDS rc0=$R0 rc1=$R1 rc2=$R2"
    cat "$BUILD/mine.lifecycle"
fi

# ------------------------------------------------------------- error paths
# Message wording and exit codes: 2 = usage error, 1 = runtime error.
err_case() {
    local name=$1 want_rc=$2; shift 2
    local out rc a

    out=$("$MINE" "$@" 2>&1)
    rc=$?
    a=$out

    if [ "$rc" -ne "$want_rc" ]; then
        bad "err path: $name (exit $rc, want $want_rc)"
        return
    fi
    case $name in
        bad\ pid*)
            [[ "$a" == *"--pid must be a non-negative integer"* ]] && ok "err path: $name" || bad "err path: $name" ;;
        unknown\ cmd*)
            [[ "$a" == *"unknown command"* ]] && ok "err path: $name" || bad "err path: $name" ;;
        bogus\ opt*)
            [[ "$a" == *"unrecognized option"* ]] && ok "err path: $name" || bad "err path: $name" ;;
        no\ cmd*)
            [[ "$a" == *"Usage:"* ]] && ok "err path: $name" || bad "err path: $name" ;;
        missing\ pid*)
            [[ "$a" == *"missing --pid"* ]] && ok "err path: $name" || bad "err path: $name" ;;
        bad\ devmap*)
            [[ "$a" == *"invalid device map"* ]] && ok "err path: $name" || bad "err path: $name" ;;
        bad\ index*)
            [[ "$a" == *"invalid device map"* ]] && ok "err path: $name" || bad "err path: $name" ;;
        bad\ combo*)
            [[ "$a" == *"--timeout is only supported by the lock command"* ]] && ok "err path: $name" || bad "err path: $name" ;;
        extra\ arg*)
            [[ "$a" == *"unexpected argument 'extra'"* ]] && ok "err path: $name" || bad "err path: $name" ;;
        wrong\ state*)
            [[ "$a" == *"run 'lock' first"* ]] && ok "err path: $name" || bad "err path: $name" ;;
        non-cuda*)
            [[ "$a" == *"failed to get state of process 1"* ]] && ok "err path: $name" || bad "err path: $name" ;;
    esac
}

err_case "bad pid"      2 state -p abc
err_case "unknown cmd"  2 frob
err_case "bogus opt"    2 state --bogus
err_case "no cmd"       2
err_case "missing pid"  2 state
err_case "bad devmap"   2 restore -p 1 -d foo=bar
err_case "bad index"    2 restore -p 1 -d 99=1
err_case "bad combo"    2 checkpoint -p 1 -t 5
err_case "extra arg"    2 state -p 1 extra
err_case "wrong state"  2 checkpoint -p "$P"
err_case "non-cuda"     1 state -p 1

# ------------------------------------------------------------------ features
if "$MINE" --version 2>&1 | grep -q "gpucheckpoint 1.1.0"; then
    ok "--version"
else
    bad "--version"
fi

if "$MINE" gpus 2>/dev/null | grep -qE '^0 +GPU-[0-9a-f]{32}$'; then
    ok "gpus lists index 0 with a 32-hex UUID"
else
    bad "gpus lists index 0 with a 32-hex UUID"
fi

if "$MINE" scan 2>/dev/null | grep -E "^\s*$P\s+running$" >/dev/null; then
    ok "scan finds the counter in state running"
else
    bad "scan finds the counter in state running"
fi

if "$MINE" scan --json 2>/dev/null | grep -q "\"pid\": $P, \"state\": \"running\""; then
    ok "scan --json includes the counter"
else
    bad "scan --json includes the counter"
fi

if [[ "$("$MINE" state -p "$P" --json 2>/dev/null)" == "{\"pid\": $P, \"state\": \"running\"}" ]]; then
    ok "state --json"
else
    bad "state --json"
fi

tid_re='^\{"pid": '"$P"', "tid": [0-9]+\}$'
if [[ "$("$MINE" restore-tid -p "$P" --json 2>/dev/null)" =~ $tid_re ]]; then
    ok "restore-tid --json"
else
    bad "restore-tid --json"
fi

if "$MINE" state -p "$P" --trace 2>&1 | grep -q "^\[trace\] cuCheckpointProcessGetState"; then
    ok "--trace prints driver calls"
else
    bad "--trace prints driver calls"
fi

# Index and UUID device maps must parse and reach the driver. The driver
# rejects a same-GPU remap, which proves the pair made it through.
check_remap() {
    local name=$1 map=$2 out rc

    "$MINE" save -p "$P" >/dev/null 2>&1
    out=$("$MINE" restore -p "$P" -d "$map" 2>&1)
    rc=$?
    "$MINE" restore -p "$P" >/dev/null 2>&1
    "$MINE" unlock -p "$P" >/dev/null 2>&1
    if [ "$rc" -eq 1 ] && [[ "$out" == *"failed to restore"* ]]; then
        ok "device map $name reaches the driver"
    else
        bad "device map $name reaches the driver (rc=$rc: $out)"
    fi
}
check_remap "by index (0=0)" "0=0"
U=$("$MINE" gpus | awk 'NR==2 {print $2}')
check_remap "by uuid ($U)" "$U=$U"
H=${U#GPU-}
UD="GPU-$(echo "$H" | sed 's/\(.\{8\}\)\(.\{4\}\)\(.\{4\}\)\(.\{4\}\)\(.\{12\}\)/\1-\2-\3-\4-\5/')"
check_remap "by dashed uuid" "$UD=$UD"

# ------------------------------------------------------------- final state
"$MINE" state -p "$P" | grep -qx running && ok "state back to running" || bad "state back to running"

# ------------------------------------------------- in-process library
if $CC -fPIC -shared -o "$BUILD/libgpucheckpoint.so" gpucheckpoint.c -ldl \
     2>"$BUILD/lib-build.err" &&
    $CC -O2 -o "$BUILD/lib-test" tests/lib-test.c -I. -L"$BUILD" \
        -lgpucheckpoint -Wl,-rpath,"$BUILD" 2>>"$BUILD/lib-build.err"; then
    if "$BUILD/lib-test" "$P" "$U" > "$BUILD/lib-test.out" 2>&1; then
        ok "in-process library (full state machine)"
    else
        bad "in-process library"
        cat "$BUILD/lib-test.out"
    fi
else
    bad "in-process library build"
    cat "$BUILD/lib-build.err"
fi

# ------------------------------------------------------------ continuity
# One full save/resume cycle: GPU work must freeze while checkpointed and
# the counter value must continue where it left off.
A=$(wc -l < "$BUILD/counter.log")
"$MINE" save -p "$P"
sleep 3
B=$(wc -l < "$BUILD/counter.log")
v_before=$(tail -1 "$BUILD/counter.log" | awk '{print $3}')
"$MINE" resume -p "$P"
sleep 3
C=$(wc -l < "$BUILD/counter.log")
v_after=$(tail -1 "$BUILD/counter.log" | awk '{print $3}')

if [ "$B" -eq "$A" ]; then
    ok "GPU work frozen during save"
else
    bad "GPU work frozen during save ($A -> $B lines)"
fi
if [ "$C" -gt "$B" ] && [ "$v_after" -gt "$v_before" ] && [ "$v_after" -le $((v_before + 10)) ]; then
    ok "GPU value continuous across resume ($v_before -> $v_after)"
else
    bad "GPU value continuous across resume (before=$v_before after=$v_after lines $A/$B/$C)"
fi

# Same, through the four low-level commands.
A=$(wc -l < "$BUILD/counter.log")
"$MINE" lock -p "$P" && "$MINE" checkpoint -p "$P"
sleep 3
B=$(wc -l < "$BUILD/counter.log")
v_before=$(tail -1 "$BUILD/counter.log" | awk '{print $3}')
"$MINE" restore -p "$P" && "$MINE" unlock -p "$P"
sleep 3
C=$(wc -l < "$BUILD/counter.log")
v_after=$(tail -1 "$BUILD/counter.log" | awk '{print $3}')

if [ "$B" -eq "$A" ]; then
    ok "GPU work frozen during lock+checkpoint"
else
    bad "GPU work frozen during lock+checkpoint ($A -> $B lines)"
fi
if [ "$C" -gt "$B" ] && [ "$v_after" -gt "$v_before" ] && [ "$v_after" -le $((v_before + 10)) ]; then
    ok "GPU value continuous across lock+checkpoint cycle ($v_before -> $v_after)"
else
    bad "GPU value continuous across lock+checkpoint cycle (before=$v_before after=$v_after lines $A/$B/$C)"
fi

echo
echo "results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
