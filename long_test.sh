#!/usr/bin/env bash
set -uo pipefail

# Usage:
#   ./run.sh                 # run pytest 10 times
#   ./run.sh 25              # run pytest 25 times
#   ./run.sh 5 -k test_name  # run pytest 5 times with extra args

runs=10
failures=()
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
    runs="$1"
    shift
fi

for ((i = 1; i <= runs; i++)); do
    echo "[run ${i}/${runs}] pytest $*"
    if ! pytest "$@"; then
        failures+=("$i")
    fi
done

if [[ ${#failures[@]} -eq 0 ]]; then
    echo "Completed ${runs} pytest run(s): all passed."
    exit 0
fi

echo "Completed ${runs} pytest run(s): ${#failures[@]} failed (runs: ${failures[*]})."
exit 1
