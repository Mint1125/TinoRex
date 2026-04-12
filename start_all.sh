#!/bin/bash
# Start AgentX services (3-service arena architecture)
# Usage: bash start_all.sh [conda_env]

ENV=${1:-test}
BASE="$(cd "$(dirname "$0")" && pwd)"

# Force UTF-8 on Windows (needed by mlebench to read description files)
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

declare -A services=(
    ["src/evaluator"]=9009
    ["src/arena"]=8000
    ["src/solver"]=8001
)

echo "Starting AgentX services (env=$ENV)..."
for svc in "${!services[@]}"; do
    port=${services[$svc]}
    cd "$BASE/$svc" && conda run -n "$ENV" python server.py --port "$port" > /dev/null 2>&1 &
    echo "  $svc → port $port (PID $!)"
done

echo "Waiting for servers to start..."
sleep 5

echo "Status:"
for svc in "${!services[@]}"; do
    port=${services[$svc]}
    if netstat -ano 2>/dev/null | grep -q ":$port.*LISTENING"; then
        echo "  ✓ $svc → :$port"
    else
        echo "  ✗ $svc → :$port FAILED"
    fi
done
