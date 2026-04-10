#!/bin/bash
# Start all AgentX services inside a single container for AgentBeats submission.
# Usage: bash start_participant.sh --host 0.0.0.0 --port 8000

HOST="0.0.0.0"
PORT="8000"

while [[ $# -gt 0 ]]; do
  case $1 in
    --host) HOST="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    *) shift;;
  esac
done

BASE="$(cd "$(dirname "$0")" && pwd)"

# Start all sub-agents in background
declare -A agents=(
    ["src/agents/critic"]=8001
    ["src/agents/error"]=8002
    ["src/agents/eda"]=8003
    ["src/agents/model"]=8005
    ["src/agents/ensemble"]=8006
    ["src/agents/hypertune"]=8007
    ["src/agents/feature"]=8008
    ["src/agents/threshold"]=8009
    ["src/agents/planner"]=8010
    ["src/agents/codegen"]=8011
    ["src/agents/stacking"]=8012
)

echo "Starting sub-agents..."
for svc in "${!agents[@]}"; do
    p=${agents[$svc]}
    cd "$BASE/$svc" && python server.py --port "$p" &
    echo "  $svc -> port $p (PID $!)"
done

# Wait for ALL sub-agents to be ready (poll each port)
echo "Waiting for all sub-agents to be ready..."
MAX_WAIT=60
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    ALL_READY=true
    for p in 8001 8002 8003 8005 8006 8007 8008 8009 8010 8011 8012; do
        if ! python -c "import socket; s=socket.socket(); s.settimeout(0.5); s.connect(('127.0.0.1',$p)); s.close()" 2>/dev/null; then
            ALL_READY=false
            break
        fi
    done
    if $ALL_READY; then
        echo "All sub-agents ready after ${WAITED}s"
        break
    fi
    sleep 1
    WAITED=$((WAITED + 1))
done

if ! $ALL_READY; then
    echo "WARNING: Not all sub-agents ready after ${MAX_WAIT}s, starting orchestrator anyway"
fi

# Start orchestrator in foreground (this is the A2A endpoint)
echo "Starting orchestrator on $HOST:$PORT..."
cd "$BASE/src/orchestrator" && exec python server.py --host "$HOST" --port "$PORT"
