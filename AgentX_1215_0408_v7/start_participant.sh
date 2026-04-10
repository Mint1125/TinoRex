#!/bin/bash
# Start all AgentX services inside a single container for AgentBeats submission.
# Usage: bash start_participant.sh --host 0.0.0.0 --port 9009

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

# Wait for sub-agents to be ready
echo "Waiting for sub-agents to start..."
sleep 5

# Start orchestrator in foreground (this is the A2A endpoint)
echo "Starting orchestrator on $HOST:$PORT..."
cd "$BASE/src/orchestrator" && exec python server.py --host "$HOST" --port "$PORT"
