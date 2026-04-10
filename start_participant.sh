#!/bin/bash
# Start AgentX v8 services inside a single container for AgentBeats submission.
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

# Start solver in background
echo "Starting solver on port 8001..."
cd "$BASE/src/solver" && python server.py --port 8001 &
SOLVER_PID=$!
echo "  solver -> port 8001 (PID $SOLVER_PID)"

# Wait for solver to be ready
echo "Waiting for solver to be ready..."
MAX_WAIT=60
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if python -c "import socket; s=socket.socket(); s.settimeout(0.5); s.connect(('127.0.0.1',8001)); s.close()" 2>/dev/null; then
        echo "Solver ready after ${WAITED}s"
        break
    fi
    sleep 1
    WAITED=$((WAITED + 1))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "WARNING: Solver not ready after ${MAX_WAIT}s, starting arena anyway"
fi

# Start arena in foreground (this is the A2A endpoint exposed to AgentBeats)
echo "Starting arena on $HOST:$PORT..."
cd "$BASE/src/arena" && exec python server.py --host "$HOST" --port "$PORT"
