#!/bin/bash
# Stop all AgentX services

echo "Stopping AgentX services..."
for port in 9009 8000 8001; do
    PID=$(netstat -ano 2>/dev/null | grep ":$port.*LISTENING" | awk '{print $5}' | head -1)
    if [ -n "$PID" ]; then
        taskkill //F //PID "$PID" > /dev/null 2>&1
        echo "  Killed port $port (PID $PID)"
    fi
done
echo "All services stopped."
