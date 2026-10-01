#!/bin/bash
# Orchestrates the whole night: wait for any running benchmark, then official batch (4 par), then agent batch (2 par).
# Launch detached:  nohup ~/work/smoke-agent/firebench/overnight.sh > /dev/null 2>&1 &
TAG=${1:-$(date +%F)}
cd "$(dirname "$0")/.."
mkdir -p firebench/runs/$TAG; LOG="firebench/runs/$TAG/overnight.log"
echo "=== $(date) overnight start (tag $TAG)" >> "$LOG"
while pgrep -f "python firebench/run.py" >/dev/null; do sleep 60; done
echo "$(date +%T) previous benchmark finished" >> "$LOG"
pkill -f "streamlit run app/streamlit_app.py" 2>/dev/null && echo "$(date +%T) stopped streamlit to free RAM" >> "$LOG"
./firebench/night.sh official 4 "$TAG"
echo "$(date +%T) official batch done" >> "$LOG"
./firebench/night.sh agent 2 "$TAG"
echo "$(date +%T) agent batch done" >> "$LOG"
echo "=== $(date) overnight end" >> "$LOG"
