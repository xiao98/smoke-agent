#!/bin/bash
# After the first overnight chain ends, redo the agent cases that were fixed (resumable: only missing result.json run)
cd "$(dirname "$0")/.."
while pgrep -f "firebench/overnight.sh" >/dev/null; do sleep 120; done
echo "$(date) second agent pass start" >> firebench/runs/2026-10-01/overnight.log
./firebench/night.sh agent 2 2026-10-01
echo "$(date) second agent pass end" >> firebench/runs/2026-10-01/overnight.log
