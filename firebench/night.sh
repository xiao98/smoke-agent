#!/bin/bash
# Overnight FireBench batch on the laptop.  Resumable: cases with result.json are skipped.
#   ./firebench/night.sh official 4      # NIST inputs, 4 concurrent (FDS only, light on RAM)
#   ./firebench/night.sh agent 2         # SmokeAgent runs, 2 concurrent (each ~1.7 GB RAM for retrieval)
# Case list: firebench/basic.txt (one id per line).  Log: firebench/runs/<date>/night_<mode>.log
MODE=${1:-official}; PAR=${2:-2}; TAG=${3:-$(date +%F)}
source ~/.smoke-agent.env; source "$HOME/FDS/FDS6/bin/FDS6VARS.sh"; source "$HOME/FDS/FDS6/bin/SMV6VARS.sh"
cd "$(dirname "$0")/.." && source .venv/bin/activate
export PYTHONPATH=src OMP_NUM_THREADS=2 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
mkdir -p "firebench/runs/$TAG"; LOG="firebench/runs/$TAG/night_$MODE.log"
echo "=== $(date) start mode=$MODE par=$PAR" >> "$LOG"
run_one() {
  id="$1"
  if [ -f "firebench/runs/$TAG/$MODE/$id/result.json" ]; then echo "$id: done, skip" >> "$LOG"; return; fi
  echo "$(date +%T) $id: start" >> "$LOG"
  python -u firebench/run.py --mode "$MODE" --only "$id" --omp 2 --max_loop 4 --timeout 36000 --run_tag "$TAG" \
     > "firebench/runs/$TAG/${MODE}_$id.log" 2>&1
  echo "$(date +%T) $id: exit $? $(grep -o 'passed=[A-Za-z]*' "firebench/runs/$TAG/${MODE}_$id.log" | tail -1)" >> "$LOG"
}
export -f run_one; export MODE TAG LOG
grep -vE '^\s*(#|$)' firebench/basic.txt | xargs -P "$PAR" -I{} bash -c 'run_one {}'
python firebench/report.py --run_tag "$TAG" >> "$LOG" 2>&1
echo "=== $(date) end" >> "$LOG"
