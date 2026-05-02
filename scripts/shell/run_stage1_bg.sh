#!/usr/bin/env bash
# Detached launcher for the stage-1 parallel sweep. Outputs go to logs/sm_sweep/.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/sm_sweep
rm -f .sm_sweep_jobs.txt logs/sm_sweep/poll.log logs/sm_sweep/run.log
nohup make sm-sweep-parallel SM_SWEEP_DIR=configs/sweep_stage1 MAX_PARALLEL=4 \
  >logs/sm_sweep/run.log 2>&1 </dev/null &
disown
echo "PID=$!"
echo "tail -f tcc_ai/logs/sm_sweep/run.log    # submission progress"
echo "tail -f tcc_ai/logs/sm_sweep/poll.log   # status table per cycle"
