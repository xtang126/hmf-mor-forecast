#!/bin/bash
# Run the CosmoSIS test sampler and save the console output alongside the
# datablock dump in output/test/hmf_counts_v1/.

# NOTE: setup-cosmosis3 relies on commands like `grep -q` legitimately
# returning non-zero as part of its own control flow, so it must be sourced
# before we turn on -e (a "no output, just fails" symptom means -e killed
# the script inside the sourced setup script).
source /dvs_ro/cfs/projectdirs/des/zuntz/cosmosis-global/setup-cosmosis3

set -eo pipefail

cd "$(dirname "$0")/.."

LOG=output/test/hmf_counts_v1/run.log
mkdir -p "$(dirname "$LOG")"

cosmosis configs/forecast_M2e14_A4000_sd10_sm10.ini 2>&1 | tee "$LOG"
