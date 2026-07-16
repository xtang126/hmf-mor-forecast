#!/bin/bash
#SBATCH --job-name=hmf_scatter_scan
#SBATCH --array=0-3
#SBATCH --output=output/log_scatter_scan_%a.out
#SBATCH --error=output/log_scatter_scan_%a.err
#SBATCH --time=02:00:00
#SBATCH --mail-user=xt52@sussex.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --constraint=cpu
#SBATCH --qos=regular          # 2h walltime exceeds debug's 30-min cap
#SBATCH --account=des

# One thread per rank (avoid oversubscription)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source /dvs_ro/cfs/projectdirs/des/zuntz/cosmosis-global/setup-cosmosis3

cd /global/u2/x/xintang/hmf-mor-forecast

# frac_scatter_model values to scan (frac_scatter_data stays fixed at 0.10).
# 0.10 already run separately (job 55939315, configs/forecast_M2e14_A4000_sd10_sm10.ini).
FRACS=(0.0  0.05 0.2  0.4)
TAGS=(sm00 sm05 sm20 sm40)

FRAC=${FRACS[$SLURM_ARRAY_TASK_ID]}
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}

start=$(date +%s)

srun -n 4 cosmosis --mpi configs/forecast_M2e14_A4000_sd10_sm10.ini \
    -p mass_function_like.frac_scatter_model=${FRAC} \
       output.filename=output/v1_${TAG}.txt

end=$(date +%s)
runtime=$((end - start))

echo "[task ${SLURM_ARRAY_TASK_ID}, frac_scatter_model=${FRAC}] runtime: ${runtime} seconds" \
    >> output/log_scatter_scan_${SLURM_ARRAY_TASK_ID}.out
