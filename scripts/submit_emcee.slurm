#!/bin/bash
#SBATCH --job-name=hmf_mor_forecast_emcee
#SBATCH --output=output/log_hmf_mor_forecast_emcee.out
#SBATCH --error=output/log_hmf_mor_forecast_emcee.err
#SBATCH --time=00:30:00          # quick/rough run (walkers=20, samples=500); bump for production
#SBATCH --mail-user=xt52@sussex.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=4               # Matches your `srun -n 4`
#SBATCH --cpus-per-task=1
#SBATCH --constraint=cpu
#SBATCH --qos=debug              # short test; switch to 'regular' for production
#SBATCH --account=des

# One thread per rank (avoid oversubscription)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Load your modules/environment if needed
# module load openmpi
source /dvs_ro/cfs/projectdirs/des/zuntz/cosmosis-global/setup-cosmosis3

# Go to the repo directory
cd /global/u2/x/xintang/hmf-mor-forecast

# Run the pipeline
start=$(date +%s)

srun -n 4 cosmosis --mpi configs/forecast_M2e14_A4000_sd10_sm10.ini

end=$(date +%s)
runtime=$((end - start))

echo "CosmoSIS run time: $runtime seconds" >> output/log_hmf_mor_forecast_emcee.out