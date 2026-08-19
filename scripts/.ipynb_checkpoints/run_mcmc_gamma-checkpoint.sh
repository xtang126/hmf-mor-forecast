#!/bin/bash -l
#SBATCH --partition=general
#SBATCH -J hmf_gamma_gd30_free_frac
#SBATCH --ntasks=21
#SBATCH --cpus-per-task=1
#SBATCH -t 08:00:00
#SBATCH --mail-user=xt52@sussex.ac.uk
#SBATCH --mail-type=ALL
#SBATCH -o /its/home/xt52/hmf-mor-forecast/log/hmf_gamma_gd30_free_frac_%j.log
#SBATCH -e /its/home/xt52/hmf-mor-forecast/log/hmf_gamma_gd30_free_frac_%j.error

# ============================================================
#
# Submit with:
#   sbatch run_mcmc.sh
#
# Monitor with:
#   squeue -u $USER
#
# ============================================================

export TOP_DIR=/its/home/aa3044/DES_cluster2
export COSMOSIS_REPO_DIR=${TOP_DIR}/cosmosis
export CSL_DIR=${TOP_DIR}/cosmosis-standard-library

export OMP_NUM_THREADS=1

# MPI fixes for Artemis InfiniBand issues
export UCX_TLS=tcp
export OMPI_MCA_btl=tcp,self
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl_openib_warn_no_device_params_found=0

source ~/setup_cosmosis.sh

module purge
module load gompi/2022b
module load mpi4py/3.1.4-gompi-2022b

# ============================================================
# Print job info
# ============================================================
echo "=============================================="
echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Working directory: $(pwd)"
echo "=============================================="

cd /its/home/xt52/hmf-mor-forecast

# Run the pipeline
start=$(date +%s)

#cosmosis configs/forecast_M2e14_A4000_sd20_g00_fixed.ini
mpirun -np $SLURM_NTASKS cosmosis --mpi configs/forecast_M2e14_A4000_sd20_gt30_free_frac.ini

end=$(date +%s)
runtime=$((end - start))

echo "=============================================="
echo "CosmoSIS run time: $runtime seconds"
echo "Job finished: $(date)"
echo "==============================================" 