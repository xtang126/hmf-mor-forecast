#!/bin/bash -l
#SBATCH --partition=general
#SBATCH -J hmf_scatter_fixed_02_area
#SBATCH --array=0-2
#SBATCH --ntasks=21
#SBATCH --cpus-per-task=1
#SBATCH -t 08:00:00
#SBATCH --mail-user=xt52@sussex.ac.uk
#SBATCH --mail-type=ALL
#SBATCH -o /its/home/xt52/hmf-mor-forecast/log/hmf_scatter_fixed_02_area_%j.log
#SBATCH -e /its/home/xt52/hmf-mor-forecast/log/hmf_scatter_fixed_02_area_%j.error

# ============================================================
#
# Submit with:
#   sbatch run_mcmc_scan.sh
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

# area_deg2 values to scan
AREA=(1000.0 2000.0 8000.0)
TAGS=(A1 A2 A8)

AREA=${AREA[$SLURM_ARRAY_TASK_ID]}
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}

# Run the pipeline
start=$(date +%s)

mpirun -np $SLURM_NTASKS cosmosis --mpi configs/forecast_M2e14_A4000_sd20_fixed.ini \
    -p mass_function_like.area_deg2=${AREA} \
       output.filename=output/scatter_fixed_02_area_${TAG}.txt

end=$(date +%s)
runtime=$((end - start))

echo "=============================================="
echo "CosmoSIS run time: $runtime seconds"
echo "Job finished: $(date)"
echo "==============================================" 