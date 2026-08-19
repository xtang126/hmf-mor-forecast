#!/bin/bash -l
#SBATCH --partition=general
#SBATCH -J hmf_gamma_scan
#SBATCH --array=0-1
#SBATCH --ntasks=21
#SBATCH --cpus-per-task=1
#SBATCH -t 08:00:00
#SBATCH --mail-user=xt52@sussex.ac.uk
#SBATCH --mail-type=ALL
#SBATCH -o /its/home/xt52/hmf-mor-forecast/log/hmf_gamma_scan_%j.log
#SBATCH -e /its/home/xt52/hmf-mor-forecast/log/hmf_gamma_scan_%j.error
# ============================================================
#
# Submit with:
#   sbatch run_mcmc_scan_gamma.sh
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
 
# gamma_scatter_model (gm) values to scan, 0.0 -> 0.3
# TRUE scatter (gamma_scatter_data) is fixed at 0.30 in the base ini below;
# gm=0.00 is the fully mis-specified fit (model assumes constant scatter),
# gm=0.30 is the correctly-specified fit. Everything in between traces out
# how the sigma8-Omega_m contour moves as the model's assumed O-dependence
# approaches the truth.
GM_VALUES=(0.00 0.30)
TAGS=(gm00 gm30)
gm=${GM_VALUES[$SLURM_ARRAY_TASK_ID]}
tag=${TAGS[$SLURM_ARRAY_TASK_ID]}
 
if [[ -z "$gm" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range" >&2
    exit 1
fi
 
echo "Task ${SLURM_ARRAY_TASK_ID}: gamma_scatter_model=${gm}  tag=${tag}"
 
# Run the pipeline
start=$(date +%s)
mpirun -np $SLURM_NTASKS cosmosis --mpi configs/forecast_M2e14_A4000_sd20_g30_fixed.ini \
    -p mass_function_like.gamma_scatter_model="${gm}" \
       output.filename=output/scatter_gt30_gm_scan_full_${tag}.txt
end=$(date +%s)
runtime=$((end - start))
echo "=============================================="
echo "CosmoSIS run time: $runtime seconds"
echo "Job finished: $(date)"
echo "=============================================="
 
