#!/bin/bash -l
#SBATCH --partition=general
#SBATCH -J hmf_scatter_free
#SBATCH --array=0-2
#SBATCH --ntasks=21
#SBATCH --cpus-per-task=1
#SBATCH -t 08:00:00
#SBATCH --mail-user=xt52@sussex.ac.uk
#SBATCH --mail-type=ALL
#SBATCH -o /its/home/xt52/hmf-mor-forecast/log/hmf_scatter_fixed_02_mt_%j.log
#SBATCH -e /its/home/xt52/hmf-mor-forecast/log/hmf_scatter_fixed_02_mt_%j.error

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

# area_deg2/others values to scan
#CONFIG=(0.05 0.00 0.10 0.20 0.40)
#TAGS=(sm05 sm00 sm10 sm20 sm40)

#CONFIG=${CONFIG[$SLURM_ARRAY_TASK_ID]}
#TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}

# Mass windows to scan (each 1 dex wide); mock is generated with sd=0.20
# Mid window (1e14-1e15) is the reference — matches the default config
MMIN_VALUES=(5e13 1e14 5e14)
MMAX_VALUES=(5e14 1e15 5e15)
TAGS=(M5e13-5e14 M1e14-1e15 M5e14-5e15)

mmin=${MMIN_VALUES[$SLURM_ARRAY_TASK_ID]}
mmax=${MMAX_VALUES[$SLURM_ARRAY_TASK_ID]}
tag=${TAGS[$SLURM_ARRAY_TASK_ID]}

if [[ -z "$mmin" || -z "$mmax" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range" >&2
    exit 1
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: mass_min=${mmin}  mass_max=${mmax}  tag=${tag}"

# Run the pipeline
start=$(date +%s)

mpirun -np $SLURM_NTASKS cosmosis --mpi configs/forecast_A4000_sd20_fixed.ini \
    -p mass_function_like.mass_min="${mmin}" \
       mass_function_like.mass_max="${mmax}" \
       output.filename=output/scatter_fixed_02_mass_${tag}.txt

end=$(date +%s)
runtime=$((end - start))

echo "=============================================="
echo "CosmoSIS run time: $runtime seconds"
echo "Job finished: $(date)"
echo "==============================================" 