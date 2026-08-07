#!/bin/bash -l
#SBATCH --partition=general
#SBATCH -J hmf_prior_scan
#SBATCH --array=0-8
#SBATCH --ntasks=21
#SBATCH --cpus-per-task=1
#SBATCH -t 08:00:00
#SBATCH --mail-user=xt52@sussex.ac.uk
#SBATCH --mail-type=ALL
#SBATCH -o /its/home/xt52/hmf-mor-forecast/log/hmf_prior_scan_%A_%a.log
#SBATCH -e /its/home/xt52/hmf-mor-forecast/log/hmf_prior_scan_%A_%a.error

# ============================================================
#
# Prior-scan grid: 3 centers × 3 widths = 9 chains
#   Centers: 0.10 (biased low), 0.20 (truth), 0.40 (biased high)
#   Widths:  0.01 (tight), 0.05 (realistic), 0.20 (loose)
# Mock data is generated with sd=0.20 (true scatter).
#
# Submit with:
#   sbatch run_prior_scan.sh
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
echo "=============================================="
echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Running on: $(hostname)"
echo "Working directory: $(pwd)"
echo "=============================================="

cd /its/home/xt52/hmf-mor-forecast

# ------------------------------------------------------------
# Prior-scan grid
# ------------------------------------------------------------
# 3 centers × 3 widths = 9 combinations
# Task index maps as (center_idx * 3 + width_idx)
#   idx 0-2: center 0.10, widths 0.01/0.05/0.20
#   idx 3-5: center 0.20, widths 0.01/0.05/0.20
#   idx 6-8: center 0.40, widths 0.01/0.05/0.20

CENTERS=(0.10 0.10 0.10  0.20 0.20 0.20  0.40 0.40 0.40)
WIDTHS=( 0.01 0.05 0.20  0.01 0.05 0.20  0.01 0.05 0.20)

# Filename tags (short form for readability)
CTAGS=(c10 c10 c10  c20 c20 c20  c40 c40 c40)
WTAGS=(w01 w05 w20  w01 w05 w20  w01 w05 w20)

center=${CENTERS[$SLURM_ARRAY_TASK_ID]}
width=${WIDTHS[$SLURM_ARRAY_TASK_ID]}
ctag=${CTAGS[$SLURM_ARRAY_TASK_ID]}
wtag=${WTAGS[$SLURM_ARRAY_TASK_ID]}

if [[ -z "$center" || -z "$width" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range" >&2
    exit 1
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: prior center=${center}  width=${width}"
echo "Tag: ${ctag}_${wtag}"

# ------------------------------------------------------------
# Build a per-task priors file so each chain sees its own Gaussian
# ------------------------------------------------------------
PRIORS_DIR=configs/priors_scan
mkdir -p ${PRIORS_DIR}
PRIORS_FILE=${PRIORS_DIR}/priors_${ctag}_${wtag}.ini

cat > ${PRIORS_FILE} << EOF
[cosmological_parameters]
omega_m = uniform 0.1 0.5
sigma8_input = uniform 0.6 1.0

[mor_parameters]
frac_scatter = gaussian ${center} ${width}
EOF

echo "Wrote priors file: ${PRIORS_FILE}"
cat ${PRIORS_FILE}

# ------------------------------------------------------------
# Run the pipeline
# ------------------------------------------------------------
start=$(date +%s)

mpirun -np $SLURM_NTASKS cosmosis --mpi configs/forecast_M2e14_A4000_sd20_free.ini \
    -p pipeline.priors="${PRIORS_FILE}" \
       output.filename=output/prior_scan_sd20_${ctag}_${wtag}.txt

end=$(date +%s)
runtime=$((end - start))

echo "=============================================="
echo "CosmoSIS run time: ${runtime} seconds"
echo "Job finished: $(date)"
echo "=============================================="