#!/bin/bash
# submit_polarmae.sh — PoLAr-MAE training submission to SDCC GPU pool.
#
# Usage:
#   bash gridutils/submit_polarmae.sh <run_name> <path/to/config.yml> [pointmae|polarmae]
#
# Outputs land under ${CONDOR_OUT}/${run_name}/.  Conda env at $ENV_PREFIX is
# activated on the worker.

set -euo pipefail

# ---- User-overridable env --------------------------------------------------

CONDOR_OUT="${CONDOR_OUT:-/gpfs01/lbne/users/fm/${USER}/CONDOR_OUT}"
REPODIR="${REPODIR:-/direct/lbne+u/hyu/PoLAr-MAE}"
ENV_PREFIX="${ENV_PREFIX:-/gpfs01/lbne/users/fm/${USER}/uvenv-polar-mae}"
CACHE_DIR="${CACHE_DIR:-/gpfs01/lbne/users/fm/${USER}/cache}"

REQUEST_MEMORY="${REQUEST_MEMORY:-32000}"
REQUEST_GPUS="${REQUEST_GPUS:-1}"
REQUEST_CPUS="${REQUEST_CPUS:-4}"
GPU_REQUIREMENTS="${GPU_REQUIREMENTS:-(GPUs_DeviceName == \"NVIDIA L40S\") && (GPUs_Capability == 8.9)}"

# ---- Args ------------------------------------------------------------------
if [ $# -lt 2 ]; then
    echo "usage: $0 <run_name> <config.yml> [polarmae|pointmae]" >&2
    exit 2
fi

run_name=$1
config=$2
task=${3:-polarmae}

if [ ! -f "$config" ]; then
    echo "ERROR: config file not found: $config" >&2
    exit 1
fi
config=$(cd "$(dirname "$config")" && pwd)/$(basename "$config")

out_dir="${CONDOR_OUT}/${run_name}"
if [ -d "$out_dir" ]; then
    echo "ERROR: ${out_dir} already exists. Pick a fresh run_name or remove it." >&2
    exit 1
fi
mkdir -p "$out_dir"
echo "Created run directory: ${out_dir}"

subfile="${out_dir}/${run_name}.sub"
cat > "$subfile" <<EOF
universe                = vanilla
notification            = never
executable              = ${REPODIR}/gridutils/trainjob_polarmae.sh
arguments               = ${REPODIR} ${ENV_PREFIX} ${config} ${out_dir} ${CACHE_DIR} ${run_name} ${task}
environment             = "CLUSTER_ID=\$(ClusterId) JOB_ID=\$(ProcId)"
output                  = ${out_dir}/\$(ClusterId).\$(ProcId).out
error                   = ${out_dir}/\$(ClusterId).\$(ProcId).err
log                     = ${out_dir}/\$(ClusterId).\$(ProcId).log
getenv                  = True
request_memory          = ${REQUEST_MEMORY}
request_cpus            = ${REQUEST_CPUS}
request_gpus            = ${REQUEST_GPUS}
Requirements            = ${GPU_REQUIREMENTS}
should_transfer_files   = NO
stream_output           = True
stream_error            = True
queue 1
EOF

echo "Submitting ${subfile}"
condor_submit "$subfile"
