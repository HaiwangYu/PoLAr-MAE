#!/bin/bash
# trainjob_polarmae.sh — worker-side launcher for PoLAr-MAE training jobs.
#
# Args (positional):
#   $1 repo_dir   -- path to PoLAr-MAE repo root
#   $2 env_prefix -- conda env prefix (e.g. /gpfs01/lbne/users/fm/hyu/uvenv-polar-mae)
#   $3 config     -- path to lightning yaml config
#   $4 out_dir    -- GPFS path to rsync outputs to
#   $5 cache_dir  -- general cache base; ${cache_dir}/data is used for APA2D index
#   $6 run_name   -- run label (passed to lightning + json probe dir)
#   $7 task       -- 'polarmae' or 'pointmae' (default polarmae)

set -euo pipefail

repo_dir=$1
env_prefix=$2
config=$3
out_dir=$4
cache_dir=$5
run_name=$6
task=${7:-polarmae}

data_cache="${cache_dir}/data"
mkdir -p "$data_cache"

echo "Running $CLUSTER_ID.$JOB_ID on $(hostname)"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "  _CONDOR_SCRATCH_DIR=${_CONDOR_SCRATCH_DIR}"
echo
echo "JOB CONFIGURATION:"
echo "  run_name=${run_name}"
echo "  repo_dir=${repo_dir}"
echo "  env_prefix=${env_prefix}"
echo "  config=${config}"
echo "  out_dir=${out_dir}"
echo "  cache_dir=${cache_dir}"
echo "  task=${task}"
echo

# Use the env's binaries directly; avoid 'conda activate' in non-interactive shells.
export PATH="${env_prefix}/bin:${PATH}"
export LD_LIBRARY_PATH="${env_prefix}/lib:${LD_LIBRARY_PATH:-}"
PY="${env_prefix}/bin/python"
test -x "$PY" || { echo "FATAL: $PY missing — run build_polarmae_env first" >&2; exit 1; }
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

scratch_root=${_CONDOR_SCRATCH_DIR}/${run_name}
scratch_ckpt=${scratch_root}/checkpoints
scratch_probes=${scratch_root}/probes
scratch_logs=${scratch_root}/lightning_logs
mkdir -p "$scratch_ckpt" "$scratch_probes" "$scratch_logs"

sync_back() {
    echo "Syncing ${scratch_root} -> ${out_dir}"
    mkdir -p "${out_dir}"
    rsync -a "${scratch_root}/" "${out_dir}/" || true
}
cleanup() {
    if [[ -n "${nvsmi_pid:-}" ]]; then
        kill "${nvsmi_pid}" 2>/dev/null || true
    fi
    sync_back
}
trap cleanup EXIT
trap 'cleanup; exit 143' SIGTERM

mkdir -p "${out_dir}"
echo "Starting nvidia-smi polling loop -> ${out_dir}/gpu.log"
nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv -l 10 > "${out_dir}/gpu.log" 2>&1 &
nvsmi_pid=$!

cd "$repo_dir"

# --trainer.callbacks.init_args.json_dir / --data.init_args.dataset_kwargs.cache_dir
# direct the per-probe JSON dumps + APA2D index cache to scratch / GPFS cache.
echo "Launching polarmae.tasks.${task} fit ..."
"$PY" -u -m polarmae.tasks.${task} fit \
    --config "$config" \
    --trainer.default_root_dir "$scratch_logs" \
    --trainer.callbacks.init_args.json_dir "$scratch_probes" \
    --data.init_args.dataset_kwargs.cache_dir "$data_cache" \
    --data.init_args.test_dataset_kwargs.cache_dir "$data_cache"

echo "Training complete!"
