#!/bin/bash
# build_polarmae_env.sh — build the PoLAr-MAE env on a GPU worker.
#
# Strategy:
#   1. micromamba creates a minimal env with python + cuda toolkit + gcc/gxx
#      + cudnn. (Avoids the channel-resolution issue where conda picks the
#      CPU build of pytorch on the polarmae env.yml.)
#   2. pip installs the CUDA-enabled pytorch 2.5.1 from the pytorch index.
#   3. pip installs the remaining python deps.
#   4. Compile pytorch3d + cnms from source against the GPU's CUDA arch.
#   5. pip install -e the polarmae package.
#
# Visible GPU is required because pytorch3d + cnms compile their CUDA kernels
# against the GPU's compute capability.
set -euo pipefail

USER=hyu
ENV_PREFIX=/gpfs01/lbne/users/fm/${USER}/uvenv-polar-mae
REPO_DIR=/direct/lbne+u/hyu/PoLAr-MAE
MAX_JOBS_ENV=${MAX_JOBS:-4}

export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS="$MAX_JOBS_ENV"

echo "Target env prefix: $ENV_PREFIX"
echo "Repo:              $REPO_DIR"
echo "MAX_JOBS:          $MAX_JOBS_ENV"
echo "GPU node:          $(hostname)"
nvidia-smi -L || true
echo

# ----- micromamba (pre-staged on GPFS — compute nodes have no outbound HTTPS to micro.mamba.pm) -----
MM_BIN="/gpfs01/lbne/users/fm/${USER}/tools/bin/micromamba"
test -x "$MM_BIN" || { echo "FATAL: $MM_BIN missing" >&2; exit 1; }
"$MM_BIN" --version

SCRATCH="${_CONDOR_SCRATCH_DIR:-/tmp/$USER}"
export MAMBA_ROOT_PREFIX="${SCRATCH}/mamba-root"
mkdir -p "$MAMBA_ROOT_PREFIX"

# ----- env create -----
# We deliberately bypass environment.yml: that file resolves to a CPU-only
# pytorch under micromamba (the pytorch channel's pytorch-cuda metapackage
# doesn't force the cuda build under our channel-priority setting). Using
# pip for torch is what build_env_hyu.sh does too, and it's reliable.
if [[ -f "$ENV_PREFIX/conda-meta/history" ]] && [[ -x "$ENV_PREFIX/bin/python" ]]; then
    echo "[skip] env already exists at $ENV_PREFIX — extending in place"
else
    rm -rf "$ENV_PREFIX"
    echo "[1/5] micromamba create minimal env"
    "$MM_BIN" create -y -p "$ENV_PREFIX" \
        -c nvidia/label/cuda-12.4.0 -c conda-forge \
        python=3.10 cuda=12.4.0 cudnn gcc=12 gxx=12 pip
fi

PY="$ENV_PREFIX/bin/python"
PIP="$ENV_PREFIX/bin/pip"
test -x "$PY"  || { echo "FATAL: $PY not executable"  >&2; exit 1; }
test -x "$PIP" || { echo "FATAL: $PIP not executable" >&2; exit 1; }

export PATH="${ENV_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="${ENV_PREFIX}"

# ----- pip-installed deps (pytorch CUDA + everything else) -----
echo "[2/5] pip install torch (CUDA 12.4) + remaining python deps"
"$PIP" install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124
"$PIP" install --no-cache-dir \
    "pytorch-lightning>=2.0" lightning-utilities \
    numpy scikit-learn scipy omegaconf h5py iopath torchmetrics pybind11 \
    "jsonargparse[signatures]>=4.27.7" ninja

echo "torch sanity:"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'arch', torch.cuda.get_arch_list()); assert torch.version.cuda is not None, 'torch CUDA missing'"

# ----- pytorch3d (compile from source) -----
# Use the env's site-packages as the install-marker. The previous skip-check
# (import pytorch3d while cwd inside the source dir) found the un-compiled
# source pkg and falsely skipped the build.
PT3D_MARKER="$ENV_PREFIX/lib/python3.10/site-packages/pytorch3d"
if [[ -d "$PT3D_MARKER" ]] || [[ -f "$ENV_PREFIX/lib/python3.10/site-packages/pytorch3d.egg-link" ]]; then
    echo "[skip] pytorch3d already installed at $PT3D_MARKER"
else
    echo "[3/5] building pytorch3d from source"
    cd "$REPO_DIR/extensions/pytorch3d"
    "$PY" setup.py install
fi

# ----- cnms (compile from source) -----
CNMS_MARKER="$ENV_PREFIX/lib/python3.10/site-packages/cnms"
if [[ -d "$CNMS_MARKER" ]] || ls "$ENV_PREFIX/lib/python3.10/site-packages/"cnms*.egg-info 1>/dev/null 2>&1; then
    echo "[skip] cnms already installed"
else
    echo "[4/5] building cnms from source"
    cd "$REPO_DIR/extensions/cnms"
    "$PY" setup.py install
fi

# ----- polarmae editable + extras -----
cd "$REPO_DIR"
echo "[5/5] pip install -e ."
"$PIP" install -e .

echo
echo "Sanity import check:"
"$PY" - <<PYEOF
import torch, pytorch_lightning, h5py, numpy, sklearn, scipy, pytorch3d, cnms
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("pytorch3d", pytorch3d.__version__ if hasattr(pytorch3d, "__version__") else "ok")
import polarmae
print("polarmae", polarmae)
from polarmae.datasets import APA2DDataModule
from polarmae.eval.probes import APA2DProbeCallback
print("APA2DDataModule + APA2DProbeCallback imported OK")
PYEOF

echo
echo "Done — env at $ENV_PREFIX"
