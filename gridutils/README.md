# PoLAr-MAE on SDCC

Scripts that mirror the `ml-dune-model/gridutils` pattern for the PoLAr-MAE
codebase.  Two stages:

## 1) Build the conda env (one-time, on a GPU node)

The `polarmae` env requires `pytorch3d` and `cnms` built from source against
the GPU's CUDA arch.  Submit the build job:

```bash
cd /direct/lbne+u/hyu/PoLAr-MAE/gridutils
mkdir -p /gpfs01/lbne/users/fm/$USER/CONDOR_OUT     # if not present
condor_submit build_polarmae_env.sub
```

This creates the env at `/gpfs01/lbne/users/fm/hyu/uvenv-polar-mae`
(= `/lbne/u/hyu/fm/uvenv-polar-mae`).  Log lands at
`CONDOR_OUT/build_polarmae_env.*.out`.  Expect ~30–60 min wall time (most of
that is pytorch3d compile).

## 2) Submit a training run

```bash
bash gridutils/submit_polarmae.sh <run_name> configs/polarmae_apa2d_smoke.yml
```

Outputs land in `${CONDOR_OUT}/${run_name}/`:
- `checkpoints/`
- `probes/probes_step*.json`     ← per-voxel SVM + immediate-SFT metrics
- `lightning_logs/`
- `gpu.log`
- `${run_name}.sub`              ← the generated submit file
- `<ClusterId>.<ProcId>.{out,err,log}`

`submit_polarmae.sh polarmae` runs PoLAr-MAE; pass `pointmae` as the third arg
to run the vanilla Point-MAE baseline on the same data.

## Overridable env

| Var                | Default                                              |
|--------------------|------------------------------------------------------|
| `CONDOR_OUT`       | `/gpfs01/lbne/users/fm/${USER}/CONDOR_OUT`           |
| `REPODIR`          | `/direct/lbne+u/hyu/PoLAr-MAE`                       |
| `ENV_PREFIX`       | `/gpfs01/lbne/users/fm/${USER}/uvenv-polar-mae`      |
| `CACHE_DIR`        | `/gpfs01/lbne/users/fm/${USER}/cache`                |
| `REQUEST_MEMORY`   | `32000`                                              |
| `REQUEST_GPUS`     | `1`                                                  |
| `REQUEST_CPUS`     | `4`                                                  |
| `GPU_REQUIREMENTS` | `(GPUs_DeviceName == "NVIDIA L40S") && (GPUs_Capability == 8.9)` |
