# PoLAr-MAE on DUNE APA wire-plane data

Adapter + training pipeline that runs PoLAr-MAE pretraining on the
2D DUNE APA pixeldata (channel × tick × charge) we have on GPFS, with a
direct apples-to-apples comparison against the sparse-CNN MAE in
`/lbne/u/hyu/ml-dune-model/mae`.

Final result (run `polarmae_apa2d_full_260523_v3`, 20 000 steps on a single
L40S, ~1.5 h):

| Probe                              | Backbone-feat F1 (peak) | Raw-charge F1 (mean) | Lift |
|------------------------------------|-------------------------|----------------------|------|
| per-voxel SVM (3-class)            | **0.940** @ step 17 000 | 0.433                | 2.17× |
| per-voxel immediate-SFT MLP        | **0.933** @ step 18 000 | 0.532                | 1.75× |

Classes derived per-pixel from the raw PDG (`pdg_to_pixel_class`):
`{track = 0, shower = 1, other = 2}`, with the γ-blip connected-component
split matching the existing MAE.

---

## Data layout

| Use            | Path                                                                                | Pixel labels |
|----------------|-------------------------------------------------------------------------------------|--------------|
| SSL pretraining| `/gpfs01/lbne/users/fm/cffm-data/prod-jay-1M-2026-02-27/13825/1/00[1-8]` (train)   | none         |
|                | `/gpfs01/lbne/users/fm/cffm-data/prod-jay-1M-2026-02-27/13825/1/009` (val)         | none         |
| Probes         | `/gpfs01/lbne/users/fm/cffm-data/prod-jay-100k-truth-2026-02-27/13874/1/00[1-8]`   | `frame_pid_1st` |
|                | `/gpfs01/lbne/users/fm/cffm-data/prod-jay-100k-truth-2026-02-27/13874/1/009`      | `frame_pid_1st` |

W-view only (channels 1600 – 2649). After per-view rebase, coords land in
ch ∈ [0, 1049], tick ∈ [0, 1125].

---

## What got added (all new files — no upstream edits)

```
PoLAr-MAE/
├── polarmae/datasets/APA2D.py          # APA2D dataset + APA2DDataModule
│                                         (emits PILArNet-format dicts: points (N,4),
│                                          lengths, semantic_id, cluster_id)
├── polarmae/eval/__init__.py
├── polarmae/eval/probes.py             # APA2DProbeCallback
│                                         (4 probes per val epoch: per-voxel SVM
│                                          on backbone feats + raw charge; immediate
│                                          SFT MLP head on backbone feats + raw charge)
├── configs/polarmae_apa2d_smoke.yml    # 800-step smoke test
├── configs/polarmae_apa2d_full.yml     # 20 000-step full run
├── gridutils/build_polarmae_env.sh     # micromamba env build (GPU job)
├── gridutils/build_polarmae_env.sub
├── gridutils/submit_polarmae.sh        # mirrors ml-dune-model/gridutils/submit_mae.sh
├── gridutils/trainjob_polarmae.sh
├── gridutils/README.md
└── polarmae/datasets/__init__.py       # registers APA2D / APA2DDataModule
```

---

## 2D → 3D mapping

PoLAr-MAE expects 3D point clouds (x, y, z, energy). DUNE W-view is 2D, so we
embed as `(channel, tick, 0, log_charge)`:

* `transformation_center: [525, 562, 0]` — midpoint of the W-view bounding box
* `transformation_scale_factor: 1/600` — uniform scale so the data lands in
  roughly [-1, 1] (the 0-th axis collapses to 0)
* `voxel_size: 5` → `group_radius: 5/600` ≈ 0.00833 in normalized coords,
  i.e. a 5 × 5 (tick × channel) ball-query radius in original units (as you
  requested)
* Rotation transformation disabled: channel and tick are different physical
  axes, so rotating them is meaningless.

FPS/k-NN/Chamfer in PyTorch3D handle the degenerate z-axis without changes.

---

## Probes (`APA2DProbeCallback`)

Fires on every `on_validation_epoch_end`. Each pass:

1. Drains the labeled probe `val_dataloader()` (capacity-capped per class).
2. Builds per-voxel backbone features by running the encoder + inverse-distance
   K-NN upsampling token → voxel (mirrors PoLAr-MAE's `PointNetFeatureUpsampling`
   but with no learnable MLP).
3. Fits **four** classifiers on the held-out class-stratified pool:

   | Name           | Input                                       | Estimator                |
   |----------------|---------------------------------------------|--------------------------|
   | `voxel_svm_feat` | per-voxel backbone features              | sklearn `LinearSVC`      |
   | `voxel_svm_raw`  | per-voxel `(x, y, z, log_q)`             | sklearn `LinearSVC`      |
   | `sft_feat`       | per-voxel backbone features              | tiny MLP head trained 30 epochs |
   | `sft_raw`        | per-voxel `(x, y, z, log_q)`             | tiny MLP head trained 30 epochs |

4. Logs `<name>_train_macro_f1` / `<name>_val_macro_f1` to Lightning + dumps a
   JSON to `${json_dir}/probes_step{N:08d}.json` with confusion matrices.

The callback owns its own `APA2DDataModule` (via `probe_data_path` etc.) so
SSL data and probe data can be independent.

---

## Pipeline plumbing

* **Env**: `micromamba` (binary pre-staged at
  `/gpfs01/lbne/users/fm/hyu/tools/bin/micromamba`) creates a conda env at
  `/gpfs01/lbne/users/fm/hyu/uvenv-polar-mae` with `python 3.10 + cuda 12.4 +
  gcc 12`. Torch 2.5.1 CUDA wheel installed via pip (avoids the upstream
  `environment.yml`'s CPU/CUDA channel resolution issue). `pytorch3d` and
  `cnms` compile from source against L40S (`TORCH_CUDA_ARCH_LIST=8.9`).

* **Submission**: `gridutils/submit_polarmae.sh <run_name> <config.yml>` ;
  mirrors `ml-dune-model/gridutils/submit_mae.sh`. Outputs go to
  `${CONDOR_OUT}/${run_name}/` with `probes/`, `checkpoints/`, `lightning_logs/`,
  `gpu.log`, and the per-job .out/.err.

* **Memory hygiene** (`APA2DProbeCallback._collect_pool` end-of-call):
  `gc.collect() + torch.cuda.empty_cache()` after each probe — a previous
  run hit the 32 GB cgroup limit at step ~13 k without this.
  `APA2D._h5` has an LRU cap of 128 open handles (was unbounded; would leak
  file descriptors over the ~8 k pixeldata files).

---

## Known limitations / follow-ups

* **`LightningCLI` `Dict[str, LightningDataModule]` wiring is unreliable.** When
  `model.svm_validation.larnet.init_args` differ from `data.init_args`, the YAML
  doesn't propagate — both DataModule instances end up sharing the
  `data:` block. We sidestep this by giving `APA2DProbeCallback` its own
  `probe_data_path` arguments (a fully separate datamodule) and leaving
  `model.svm_validation: {}` so the upstream per-token SVM doesn't try to
  iterate unlabeled SSL data.

* **Upstream `ModelCheckpoint(monitor='svm_val_acc_larnet')`** is hardcoded in
  `polarmae/tasks/polarmae.py`. We satisfy it by having `APA2DProbeCallback`
  alias `voxel_svm_feat_val_macro_f1 → svm_val_acc_larnet` so the checkpoint
  callback has a metric to monitor (and the resulting "best" checkpoint
  filename includes the actual SVM val F1).

* **Dense events overflow FPS grouping.** Some W-view events have ~25 k pixels
  and produce > 512 groups → CUDA misaligned-address kernel crash. `APA2D` now
  takes a `max_points` argument (random subsample at __getitem__ time) and we
  set `max_points: 8000` in the config.

* **Smoke-test artifacts** preserved separately at
  `${CONDOR_OUT}/polarmae_apa2d_full_260523_v2/` (held by 32 GB cgroup at step
  ~13 k — the F1 curve up to that point was salvaged from the err log to
  `probes/probes_from_err.{json,csv}` + `probes_curve.png`).

---

## How to run

### One-time: build the env (on a GPU node)

```bash
cd /direct/lbne+u/hyu/PoLAr-MAE/gridutils
condor_submit build_polarmae_env.sub
# ~10–15 min on an L40S
```

### Smoke test (a few hundred steps, < 30 min)

```bash
bash gridutils/submit_polarmae.sh polarmae_apa2d_smoke_$(date +%y%m%d) configs/polarmae_apa2d_smoke.yml
```

### Full run (20 000 steps, ~1.5 h, 64 GB RAM)

```bash
REQUEST_MEMORY=64000 bash gridutils/submit_polarmae.sh polarmae_apa2d_full_$(date +%y%m%d) configs/polarmae_apa2d_full.yml
```

Each run gets its own `${CONDOR_OUT}/${run_name}/` directory — old results
are never overwritten.

---

## Comparison to the sparse-CNN MAE

Both pipelines now report the same 4 probe metrics on the same labeled subset,
so the numbers are directly comparable. The PoLAr-MAE pretraining gives
**val macro-F1 ≈ 0.94 (SVM) / 0.93 (SFT)** on the 3-class pixel ID task —
substantially above the **0.43 / 0.53** raw-charge baselines and a 2× lift
overall.
