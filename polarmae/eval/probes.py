"""Validation-time probes for the APA2D comparison runs.

Three probes are bundled into a single Lightning callback so they share the
expensive backbone forward pass:

  1. ``per_voxel_svm``: upsample tokens to per-voxel features via inverse-
     distance K-NN, fit a balanced LinearSVC on per-voxel labels, report F1.
     Mirrors ``ml-dune-model/mae/diagnostics/svm_probe.py``.

  2. ``immediate_sft_feat``: train a small linear classifier on a held-out
     subset of per-voxel backbone features for K epochs, report
     accuracy / efficiency / purity. Mirrors the "immediate-SFT" head in
     ``ml-dune-model/mae/scripts/train_mae.py``.

  3. ``immediate_sft_raw``: same training routine as (2) but the "feature" is
     the raw per-voxel ``(x, y, z, log_charge)`` -- the fairness baseline.

All three run on PoLAr-MAE's ``svm_validation`` datamodule. Class names are
read off the datamodule, so a 3-class APA2DDataModule and a 5-class
PILArNetDataModule both work.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

from polarmae.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Plain count confusion matrix (rows=true, cols=pred)."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    valid = (y_true >= 0)
    np.add.at(cm, (y_true[valid], y_pred[valid]), 1)
    return cm


def _eff_purity(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-class efficiency (recall) and purity (precision) from a confusion matrix."""
    tp = np.diag(cm).astype(np.float64)
    fn = cm.sum(axis=1).astype(np.float64) - tp
    fp = cm.sum(axis=0).astype(np.float64) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        pur = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
    return eff, pur


def _macro_f1(eff: np.ndarray, pur: np.ndarray) -> float:
    """Macro F1 across classes; ignores NaNs."""
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * eff * pur / (eff + pur)
    valid = ~np.isnan(f1)
    return float(np.nanmean(f1[valid])) if valid.any() else float("nan")


def _print_probe_summary(name: str, cm: np.ndarray, class_names: List[str]) -> None:
    eff, pur = _eff_purity(cm)
    width = max(len(n) for n in class_names) + 2
    log.info(f"\n{name}  confusion (rows=true, cols=pred):")
    header = " " * (width + 2) + "  ".join(f"{n:>{width}}" for n in class_names)
    log.info(header)
    for i, n in enumerate(class_names):
        row = f"  {n:>{width}}  " + "  ".join(
            f"{cm[i, j]:>{width}d}" for j in range(len(class_names))
        )
        log.info(row)
    log.info(f"{name}  per-class:")
    log.info(f"  {'class':>10}  {'efficiency':>12}  {'purity':>10}")
    for i, n in enumerate(class_names):
        log.info(f"  {n:>10}  {eff[i]:>12.4f}  {pur[i]:>10.4f}")
    log.info(f"{name}  macro-F1 = {_macro_f1(eff, pur):.4f}")


# -----------------------------------------------------------------------------
# Per-voxel feature extraction
# -----------------------------------------------------------------------------

@dataclass
class _VoxelBatch:
    feats:  np.ndarray   # (N, D) per-voxel feature
    labels: np.ndarray   # (N,)   per-voxel label (-1 = ignore)


@torch.no_grad()
def _extract_per_voxel_features(
    model:        pl.LightningModule,
    points:       torch.Tensor,            # (B, N_max, 4)
    lengths:      torch.Tensor,            # (B,)
    semantic_id:  torch.Tensor,            # (B, N_max, 1)
) -> Tuple[np.ndarray, np.ndarray]:
    """Run encoder, upsample tokens -> voxels via inverse-distance K-NN.

    Returns (feats[N_total, D], labels[N_total]) flattened over the batch.
    """
    encoder = model.encoder
    # Apply the same val transformation the model was trained against (centering+scaling).
    points_xform = model.val_transformations(points)
    out = encoder.prepare_tokens(points_xform, lengths, ids=semantic_id)
    out_t = encoder.transformer(out["x"], out["pos_embed"], out["emb_mask"])
    tokens = out_t.last_hidden_state                       # (B, T, D)
    centers = out["centers"][..., :3]                      # (B, T, 3)
    emb_lens = out["emb_mask"].sum(dim=1)                  # (B,)

    # Per-voxel xyz (post-transform); pad mask
    xyz_pts  = points_xform[..., :3]                       # (B, N_max, 3)
    pt_mask  = (torch.arange(points.shape[1], device=points.device)
                .unsqueeze(0) < lengths.unsqueeze(1))      # (B, N_max)

    K = min(5, int(emb_lens.min().item()))
    K = max(K, 1)
    dists, idx, _ = knn_points(
        xyz_pts, centers,
        lengths1=lengths, lengths2=emb_lens,
        K=K, return_sorted=False,
    )                                                       # both (B, N_max, K)
    weight = 1.0 / (dists + torch.finfo(dists.dtype).eps)
    weight = weight / weight.sum(dim=2, keepdim=True)       # (B, N_max, K)

    # Gather token features at idx: tokens[b, idx[b, n, k]]
    B, N_max, _ = xyz_pts.shape
    D = tokens.shape[-1]
    idx_flat = idx.clamp(min=0)                              # (B, N_max, K)
    gathered = torch.gather(
        tokens.unsqueeze(1).expand(-1, N_max, -1, -1),       # (B, N_max, T, D)
        2,
        idx_flat.unsqueeze(-1).expand(-1, -1, -1, D),        # (B, N_max, K, D)
    )                                                        # (B, N_max, K, D)
    voxel_feats = (gathered * weight.unsqueeze(-1)).sum(dim=2)  # (B, N_max, D)

    # Flatten
    feats_flat   = voxel_feats[pt_mask].cpu().numpy().astype(np.float32, copy=False)
    labels_flat  = semantic_id.squeeze(-1)[pt_mask].cpu().numpy().astype(np.int64, copy=False)
    return feats_flat, labels_flat


@torch.no_grad()
def _extract_per_voxel_raw(
    points:      torch.Tensor,
    lengths:     torch.Tensor,
    semantic_id: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    pt_mask = (torch.arange(points.shape[1], device=points.device)
               .unsqueeze(0) < lengths.unsqueeze(1))
    feats = points[pt_mask].cpu().numpy().astype(np.float32, copy=False)        # (N, 4)
    labels = semantic_id.squeeze(-1)[pt_mask].cpu().numpy().astype(np.int64, copy=False)
    return feats, labels


# -----------------------------------------------------------------------------
# Stratified sampler
# -----------------------------------------------------------------------------

def _stratified_sample(
    feats:  np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    cap_per_class: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    out_X, out_y = [], []
    for c in range(n_classes):
        idx = np.flatnonzero(labels == c)
        if idx.size == 0:
            continue
        take = min(cap_per_class, idx.size)
        chosen = rng.choice(idx, size=take, replace=False)
        out_X.append(feats[chosen])
        out_y.append(labels[chosen])
    if not out_X:
        return (np.zeros((0, feats.shape[1]), dtype=feats.dtype),
                np.zeros((0,), dtype=labels.dtype))
    return np.concatenate(out_X, axis=0), np.concatenate(out_y, axis=0)


# -----------------------------------------------------------------------------
# Tiny SFT head: 2-layer MLP
# -----------------------------------------------------------------------------

class _SmallHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_head(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> _SmallHead:
    """Fit a fresh head with class-balanced CE.

    Called from a Lightning ``on_validation_epoch_end`` hook, which runs under
    ``torch.no_grad()``. We must re-enable grad locally so ``loss.backward()``
    can build a graph.
    """
    head = _SmallHead(X.shape[1], n_classes).to(device)
    opt  = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    X_t  = torch.from_numpy(X).to(device)
    y_t  = torch.from_numpy(y).to(device)

    cw = np.zeros(n_classes, dtype=np.float32)
    for c in range(n_classes):
        n_c = int((y == c).sum())
        cw[c] = 0.0 if n_c == 0 else 1.0 / n_c
    if cw.sum() > 0:
        cw = cw / cw.sum() * n_classes
    cw_t = torch.from_numpy(cw).to(device)

    n = X_t.shape[0]
    with torch.enable_grad():
        for _ in range(epochs):
            perm = torch.randperm(n, device=device)
            for s in range(0, n, batch_size):
                sel = perm[s:s + batch_size]
                logits = head(X_t[sel])
                loss = F.cross_entropy(logits, y_t[sel], weight=cw_t, ignore_index=-1)
                opt.zero_grad(); loss.backward(); opt.step()
    return head


@torch.no_grad()
def _head_predict(
    head:   _SmallHead,
    X:      np.ndarray,
    device: torch.device,
    chunk:  int = 16384,
) -> np.ndarray:
    head.eval()
    preds = []
    X_t = torch.from_numpy(X).to(device)
    for s in range(0, X_t.shape[0], chunk):
        preds.append(head(X_t[s:s + chunk]).argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds) if preds else np.zeros((0,), dtype=np.int64)


# -----------------------------------------------------------------------------
# Per-voxel SVM
# -----------------------------------------------------------------------------

def _fit_svm(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    class_names: List[str],
    C: float = 1.0,
) -> Dict[str, float]:
    svm = LinearSVC(C=C, class_weight="balanced", random_state=0, max_iter=5000)
    svm.fit(X_tr, y_tr)
    n = len(class_names)
    pred_tr = svm.predict(X_tr); pred_va = svm.predict(X_va)
    cm_tr = _confusion(y_tr, pred_tr, n)
    cm_va = _confusion(y_va, pred_va, n)
    eff_tr, pur_tr = _eff_purity(cm_tr)
    eff_va, pur_va = _eff_purity(cm_va)
    return {
        "train_acc": float(svm.score(X_tr, y_tr)),
        "val_acc":   float(svm.score(X_va, y_va)),
        "train_macro_f1": _macro_f1(eff_tr, pur_tr),
        "val_macro_f1":   _macro_f1(eff_va, pur_va),
        "train_cm": cm_tr.tolist(),
        "val_cm":   cm_va.tolist(),
        "train_eff": eff_tr.tolist(),
        "val_eff":   eff_va.tolist(),
        "train_pur": pur_tr.tolist(),
        "val_pur":   pur_va.tolist(),
    }


# -----------------------------------------------------------------------------
# Main callback
# -----------------------------------------------------------------------------

class APA2DProbeCallback(pl.Callback):
    """Run per-voxel SVM + per-voxel immediate-SFT (feat & raw) at each val end.

    Two ways to provide labeled probe data:

    1. ``probe_data_path`` / ``probe_val_data_path`` / ``probe_dataset_kwargs``:
       the callback builds and owns its own APA2DDataModule. This is the robust
       path — independent of how LightningCLI wires ``model.svm_validation``.
       Use this when SSL pretraining data has no per-pixel labels but you want
       probes on a different (labeled) dataset.

    2. Legacy: ``datamodule_key`` looks up ``pl_module.hparams.svm_validation``
       and uses the registered datamodule. Works only when LightningCLI
       correctly applies the YAML init_args (which is brittle for
       ``Dict[str, LightningDataModule]``).

    Args:
      datamodule_key:  key in ``model.hparams.svm_validation`` (mode 2).
      probe_data_path / probe_val_data_path / probe_dataset_kwargs:
                       owned-datamodule kwargs (mode 1). Wins over mode 2.
      probe_batch_size / probe_num_workers: dataloader kwargs for mode 1.
      max_pixels_per_class: per-class cap when subsampling.
      svm_C:           regularization for LinearSVC.
      sft_epochs / sft_batch / sft_lr: hyperparams for the small head.
      train_frac:      train / val split of the collected pool.
      json_dir:        directory to dump per-epoch probe metrics as JSON.
    """

    def __init__(
        self,
        datamodule_key: str = "larnet",
        probe_data_path: Optional[str] = None,
        probe_val_data_path: Optional[str] = None,
        probe_dataset_kwargs: Optional[dict] = None,
        probe_batch_size: int = 8,
        probe_num_workers: int = 2,
        max_pixels_per_class: int = 5000,
        svm_C: float = 1.0,
        sft_epochs: int = 30,
        sft_batch: int = 256,
        sft_lr: float = 5e-3,
        train_frac: float = 0.8,
        json_dir: Optional[str] = None,
    ):
        super().__init__()
        self.datamodule_key       = datamodule_key
        self.probe_data_path      = probe_data_path
        self.probe_val_data_path  = probe_val_data_path
        self.probe_dataset_kwargs = dict(probe_dataset_kwargs or {})
        self.probe_batch_size     = int(probe_batch_size)
        self.probe_num_workers    = int(probe_num_workers)
        self.max_pixels_per_class = int(max_pixels_per_class)
        self.svm_C                = float(svm_C)
        self.sft_epochs           = int(sft_epochs)
        self.sft_batch            = int(sft_batch)
        self.sft_lr               = float(sft_lr)
        self.train_frac           = float(train_frac)
        self.json_dir             = Path(json_dir) if json_dir else None
        self._epoch               = 0
        self._owned_datamodule: Optional[pl.LightningDataModule] = None

    def _get_datamodule(self, pl_module: pl.LightningModule) -> Optional[pl.LightningDataModule]:
        if self.probe_data_path is not None:
            if self._owned_datamodule is None:
                from polarmae.datasets import APA2DDataModule
                self._owned_datamodule = APA2DDataModule(
                    data_path=self.probe_data_path,
                    val_data_path=self.probe_val_data_path,
                    batch_size=self.probe_batch_size,
                    num_workers=self.probe_num_workers,
                    dataset_kwargs=self.probe_dataset_kwargs,
                )
                self._owned_datamodule.setup("fit")
                log.info(
                    f"APA2DProbeCallback: owned datamodule built — "
                    f"train={self.probe_data_path!r} val={self.probe_val_data_path!r}"
                )
            return self._owned_datamodule
        svm_validation = getattr(pl_module.hparams, "svm_validation", None)
        if svm_validation:
            return svm_validation.get(self.datamodule_key)
        return None

    @torch.no_grad()
    def _collect_pool(
        self, pl_module: pl.LightningModule, datamodule: pl.LightningDataModule,
    ) -> Tuple[_VoxelBatch, _VoxelBatch]:
        """Drain enough of the val loader to fill class quotas; return raw + feat."""
        device = pl_module.device
        n_cls  = datamodule.num_seg_classes
        cap    = self.max_pixels_per_class

        # Per-class pools for both feature types.
        feat_pool:  List[List[np.ndarray]] = [[] for _ in range(n_cls)]
        raw_pool:   List[List[np.ndarray]] = [[] for _ in range(n_cls)]
        feat_cnt   = np.zeros(n_cls, dtype=np.int64)
        raw_cnt    = np.zeros(n_cls, dtype=np.int64)
        rng        = np.random.default_rng(0)

        loader = datamodule.val_dataloader()
        warned_no_labels = False
        for batch in loader:
            if feat_cnt.min() >= cap and raw_cnt.min() >= cap:
                break
            if batch.get("semantic_id") is None:
                if not warned_no_labels:
                    log.warning(
                        f"APA2DProbeCallback: datamodule '{self.datamodule_key}' "
                        f"yielded a batch with semantic_id=None (no pixel labels). "
                        f"Probes need labeled data — check the datamodule wiring "
                        f"(return_semantic_id and data_path). Skipping probes."
                    )
                    warned_no_labels = True
                continue
            points      = batch["points"].to(device)
            lengths     = batch["lengths"].to(device)
            semantic_id = batch["semantic_id"].to(device)

            feats, labels = _extract_per_voxel_features(
                pl_module, points, lengths, semantic_id,
            )
            raw_feats, _  = _extract_per_voxel_raw(points, lengths, semantic_id)

            for c in range(n_cls):
                need = cap - int(feat_cnt[c])
                if need > 0:
                    m = labels == c
                    n = int(m.sum())
                    if n > 0:
                        take = min(n, need)
                        sel = rng.choice(np.flatnonzero(m), size=take, replace=False) \
                              if n > take else np.flatnonzero(m)
                        feat_pool[c].append(feats[sel])
                        feat_cnt[c] += sel.size

                need = cap - int(raw_cnt[c])
                if need > 0:
                    m = labels == c
                    n = int(m.sum())
                    if n > 0:
                        take = min(n, need)
                        sel = rng.choice(np.flatnonzero(m), size=take, replace=False) \
                              if n > take else np.flatnonzero(m)
                        raw_pool[c].append(raw_feats[sel])
                        raw_cnt[c] += sel.size

        def _stack(pool, cnt):
            X_parts, y_parts = [], []
            for c in range(n_cls):
                if pool[c]:
                    arr = np.concatenate(pool[c], axis=0)
                    X_parts.append(arr)
                    y_parts.append(np.full(arr.shape[0], c, dtype=np.int64))
            X = np.concatenate(X_parts, axis=0) if X_parts else np.zeros((0, 1), np.float32)
            y = np.concatenate(y_parts, axis=0) if y_parts else np.zeros((0,), np.int64)
            return _VoxelBatch(feats=X, labels=y)

        return _stack(feat_pool, feat_cnt), _stack(raw_pool, raw_cnt)

    def _split(self, vb: _VoxelBatch, rng: np.random.Generator) -> Tuple[_VoxelBatch, _VoxelBatch]:
        n = vb.feats.shape[0]
        if n == 0:
            return vb, vb
        perm = rng.permutation(n)
        cut  = int(self.train_frac * n)
        tr_idx, va_idx = perm[:cut], perm[cut:]
        return (_VoxelBatch(vb.feats[tr_idx], vb.labels[tr_idx]),
                _VoxelBatch(vb.feats[va_idx], vb.labels[va_idx]))

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        datamodule = self._get_datamodule(pl_module)
        if datamodule is None:
            log.warning(
                "APA2DProbeCallback: no datamodule available (neither "
                f"probe_data_path nor svm_validation['{self.datamodule_key}']); "
                "skipping probes."
            )
            return

        class_names = [datamodule.seg_class_to_category[i]
                       for i in range(datamodule.num_seg_classes)]
        n_cls = len(class_names)
        was_training = pl_module.training
        pl_module.eval()
        try:
            feat_pool, raw_pool = self._collect_pool(pl_module, datamodule)
        finally:
            pl_module.train(was_training)

        rng = np.random.default_rng(0)
        feat_tr, feat_va = self._split(feat_pool, rng)
        raw_tr,  raw_va  = self._split(raw_pool,  rng)

        results: Dict[str, dict] = {}

        # ---- 1. Per-voxel SVM on backbone features ----
        if feat_tr.feats.shape[0] > 0 and feat_va.feats.shape[0] > 0:
            svm_metrics = _fit_svm(
                feat_tr.feats, feat_tr.labels,
                feat_va.feats, feat_va.labels,
                class_names, C=self.svm_C,
            )
            results["voxel_svm_feat"] = svm_metrics
            self._log_scalar(pl_module, "voxel_svm_feat_train_macro_f1",
                             svm_metrics["train_macro_f1"])
            self._log_scalar(pl_module, "voxel_svm_feat_val_macro_f1",
                             svm_metrics["val_macro_f1"])
            # Alias under the name upstream's hardcoded ModelCheckpoint
            # (monitor='svm_val_acc_larnet') expects, so we don't have to
            # carry the upstream per-token SVM (which crashes on unlabeled SSL data).
            self._log_scalar(pl_module, "svm_val_acc_larnet",
                             svm_metrics["val_macro_f1"])
            _print_probe_summary(
                "voxel_svm_feat (val)",
                np.asarray(svm_metrics["val_cm"]),
                class_names,
            )

        # ---- 2. Per-voxel SVM on raw (x, y, z, log_charge) ----
        if raw_tr.feats.shape[0] > 0 and raw_va.feats.shape[0] > 0:
            svm_metrics = _fit_svm(
                raw_tr.feats, raw_tr.labels,
                raw_va.feats, raw_va.labels,
                class_names, C=self.svm_C,
            )
            results["voxel_svm_raw"] = svm_metrics
            self._log_scalar(pl_module, "voxel_svm_raw_train_macro_f1",
                             svm_metrics["train_macro_f1"])
            self._log_scalar(pl_module, "voxel_svm_raw_val_macro_f1",
                             svm_metrics["val_macro_f1"])
            _print_probe_summary(
                "voxel_svm_raw (val)",
                np.asarray(svm_metrics["val_cm"]),
                class_names,
            )

        # ---- 3. Immediate-SFT head on backbone features ----
        device = pl_module.device
        if feat_tr.feats.shape[0] > 0 and feat_va.feats.shape[0] > 0:
            head = _train_head(
                feat_tr.feats, feat_tr.labels, n_cls,
                epochs=self.sft_epochs, batch_size=self.sft_batch,
                lr=self.sft_lr, device=device,
            )
            pred_tr = _head_predict(head, feat_tr.feats, device)
            pred_va = _head_predict(head, feat_va.feats, device)
            cm_tr = _confusion(feat_tr.labels, pred_tr, n_cls)
            cm_va = _confusion(feat_va.labels, pred_va, n_cls)
            eff_va, pur_va = _eff_purity(cm_va)
            sft_metrics = {
                "train_cm": cm_tr.tolist(),
                "val_cm":   cm_va.tolist(),
                "val_eff":  eff_va.tolist(),
                "val_pur":  pur_va.tolist(),
                "val_macro_f1": _macro_f1(*_eff_purity(cm_va)),
                "train_macro_f1": _macro_f1(*_eff_purity(cm_tr)),
            }
            results["sft_feat"] = sft_metrics
            self._log_scalar(pl_module, "sft_feat_train_macro_f1",
                             sft_metrics["train_macro_f1"])
            self._log_scalar(pl_module, "sft_feat_val_macro_f1",
                             sft_metrics["val_macro_f1"])
            _print_probe_summary("sft_feat (val)", cm_va, class_names)

        # ---- 4. Immediate-SFT head on raw (x, y, z, log_charge) ----
        if raw_tr.feats.shape[0] > 0 and raw_va.feats.shape[0] > 0:
            head = _train_head(
                raw_tr.feats, raw_tr.labels, n_cls,
                epochs=self.sft_epochs, batch_size=self.sft_batch,
                lr=self.sft_lr, device=device,
            )
            pred_tr = _head_predict(head, raw_tr.feats, device)
            pred_va = _head_predict(head, raw_va.feats, device)
            cm_tr = _confusion(raw_tr.labels, pred_tr, n_cls)
            cm_va = _confusion(raw_va.labels, pred_va, n_cls)
            sft_metrics = {
                "train_cm": cm_tr.tolist(),
                "val_cm":   cm_va.tolist(),
                "val_eff":  _eff_purity(cm_va)[0].tolist(),
                "val_pur":  _eff_purity(cm_va)[1].tolist(),
                "val_macro_f1": _macro_f1(*_eff_purity(cm_va)),
                "train_macro_f1": _macro_f1(*_eff_purity(cm_tr)),
            }
            results["sft_raw"] = sft_metrics
            self._log_scalar(pl_module, "sft_raw_train_macro_f1",
                             sft_metrics["train_macro_f1"])
            self._log_scalar(pl_module, "sft_raw_val_macro_f1",
                             sft_metrics["val_macro_f1"])
            _print_probe_summary("sft_raw (val)", cm_va, class_names)

        # ---- Dump JSON ----
        if self.json_dir is not None:
            self.json_dir.mkdir(parents=True, exist_ok=True)
            stub = self.json_dir / f"probes_step{trainer.global_step:08d}.json"
            with open(stub, "w") as fp:
                json.dump(
                    {"global_step": int(trainer.global_step),
                     "epoch": self._epoch,
                     "class_names": class_names,
                     "results": results},
                    fp, indent=2, default=float,
                )
            log.info(f"APA2DProbeCallback: wrote {stub}")

        # Memory hygiene: pools + heads are large numpy/torch buffers; release
        # them and ask the CUDA caching allocator to give memory back so we
        # don't accumulate across many val cycles (caused a 32 GB cgroup OOM
        # at step ~13k on a previous run).
        del feat_pool, raw_pool, feat_tr, feat_va, raw_tr, raw_va, results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._epoch += 1

    @staticmethod
    def _log_scalar(pl_module: pl.LightningModule, key: str, value: float) -> None:
        if not np.isfinite(value):
            return
        pl_module.log(key, float(value), sync_dist=True, prog_bar=True)
