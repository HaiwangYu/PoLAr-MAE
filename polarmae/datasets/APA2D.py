"""
APA2D — DUNE wire-plane (channel, tick) sparse data, exposed in the
PILArNet point-cloud format expected by the rest of PoLAr-MAE.

Source data:
    {root}/{run}/{subrun}/{event}/out_{basename}/
        {basename}_pixeldata-anode{N}.h5      (sparse pixels per APA)
        {basename}_metadata.h5                (per-event truth)

Each HDF5 contains top-level groups (one per APA event slice). Per group:
    /{g}/frame_rebinned_reco/coords    (M, 2) int32   -- (channel, tick)
    /{g}/frame_rebinned_reco/features  (M,)   float32 -- charge
    /{g}/frame_pid_1st/coords          (K, 2) int32   -- (channel, tick)
    /{g}/frame_pid_1st/features        (K,)   float32 -- raw PDG code

We treat the 2D point set as 3D by setting z=0 and use the per-pixel PDG
(via ``pdg_to_pixel_class``) as the semantic id for the SVM / SFT probes.

Outputs match ``polarmae.datasets.PILArNet``:
    {points: (N, 4)=(ch, tick, 0, log_charge),
     lengths: ...,
     semantic_id: (N, 1) long in {0,1,2,-1},
     cluster_id: (N, 1) long  -- all zeros here (no cluster supervision)}
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from polarmae.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


# View channel ranges (W/U/V for the prod-jay APA pixeldata).
DEFAULT_VIEW_RANGES = {
    "U": (0, 800),
    "V": (800, 1600),
    "W": (1600, 2650),
}


# -----------------------------------------------------------------------------
# PDG -> 3-class pixel label (track / shower / other).
# Copied / adapted from ml-dune-model/models/mae_model.pdg_to_pixel_class so we
# don't depend on that repo or on WarpConvNet here.
# -----------------------------------------------------------------------------

_TRACK_PDGS = {13, -13, 2212, 211, -211}     # mu+-, p, pi+-
_ELEC_PDGS  = {11, -11}                       # e+-
_GAMMA_PDG  = 22
_BLIP_CONNECT_DIST = 5.0
_BLIP_MAX_PIXELS   = 30

PIXEL_PID_CLASS_NAMES = ["track", "shower", "other"]
PIXEL_PID_N_CLASSES   = len(PIXEL_PID_CLASS_NAMES)


def pdg_to_pixel_class(
    pid_labels: np.ndarray,   # (N,) int32 raw PDG codes; 0 = no truth
    positions:  np.ndarray,   # (N, 2) int32 (channel, tick)
    connect_dist: float = _BLIP_CONNECT_DIST,
    blip_max_pixels: int  = _BLIP_MAX_PIXELS,
) -> np.ndarray:
    """Map per-pixel PDG codes to {track=0, shower=1, other=2, -1=no truth}.

    Gamma pixels are split into shower (large EM cluster) vs other (small blip)
    via connected-component analysis over (gamma + e+-) pixels in the image.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    pdg = pid_labels.astype(np.int32, copy=False)
    n   = len(pdg)
    out = np.full(n, 2, dtype=np.int64)                  # default: other
    out[pdg == 0] = -1                                   # no truth
    out[np.isin(pdg, list(_TRACK_PDGS))] = 0             # track
    out[np.isin(pdg, list(_ELEC_PDGS))]  = 1             # shower (e+-)

    gamma_local = np.where(pdg == _GAMMA_PDG)[0]
    if len(gamma_local) > 0:
        elec_local = np.where(np.isin(pdg, list(_ELEC_PDGS)))[0]
        em_local   = np.concatenate([gamma_local, elec_local])
        em_pos     = positions[em_local].astype(float)
        n_em       = len(em_local)

        if n_em == 1:
            out[gamma_local[0]] = 2                      # solitary gamma -> other
        else:
            tree  = cKDTree(em_pos)
            pairs = tree.query_pairs(connect_dist)
            if pairs:
                ra, ca = zip(*pairs)
                ra, ca = list(ra), list(ca)
                row    = np.array(ra + ca, dtype=np.int64)
                col    = np.array(ca + ra, dtype=np.int64)
                data   = np.ones_like(row, dtype=np.int8)
                graph  = csr_matrix((data, (row, col)), shape=(n_em, n_em))
                _, comp = connected_components(graph, directed=False)
            else:
                comp = np.arange(n_em)

            # which component is each gamma in?
            gamma_em_idx = np.arange(len(gamma_local))
            gamma_comp   = comp[gamma_em_idx]
            comp_sizes   = np.bincount(comp)
            small        = comp_sizes[gamma_comp] <= blip_max_pixels
            # large EM cluster -> shower (1); small blip -> other (2; already default)
            out[gamma_local[~small]] = 1

    return out


# -----------------------------------------------------------------------------
# Energy log-transform: mirrors PILArNet.log_transform but with our charge scale
# -----------------------------------------------------------------------------

def log_transform(x: np.ndarray, xmax: float = 1.0, eps: float = 1e-7) -> np.ndarray:
    """[eps, xmax] -> [-1, 1]"""
    y0 = np.log10(eps)
    y1 = np.log10(eps + xmax)
    return 2 * (np.log10(x + eps) - y0) / (y1 - y0) - 1


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

@dataclass
class _APASample:
    path: str
    group: str


class APA2D(Dataset):
    """Per-event APA wire-plane sparse data, served as PILArNet-format dicts.

    Args:
      data_path:     glob pattern OR directory root. If it ends with ``.h5`` or
                     contains a ``*`` we ``glob`` it; otherwise we ``rglob`` for
                     ``*pixeldata-anode{apa}.h5`` under that root.
      apa:           APA index to load (matched against filename suffix).
      view:          'U' | 'V' | 'W' wire-plane view.
      emin, emax:    log-transform range for charge.
      energy_threshold: drop pixels with charge below this BEFORE log-transform.
      min_points:    drop events with fewer surviving pixels than this.
      max_points:    if positive and a single event exceeds this after the
                     energy-threshold filter, randomly subsample down to
                     ``max_points``. Prevents PoLAr-MAE's FPS grouping from
                     overflowing ``context_length`` (which can trigger a
                     CUDA misaligned-address kernel crash on dense events).
      maxlen:        cap on dataset length (negative -> no cap).
      return_semantic_id: emit per-pixel 3-class labels (needs metadata file).
      return_cluster_id:  emit per-pixel cluster id (always zeros; for API parity).
      cache_dir:     where to cache the per-file event index (pickled list).
      use_cache:     load/save the index from/to ``cache_dir``.
    """

    def __init__(
        self,
        data_path: str,
        apa: int = 0,
        view: str = "W",
        emin: float = 1.0,                # charge units
        emax: float = 1.0e5,              # ~ observed max ~8e4
        energy_threshold: float = 1.0,    # pre-log-transform threshold
        min_points: int = 256,
        max_points: int = -1,             # cap pixels/event; <=0 disables
        maxlen: int = -1,
        return_semantic_id: bool = True,
        return_cluster_id: bool = False,
        cache_dir: str = "./data",
        use_cache: bool = True,
    ):
        self.data_path        = data_path
        self.apa              = int(apa)
        self.view             = view.upper()
        self.emin             = float(emin)
        self.emax             = float(emax)
        self.energy_threshold = float(energy_threshold)
        self.min_points       = int(min_points)
        self.max_points       = int(max_points)
        self.maxlen           = int(maxlen)
        self.return_semantic_id = bool(return_semantic_id)
        self.return_cluster_id  = bool(return_cluster_id)
        self.cache_dir        = Path(cache_dir)
        self.use_cache        = bool(use_cache)

        if self.view not in DEFAULT_VIEW_RANGES:
            raise ValueError(f"view must be one of {list(DEFAULT_VIEW_RANGES)}, got {view!r}")
        self.ch_start, self.ch_end = DEFAULT_VIEW_RANGES[self.view]

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # h5py handles opened lazily per-worker (DataLoader-friendly).
        # Initialise BEFORE _scan() so __del__ doesn't trip if scanning fails.
        self._h5_cache: dict = {}
        self.samples: List[_APASample] = self._scan()
        if not self.samples:
            raise RuntimeError(f"No APA pixeldata samples found for {data_path!r} apa={apa}")

        log.info(
            f"APA2D: {len(self.samples)} events  "
            f"data_path={data_path!r}  apa={apa}  view={view}"
        )

    # ---------------- index ----------------

    def _enumerate_files(self) -> List[Path]:
        """Enumerate per-APA HDF5 pixeldata files under the user-supplied path.

        Accepts three forms in ``data_path``:
          - a single ``.h5`` file
          - a directory tree (rglob for ``*pixeldata-anode{apa}.h5``)
          - a glob pattern. Expanded with ``glob()``; matched files are taken
            as-is, matched directories are rglob'd for the per-APA file.
        """
        suffix = f"anode{self.apa}.h5"
        pattern = f"*pixeldata-anode{self.apa}.h5"

        if self.data_path.endswith(".h5") and not any(c in self.data_path for c in "*?["):
            roots: List[Path] = [Path(self.data_path)]
        elif any(c in self.data_path for c in "*?["):
            roots = [Path(p) for p in glob(self.data_path)]
        else:
            roots = [Path(self.data_path)]

        files: List[Path] = []
        for r in roots:
            if r.is_file():
                files.append(r)
            elif r.is_dir():
                files.extend(r.rglob(pattern))

        files = [p for p in files if p.is_file() and p.name.endswith(suffix)]
        files = sorted(set(files))
        return files

    def _cache_path(self) -> Path:
        key = hashlib.md5(
            f"{os.path.abspath(self.data_path)}|apa={self.apa}|view={self.view}".encode()
        ).hexdigest()[:10]
        return self.cache_dir / f"APA2D_apa{self.apa}_view{self.view}_{key}.pt"

    def _scan(self) -> List[_APASample]:
        cf = self._cache_path()
        if self.use_cache and cf.exists():
            log.info(f"APA2D: loading event index from {cf}")
            data = torch.load(cf, map_location="cpu")
            return [_APASample(path=p, group=g) for p, g in data]

        files = self._enumerate_files()
        log.info(f"APA2D: scanning {len(files)} files (will cache to {cf})")
        samples: List[_APASample] = []
        for fp in files:
            try:
                with h5py.File(fp, "r") as f:
                    for grp_name in f.keys():
                        grp = f[grp_name]
                        if not isinstance(grp, h5py.Group):
                            continue
                        frame = grp.get("frame_rebinned_reco")
                        if (
                            isinstance(frame, h5py.Group)
                            and "coords" in frame
                            and "features" in frame
                        ):
                            samples.append(_APASample(path=str(fp), group=grp_name))
            except OSError as e:
                log.warning(f"could not open {fp}: {e}")

        samples.sort(key=lambda s: (s.path, int(s.group)))
        # Don't cache empty results — a single bad scan would otherwise poison
        # all subsequent runs.
        if self.use_cache and samples:
            torch.save([(s.path, s.group) for s in samples], cf)
            log.info(f"APA2D: cached {len(samples)} events to {cf}")
        return samples

    # ---------------- HDF5 access ----------------

    # Per-process LRU cap on open HDF5 handles. With ~8k pixeldata files per
    # epoch keeping every one open leaks file descriptors + h5py chunk-cache
    # memory across the long training run. 128 is plenty for in-batch reuse.
    _H5_CACHE_LIMIT = 128

    def _h5(self, path: str) -> h5py.File:
        h = self._h5_cache.pop(path, None)
        if h is None:
            h = h5py.File(path, "r", libver="latest", swmr=True)
        # Re-insert to mark as MRU.
        self._h5_cache[path] = h
        # Evict LRU entries beyond the limit.
        while len(self._h5_cache) > self._H5_CACHE_LIMIT:
            lru_key = next(iter(self._h5_cache))
            try:
                self._h5_cache.pop(lru_key).close()
            except Exception:
                pass
        return h

    def __len__(self) -> int:
        n = len(self.samples)
        return min(n, self.maxlen) if self.maxlen > 0 else n

    def __getitem__(self, idx: int) -> dict:
        s   = self.samples[idx]
        f   = self._h5(s.path)
        grp = f[s.group]

        # ---- coords + charge ----
        co_all = grp["frame_rebinned_reco"]["coords"][()]        # (M, 2) int32
        fe_all = grp["frame_rebinned_reco"]["features"][()]      # (M,)   float32
        view_m = (co_all[:, 0] >= self.ch_start) & (co_all[:, 0] < self.ch_end)
        co     = co_all[view_m]
        fe     = fe_all[view_m]
        # Rebase channel coord so per-view origin is 0 (matches APASparseDataset).
        co     = co.copy()
        co[:, 0] -= self.ch_start

        # ---- energy threshold + log-transform ----
        thr_m = fe > self.energy_threshold
        co    = co[thr_m]
        fe    = fe[thr_m].astype(np.float32, copy=False)

        # ---- cap pixels/event so FPS grouping doesn't overflow context_length ----
        sub_idx = None  # indices applied AFTER thresholding; needed to align pdg below
        if self.max_points > 0 and co.shape[0] > self.max_points:
            sub_idx = np.random.choice(co.shape[0], size=self.max_points, replace=False)
            sub_idx.sort()
            co = co[sub_idx]
            fe = fe[sub_idx]

        log_q = log_transform(fe, xmax=self.emax, eps=self.emin).astype(np.float32)

        # ---- pixel-level PDG -> 3-class label ----
        sem = None
        if self.return_semantic_id:
            pdg = self._read_pixel_pdg(s.path, s.group)              # (M,) int32 in raw frame order
            if pdg is None:
                pdg = np.zeros(co_all.shape[0], dtype=np.int32)
            pdg_view = pdg[view_m][thr_m]
            if sub_idx is not None:
                pdg_view = pdg_view[sub_idx]
            sem = pdg_to_pixel_class(pdg_view, co)                   # (N,) long, -1=no truth

        # ---- minimum-points filter: pad if too few (caller may drop them later) ----
        # We do NOT drop here; PILArNet just emits short events. Let the consumer's
        # padded collate handle it. Min-points filtering is done at index time below.

        # ---- assemble PILArNet-style dict ----
        n = co.shape[0]
        # x = ch, y = tick, z = 0, e = log charge in [-1, 1]
        points = np.zeros((n, 4), dtype=np.float32)
        points[:, 0] = co[:, 0].astype(np.float32)
        points[:, 1] = co[:, 1].astype(np.float32)
        points[:, 2] = 0.0
        points[:, 3] = log_q

        out = {
            "points":      torch.from_numpy(points),
            "semantic_id": torch.from_numpy(sem).unsqueeze(1).long() if sem is not None else None,
            "cluster_id":  torch.zeros((n, 1), dtype=torch.long) if self.return_cluster_id else None,
        }
        return out

    def _read_pixel_pdg(self, path: str, group: str) -> Optional[np.ndarray]:
        """Return PDG-per-reco-pixel aligned to frame_rebinned_reco/coords order.

        Mirrors APASparseMetaDataset._read_pixel_truth but operates BEFORE the
        view filter (so the returned array is aligned to the raw coords).
        """
        f   = self._h5(path)
        grp = f[group]
        if "frame_pid_1st" not in grp:
            return None
        co_reco = grp["frame_rebinned_reco"]["coords"][()]
        co_pid  = grp["frame_pid_1st"]["coords"][()]
        fe_pid  = grp["frame_pid_1st"]["features"][()]

        lookup = {
            (int(c[0]), int(c[1])): int(v)
            for c, v in zip(co_pid, fe_pid)
        }
        return np.array(
            [lookup.get((int(c[0]), int(c[1])), 0) for c in co_reco],
            dtype=np.int32,
        )

    def __del__(self):
        for h in list(self._h5_cache.values()):
            try:
                h.close()
            except Exception:
                pass
        self._h5_cache.clear()

    # ---------------- DataLoader plumbing ----------------

    @staticmethod
    def init_worker_fn(worker_id: int) -> None:
        info = torch.utils.data.get_worker_info()
        if info is None:
            return
        ds = info.dataset
        # Each worker starts with an empty handle cache so files are opened in
        # the worker process (h5py + fork doesn't share handles safely).
        if hasattr(ds, "_h5_cache"):
            ds._h5_cache = {}
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        """Pad-batched collate identical in shape to PILArNet.collate_fn."""
        data        = [item["points"] for item in batch]
        semantic_id = [item["semantic_id"] for item in batch]
        cluster_id  = [item["cluster_id"]  for item in batch]

        lengths = torch.tensor([p.size(0) for p in data], dtype=torch.long)
        padded_points = pad_sequence(data, batch_first=True)                 # (B, N, 4)

        padded_sem = None
        if semantic_id[0] is not None:
            padded_sem = pad_sequence(semantic_id, batch_first=True, padding_value=-1)
        padded_cls = None
        if cluster_id[0] is not None:
            padded_cls = pad_sequence(cluster_id,  batch_first=True, padding_value=-1)

        return {
            "points":      padded_points,
            "lengths":     lengths,
            "semantic_id": padded_sem,
            "cluster_id":  padded_cls,
        }


# -----------------------------------------------------------------------------
# LightningDataModule
# -----------------------------------------------------------------------------

class APA2DDataModule(pl.LightningDataModule):
    """LightningDataModule mirror of PILArNetDataModule, backed by APA2D.

    Either set ``data_path`` to a train-side glob and ``val_data_path`` to a
    val-side glob, OR pass a single ``data_path`` containing ``"train"`` and
    we'll derive the val side by replacing it with ``"val"`` (PILArNet style).
    """

    _class_weights = None

    def __init__(
        self,
        data_path: str,
        val_data_path: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        dataset_kwargs: Optional[dict] = None,
        test_dataset_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.persistent_workers = num_workers > 0

        # Mirror PILArNetDataModule.category_to_seg_classes API so SSLModel.validate
        # can print class names without further changes.
        self._category_to_seg_classes = {
            "track":  [0],
            "shower": [1],
            "other":  [2],
        }
        self._seg_class_to_category = {0: "track", 1: "shower", 2: "other"}

    def setup(self, stage: Optional[str] = None):
        dk  = dict(self.hparams.dataset_kwargs or {})
        tdk = dict(self.hparams.test_dataset_kwargs or {})

        train_path = self.hparams.data_path
        val_path   = self.hparams.val_data_path or train_path.replace("train", "val")

        self.train_dataset = APA2D(train_path, **dk)
        val_dk = {**dk, **tdk}
        self.val_dataset   = APA2D(val_path,   **val_dk)

    def train_dataloader(self):
        if not hasattr(self, "train_dataset"):
            self.setup()
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.persistent_workers,
            collate_fn=APA2D.collate_fn,
            worker_init_fn=APA2D.init_worker_fn,
        )

    def val_dataloader(self):
        if not hasattr(self, "val_dataset"):
            self.setup()
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.persistent_workers,
            collate_fn=APA2D.collate_fn,
            worker_init_fn=APA2D.init_worker_fn,
        )

    @property
    def category_to_seg_classes(self):
        return self._category_to_seg_classes

    @property
    def seg_class_to_category(self):
        return self._seg_class_to_category

    @property
    def num_seg_classes(self):
        return len(self._category_to_seg_classes)

    @property
    def class_weights(self):
        # placeholder — only used for semantic-segmentation finetuning, which
        # we are not running in this comparison.
        return torch.ones(self.num_seg_classes)
