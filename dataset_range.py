import os
import glob
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from physics import hu_volume_to_wet

GRID_SHAPE = (128, 128, 128)
CT_CACHE = {}
WET_TRUE_CACHE = {}
WET_BASE_CACHE = {}


def load_sparse_tensor_csv(csv_path: str):
    data = np.genfromtxt(csv_path, delimiter=",", dtype=np.float32, skip_header=1)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] == 1:
        idx = data[:, 0].astype(np.int64)
        vals = np.ones_like(idx, dtype=np.float32)
    else:
        idx = data[:, 0].astype(np.int64)
        vals = data[:, 1].astype(np.float32)

    vals = np.nan_to_num(vals, nan=0.0)
    return idx, vals


def load_ct_volume_from_sparse(pat_dir: str) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    ct_path = os.path.join(pat_dir, "ct.csv")
    d, h, w = GRID_SHAPE
    total = d * h * w

    idx, vals = load_sparse_tensor_csv(ct_path)

    mask = (idx >= 0) & (idx < total)
    idx = idx[mask]
    vals = vals[mask]

    ct_flat = np.zeros(total, dtype=np.float32)
    ct_flat[idx] = vals

    ct_vol = ct_flat.reshape(GRID_SHAPE, order="C")
    return ct_vol, GRID_SHAPE


def compute_simple_baseline_wet(ct_vol: np.ndarray,
                                axis: int = 0,
                                voxel_mm: float = 2.0) -> np.ndarray:
    d, h, w = ct_vol.shape
    
    body_mask = (ct_vol > -300).astype(np.float32)

    if axis != 0:
        raise NotImplementedError("Only axis=0 supported for now")

    depth_idx = np.arange(d, dtype=np.float32).reshape(d, 1, 1)
    depth_mm = depth_idx * voxel_mm
    baseline_wet = depth_mm * body_mask

    return baseline_wet.astype(np.float32)


class OpenKBPRangeSliceDataset(Dataset):
    def __init__(self,
                 root: str,
                 split: str = "train",
                 axis: int = 0,
                 voxel_mm: float = 2.0,
                 margin: int = 5,
                 max_pts: int | None = None):

        self.axis = axis
        self.voxel_mm = voxel_mm
        self.margin = margin

        base = "train-pats" if split == "train" else "valid-pats"
        pats_dir = os.path.join(root, "provided-data", base)

        self.patient_dirs: List[str] = sorted(glob.glob(os.path.join(pats_dir, "pt_*")))
        if max_pts is not None:
            self.patient_dirs = self.patient_dirs[:max_pts]

        self.ct_shape = GRID_SHAPE
        d, _, _ = GRID_SHAPE

        self.indices: List[Tuple[int, int]] = []
        for p in range(len(self.patient_dirs)):
            for s in range(d):
                self.indices.append((p, s))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        p_idx, s_idx = self.indices[idx]
        p_dir = self.patient_dirs[p_idx]

        if p_dir in CT_CACHE:
            ct_vol = CT_CACHE[p_dir]
            wet_true = WET_TRUE_CACHE[p_dir]
            wet_base = WET_BASE_CACHE[p_dir]
        else:
            ct_vol, _ = load_ct_volume_from_sparse(p_dir)
            wet_true = hu_volume_to_wet(ct_vol, axis=self.axis, voxel_size_mm=self.voxel_mm)
            wet_base = compute_simple_baseline_wet(ct_vol, axis=self.axis, voxel_mm=self.voxel_mm)

            CT_CACHE[p_dir] = ct_vol
            WET_TRUE_CACHE[p_dir] = wet_true
            WET_BASE_CACHE[p_dir] = wet_base

        d, h, w = ct_vol.shape

        if s_idx < self.margin or s_idx >= d - self.margin:
            s_idx = d // 2

        ct_slice = ct_vol[s_idx]
        wet_true_slice = wet_true[s_idx]
        wet_base_slice = wet_base[s_idx]

        ct_slice = np.clip(ct_slice, -1000.0, 2000.0)
        ct_slice = (ct_slice - 500.0) / 1500.0

        wet_true_cm = wet_true_slice / 10.0
        wet_base_cm = wet_base_slice / 10.0

        wet_true_norm = wet_true_cm / 20.0
        wet_base_norm = wet_base_cm / 20.0

        ct_t = torch.from_numpy(ct_slice.astype(np.float32)).unsqueeze(0)
        wet_true_t = torch.from_numpy(wet_true_norm.astype(np.float32)).unsqueeze(0)
        wet_base_t = torch.from_numpy(wet_base_norm.astype(np.float32)).unsqueeze(0)

        return ct_t, wet_true_t, wet_base_t