import numpy as np


def hu_to_relative_electron_density(hu: np.ndarray) -> np.ndarray:
    hu = hu.astype(np.float32)

    rho = np.where(hu < -950, 0.0, np.nan)

    lung_mask = (hu >= -950) & (hu < -500)
    rho[lung_mask] = 0.2 + 0.3 * (hu[lung_mask] + 950) / 450.0

    soft_mask = (hu >= -500) & (hu < 100)
    rho[soft_mask] = 0.5 + 0.5 * (hu[soft_mask] + 500) / 600.0

    bone_mask = hu >= 100
    rho[bone_mask] = 1.0 + 1.5 * (np.minimum(hu[bone_mask], 2000) - 100) / 1900.0

    rho = np.nan_to_num(rho, nan=1.0)

    return rho.astype(np.float32)


def relative_electron_density_to_spr(rho: np.ndarray) -> np.ndarray:
    return rho.astype(np.float32)


def compute_wet_map(spr_vol: np.ndarray,
                    axis: int = 0,
                    voxel_mm: float = 2.0) -> np.ndarray:
    spr = spr_vol.astype(np.float32)

    spr_moved = np.moveaxis(spr, axis, 0)
    wet_moved = np.cumsum(spr_moved * voxel_mm, axis=0)
    wet = np.moveaxis(wet_moved, 0, axis)

    return wet.astype(np.float32)


def hu_volume_to_wet(hu_vol: np.ndarray,
                     axis: int = 0,
                     voxel_size_mm: float = 2.0) -> np.ndarray:
    rho = hu_to_relative_electron_density(hu_vol)
    spr = relative_electron_density_to_spr(rho)
    wet = compute_wet_map(spr, axis=axis, voxel_mm=voxel_size_mm)
    return wet