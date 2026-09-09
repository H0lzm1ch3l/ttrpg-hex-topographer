"""Aspect from the DEM, for when no gdaldem aspect raster is supplied."""

from __future__ import annotations

import numpy as np


def aspect_from_dem(fld) -> np.ndarray:
    """Compass bearing the ground faces (downhill), 0-360.

    Matches gdaldem's convention so a supplied raster and a computed one are
    interchangeable. Flat cells return -1 and are skipped by the hachure pass.
    """
    e = np.nan_to_num(fld.elev, nan=0.0).astype(np.float64)
    d_row, d_col = np.gradient(e, fld.m_per_px, fld.m_per_px)
    dz_de, dz_dn = d_col, -d_row
    flat = np.hypot(dz_de, dz_dn) < 1e-9
    asp = (np.degrees(np.arctan2(-dz_de, -dz_dn)) + 360.0) % 360.0
    return np.where(flat, -1.0, asp)
