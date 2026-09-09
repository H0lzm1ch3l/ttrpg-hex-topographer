"""Hex thumbnail rendering: hillshade under an elevation ramp, clipped to the hex.

Thumbnails are cached on the tile row.  Back/forward through roll history has to
be instant, and re-rendering means re-fetching.
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image

from .zonal import HexRaster, hillshade

# Coarse hypsometric ramp: (stop, r, g, b).  Deliberately muted -- the hillshade
# is doing the work of showing shape, and a loud ramp fights it.
RAMP = [
    (-0.02, (58, 84, 110)),    # water
    (0.00, (168, 186, 156)),
    (0.18, (146, 168, 122)),
    (0.40, (188, 180, 132)),
    (0.62, (176, 148, 112)),
    (0.82, (156, 138, 128)),
    (1.00, (238, 238, 240)),
]


def _ramp(t: np.ndarray) -> np.ndarray:
    stops = np.array([s for s, _ in RAMP])
    cols = np.array([c for _, c in RAMP], dtype=float)
    out = np.zeros(t.shape + (3,), dtype=float)
    for i in range(len(stops) - 1):
        lo, hi = stops[i], stops[i + 1]
        m = (t >= lo) & (t <= hi)
        if not m.any():
            continue
        f = ((t[m] - lo) / max(hi - lo, 1e-9))[:, None]
        out[m] = cols[i] * (1 - f) + cols[i + 1] * f
    out[t < stops[0]] = cols[0]
    out[t > stops[-1]] = cols[-1]
    return out


def render_hex_png(hr: HexRaster, size_px: int = 512,
                   sea_level: float = 0.0) -> bytes:
    """``size_px`` is the output *width*; height follows the hex's 2 : sqrt(3)
    proportions so a flat-top hex renders flat-top."""
    inside = hr.inside
    lo, hi = float(inside.min()), float(inside.max())
    span = max(hi - lo, 1.0)

    t = (hr.elev - lo) / span
    t = np.where(hr.elev <= sea_level, -0.05, t)
    rgb = _ramp(t)

    hs = hillshade(hr)[..., None]
    rgb = np.clip(rgb * (0.45 + 0.75 * hs), 0, 255)

    alpha = np.where(hr.mask, 255, 0).astype(np.uint8)
    img = np.dstack([rgb.astype(np.uint8), alpha])

    im = Image.fromarray(img, mode="RGBA")
    target = (size_px, max(int(round(size_px * math.sqrt(3) / 2.0)), 1))
    if im.size != target:
        im = im.resize(target, Image.LANCZOS)

    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
