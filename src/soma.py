# -*- coding: utf-8 -*-
"""Segment the soma (cell body) and measure its shape.

The soma is a compact bright blob. Two failure modes are handled explicitly:
  * large out-of-focus flare / pipette glare that is brighter and bigger than the
    soma -> the centre is found with a white top-hat (on a downsampled image),
    which suppresses bright regions larger than a soma so the compact soma wins;
  * the thin patch-pipette streak and proximal dendrites bridging out of the soma
    -> severed with a scale-adaptive opening (structuring element ~0.45 * the soma
    radius from the distance transform), so shape (solidity/roundness) is kept.

Within a local window around the centre an adaptive (Otsu, floored above the local
background) threshold captures the true boundary rather than just the saturated
core. Validated against the lab's golden manual tracing on 121 held-out cells:
area r=0.89, perimeter r=0.85, roundness r=0.76 (all essentially unbiased).

Returns a mask, a contour for drawing, the soma centre, and the shape metrics the
lab captures in Fiji (area, perimeter, circularity, roundness, solidity, diameter).
"""
import numpy as np
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import opening, disk, white_tophat, binary_dilation
from skimage.filters import threshold_otsu, gaussian
from scipy.ndimage import binary_fill_holes, distance_transform_edt, label as ndlabel

SOMA_AREA_MAX_UM2 = 1600.0   # a real iN soma is < ~1200 um^2; larger => glare bridged in



REF_VOX = 0.1625


def _px(n, vox):
    """pixel constant `n` was tuned at REF_VOX um/px; rescale it to `vox` (rounded, at least 1; exactly n at the reference)."""
    return max(1, int(round(n * REF_VOX / float(vox))))

def bright_soma_region(g, cy, cx, vox, frac=0.40, dilate=6):
    """The bright saturated soma blob + halo (largest bright region containing the centre). The neurite
    skeleton inside this glows-out region is saturation noise that spuriously links proximal neurites
    into one web -> the tracer excises it so each neurite can attach to the soma independently."""
    H, W = g.shape
    R = int(48 / vox)
    y0, y1 = max(int(cy) - R, 0), min(int(cy) + R, H)
    x0, x1 = max(int(cx) - R, 0), min(int(cx) + R, W)
    crop = g[y0:y1, x0:x1].astype(float)
    if crop.size == 0:
        return np.zeros((H, W), bool)
    pk = np.percentile(crop, 99.5); bg = np.median(crop)
    m = crop >= (bg + frac * (pk - bg))
    lab, n = ndlabel(m)
    full = np.zeros((H, W), bool)
    if n:
        cl = lab[int(cy) - y0, int(cx) - x0]
        if cl == 0:                                        # centre not in a bright blob: take the largest
            sizes = np.bincount(lab.ravel()); sizes[0] = 0; cl = int(sizes.argmax())
        full[y0:y1, x0:x1] = (lab == cl)
        full = binary_dilation(full, disk(_px(dilate, vox)))
    return full


def _regionprop_val(p, *names):
    for n in names:
        if hasattr(p, n):
            return getattr(p, n)
    raise AttributeError(names)


def soma_center(g, vox):
    """Brightest COMPACT blob, glare-suppressed via white top-hat on a downsampled image."""
    ds = 4
    small = g[::ds, ::ds].astype(float)
    r_px = max(6, int((22.0 / vox) / ds))          # ~ soma radius in downsampled px
    th = white_tophat(small, disk(r_px))
    hi = th >= max(np.percentile(th, 99.3), 0.12 * th.max())
    hi = opening(hi, disk(_px(2, vox)))
    lab = label(hi)
    if lab.max() == 0:
        cy, cx = np.unravel_index(int(np.argmax(small)), small.shape)
        return int(cy * ds), int(cx * ds)
    props = regionprops(lab, intensity_image=th)
    p = max(props, key=lambda r: _regionprop_val(r, 'intensity_mean', 'mean_intensity') * (r.area ** 0.5))
    cy, cx = p.centroid
    return int(cy * ds), int(cx * ds)


def segment_soma(gray_u8, vox, center=None, win_um=45, open_px=3):
    g = gaussian(gray_u8.astype(float), 1.0, preserve_range=True)
    H, W = g.shape
    if center is None:
        center = soma_center(g, vox)
    cy, cx = int(center[0]), int(center[1])
    R = int(win_um / vox)
    y0, y1 = max(cy - R, 0), min(cy + R, H)
    x0, x1 = max(cx - R, 0), min(cx + R, W)
    crop = g[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    ccy, ccx = cy - y0, cx - x0
    try:
        t_otsu = threshold_otsu(crop)
    except Exception:
        t_otsu = np.percentile(crop, 90)
    bg = np.median(np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]]))

    def nearest_comp(lab):
        best = None
        for p in regionprops(lab):
            d = np.hypot(p.centroid[0] - ccy, p.centroid[1] - ccx)
            if best is None or d < best[0]:
                best = (d, p)
        return best[1] if best else None

    def seg_at(frac):
        thr = max(t_otsu, bg + frac * (crop.max() - bg))
        m = binary_fill_holes(crop >= thr)
        m = opening(m, disk(_px(open_px, vox)))
        lab1 = label(m)
        if lab1.max() == 0:
            return None
        p0 = nearest_comp(lab1)
        if p0 is None:
            return None
        m1 = (lab1 == p0.label)
        # scale-adaptive opening: sever dendrites / pipette thinner than the soma bulk
        dt = distance_transform_edt(m1)
        op = int(max(_px(open_px, vox), round(0.45 * float(dt.max()))))
        m2 = binary_fill_holes(opening(m1, disk(op)))
        lab2 = label(m2)
        if lab2.max() == 0:
            lab2, m2 = lab1, m1
        p = nearest_comp(lab2)
        if p is None:
            return None
        return p, m2

    res = seg_at(0.35)
    frac = 0.35
    while res is not None and res[0].area * vox * vox > SOMA_AREA_MAX_UM2 and frac < 0.8:
        frac += 0.12
        res = seg_at(frac)
    if res is None:
        return None
    p, mm = res

    area = p.area * vox * vox                              # um^2
    perim = float(p.perimeter) * vox                       # um
    circ = min(1.0, 4 * np.pi * p.area / (p.perimeter ** 2 + 1e-8))
    major = float(p.axis_major_length); minor = float(p.axis_minor_length)
    roundness = min(1.0, 4.0 * p.area / (np.pi * major ** 2 + 1e-8))
    diam = float(_regionprop_val(p, 'equivalent_diameter_area', 'equivalent_diameter')) * vox
    aspect = major / max(minor, 1e-6)
    metrics = {
        "soma_area_um2": round(area, 2),
        "soma_perimeter_um": round(perim, 2),
        "soma_circularity": round(circ, 3),
        "soma_roundness": round(roundness, 3),
        "soma_solidity": round(float(p.solidity), 3),
        "soma_aspect_ratio": round(aspect, 3),
        "soma_diameter_um": round(diam, 2),
    }
    mask = np.zeros((H, W), bool)
    mask[y0:y0 + mm.shape[0], x0:x0 + mm.shape[1]] = mm
    contours = find_contours(mask.astype(float), 0.5)
    contour = max(contours, key=len) if contours else None   # (row,col) points
    flag = area > SOMA_AREA_MAX_UM2
    # centre = centroid of the segmented soma (STABLE); the coarse detector point drifts with glare
    ys, xs = np.nonzero(mask)
    ccy, ccx = (float(ys.mean()), float(xs.mean())) if ys.size else (float(cy), float(cx))
    bright = bright_soma_region(g, ccy, ccx, vox)            # region to excise from the neurite skeleton
    return dict(mask=mask, metrics=metrics, contour=contour, center=(ccy, ccx),
                bright_mask=bright, flag_large=flag)
