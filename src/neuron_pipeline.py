# -*- coding: utf-8 -*-
"""End-to-end: image -> segmentation -> SWC tree + soma -> morphometrics.
Single image (run) and whole-folder batch (run_batch). No torch/nnunet needed."""
import os, csv, glob, gc
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
import onnxruntime as ort
import infer as _infer
import tracer as _tracer
import soma as _soma

_SESS = None
IMG_EXTS = ('.png', '.tif', '.tiff', '.jpg', '.jpeg')
# derived/non-raw images to skip in batch (our outputs + common SNT/Fiji artifacts)
EXCLUDE_PAT = ('_mask', '_overlay', '_prob', 'traced', 'sholl', 'patching',
               'post collection', 'post_collection', 'profile', '_annotated')

def find_images(folder):
    """Recursively find candidate raw images in a folder, skipping our own
    outputs and common derived images (traced overlays, sholl, patching, ...).
    Hidden files (macOS '._name' resource forks, '.DS_Store') and '__MACOSX' folders are skipped too."""
    out = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d != '__MACOSX' and not d.startswith('.')]
        if '_neuron_out' in root.lower():
            continue
        for f in files:
            if f.startswith('.'):
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext in IMG_EXTS and not any(p in f.lower() for p in EXCLUDE_PAT):
                out.append(os.path.join(root, f))
    return sorted(out)

def _model_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "neurite_model.onnx")

def get_session():
    global _SESS
    if _SESS is None:
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
        so.enable_cpu_mem_arena = False      # release scratch memory between images (batch-safe)
        so.enable_mem_pattern = False
        _SESS = ort.InferenceSession(_model_path(), sess_options=so,
                                     providers=['CPUExecutionProvider'])
    return _SESS

def load_raw_float(path):
    """The image at its native bit depth as float32 (same channel choice as load_gray), for
    measurements that need the full dynamic range (neurite width). None if unreadable."""
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.tif', '.tiff'):
            import tifffile
            a = tifffile.imread(path)
            if a.ndim == 3:
                a = a[..., 0] if a.shape[-1] in (3, 4) else a[0]
            return a.astype(np.float32)
        im = Image.open(path)
        if im.mode in ('I;16', 'I;16B', 'I;16L', 'I', 'F'):
            return np.asarray(im, dtype=np.float32)
        return np.asarray(im.convert('L')).astype(np.float32)
    except Exception:
        return None


def load_gray(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff'):
        try:
            import tifffile
            a = tifffile.imread(path)
            if a.ndim == 3:
                a = a[..., 0] if a.shape[-1] in (3, 4) else a[0]
            a = a.astype(np.float64)
            a = (a - a.min()) / max(a.max() - a.min(), 1) * 255
            return a.astype(np.uint8)
        except Exception:
            pass
    return np.asarray(Image.open(path).convert('L'))

def overlay_rgb(gray_u8, tree_edges, soma_contour, soma_center, branch_points=None, tips=None, crossings=None):
    """Reconstruction overlay for auditing topology decisions. Neurites red; soma outlined red;
    soma centre cyan dot. Branch nodes = hollow YELLOW circles; terminal tips = small GREEN dots;
    points the tracer resolved as X-CROSSINGS (two neurites passing through, NOT a branch) = BLUE X.
    So a yellow ring = 'called a branch here', a blue X = 'called a crossing here'."""
    lo, hi = np.percentile(gray_u8, 0.5), np.percentile(gray_u8, 99.5)
    g = np.clip((gray_u8.astype(float) - lo) / max(hi - lo, 1), 0, 1)
    im = Image.fromarray((np.stack([g, g, g], -1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for (y0, x0), (y1, x1) in tree_edges:
        d.line([x0, y0, x1, y1], fill=(255, 40, 40), width=2)
    if soma_contour is not None:
        pts = [(x, y) for y, x in soma_contour]
        d.line(pts + [pts[0]], fill=(255, 40, 40), width=3)   # soma outlined red
    for (ty, tx) in (tips or []):                             # terminal tips: small green dots
        d.ellipse([tx - 4, ty - 4, tx + 4, ty + 4], fill=(50, 255, 90))
    for (by, bx) in (branch_points or []):                    # branch nodes: hollow yellow rings
        d.ellipse([bx - 8, by - 8, bx + 8, by + 8], outline=(255, 235, 0), width=3)
    for (xy, xx) in (crossings or []):                        # resolved crossings: blue X
        d.line([xx - 7, xy - 7, xx + 7, xy + 7], fill=(40, 120, 255), width=3)
        d.line([xx - 7, xy + 7, xx + 7, xy - 7], fill=(40, 120, 255), width=3)
    cy, cx = soma_center
    d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(0, 230, 255))
    return im

# metric order: (key, label, unit, note). note='preliminary' = topology-dependent
_METRIC_SPEC = [
    ("total_length_um", "total_length", "um", ""),
    ("num_branch_points", "num_branch_points", "count", ""),
    ("num_tips", "num_tips", "count", ""),
    ("num_branches", "num_branches", "count", ""),
    ("mean_intersections", "mean_intersections", "count", ""),
    ("enclosing_radius_um", "enclosing_radius", "um", ""),
    ("polarity_index", "polarity_index", "0-1", ""),
    ("soma_area_um2", "soma_area", "um^2", ""),
    ("soma_perimeter_um", "soma_perimeter", "um", ""),
    ("soma_circularity", "soma_circularity", "0-1", ""),
    ("soma_roundness", "soma_roundness", "0-1", ""),
    ("soma_solidity", "soma_solidity", "0-1", ""),
    ("soma_aspect_ratio", "soma_aspect_ratio", "ratio", ""),
    ("soma_diameter_um", "soma_diameter", "um", ""),
    ("neurite_width_um", "neurite_width", "um", "median FWHM across neurites, from the image; includes ~+0.2 um optical blur"),
    ("highest_branch_order", "highest_branch_order", "order", "preliminary; SNT path order (straightest daughter keeps the order)"),
    ("highest_branch_order_centrifugal", "highest_branch_order_centrifugal", "order", "preliminary; both daughters +1 at every bifurcation"),
    ("avg_branch_angle_deg", "avg_branch_angle", "deg", "preliminary; angle between the two daughters measured to their segment ends (SNT remote bifurcation angle)"),
    ("avg_parent_daughter_angle_deg", "avg_parent_daughter_angle", "deg", "preliminary; parent direction vs each daughter"),
    ("avg_len_before_first_bif_um", "avg_length_before_first_bifurcation", "um", "preliminary"),
    ("avg_segment_len_um", "avg_segment_length_before_bifurcation", "um", "preliminary"),
]

def _collect_metrics(res, soma_metrics, vox):
    m = dict(res)
    if soma_metrics:
        m.update(soma_metrics)
    m["pixel_size"] = vox
    return m

# Input-modality QC. Fluorescence images are mostly dark background with a bright
# cell, so the median sits near the bottom of the intensity range. Brightfield/DIC
# snapshots (patch-pipette photos etc.) are bright everywhere. Measured on the 93-cell
# batch96 set: fluorescence scores <= 0.24, brightfield >= 0.63 -> threshold 0.45.
# Tracing a brightfield image produces garbage morphometrics (verified: 5/25 of Liz's
# corrected cells were exactly this failure), so we refuse instead.
# 2026-10-18: the score is now computed after removing the slow illumination gradient
# (Gaussian blur, sigma 64 px, subtracted). Fluorescence images with a bright uneven
# background (CL_3_NTC_44, L1_14, L1_15) scored 0.53-0.66 before and were refused.
# Measured on 596 manifest images + the 5 known brightfield cells: fluorescence <= 0.34,
# brightfield 0.52-0.60, blank images ~0.97 -> the same threshold 0.45 still separates them.
FLUOR_QC_THRESHOLD = 0.45

def fluorescence_score(gray):
    """(median - p0.1) / (p99.9 - p0.1) of the image minus its smooth background;
    ~0.1-0.3 for fluorescence, ~0.5+ for brightfield, ~1 for blank images."""
    from scipy.ndimage import gaussian_filter, zoom
    a = gray.astype(np.float64)
    small = zoom(a, 0.25, order=1)
    bg = zoom(gaussian_filter(small, 16), 4, order=1)
    bg = np.pad(bg, ((0, max(0, a.shape[0] - bg.shape[0])), (0, max(0, a.shape[1] - bg.shape[1]))), mode='edge')[:a.shape[0], :a.shape[1]]
    r = a - bg
    lo = np.percentile(r, 0.1)
    top = np.percentile(r, 99.9)
    return float((np.median(r) - lo) / max(top - lo, 1e-9))

def run(image_path, vox=0.1625, mirror=True, out_dir=None, progress=None, qc=True):
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(image_path),
                               os.path.splitext(os.path.basename(image_path))[0] + "_neuron_out")
    os.makedirs(out_dir, exist_ok=True)
    gray = load_gray(image_path)
    if qc:
        score = fluorescence_score(gray)
        if score > FLUOR_QC_THRESHOLD:
            raise RuntimeError(
                "NOT FLUORESCENCE? image looks brightfield/DIC (QC score %.2f > %.2f). "
                "Tracing skipped - pick the fluorescence channel or rerun with QC off." % (score, FLUOR_QC_THRESHOLD))
    if progress:
        progress("Segmenting neurites (CPU)...", 0.05)
    sess = get_session()
    prob = _infer.infer(sess, gray, mirror=mirror,
                        progress=(lambda dn, tt: progress("Segmenting patch %d/%d..." % (dn, tt), 0.05 + 0.65 * dn / tt)) if progress else None)
    if progress:
        progress("Tracing tree and soma...", 0.78)
    soma_res = _soma.segment_soma(gray, vox)
    if soma_res:
        soma_center = soma_res["center"]
        soma_radius = float(np.sqrt(soma_res["mask"].sum() / np.pi))   # equivalent radius, px
        soma_bright = soma_res.get("bright_mask")
        soma_body = soma_res.get("mask")
    else:
        soma_center = _tracer.detect_soma(gray, vox); soma_radius = 90.0; soma_bright = None; soma_body = None
    res = _tracer.trace(prob, vox, soma_center, soma_radius, gray=gray, soma_mask=soma_bright, soma_body=soma_body,
                        gray_raw=load_raw_float(image_path))
    if res is None:
        raise RuntimeError("No neurite structure detected in this image.")
    soma_metrics = soma_res["metrics"] if soma_res else None
    soma_contour = soma_res["contour"] if soma_res else None
    if progress:
        progress("Saving outputs...", 0.9)
    base = os.path.splitext(os.path.basename(image_path))[0]
    Image.fromarray((res['mask'] * 255).astype(np.uint8)).save(os.path.join(out_dir, base + "_mask.png"))
    Image.fromarray((prob * 65535).astype(np.uint16)).save(os.path.join(out_dir, base + "_prob.png"))
    # normalized grayscale preview -> lets the annotator toggle the tracing on/off (it needs the raw layer)
    _lo, _hi = np.percentile(gray, 0.5), np.percentile(gray, 99.5)
    _g = np.clip((gray.astype(float) - _lo) / max(_hi - _lo, 1), 0, 1)
    Image.fromarray((_g * 255).astype(np.uint8)).save(os.path.join(out_dir, base + "_raw.png"))
    ov = overlay_rgb(gray, res['tree_edges_px'], soma_contour, res['soma_rc'],
                     res.get('branch_points_px'), res.get('tips_px'), res.get('crossings_px'))
    ov.save(os.path.join(out_dir, base + "_overlay.png"))
    # soma outline for the edit window (the SWC only holds the centre; the overlay burns the outline into pixels)
    if soma_contour is not None:
        import json
        with open(os.path.join(out_dir, base + "_soma.json"), 'w') as fh:
            json.dump({"contour_xy": [[round(float(x), 2), round(float(y), 2)] for y, x in soma_contour],
                       "center_xy": [float(res['soma_rc'][1]), float(res['soma_rc'][0])],
                       "radius_px": float(soma_radius), "pixel_size_um": vox}, fh)
    with open(os.path.join(out_dir, base + ".swc"), 'w') as fh:
        fh.write(res['swc'])
    allm = _collect_metrics(res, soma_metrics, vox)
    with open(os.path.join(out_dir, base + "_metrics.csv"), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value", "unit", "note"])
        for key, label, unit, note in _METRIC_SPEC:
            if key in allm:
                w.writerow([label, allm[key], unit, note])
    with open(os.path.join(out_dir, base + "_sholl.csv"), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(["radius_um", "intersections"])
        for r, n in zip(res['sholl_radii'], res['sholl_intersections']):
            w.writerow(["%.1f" % r, int(n)])
    if progress:
        progress("Done.", 1.0)
    res['out_dir'] = out_dir; res['overlay'] = ov; res['gray'] = gray
    res['soma_metrics'] = soma_metrics or {}
    res['all_metrics'] = allm
    return res

def run_batch(folder, out_root, vox=0.1625, mirror=True, progress=None):
    """Process every image in a folder. ALL outputs go under out_root (never into
    the source folder): one subfolder per cell + a combined summary.csv."""
    imgs = find_images(folder)
    os.makedirs(out_root, exist_ok=True)
    rows = []
    n = len(imgs)
    for i, img in enumerate(imgs):
        rel = os.path.splitext(os.path.relpath(img, folder))[0].replace(os.sep, '_').replace('/', '_')
        if progress:
            progress("Cell %d/%d: %s" % (i + 1, n, os.path.basename(img)), i / max(n, 1))
        try:
            out_dir = os.path.join(out_root, rel + "_neuron_out")
            res = run(img, vox=vox, mirror=mirror, out_dir=out_dir)
            row = {"cell": rel}
            for key, label, unit, note in _METRIC_SPEC:
                row[label] = res['all_metrics'].get(key, "")
            rows.append(row)
            # free everything this cell allocated before the next one (batch memory safety)
            res.clear()
            del res, row
        except MemoryError:
            rows.append({"cell": rel, "error": "out of memory on this image"})
        except Exception as e:
            rows.append({"cell": rel, "error": str(e)[:80]})
        gc.collect()
    # write combined summary
    cols = ["cell"] + [label for _, label, _, _ in _METRIC_SPEC] + ["error"]
    summary = os.path.join(out_root, "summary.csv")
    with open(summary, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    if progress:
        progress("Batch done: %d cells -> summary.csv" % n, 1.0)
    return dict(n=n, summary=summary, rows=rows, folder=out_root)


if __name__ == '__main__':
    import sys, time
    if '--batch' in sys.argv:
        folder = [a for a in sys.argv[1:] if not a.startswith('-')][0]
        t = time.time()
        r = run_batch(folder, folder.rstrip('/\\') + "_output", mirror=('--fast' not in sys.argv),
                      progress=lambda m, f: print("  [%3d%%] %s" % (int(f * 100), m)))
        print("batch %d cells in %.0fs -> %s" % (r['n'], time.time() - t, r['summary']))
    else:
        p = [a for a in sys.argv[1:] if not a.startswith('-')][0]
        t = time.time()
        r = run(p, mirror=('--fast' not in sys.argv),
                progress=lambda m, f: print("  [%3d%%] %s" % (int(f * 100), m)))
        print("length=%.1f branches=%d order=%d soma_area=%s | %.0fs -> %s" % (
            r['total_length_um'], r.get('num_branches', -1), r.get('highest_branch_order', -1),
            r['soma_metrics'].get('soma_area_um2', 'NA'), time.time() - t, r['out_dir']))
