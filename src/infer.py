# -*- coding: utf-8 -*-
"""CPU inference core for the app: nnU-Net-faithful sliding window + gaussian
blend + mirror TTA, using onnxruntime. No torch/nnunet at runtime."""
import numpy as np
from scipy.ndimage import gaussian_filter

PATCH = (1280, 1024)   # from nnUNetPlans 2d

def preprocess(img_u8):
    a = img_u8.astype(np.float64)
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    a = np.clip((a - lo) / max(hi - lo, 1), 0, 1)
    a = (a - a.mean()) / (a.std() + 1e-8)
    return a.astype(np.float32)

def _steps(size, patch, overlap=0.5):
    if size <= patch:
        return [0]
    n = int(np.ceil((size - patch) / (patch * overlap))) + 1
    step = (size - patch) / (n - 1)
    return [int(round(i * step)) for i in range(n)]

def _gaussian(shape, sigma_scale=1. / 8):
    tmp = np.zeros(shape, np.float32); tmp[shape[0] // 2, shape[1] // 2] = 1
    g = gaussian_filter(tmp, [s * sigma_scale for s in shape], mode='constant')
    g /= g.max()
    g = np.maximum(g, g.max() * 0.1)
    return g.astype(np.float32)

def _softmax(logits):
    e = np.exp(logits - logits.max(0, keepdims=True))
    return e / e.sum(0, keepdims=True)

def _predict_patch(sess, patch_img, mirror):
    """patch_img: (H,W) float32 -> prob (2,H,W), averaged over mirror flips."""
    flips = [()]
    if mirror:
        flips = [(), (2,), (3,), (2, 3)]
    acc = None
    for fl in flips:
        x = patch_img[None, None]
        for ax in fl:
            x = np.flip(x, ax)
        logit = sess.run(None, {'input': np.ascontiguousarray(x)})[0]
        for ax in fl:
            logit = np.flip(logit, ax)
        p = _softmax(logit[0])
        acc = p if acc is None else acc + p
    return acc / len(flips)

def infer(sess, img_u8, mirror=True, progress=None):
    """Return neurite probability map (H,W) in [0,1]."""
    x = preprocess(img_u8)
    H, W = x.shape
    ph, pw = PATCH
    padH, padW = max(0, ph - H), max(0, pw - W)
    xp = np.pad(x, ((0, padH), (0, padW)))
    Hp, Wp = xp.shape
    sh, sw = _steps(Hp, ph), _steps(Wp, pw)
    gmap = _gaussian((ph, pw))
    acc = np.zeros((2, Hp, Wp), np.float32)
    wacc = np.zeros((Hp, Wp), np.float32)
    total = len(sh) * len(sw); done = 0
    for sy in sh:
        for sx in sw:
            patch = xp[sy:sy + ph, sx:sx + pw]
            p = _predict_patch(sess, patch, mirror)
            acc[:, sy:sy + ph, sx:sx + pw] += p * gmap
            wacc[sy:sy + ph, sx:sx + pw] += gmap
            done += 1
            if progress:
                progress(done, total)
    prob = acc / np.maximum(wacc, 1e-8)
    return prob[1, :H, :W]


if __name__ == '__main__':
    import os, onnxruntime as ort
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    ONNX = os.path.join(os.path.dirname(__file__), "neurite_model.onnx")
    DS = "C:/Users/75690/nnUNet_raw/Dataset101_Neurite"
    sess = ort.InferenceSession(ONNX, providers=['CPUExecutionProvider'])
    # validate on a few test images vs nnU-Net pred
    cids = sorted(os.path.basename(f)[:-9] for f in os.listdir(DS + "/imagesTs"))[:3]
    import time
    for cid in cids:
        img = np.asarray(Image.open(DS + "/imagesTs/" + cid + "_0000.png").convert('L'))
        t = time.time()
        prob = infer(sess, img, mirror=True)
        dt = time.time() - t
        onnx_seg = prob > 0.5
        nn_seg = np.asarray(Image.open(DS + "/pred_Ts/" + cid + ".png")) > 0
        dice = 2 * (onnx_seg & nn_seg).sum() / (onnx_seg.sum() + nn_seg.sum() + 1e-8)
        print("%s: Dice(CPU-ONNX vs GPU-nnUNet)=%.4f  | %.1fs (mirror TTA, sliding)" % (cid, dice, dt))
