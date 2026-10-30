# -*- coding: utf-8 -*-
"""Export the trained nnU-Net 2D model to ONNX (CPU) and validate against
nnU-Net's own GPU prediction on one test image."""
import os, json
import numpy as np
import torch
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

PREP = "C:/Users/75690/nnUNet_preprocessed/Dataset101_Neurite"
FOLD = "C:/Users/75690/nnUNet_results/Dataset101_Neurite/nnUNetTrainer_250epochs__nnUNetPlans__2d/fold_0"
DS = "C:/Users/75690/nnUNet_raw/Dataset101_Neurite"
OUT = "C:/Users/75690/AppData/Local/Temp/claude/c--Users-75690-OneDrive----morphology-data-for-Dr--Tang-selected/ce27a5a6-618c-4c31-8bad-aa03adaa03a7/scratchpad/neurite_model.onnx"

plans = json.load(open(PREP + "/nnUNetPlans.json"))
cfg = plans['configurations']['2d']
arch = cfg['architecture']
print("building network:", arch['network_class_name'])
net = get_network_from_plans(arch['network_class_name'], arch['arch_kwargs'],
                             arch['_kw_requires_import'], 1, 2,
                             allow_init=True, deep_supervision=False)
ckpt = torch.load(FOLD + "/checkpoint_best.pth", map_location='cpu', weights_only=False)
net.load_state_dict(ckpt['network_weights'])
net.eval()
print("weights loaded (epoch %s, best EMA %.3f)" % (ckpt.get('current_epoch', '?'),
      ckpt.get('_best_ema') or -1))

dummy = torch.zeros(1, 1, 512, 512)
torch.onnx.export(net, dummy, OUT, input_names=['input'], output_names=['logits'],
                  dynamic_axes={'input': {0: 'b', 2: 'h', 3: 'w'}, 'logits': {0: 'b', 2: 'h', 3: 'w'}},
                  opset_version=17)
sz = os.path.getsize(OUT) / 1e6
print("ONNX exported: %.1f MB -> %s" % (sz, OUT))

# ---- validate on one test image ----
import onnxruntime as ort
sess = ort.InferenceSession(OUT, providers=['CPUExecutionProvider'])

def preprocess(img_u8):
    a = img_u8.astype(np.float64)
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    a = np.clip((a - lo) / max(hi - lo, 1), 0, 1)
    a = (a - a.mean()) / (a.std() + 1e-8)      # z-score (matches nnU-Net)
    return a.astype(np.float32)

def pad256(a):
    h, w = a.shape
    H = ((h + 255) // 256) * 256; W = ((w + 255) // 256) * 256
    out = np.zeros((H, W), np.float32); out[:h, :w] = a
    return out, (h, w)

# take a test image and compare argmax to nnU-Net's pred_Ts
cid = sorted(os.path.basename(f)[:-9] for f in os.listdir(DS + "/imagesTs"))[0]
img = np.asarray(Image.open(DS + "/imagesTs/" + cid + "_0000.png").convert('L'))
x = preprocess(img)
xp, (h, w) = pad256(x)
inp = xp[None, None]
logits = sess.run(None, {'input': inp})[0][0]     # (2,H,W)
prob = np.exp(logits) / np.exp(logits).sum(0, keepdims=True)
onnx_seg = (prob[1] > 0.5)[:h, :w]

nn_seg = np.asarray(Image.open(DS + "/pred_Ts/" + cid + ".png")) > 0
inter = (onnx_seg & nn_seg).sum()
dice = 2 * inter / (onnx_seg.sum() + nn_seg.sum() + 1e-8)
print("\nVALIDATION on %s:" % cid)
print("  ONNX(CPU) fg px %d | nnU-Net(GPU) fg px %d" % (onnx_seg.sum(), nn_seg.sum()))
print("  Dice(ONNX-CPU vs nnUNet-GPU) = %.4f" % dice)
print("  (>0.95 = export is faithful; small diff expected: no TTA/gaussian-window)")
