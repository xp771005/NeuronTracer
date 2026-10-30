# NeuronTracer tracer, release "New" (2026-10-15): Steps 1 to 3 (v3) plus Step 6 turn-back rule over the whole image (TURNBACK_SOMA_UM = 10000), all ON by default. Same model as before.
# -*- coding: utf-8 -*-
"""Probability map -> connected SWC arbor + morphometrics + Sholl.
Tracer v1.2: evidence-based conservative bridging (keeps a forest, like SNT's
own traces) + clustered-branch-point collapse. Validated against SNT .traces
ground truth (branch-point F1 0.42 -> 0.48, precision 0.34 -> 0.51 vs v1.1).
Standalone: numpy / scipy / scikit-image / networkx only."""
import numpy as np
from skimage.morphology import skeletonize, binary_closing, binary_dilation, opening, disk
from skimage.filters import apply_hysteresis_threshold
from scipy.ndimage import label as cc_label, distance_transform_edt, convolve as _convolve, maximum_filter as _max_filter
from scipy.spatial import cKDTree
import networkx as nx

LOW, HIGH = 0.2, 0.5
_STASH = None            # debug: set to {} before trace() to capture forest/pts/hub/roots
MIN_COMP = 15
MIN_COMP_EARLY = 4       # DEFER_DROP: kill only tiny noise specks up front, keep short REAL fragments
DEFER_DROP = False       # alive so the bridges/stitch can reconnect them; drop still-isolated ones after.
                         # Tested: +1 Liz but +1 golden branch/tip (noise fragments stitch too) -> net wash;
                         # left OFF for cleanest golden. Flip True to trade a little precision for recall.
SPUR = 20                # px, prune leaf spurs shorter than this (false tips + short false branch arms).
                         # 14->20 measured: golden branches & tips DOWN toward golden, length stable, and
                         # Liz continuity UP (removes spurious tips sitting on her gap marks).
# --- prob-guided mask healing: the nnU-Net predicts FAINT signal across many real neurite gaps, but the
# hysteresis threshold cuts it -> a break, a false tip, and a false branch. Heal the gap in the MASK
# (grow each dangling tip along the least-cost prob path until it rejoins a neurite) BEFORE skeletonizing,
# so the neurite becomes one continuous line with no false tip/branch. Only where the model actually
# predicted a continuous faint path (bottleneck prob >= floor) -> evidence-based, low false-merge risk. ---
PROB_HEAL = True
HEAL_GAP = 150           # px, max tip->neurite gap to heal
HEAL_LEN = 120           # px, max healed path length
HEAL_FLOOR = 0.12        # ridge (max-pooled prob) bottleneck must stay >= this; only faint-BUT-continuous
                         # gaps get healed. 96% of Liz's flagged gaps are truly dark (prob<0.05) in the
                         # gap -> a SEGMENTATION limit no post-proc can fix; heal catches the few with a ridge.
BRIDGE_MAX = 45          # px, only bridge short gaps
BRIDGE_PROB = 0.12       # mean probability along a bridge must exceed this (evidence)
# --- direction-continuity bridge: connect broken fragments a human would join by "good
# continuation" (aligned tangents + short gap), even where the image has little signal. Runs
# AFTER the evidence bridge, so signal-backed gaps are joined first. Conservative thresholds
# (both endpoint tangents must point at each other) to limit false links from the 2D projection. ---
CONTINUITY_BRIDGE = True
CONT_MAX_GAP = 110       # px, max endpoint gap to consider (swept: 110/0.55 = good balance; 150/0.50
CONT_MIN_COS = 0.55      #   cut fragments the most but doubles the gap -> more false-merge risk the
                         #   aggregate guards can't see. 110/0.55: -33% fragments, branch/length stable.
CONT_DIR_STEPS = 6       # skeleton steps used to estimate an endpoint's outgoing direction
# --- endpoint-to-neurite join: most continuity breaks leave a dangling tip whose continuation is the
# MIDDLE of another traced neurite (or a branch), not a second dangling tip -- the endpoint-to-endpoint
# bridge can't see those. So a second pass connects a tip to the nearest point on ANOTHER neurite when
# the tip clearly points at it (good continuation) over a short gap. Guarded to avoid false merges. ---
CONT_JOIN = False        # PARKED: tested vs Liz's 14-cell ground truth -- endpoint-to-neurite joins
CONT_JOIN_GAP = 130      #   over-connect and INFLATE branches (br 82->96 on n278) without cleanly
CONT_JOIN_COS = 0.60     #   fixing her gaps. Superseded by FRAG_STITCH below (fragment-scoped, safer).
# --- fragment stitch: Liz's breaks are SHORT (median 17px=2.7um; 89% <=60px) and 78% leave a DISCRETE
# isolated fragment -- a floating piece is definitely wrong (real broken neurite or noise), so attach it
# by GEOMETRY (no prob signal needed; the gap is dark anyway). Attach each dangling endpoint to the
# nearest node of ANOTHER component: <=STITCH_SNAP px snap unconditionally (a 2-3um gap to a floating
# piece is the same neurite); up to STITCH_GAP with a direction gate. Mid-neurite (branch-making) attach
# ONLY when very short, so endpoint->endpoint (no new branch) is preferred for longer gaps. Validated vs
# Liz's 14 cells: recovery 28->38/82 (+10), golden length stable (0.96), free tips DOWN. ---
FRAG_STITCH = True
STITCH_SNAP = 18         # px, snap a fragment across a gap this short regardless of angle
STITCH_GAP = 60          # px, max gap for a directional endpoint->endpoint stitch (adds NO branch)
STITCH_COS = 0.25        # min cos(tip tangent, gap direction) for endpoint->endpoint beyond SNAP
STITCH_TGAP = 40         # px, max gap for an endpoint->mid-span T-junction (an L_JUNCTION: a real side
STITCH_TCOS = 0.45       #   branch broken off). Needs a STRONGER direction gate (the tip must clearly
STITCH_BRANCH_PEN = 2.0  #   point at the neurite) since a T adds a branch -- guards against false merges.
STITCH_FRAG_SIZE = 40    # a component <= this many skeleton nodes is a FLOATING FRAGMENT: it MUST attach
STITCH_FRAG_COS = -0.20  #   to something, so relax its endpoint->endpoint direction gate (a short break to
                         #   a floating piece is the same neurite even if the tip's tangent estimate is noisy).
# --- soma-edge attach: a neurite broken right at the soma (buried in the bright glow) leaves a dangling
# tip near the soma surface that the proximal-stub attach misses. Attach any near-soma endpoint whose
# tangent points AT the soma to the hub -- but only if its component isn't already soma-connected, so we
# never create a LOOP (a genuinely soma-connected arbor is left alone). Angle-gated to avoid grabbing a
# distal neurite that merely passes near the soma. ---
SOMA_ANGLE_ATTACH = True
SOMA_ATTACH_R_MULT = 2.0  # attach endpoints within this * excised soma radius
SOMA_ATTACH_COS = 0.20    # endpoint tangent . (toward soma) must exceed this (points back at the soma)
EXCISE_FULL_SOMA = False  # (tried for near-soma tips; no effect -- the chain-rooting is arbor-wide, not a
SOMA_EXCISE_PAD = 6       #  local soma tangle. The real fix is multi-spoke, see SOMA_ONE_SPOKE below.)
SOMA_SEVER_DISK = False   # (tried: excising a disk just deletes near-soma neurite, doesn't reconnect)
SOMA_SEVER_MULT = 1.35
SOMA_RADIAL_CUT = False    # (tried: cutting tangential near-soma edges; no effect on the detour tips)
SOMA_RADIAL_R_MULT = 1.9
SOMA_RADIAL_COS = 0.34
SOMA_NEAR_MULTISPOKE = True  # one-spoke keeps continuity but leaves near-soma proximal ends dangling as
SOMA_NEAR_MULT = 1.5         # tips (rooted through a long detour). So within this*excise_r, ALSO give every
                             # dangling stub its own hub spoke (radial attach); the MST prefers the short
                             # direct spoke over the long detour and drops the (false) near-soma cross-link.
                             # Scoped tight so it doesn't manufacture the far breaks full multi-spoke does.
SOMA_ONE_SPOKE = True     # attach each in-zone component to the soma with exactly ONE spoke (its best
                          # point), not every proximal stub. Multiple spokes on one component form a hub
                          # cycle whose weighted MST then cuts a real neurite edge = a soma break we made.
SOMA_LOOP_BREAK = True    # if a near-soma tip's component is ALREADY soma-connected, attach it anyway and
                          # break ONE edge -- the component's FARTHEST existing hub spoke (a redundant
                          # spoke, never a real neurite), so the marked tip roots directly at the soma
                          # (user: "如果是环的话需要断开一个"). The tip stays connected -> no new break.
# --- STEP 1 (2026-10-11): soma HAIRPIN cut. Two adjacent primary neurites can have their soma-facing ends
# joined by a short U-shaped skeleton chain whose innermost point (the apex) sits at the soma edge. Neither
# end is a dangling stub, so SOMA_ONE_SPOKE / SOMA_NEAR_MULTISPOKE never give them a hub spoke, and each
# primary reaches the soma only by a long detour (path/straight 4-14x on the lab's manual primaries; e.g.
# CL_2_NTC_136 8 manual primaries, only 4 own spokes). Cut the chain at its apex; the two new stubs are then
# attached by SOMA_NEAR_MULTISPOKE. Apex = a degree-2 node, within HAIRPIN_ZONE*excise_r of the hub, that is the
# minimum of the distance to the hub over a window of HAIRPIN_ARM_PX skeleton steps on each side, with both
# window ends at least HAIRPIN_RISE_MULT*excise_r farther out (so a neurite hugging the soma edge is not cut).
SOMA_HAIRPIN_CUT = True  # ON in the released tracer (Step 1)
                          # MST cut each cycle at an ARBITRARY heaviest edge = a break in the middle of a neurite
                          # (CL_2_NTC_136: tips 32 -> 36, two manufactured breaks). Superseded by SOMA_DETOUR_REROOT.
HAIRPIN_ZONE = 1.5        # apex must lie within this * excise_r of the hub (= the NEAR_MULTISPOKE reach)
HAIRPIN_ARM_PX = 14       # skeleton steps walked on each side of the candidate apex
HAIRPIN_RISE_MULT = 0.25  # both arms must end at least this * excise_r farther from the hub than the apex
# --- STEP 1 v2: soma DETOUR re-root. A primary neurite's soma-facing end can reach the hub only by a long detour
# (outward along the neurite, across a false junction/crossing, then back inward along another neurite): tree path
# length / straight distance > DETOUR_MIN for a node within DETOUR_ZONE*excise_r of the hub. Rule: primaries connect
# to the soma. Fix, applied on the FINAL tree: take the node p nearest the hub among those detoured nodes, give it its
# own hub spoke, and cut the detour path at its FARTHEST point from the hub (the false junction where the two neurites
# were linked) -- a deterministic cut at the junction, not an MST-chosen edge in the middle of a neurite. The piece that
# contains p is re-rooted at p so every path runs outward from the soma. Repeats until no detoured node is left.
SOMA_DETOUR_REROOT = False
DETOUR_ZONE = 2.2         # only nodes within this * excise_r of the hub can be a primary's soma end
DETOUR_MIN = 3.0          # tree path / straight distance above this = detour (manual primaries: 4-14x before the fix)
DETOUR_MIN_EXTRA = 0.3    # ...and the path must exceed the straight distance by at least this * excise_r (tiny ratios)
DETOUR_MAX_ITER = 80
# --- STEP 3 (lab rule 3: branches should never trace from outward to inward, toward the soma). Only the RETURN PATH NEAR THE SOMA is handled; outer wavy
# neurites need not get monotonically farther. A near-soma end e that is reached from a farther node p by a chain whose distance to the hub keeps shrinking
# (net >= INWARD_MIN_NET_UM) has a missing/wrong proximal connection. Two readings are tested with evidence, and the one with the shorter gap wins:
#   'soma' : e is the proximal end of an independent primary: its tangent points at the soma (cos >= INWARD_DIR_COS), no hub spoke already within INWARD_DUP_UM,
#            mean probability along e -> soma boundary >= INWARD_PROB_MIN
#   'join' : e is a branch of an existing neurite: a node q of another path within INWARD_JOIN_UM, the tangent points at q (cos >= INWARD_JOIN_COS), probability ok
# Nothing is deleted: the old far entry edge (p, first node) is cut, the chain is re-rooted at e, e is hung on hub or q. Insufficient evidence = unchanged.
FIX_INWARD_NEAR_SOMA = True
TURNBACK_SOMA_UM = 10000.0        # Step 6: an end within this distance of the soma centre (um) that is >= TURNBACK_RISE_UM closer to the soma than the start of its own stretch is a turn-back (None = off)
TURNBACK_RISE_UM = 10.0
TURNBACK_JOIN_UM = 2.0         # a reliable new start: another path within this distance of the end ...
TURNBACK_JOIN_PROB = 0.2       # ... with at least this mean neurite probability on the straight gap
PIECE_CONT_UM = 10.0           # Step 7 (2026-10-25): a separate piece whose root lies within this of a main-tree END, with the two
PIECE_CONT_DEG = 45.0          #   directions collinear within this angle, is a continuation -> joined end to end (6 of 6 real on 115 cells)
PIECE_SIDE_UM = 5.0            # a piece root within this of the main tree's SIDE is only a CANDIDATE (half were wrong): listed for the editor
PIECE_DROP_UM = 5.0            # pieces shorter than this are deleted (15 fake vs 2 real on 115 cells)
INWARD_ZONE_UM = 35.0        # e must lie within this of the soma surface (excise_r + this)
INWARD_MIN_NET_UM = 10.0
INWARD_DIR_PX = 20           # walk this many px back along the chain to get the heading at e
INWARD_DIR_COS = 0.5
INWARD_PROB_MIN = 0.10
INWARD_FREE_UM = 2.0         # connections shorter than this need no probability evidence
INWARD_DUP_UM = 3.0
INWARD_JOIN_UM = 6.0
INWARD_JOIN_COS = 0.3
INWARD_MAX_ITER = 60
INWARD_MASK_AWARE = False      # A: image evidence only on the part of the segment OUTSIDE the excised soma region (the model cannot see inside it); an end inside the region is not connected
INWARD_ALL_ARMS = False        # B: every arm meeting at an interior/branch end is evaluated separately (own heading, own radial approach)
INWARD_SOMA_FREE_UM = 6.0     # gap (um, end -> soma boundary) up to which the SOMA option needs no image evidence (None = INWARD_FREE_UM). The model cannot see the glow next to the soma.
INWARD_RIDGE_LAT = 3          # px, lateral offset for the raw-image ridge test
INWARD_RIDGE_FRAC = 0.75      # fraction of samples whose centre is brighter than both sides
INWARD_SLOPE_MIN = 0.4        # B: radial approach per px of arm (excludes neurites that pass the soma tangentially)
DETOUR_DIR_COS = 0.5      # direction gate (None = off): heading of the neurite running into p must point at the hub, cos >= this
DETOUR_DIR_PX = 25        # px of neurite walked away from p to measure that heading
# --- STEP 1 v3: soma BRIDGE cut. A long straight single edge lying along the soma edge is not skeleton (adjacent skeleton pixels are
# 1-1.4 px apart): it is a bridge drawn by a stitch step between the soma-facing ends of two neighbouring primaries. Checked against the
# lab's manual traces on the 11 test cells: 5 of 7 such 2-node edges are covered by TWO different manual paths, one per end (= two
# separate neurites joined by the bridge). Delete them; the two ends then get their own hub spokes (NEAR_MULTISPOKE / DETOUR_REROOT).
# A multi-node tangential chain is skeleton (a real neurite hugging the soma) and is never touched.
SOMA_BRIDGE_CUT = True
BRIDGE_ZONE = 1.5         # both ends within this * excise_r of the hub
BRIDGE_TAN_COS = 0.5      # |cos(edge direction, radial direction)| below this = tangential
BRIDGE_MIN_EX = 0.3       # edge length >= this * excise_r
BRIDGE_MIN_PX = 6.0       # ...and at least this many px (a skeleton edge is <= 1.42 px)
# --- STEP 1 v3.1 (from CL_2_NTC_77, user: "split, but one line too many, only two should be there"). Two more limits:
# (a) DETOUR_REROOT may only give a spoke to a node at most DETOUR_MAX_GAP_UM from the soma SURFACE. Without it, it spoke-linked a node
#     14.7 um out (a long straight line through the glow = a bridge). New spokes of v3 on the 11 cells: all 13 within 2.1-6.8 um had a manual
#     primary root nearby; of the 3 beyond 12 um two had none (CL_1_NTC_161 12.1, CL_2_NTC_77 16.5), one did (CL_3_NTC_46 13.5, lost by this cap).
# (b) SOMA_SPOKE_MERGE: a hub child whose whole subtree is small (< SPOKE_SMALL_NODES nodes: a stub of the skeleton fork beside a main trunk) and that
#     lies within SPOKE_MERGE_UM of a bigger hub child's attach point does not get its own spoke: it is hung on the nearest node of that bigger subtree.
#     CL_2_NTC_77: a 45-node stub 2.3 um beside the 276-node trunk had its own spoke = the third line of the fan.
DETOUR_MAX_GAP_UM = 10.0
SOMA_SPOKE_MERGE = False  # OFF (2026-10-11): in CL_2_NTC_77 the 45-node 'stub' beside the trunk is the real end of a manual primary (3 manual paths reach
                          # the soma there: root3/4/5). Merging or deleting it removed a real primary. v3.1 (merge not triggering) was right for 77.
SPOKE_SMALL_NODES = 80
SPOKE_MERGE_UM = 5.0      # 4.0 missed CL_2_NTC_77: its 45-node stub is 4.6 um (28 px) from the 276-node trunk spoke
SPOKE_SMALL_DROP = True   # delete the small stub instead of hanging it on the trunk (hanging it left a visible third line, v3.2)
# --- hair removal: a short side-arm (a "hair") hanging off a node whose OTHER two arms are both
# long main lines is a skeletonization bump on a continuous neurite, not a real branch. It survives
# SPUR (just over 14px) and shows up as a spurious yellow-branch + green-tip pair sitting on an
# otherwise continuous line. Remove it. Guarded by HAIR_MAIN (both siblings must be long main lines)
# so genuine bifurcations of two comparable branches are never touched. ---
HAIR_STUB = 26           # px, a leaf side-arm this short or shorter is a candidate hair
HAIR_MAIN = 70           # px, ...only if the node's other two arms are BOTH at least this long (90->70:
                         # removes more skeleton-bump hairs on continuous neurites; siblings still long)
JUNCTION_MERGE = 25      # px, collapse branch points closer than this (6->12: merges close double-
                         # junctions that skeletonize renders as two branch points on one real branch)
SOMA_MARGIN = 150        # px (~24 um) beyond the excised soma region: stubs here attach to the soma hub
SOMA_HUB_W = 0.5         # weight of a hub<->stub edge; < the 1px min skeleton edge so the weighted MST
                         #   prefers rooting each proximal neurite at the soma over chaining neighbours
# --- crossing resolution (per Liz Peker's criteria: X-crossing vs Y-branch) ---
RESOLVE_CROSSINGS = True
# Primary cue is CONTINUITY (Liz Peker): a crossing's four arms pair into TWO nearly
# anti-parallel straight pass-throughs; a Y-branch cannot (its daughters diverge, so at
# most one pair is straight). At our 2D/segmentation resolution the "webbing" thickness
# cue does NOT separate the two (overlapping neurites widen the mask at crossings too),
# so thickness is used only as a weak veto against extreme blobs.
CROSS_STRAIGHT_DEG = 152  # both arm-pairs must be anti-parallel (angle >= this) to be an X-crossing
CROSS_THICK_RATIO = 2.2   # node/branch width above this = a real dense blob (soma-like) -> not a crossing
CROSS_CLUSTER_R = 4       # px, cluster nearby skeleton junction pixels into one node
CROSS_STUB = 8            # px, walk this far out of a cluster to estimate a branch's direction
CROSS_DOUBLET_MAX = 40    # px, max connecting-segment length to merge two 3-way T-junctions
                          # (14 -> 40 on 2026-10-01: clean A/B on Liz's 14-cell flag set, final model:
                          #  crossings 0/9 -> 4/9, continuity 34/60 -> 39/60, nothing worse; cost on her
                          #  25 corrected traces ~0.16 real branch-point pairs per cell wrongly merged)
                          # (skeletonization often splits one X-crossing into two nearby T's)
# --- optional learned crossing classifier (soft prior on the raw grayscale image) ---
# A small CNN (cnn_crossing.onnx, ~AUC 0.72 on golden branch/crossing patches) refines the
# X/Y call in the geometric borderline band. Geometry stays primary; the CNN only rescues
# borderline pass-throughs and vetoes clear ones it is very confident are branches. Falls
# back to geometry-only if onnxruntime or the model file is missing.
# DISABLED: a rigorous golden-truth test (measure_cnn_help.py, 28 cells) showed the CNN does
# NOT transfer to the tracer's actual candidates (AUC 0.38 there vs 0.72 on golden patches --
# distribution shift: golden-point vs skeleton-centroid patches) and made accuracy WORSE
# (0.625 geometry -> 0.500). Geometry-only is better. Re-enable only after retraining the CNN
# on the ACTUAL tracer candidate patches. The onnx stays bundled but unused.
USE_CNN_CROSSING = False
CNN_PATCH = 48
CNN_STRAIGHT_MARGIN = 12  # deg below CROSS_STRAIGHT_DEG that the CNN may still rescue
CNN_RESCUE_P = 0.55       # borderline uncrossed only if CNN P(crossing) >= this
CNN_VETO_P = 0.15         # clear-straight crossing vetoed if CNN P(crossing) < this
_CROSS_SESS = "unset"


def _crossing_session():
    global _CROSS_SESS
    if _CROSS_SESS != "unset":
        return _CROSS_SESS
    _CROSS_SESS = None
    if not USE_CNN_CROSSING:
        return None
    try:
        import os, sys, onnxruntime as ort
        cands = []
        base = getattr(sys, "_MEIPASS", None)
        if base:
            cands.append(os.path.join(base, "cnn_crossing.onnx"))
        here = os.path.dirname(os.path.abspath(__file__))
        cands += [os.path.join(here, "cnn_crossing.onnx"),
                  os.path.join(here, "..", "cnn_crossing.onnx"),
                  os.path.join(os.getcwd(), "cnn_crossing.onnx")]
        for p in cands:
            if os.path.exists(p):
                _CROSS_SESS = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
                break
    except Exception:
        _CROSS_SESS = None
    return _CROSS_SESS


def _cnn_p_crossing(gray, rc):
    """P(crossing) for a junction at (row, col) from the raw grayscale image, or None."""
    sess = _crossing_session()
    if sess is None or gray is None:
        return None
    g = gray[..., 0] if getattr(gray, "ndim", 2) == 3 else gray
    h, w = g.shape; r, c = int(round(rc[0])), int(round(rc[1])); half = CNN_PATCH // 2
    if r - half < 0 or c - half < 0 or r + half > h or c + half > w:
        return None
    p = g[r - half:r + half, c - half:c + half].astype(np.float32)
    if p.shape != (CNN_PATCH, CNN_PATCH):
        return None
    p = (p - p.mean()) / (p.std() + 1e-5)
    try:
        return float(sess.run(None, {'patch': p[None, None]})[0].ravel()[0])
    except Exception:
        return None


# --------------------------------------------------------------------------------------------------------------------
# Resolution scaling (2026-10-25). Every pixel-valued constant above is defined at the reference resolution REF_VOX = 0.1625 um/px
# (the lab's images, on which all rules were tuned). trace() first rescales the PIXEL constants by REF_VOX / vox: lengths linearly
# (ints rounded, at least 1), kernel radii rounded, nothing else touched (angles, cosines, probabilities, ratios and multiples of the
# soma radius are unitless; *_UM constants are already in micrometres). At vox == REF_VOX the factor is exactly 1.0, so the output
# is byte-identical to the unscaled tracer (checked by yuqi/vox_scaling/regression.py).
REF_VOX = 0.1625
CLOSE_DISK = 2            # px, closing of the hysteresis mask
HEAL_DISK = 1             # px, dilation of healed gap paths
SOMA_DETECT_DISK = 6      # px, opening in detect_soma / detect_somas
_PX_INT = ['MIN_COMP', 'MIN_COMP_EARLY', 'SPUR', 'HEAL_GAP', 'HEAL_LEN', 'BRIDGE_MAX', 'CONT_MAX_GAP', 'CONT_DIR_STEPS', 'CONT_JOIN_GAP',
           'STITCH_SNAP', 'STITCH_GAP', 'STITCH_TGAP', 'STITCH_FRAG_SIZE', 'SOMA_EXCISE_PAD', 'HAIRPIN_ARM_PX', 'INWARD_DIR_PX',
           'INWARD_RIDGE_LAT', 'DETOUR_DIR_PX', 'SPOKE_SMALL_NODES', 'HAIR_STUB', 'HAIR_MAIN', 'JUNCTION_MERGE', 'SOMA_MARGIN',
           'CROSS_CLUSTER_R', 'CROSS_STUB', 'CROSS_DOUBLET_MAX', 'FWHM_HALF', 'CLOSE_DISK', 'HEAL_DISK', 'SOMA_DETECT_DISK', 'MULTI_BLANK_PAD_PX']
_PX_FLOAT = ['BRIDGE_MIN_PX', 'SOMA_HUB_W']
_PX_BASE = None
_CUR_VOX = REF_VOX


def set_vox(vox):
    """rescale the pixel constants to `vox` um/px (no-op at the reference resolution)."""
    global _PX_BASE, _CUR_VOX
    g = globals()
    if _PX_BASE is None:
        _PX_BASE = {k: g[k] for k in _PX_INT + _PX_FLOAT}
    if vox is None or abs(vox - _CUR_VOX) < 1e-12:
        return
    f = REF_VOX / float(vox)
    for k in _PX_INT:
        g[k] = max(1, int(round(_PX_BASE[k] * f)))
    for k in _PX_FLOAT:
        g[k] = float(_PX_BASE[k] * f)
    _CUR_VOX = float(vox)


def detect_soma(gray_u8, vox):
    set_vox(vox)
    g = gray_u8.astype(float)
    b = opening(g >= np.percentile(g, 99.7), disk(SOMA_DETECT_DISK))
    lab, n = cc_label(b)
    if n == 0:
        return (gray_u8.shape[0] / 2.0, gray_u8.shape[1] / 2.0)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    ys, xs = np.where(lab == sizes.argmax())
    return (ys.mean(), xs.mean())


def _build_graph(skel):
    yy, xx = np.nonzero(skel)
    pts = list(zip(yy.tolist(), xx.tolist()))
    idx = {p: i for i, p in enumerate(pts)}
    G = nx.Graph(); G.add_nodes_from(range(len(pts)))
    for i, (y, x) in enumerate(pts):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    j = idx.get((y + dy, x + dx))
                    if j is not None and j > i:
                        G.add_edge(i, j)
    return G, pts


def _drop_small(G, m):
    keep = set()
    for c in nx.connected_components(G):
        if len(c) >= m:
            keep |= c
    return G.subgraph(keep).copy()


def _line_prob(prob, a, b, n=24):
    ys = np.clip(np.linspace(a[0], b[0], n).round().astype(int), 0, prob.shape[0] - 1)
    xs = np.clip(np.linspace(a[1], b[1], n).round().astype(int), 0, prob.shape[1] - 1)
    return prob[ys, xs].mean()


def _evidence_bridge(G, pts, prob):
    """Bridge only short gaps that have probability evidence. Forest is fine."""
    comps = list(nx.connected_components(G))
    if len(comps) <= 1:
        return G
    compid = {n: c for c, nodes in enumerate(comps) for n in nodes}
    deg = dict(G.degree())
    ends = [n for n in G.nodes if deg[n] <= 1] or list(G.nodes)
    ec = np.array([pts[n] for n in ends])
    tree = cKDTree(ec)
    cand = []
    for ai, a in enumerate(ends):
        for bi in tree.query_ball_point(ec[ai], BRIDGE_MAX):
            b = ends[bi]
            if b <= a or compid[a] == compid[b]:
                continue
            d = np.hypot(*(np.array(pts[a]) - np.array(pts[b])))
            ev = _line_prob(prob, pts[a], pts[b])
            if ev >= BRIDGE_PROB:
                cand.append((d / (ev + 1e-3), a, b))
    cand.sort()
    parent = {c: c for c in range(len(comps))}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for _, a, b in cand:
        ra, rb = find(compid[a]), find(compid[b])
        if ra != rb:
            parent[ra] = rb; G.add_edge(a, b)
    return G


def _endpoint_dir(G, pts, a, k=CONT_DIR_STEPS):
    """Outward unit tangent at a dangling endpoint a: walk up to k skeleton steps into the
    fragment and point from there back out through a."""
    path = [a]; prev = None; cur = a
    for _ in range(k):
        nbrs = [n for n in G.neighbors(cur) if n != prev]
        if len(nbrs) != 1:
            break
        prev, cur = cur, nbrs[0]; path.append(cur)
    v = np.array(pts[a], float) - np.array(pts[path[-1]], float)
    nn = np.hypot(*v)
    return v / nn if nn > 1e-6 else np.array([0.0, 0.0])


def _continuity_bridge(G, pts, prob):
    """Join broken fragments by good continuation: connect two dangling endpoints only if BOTH
    tangents point at each other (smooth, low-curvature) and the gap is short. Image evidence, if
    present, discounts the cost but is not required (this is what recovers faint gaps)."""
    comps = list(nx.connected_components(G))
    if len(comps) <= 1:
        return G
    compid = {n: c for c, nodes in enumerate(comps) for n in nodes}
    deg = dict(G.degree())
    ends = [n for n in G.nodes if deg[n] == 1]
    if len(ends) < 2:
        return G
    dirs = {a: _endpoint_dir(G, pts, a) for a in ends}
    ec = np.array([pts[a] for a in ends], float)
    tree = cKDTree(ec)
    cand = []
    for ai, a in enumerate(ends):
        ta = dirs[a]
        for bi in tree.query_ball_point(ec[ai], CONT_MAX_GAP):
            b = ends[bi]
            if b <= a or compid[a] == compid[b]:
                continue
            g = np.array(pts[b], float) - np.array(pts[a], float)
            d = np.hypot(*g)
            if d < 1e-3:
                continue
            u = g / d
            cos_a = float(np.dot(ta, u)); cos_b = float(np.dot(dirs[b], -u))
            if cos_a >= CONT_MIN_COS and cos_b >= CONT_MIN_COS:      # both continue toward each other
                ev = _line_prob(prob, pts[a], pts[b])
                cost = d * (2.1 - cos_a - cos_b) / (3 * ev + 1)      # short + straight + (evidence) preferred
                cand.append((cost, a, b))
    cand.sort()
    parent = {c: c for c in range(len(comps))}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    used = set()
    for _, a, b in cand:
        if a in used or b in used:
            continue
        ra, rb = find(compid[a]), find(compid[b])
        if ra != rb:
            parent[ra] = rb; G.add_edge(a, b); used.add(a); used.add(b)

    # ---- second pass: connect a still-dangling tip to the nearest point on ANOTHER neurite ----
    if CONT_JOIN:
        comp2 = {n: c for c, nodes in enumerate(nx.connected_components(G)) for n in nodes}
        alln = list(G.nodes)
        allpos = np.array([pts[n] for n in alln], float)
        ntree = cKDTree(allpos)
        joins = []
        for a in [n for n in G.nodes if G.degree(n) == 1 and n not in used]:
            ta = _endpoint_dir(G, pts, a)
            pa = np.array(pts[a], float)
            best = None
            for bi in ntree.query_ball_point(pts[a], CONT_JOIN_GAP):
                b = alln[bi]
                if b == a or comp2[b] == comp2[a] or G.degree(b) == 0:
                    continue
                g = np.array(pts[b], float) - pa; d = float(np.hypot(*g))
                if d < 3:
                    continue
                cos_a = float(np.dot(ta, g / d))            # does the tip head toward this point?
                if cos_a < CONT_JOIN_COS:
                    continue
                ev = _line_prob(prob, pts[a], pts[b])
                cost = d * (1.6 - cos_a) / (3 * ev + 1)
                if best is None or cost < best[0]:
                    best = (cost, b)
            if best is not None:
                joins.append((best[0], a, best[1]))
        joins.sort()
        jused = set()
        for _, a, b in joins:
            if a in jused or comp2[a] == comp2[b]:
                continue
            G.add_edge(a, b); jused.add(a)
            cb = comp2[b]
            for n in [n for n in comp2 if comp2[n] == comp2[a]]:
                comp2[n] = cb                              # keep components current so we don't re-merge
    return G


def _fragment_stitch(G, pts):
    """Attach floating fragments to the arbor by geometry (short-gap breaks; see FRAG_STITCH notes).
    For each dangling endpoint, find the nearest node of another component within STITCH_GAP: <=SNAP snap
    unconditionally; beyond that require the tip to point at the target (endpoint->endpoint, no new
    branch). Mid-neurite attach (adds a branch) only within SNAP. Greedy shortest-cost first, union-find
    so fragments chain into the tree. Prob-free on purpose: these gaps are dark, the geometry is the cue."""
    comps = list(nx.connected_components(G))
    if len(comps) <= 1:
        return G
    compid = {n: c for c, nodes in enumerate(comps) for n in nodes}
    csize = [len(c) for c in comps]
    deg = dict(G.degree())
    ends = [n for n in G.nodes if deg[n] == 1]
    if not ends:
        return G
    alln = list(G.nodes)
    ntree = cKDTree(np.array([pts[n] for n in alln], float))
    cand = []
    for a in ends:
        ta = _endpoint_dir(G, pts, a); pa = np.array(pts[a], float)
        a_frag = csize[compid[a]] <= STITCH_FRAG_SIZE       # a is on a small floating fragment
        cos_req = STITCH_FRAG_COS if a_frag else STITCH_COS
        best = None
        for bi in ntree.query_ball_point(pts[a], STITCH_GAP):
            b = alln[bi]
            if b == a or compid[b] == compid[a]:
                continue
            g = np.array(pts[b], float) - pa; d = float(np.hypot(*g))
            if d < 1e-3:
                continue
            cos_a = float(np.dot(ta, g / d))
            interior = deg.get(b, 0) >= 2
            if interior:                                    # endpoint -> mid-span T (a broken side branch)
                if d > STITCH_TGAP:
                    continue
                if d > STITCH_SNAP and cos_a < STITCH_TCOS:
                    continue                                # beyond a snap, the tip must clearly point at it
                cost = d * (1.6 - cos_a) * STITCH_BRANCH_PEN
            else:                                           # endpoint -> endpoint (adds no branch)
                if d > STITCH_SNAP and cos_a < cos_req:     # fragments get the relaxed gate (must attach)
                    continue
                cost = d * (1.6 - cos_a)
            if best is None or cost < best[0]:
                best = (cost, b)
        if best is not None:
            cand.append((best[0], a, best[1]))
    cand.sort()
    parent = {c: c for c in range(len(comps))}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    used = set()
    for _, a, b in cand:
        if a in used:
            continue
        ra, rb = find(compid[a]), find(compid[b])
        if ra != rb:
            parent[ra] = rb; G.add_edge(a, b); used.add(a)
    return G


def _collapse_junctions(T, pts, roots, r):
    changed = True
    while changed:
        changed = False
        bps = [n for n in T.nodes if T.out_degree(n) >= 2]
        for a in bps:
            if a not in T:
                continue
            done = False
            for b in list(T.successors(a)):
                path = [b]; cur = b
                while T.out_degree(cur) == 1:
                    cur = next(iter(T.successors(cur))); path.append(cur)
                end = path[-1]
                if T.out_degree(end) >= 2 and end in T and a in T:
                    if np.hypot(*(np.array(pts[end]) - np.array(pts[a]))) < r:
                        for c in list(T.successors(end)):
                            T.add_edge(a, c)
                        for n in path:
                            if n in T and n not in roots:
                                T.remove_node(n)
                        changed = True; done = True; break
            if done:
                break
    return T


def _prune(T, pts, roots, spur):
    changed = True
    while changed:
        changed = False
        for lf in [n for n in list(T.nodes) if T.out_degree(n) == 0 and n not in roots]:
            path = [lf]; cur = lf
            while True:
                pr = list(T.predecessors(cur))
                if not pr:
                    break
                p = pr[0]; path.append(p)
                if T.out_degree(p) > 1 or p in roots:
                    break
                cur = p
            length = sum(np.hypot(*(np.array(pts[path[k]]) - np.array(pts[path[k + 1]]))) for k in range(len(path) - 1))
            if length < spur:
                for n in path[:-1]:
                    if n in T and n not in roots:
                        T.remove_node(n)
                changed = True
    return T


def _turnback_near_soma(forest, pts, hub, roots, cy, cx, vox, prob):
    """Step 6 (user rule): a neurite may start without a visible origin, but it can never run on its own toward the soma.
    For an end within TURNBACK_SOMA_UM of the soma centre that is >= TURNBACK_RISE_UM closer to the soma than the start of its own
    unbranched stretch: detach the stretch from that start and reverse it so it runs outward. If another path lies within
    TURNBACK_JOIN_UM of the end with gap probability >= TURNBACK_JOIN_PROB, the end is attached there (a new branch point);
    otherwise the stretch is kept whole as a separate piece ('near the soma, connection pending'). Nothing is deleted.
    Returns [(end, old_start, 'join'|'pending', end_um_from_soma)]; pending roots are added to roots."""
    if TURNBACK_SOMA_UM is None:
        return []
    P = np.array([pts[n] for n in forest.nodes], float); ids = list(forest.nodes); kd = cKDTree(P)
    D = lambda n: float(np.hypot(pts[n][0] - cy, pts[n][1] - cx)) * vox
    H, W = prob.shape
    out = []
    leaves = [n for n in forest.nodes if forest.out_degree(n) == 0 and n not in roots]
    for e in leaves:
        if forest.out_degree(e) != 0 or forest.in_degree(e) != 1 or D(e) > TURNBACK_SOMA_UM:
            continue
        chain = [e]; u = e
        while True:
            p = next(iter(forest.predecessors(u)), None)
            if p is None or p in roots or forest.out_degree(p) != 1:
                break
            chain.append(p); u = p
        start = next(iter(forest.predecessors(u)), None)
        if start is None or start == hub or D(start) - D(e) < TURNBACK_RISE_UM:
            continue
        cs = set(chain); best = None
        for j in sorted(kd.query_ball_point(P[ids.index(e)] if False else np.array(pts[e], float), TURNBACK_JOIN_UM / vox),
                        key=lambda j: np.hypot(*(P[j] - np.array(pts[e], float)))):
            m = ids[j]
            if m in cs or m == hub or m not in forest:
                continue
            a = np.array(pts[e], float); b = P[j]; k = max(3, int(np.hypot(*(b - a))) + 1)
            pv = np.mean([prob[min(H - 1, int(round(a[0] + (b[0] - a[0]) * t))), min(W - 1, int(round(a[1] + (b[1] - a[1]) * t)))] for t in np.linspace(0, 1, k)])
            if pv >= TURNBACK_JOIN_PROB:
                best = m; break
        w = forest[start][u].get('weight', 1.0) if forest.has_edge(start, u) else 1.0
        forest.remove_edge(start, u)
        for i in range(len(chain) - 1):              # chain = [e, ..., u]; edges parent chain[i+1] -> child chain[i]
            a, b = chain[i + 1], chain[i]
            ww = forest[a][b].get('weight', 1.0); forest.remove_edge(a, b); forest.add_edge(b, a, weight=ww)
        if best is not None:
            forest.add_edge(best, e, weight=w); out.append((e, start, 'join', round(D(e), 1)))
        else:
            roots.add(e); out.append((e, start, 'pending', round(D(e), 1)))
    return out


def _piece_rules(forest, pts, hub, roots, vox):
    """Step 7: what to do with separate pieces (trees whose root is not the soma). Measured on the 115 evaluation cells
    (yuqi/rules_round/pieces_diag2.csv): the gap between a piece and the main tree has ~zero probability, so only geometry is used.
      continuation : piece root within PIECE_CONT_UM of a main-tree END and the piece's first 5 um collinear (<= PIECE_CONT_DEG) with the
                     end's last 5 um -> the end is joined to the root (the piece becomes the continuation of that branch).
      drop         : pieces shorter than PIECE_DROP_UM are removed.
      side         : piece root within PIECE_SIDE_UM of the main tree's side -> NOT joined; returned as a candidate for the editor.
    Edits forest and roots in place. Returns dict(joined=[(root, end)], dropped=[root], candidates=[(root_rc, target_rc)])."""
    out = dict(joined=[], dropped=[], candidates=[])
    if hub not in forest:
        return out
    main = set(nx.descendants(forest, hub)) | {hub}

    def direction(start, step, um=5.0):
        u = start; last = np.array(pts[start], float); s0 = last.copy()
        while True:
            n = step(u)
            if n is None or np.hypot(*(last - s0)) * vox >= um:
                break
            u = n; last = np.array(pts[u], float)
        v = last - s0; nv = np.linalg.norm(v)
        return v / nv if nv > 0 else None

    def piece_len(r):
        L = 0.0; st = [r]
        while st:
            u = st.pop()
            for v in forest.successors(u):
                L += np.hypot(pts[u][0] - pts[v][0], pts[u][1] - pts[v][1]); st.append(v)
        return L * vox

    main_ends = [n for n in main if forest.out_degree(n) == 0 and n != hub]
    end_kd = cKDTree(np.array([pts[n] for n in main_ends], float)) if main_ends else None
    main_list = list(main); main_kd = cKDTree(np.array([pts[n] for n in main_list], float))
    for r in [r for r in list(roots) if r != hub and r in forest]:
        if r in main:
            continue
        L = piece_len(r)
        if L < PIECE_DROP_UM:
            sub = list(nx.descendants(forest, r)) + [r]; forest.remove_nodes_from(sub); roots.discard(r); out['dropped'].append(r); continue
        joined = False
        if end_kd is not None:
            d_end, k = end_kd.query(pts[r])
            if d_end * vox <= PIECE_CONT_UM:
                e = main_ends[k]
                pdir = direction(r, lambda u: next(iter(forest.successors(u)), None))
                edir = direction(e, lambda u: next(iter(forest.predecessors(u)), None))   # from the end back toward its parent
                if pdir is not None and edir is not None:
                    ang = float(np.degrees(np.arccos(np.clip(np.dot(pdir, -edir), -1, 1))))
                    if ang <= PIECE_CONT_DEG:
                        forest.add_edge(e, r, weight=float(d_end)); roots.discard(r); out['joined'].append((r, e)); joined = True
                        main |= set(nx.descendants(forest, r)) | {r}
        if joined:
            continue
        d_side, k = main_kd.query(pts[r])
        if d_side * vox <= PIECE_SIDE_UM:
            out['candidates'].append((tuple(pts[r]), tuple(pts[main_list[k]])))
    return out


def _remove_hairs(T, pts, roots, stub_max, main_min):
    """Remove a short leaf side-arm ("hair") at a degree-3 node when the other two arms are both long
    main lines -- a skeletonization bump on a continuous neurite that would otherwise show as a spurious
    branch+tip. Genuine bifurcations (two comparable arms) are kept because HAIR_MAIN requires BOTH
    siblings to be long. Walks undirected; only removes the hair if it terminates in a leaf tip."""
    U = T.to_undirected()

    def walk(a, b):                              # from a into neighbour b: (length, node chain, ends_in_leaf)
        L = np.hypot(*(np.array(pts[b], float) - np.array(pts[a], float))); prev, cur = a, b; chain = [b]
        while True:
            nb = [w for w in U.neighbors(cur) if w != prev]
            if len(nb) != 1:
                break
            L += np.hypot(*(np.array(pts[nb[0]], float) - np.array(pts[cur], float))); prev, cur = cur, nb[0]; chain.append(cur)
        return L, chain, (U.degree(cur) == 1 and cur not in roots)

    changed = True
    while changed:
        changed = False
        for b in [n for n in list(T.nodes) if n in U and U.degree(n) == 3 and n not in roots]:
            arms = sorted((walk(b, w) for w in U.neighbors(b)), key=lambda x: x[0])
            if arms[0][0] <= stub_max and arms[0][2] and arms[1][0] >= main_min and arms[2][0] >= main_min:
                for n in arms[0][1]:
                    if n in T and n not in roots:
                        T.remove_node(n)
                    if n in U:
                        U.remove_node(n)
                changed = True
    return T


def _unit(v):
    n = np.hypot(*v)
    return v / n if n > 0 else v


def _dir_down(F, pts, a, c, steps=5):
    """unit direction from a toward successor c, averaged a few px down the branch."""
    cur = c
    for _ in range(steps):
        s = list(F.successors(cur))
        if len(s) != 1:
            break
        cur = s[0]
    return _unit(np.array(pts[cur], float) - np.array(pts[a], float))


def _edge_um(pts, u, v, vox):
    return np.hypot(*(np.array(pts[u]) - np.array(pts[v]))) * vox


FWHM_HALF = 14         # px, half-length of the intensity profile taken across each node
FWHM_NOISE_K = 3.0     # a node is measurable only if its peak-over-baseline >= K x the image noise sd
                       # (noise sd = robust MAD of horizontal pixel differences). Measured on the RAW
                       # image: on the 8-bit normalised one neurites are only 1-5 grey levels above
                       # background. Higher K selects bright = wide neurites and biases the median up.


def _noise_sd(g):
    d = (g[:, 1:] - g[:, :-1]).ravel()
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))


def _fwhm_radii_um(gray, pts, order, parent, vox):
    """Per-node neurite radius (um) = half the full-width-at-half-maximum of the intensity
    profile perpendicular to the local direction (node -> parent). Vectorised: one
    map_coordinates call for every node. Returns {node: radius_um}, nodes omitted where unmeasurable."""
    from scipy.ndimage import gaussian_filter, map_coordinates
    nodes = [n for n in order if parent.get(n, -1) != -1]
    if not nodes:
        return {}
    g = gaussian_filter(np.asarray(gray, dtype=np.float32), 0.8)
    min_contrast = FWHM_NOISE_K * _noise_sd(g)
    P = np.array([pts[n] for n in nodes], float)                 # (row, col)
    Q = np.array([pts[parent[n]] for n in nodes], float)
    d = P - Q
    nrm = np.hypot(d[:, 0], d[:, 1]); ok = nrm > 1e-6
    d[ok] /= nrm[ok, None]
    perp = np.stack([-d[:, 1], d[:, 0]], 1)                     # perpendicular unit vectors
    t = np.arange(-FWHM_HALF, FWHM_HALF + 0.5, 0.5)
    rows = P[:, 0:1] + perp[:, 0:1] * t[None, :]
    cols = P[:, 1:2] + perp[:, 1:2] * t[None, :]
    prof = map_coordinates(g, [rows.ravel(), cols.ravel()], order=1, mode='nearest').reshape(len(nodes), -1)
    base = np.minimum(prof[:, 0], prof[:, -1]); peak = prof.max(1)
    kpk = prof.argmax(1); hm = base + 0.5 * (peak - base)
    above = prof > hm[:, None]
    out = {}
    for i, n in enumerate(nodes):
        if not ok[i] or peak[i] - base[i] < min_contrast:
            continue
        a = kpk[i]
        while a > 0 and above[i, a]:
            a -= 1
        b = kpk[i]
        while b < len(t) - 1 and above[i, b]:
            b += 1
        w = t[b] - t[a]
        if 0 < w < 2 * FWHM_HALF:
            out[n] = float(0.5 * w * vox)
    return out


def _cluster_junctions(G, pts, r):
    """Group skeleton junction pixels (degree>=3) that lie within r px into single nodes
    (skeletonize renders a crossing/branch as a small blob of high-degree pixels)."""
    junc = [n for n in G.nodes if G.degree(n) >= 3]
    if not junc:
        return []
    P = np.array([pts[n] for n in junc], float)
    tree = cKDTree(P)
    uf = {n: n for n in junc}
    def find(a):
        while uf[a] != a:
            uf[a] = uf[uf[a]]; a = uf[a]
        return a
    for i, j in tree.query_pairs(r):
        uf[find(junc[i])] = find(junc[j])
    out = {}
    for n in junc:
        out.setdefault(find(n), []).append(n)
    return list(out.values())


def _stub_dir(G, pts, cset, entry, stub_len):
    """Unit direction leaving the cluster along a branch, plus its far endpoint."""
    prev, cur = None, entry
    for _ in range(stub_len):
        nbrs = [x for x in G.neighbors(cur) if x != prev and x not in cset]
        if len(nbrs) != 1:
            break
        prev, cur = cur, nbrs[0]
    return _unit(np.array(pts[cur], float) - np.array(pts[entry], float))


def _four_arm_xy(G, pts, dt, cset, entries, gray):
    """For a 4-armed junction (node set cset, four external arm nodes 'entries'), return the
    two pass-through arm-pairs if it is an X-crossing, else None. Shared by clean degree-4
    nodes and merged double-T junctions."""
    if len(entries) != 4:
        return None
    d = {e: _stub_dir(G, pts, cset, e, CROSS_STUB) for e in entries}
    if any(np.hypot(*d[e]) == 0 for e in entries):
        return None
    node_w = max(dt[pts[n][0], pts[n][1]] for n in cset)
    branch_w = float(np.median([dt[pts[e][0], pts[e][1]] for e in entries])) + 1e-6
    if node_w >= CROSS_THICK_RATIO * branch_w:      # dense soma-like blob -> never a crossing
        return None
    e = list(entries)
    best = None
    for (a, b), (c, f) in [((e[0], e[1]), (e[2], e[3])),
                           ((e[0], e[2]), (e[1], e[3])),
                           ((e[0], e[3]), (e[1], e[2]))]:
        a1 = np.degrees(np.arccos(np.clip(np.dot(d[a], d[b]), -1, 1)))
        a2 = np.degrees(np.arccos(np.clip(np.dot(d[c], d[f]), -1, 1)))
        if best is None or min(a1, a2) > best[0]:
            best = (min(a1, a2), (a, b), (c, f))
    straight = best[0] >= CROSS_STRAIGHT_DEG
    p_cross = _cnn_p_crossing(gray, np.mean([pts[n] for n in cset], axis=0))
    if p_cross is None:                             # geometry only (CNN unavailable/disabled)
        is_cross = straight
    elif straight:                                  # trust geometry, unless CNN is sure it is a branch
        is_cross = p_cross >= CNN_VETO_P
    else:                                           # borderline geometry: CNN may rescue a pass-through
        is_cross = best[0] >= (CROSS_STRAIGHT_DEG - CNN_STRAIGHT_MARGIN) and p_cross >= CNN_RESCUE_P
    return (best[1], best[2]) if is_cross else None


def _walk_to_cluster(G, pts, cset, entry, node2cl, self_ci, maxlen):
    """Walk the degree-2 skeleton chain leaving a junction; if it reaches another junction
    cluster within maxlen px, return (connecting_path_nodes, other_cluster_index), else (None, None)."""
    prev, cur, path, length = None, entry, [], 0.0
    while True:
        if cur in node2cl and node2cl[cur] != self_ci:
            return path, node2cl[cur]
        nbrs = [x for x in G.neighbors(cur) if x != prev and x not in cset]
        if len(nbrs) != 1:
            return None, None
        path.append(cur)
        nxt = nbrs[0]
        length += float(np.hypot(*(np.array(pts[nxt], float) - np.array(pts[cur], float))))
        if length > maxlen:
            return None, None
        prev, cur = cur, nxt


def _resolve_crossings(G, pts, dt, gray=None):
    """Distinguish an X-crossing (two branches passing over, no branch node) from a
    Y-branch, using Liz Peker's morphological cues: a crossing's four arms pair into
    two nearly-straight (anti-parallel) pass-throughs AND the node does not thicken/web
    (branch nodes are denser -> larger distance-transform radius). Uncross the crossings
    so branches stay continuous and no false branch point is created. When the raw
    grayscale image and the crossing CNN are available, the CNN acts as a soft prior in
    the geometric borderline band (rescue/veto); geometry stays primary."""
    clusters = _cluster_junctions(G, pts, CROSS_CLUSTER_R)
    node2cl = {n: ci for ci, cl in enumerate(clusters) for n in cl}
    ent = {ci: list({nb for n in cl for nb in G.neighbors(n) if nb not in set(cl)})
           for ci, cl in enumerate(clusters)}
    removed = set()
    crossings = []                                       # (row,col) of each resolved X-crossing

    def uncross(nodes, pairs):
        G.remove_nodes_from(nodes); removed.update(nodes)
        (a, b), (c, f) = pairs
        G.add_edge(a, b); G.add_edge(c, f)
        crossings.append(tuple(np.mean([pts[n] for n in nodes], axis=0)))

    # 1) clean degree-4 crossings
    for ci, cl in enumerate(clusters):
        if len(ent[ci]) != 4 or any(n in removed for n in cl):
            continue
        r = _four_arm_xy(G, pts, dt, set(cl), ent[ci], gray)
        if r:
            uncross(cl, r)
    # 2) double-T crossings: two 3-way junctions joined by a short segment -> one crossing
    for ci, cl in enumerate(clusters):
        if len(ent[ci]) != 3 or any(n in removed for n in cl):
            continue
        cset = set(cl)
        for e in ent[ci]:
            path, cj = _walk_to_cluster(G, pts, cset, e, node2cl, ci, CROSS_DOUBLET_MAX)
            if cj is None or cj == ci or len(ent[cj]) != 3 or any(n in removed for n in clusters[cj]):
                continue
            merged = cset | set(clusters[cj]) | set(path)
            extra = list({nb for n in merged for nb in G.neighbors(n) if nb not in merged})
            if len(extra) != 4:                     # the two T's must expose exactly 4 external arms
                continue
            r = _four_arm_xy(G, pts, dt, merged, extra, gray)
            if r:
                uncross(merged, r)
            break
    return G, crossings


def tree_metrics(F, pts, roots, soma_rc, vox, radii, inters):
    """Morphometrics derived from the reconstructed forest. Topology-dependent
    ones (order, angle, segment lengths) are only as reliable as the branch
    structure; see report."""
    # branch points exclude roots: the soma is the root, not a bifurcation
    branch_pts = [n for n in F.nodes if F.out_degree(n) >= 2 and n not in roots]
    tips = [n for n in F.nodes if F.out_degree(n) == 0 and n not in roots]
    bset = set(branch_pts)

    # centrifugal branch order: +1 at each real (non-root) branch point
    order = {}
    for r in roots:
        order[r] = 1
        for u, v in nx.bfs_edges(F, r):
            order[v] = order[u] + (1 if (F.out_degree(u) >= 2 and u not in roots) else 0)
    highest_order = max(order.values()) if order else 0

    # segments: root/branch-point -> next branch-point/tip
    segments = []  # (start, end, len_um, ends_in_branch)
    for t in (set(roots) | set(branch_pts)):
        for c in F.successors(t):
            length = _edge_um(pts, t, c, vox); cur = c
            while F.out_degree(cur) == 1:
                nxt = next(iter(F.successors(cur)))
                length += _edge_um(pts, cur, nxt, vox); cur = nxt
            segments.append((t, cur, length, cur in bset))
    seg_len = [s[2] for s in segments]
    internal_len = [s[2] for s in segments if s[3]]
    first_bif_len = [s[2] for s in segments if s[0] in roots and s[3]]

    # branch angle, two definitions. SNT's "remote bifurcation angle" (what the lab measures by
    # hand) is the angle BETWEEN the two daughters, each taken as the vector from the bifurcation
    # to the END of its segment (next branch point or tip). The older definition here was
    # parent-direction vs each daughter, 5 px out. Both are reported; avg_branch_angle_deg follows SNT.
    angles = []          # parent-direction vs daughter (old definition)
    bif_angles = []      # daughter vs daughter, remote (SNT remote bifurcation angle)
    straightest = {}     # per bifurcation: the daughter that continues the parent most directly
    seg_end_of = {}      # (start node, first child) -> end node of that daughter segment
    for t in (set(roots) | set(branch_pts)):
        for c in F.successors(t):
            cur = c
            while F.out_degree(cur) == 1:
                cur = next(iter(F.successors(cur)))
            seg_end_of[(t, c)] = cur
    for b in branch_pts:
        preds = list(F.predecessors(b))
        if not preds:
            continue
        up = preds[0]
        for _ in range(4):
            pp = list(F.predecessors(up))
            if len(pp) != 1:
                break
            up = pp[0]
        pin = _unit(np.array(pts[b], float) - np.array(pts[up], float))
        kids = list(F.successors(b))
        douts = {c: _dir_down(F, pts, b, c) for c in kids}
        for c in kids:
            angles.append(float(np.degrees(np.arccos(np.clip(np.dot(pin, douts[c]), -1, 1)))))
        remote = {}
        for c in kids:
            e = seg_end_of.get((b, c), c)
            remote[c] = _unit(np.array(pts[e], float) - np.array(pts[b], float))
        for i in range(len(kids)):
            for j in range(i + 1, len(kids)):
                bif_angles.append(float(np.degrees(np.arccos(np.clip(np.dot(remote[kids[i]], remote[kids[j]]), -1, 1)))))
        if kids:
            straightest[b] = max(kids, key=lambda c: float(np.dot(pin, douts[c])))

    # SNT-style path order (what "highest branch order" means in the lab's SNT output): a path
    # keeps its order through a bifurcation and only the side branch goes up one. We approximate
    # SNT's hand-traced paths by letting the straightest daughter continue the parent path.
    porder = {}
    for r in roots:
        porder[r] = 1
        for u, v in nx.bfs_edges(F, r):
            side = (F.out_degree(u) >= 2 and u not in roots and straightest.get(u, v) != v)
            porder[v] = porder[u] + (1 if side else 0)
    highest_path_order = max(porder.values()) if porder else 0

    # polarity: how one-sided the arbor is around the soma (0 balanced, 1 one-sided)
    soma = np.array(soma_rc, float)
    vs = []
    for n in F.nodes:
        d = np.array(pts[n], float) - soma
        nn = np.hypot(*d)
        if nn > 1:
            vs.append(d / nn)
    polarity = float(np.hypot(*np.mean(vs, axis=0))) if vs else 0.0

    # Sholl-derived
    nz = np.nonzero(inters > 0)[0]
    enclosing = float(radii[nz[-1]]) if len(nz) else 0.0
    mean_inters = float(np.mean(inters[:nz[-1] + 1])) if len(nz) else 0.0

    def _m(x):
        return round(float(np.mean(x)), 2) if len(x) else 0.0

    return {
        "num_branches": len(segments),
        "highest_branch_order": int(highest_path_order),
        "highest_branch_order_centrifugal": int(highest_order),
        "mean_intersections": round(mean_inters, 2),
        "enclosing_radius_um": round(enclosing, 1),
        "avg_branch_angle_deg": _m(bif_angles),
        "avg_parent_daughter_angle_deg": _m(angles),
        "avg_len_before_first_bif_um": _m(first_bif_len),
        "avg_segment_len_um": _m(internal_len if internal_len else seg_len),
        "polarity_index": round(polarity, 3),
    }


def _weigh(G, pts):
    """set every edge weight to its Euclidean length so the MST is the shortest/most natural tree
    (the graph is otherwise unweighted, giving an arbitrary spanning tree near the soma)."""
    for u, v in G.edges():
        G[u][v]['weight'] = float(np.hypot(pts[u][0] - pts[v][0], pts[u][1] - pts[v][1]))
    return G


def _cut_soma_hairpins(G, pts, cy, cx, excise_r):
    """STEP 1: cut every soma-edge U-turn (hairpin) at its apex; return (G, number of cuts).
    See SOMA_HAIRPIN_CUT. Walks only through degree-2 nodes, so junctions are never cut through."""
    def d(n):
        return float(np.hypot(pts[n][0] - cy, pts[n][1] - cx))
    zone = HAIRPIN_ZONE * excise_r
    rise = HAIRPIN_RISE_MULT * excise_r
    def arm(a, first):
        prev, cur, steps = a, first, 0
        dmin = d(cur)
        while steps < HAIRPIN_ARM_PX:
            nb = [x for x in G.neighbors(cur) if x != prev]
            if G.degree(cur) != 2 or not nb:
                break
            prev, cur = cur, nb[0]
            dmin = min(dmin, d(cur)); steps += 1
        return cur, steps, dmin
    cuts = 0
    cands = sorted([n for n in list(G.nodes) if G.degree(n) == 2 and d(n) <= zone], key=d)
    for a in cands:
        if a not in G or G.degree(a) != 2:
            continue
        n1, n2 = list(G.neighbors(a))
        e1, s1, m1 = arm(a, n1)
        e2, s2, m2 = arm(a, n2)
        if s1 < HAIRPIN_ARM_PX or s2 < HAIRPIN_ARM_PX:
            continue                                    # arm ran into a junction/leaf before the window ended
        da = d(a)
        if min(m1, m2) < da - 1e-9:
            continue                                    # a is not the minimum over the window
        if d(e1) < da + rise or d(e2) < da + rise:
            continue                                    # arms do not both turn clearly outward
        G.remove_edge(a, n2)                            # apex splits the chain: a and n2 become the two new stubs
        cuts += 1
    return G, cuts


def _merge_small_spokes(forest, pts, hub, vox):
    """STEP 1 v3.1 (see SOMA_SPOKE_MERGE): hang each small hub child on the nearest node of a bigger neighbouring hub child.
    Edits `forest` in place; returns [(small_child, joined_node)]."""
    def sub(n):
        out = [n]; k = 0
        while k < len(out):
            out.extend(forest.successors(out[k])); k += 1
        return out
    lim = SPOKE_MERGE_UM / vox
    merged = []
    for _ in range(40):
        kids = list(forest.successors(hub))
        size = {k: len(sub(k)) for k in kids}
        done = True
        for q in sorted(kids, key=lambda k: size[k]):
            if size[q] >= SPOKE_SMALL_NODES:
                break
            big = [r for r in kids if r != q and size[r] > size[q]
                   and np.hypot(pts[q][0] - pts[r][0], pts[q][1] - pts[r][1]) <= lim]
            if not big:
                continue
            r = max(big, key=lambda r: size[r])
            nodes = sub(r)
            n = min(nodes, key=lambda x: np.hypot(pts[q][0] - pts[x][0], pts[q][1] - pts[x][1]))
            forest.remove_edge(hub, q)
            if SPOKE_SMALL_DROP:
                forest.remove_nodes_from(sub(q))        # the stub is a duplicate line of the big trunk's skeleton fork: delete it
            else:
                forest.add_edge(n, q)                   # (v3.1/v3.2 behaviour) hang it on the trunk: leaves a hook = still a third line
            merged.append((q, n)); done = False
            break
        if done:
            break
    return merged


def _reroot_soma_detours(forest, pts, hub, cy, cx, excise_r, vox=0.1625):
    """STEP 1 v2 (see SOMA_DETOUR_REROOT). Edits `forest` (DiGraph, parent -> child) in place; returns the list of fixes
    [(p, cut_node, detour_ratio)] for diagnostics."""
    def d(n):
        return float(np.hypot(pts[n][0] - cy, pts[n][1] - cx))
    zone = DETOUR_ZONE * excise_r
    max_d = excise_r + DETOUR_MAX_GAP_UM / vox          # soma surface (equal-area radius) + the allowed gap
    fixes = []
    skip = set()                                # candidates rejected by the direction gate
    for _ in range(DETOUR_MAX_ITER):
        pl = {hub: 0.0}
        order = [hub]
        k = 0
        while k < len(order):
            u = order[k]; k += 1
            for v in forest.successors(u):
                pl[v] = pl[u] + float(np.hypot(pts[u][0] - pts[v][0], pts[u][1] - pts[v][1]))
                order.append(v)
        cand = [n for n in order[1:]
                if n not in skip and d(n) <= min(zone, max_d) and pl[n] > DETOUR_MIN * max(d(n), 1.0)
                and pl[n] - d(n) > DETOUR_MIN_EXTRA * excise_r]
        if not cand:
            break
        p = min(cand, key=d)
        path = [p]
        while True:
            pa = next(iter(forest.predecessors(path[-1])), None)
            if pa is None or pa == hub:
                break
            path.append(pa)
        if len(path) < 2:
            skip.add(p); continue
        jn = max(path[1:], key=d)               # farthest node from the hub on the detour path = the false junction
        if jn == p:
            skip.add(p); continue
        if DETOUR_DIR_COS is not None:
            # direction gate: a primary runs radially INTO the soma. Walk ~DETOUR_DIR_PX along the neurite away from p; the heading
            # from that point to p must point at the hub. A neurite from far away that merely ENDS near the soma (a free tip whose
            # heading is tangential or away) is not a primary and must not get a spoke (CL_4_NTC_6: false bridge, manual trace ends there).
            acc, q = 0.0, p
            for nxt in path[1:path.index(jn) + 1]:
                acc += float(np.hypot(pts[q][0] - pts[nxt][0], pts[q][1] - pts[nxt][1])); q = nxt
                if acc >= DETOUR_DIR_PX:
                    break
            vx, vy = pts[p][0] - pts[q][0], pts[p][1] - pts[q][1]
            hx, hy = cy - pts[p][0], cx - pts[p][1]
            nv, nh = float(np.hypot(vx, vy)), float(np.hypot(hx, hy))
            if nv < 1e-6 or nh < 1e-6 or (vx * hx + vy * hy) / (nv * nh) < DETOUR_DIR_COS:
                skip.add(p); continue
        c = path[path.index(jn) - 1]            # the node just before jn on p's side
        chain = path[:path.index(jn)]           # p ... c   (upwards)
        old_parent = jn
        forest.remove_edge(old_parent, c)
        # re-root the detached piece at p: reverse the edges on the chain c -> ... -> p
        for i in range(len(chain) - 1, 0, -1):  # chain[i] is the parent of chain[i-1]; reverse that edge
            forest.remove_edge(chain[i], chain[i - 1])
            forest.add_edge(chain[i - 1], chain[i])
        forest.add_edge(hub, p)
        fixes.append((p, jn, round(pl[p] / max(d(p), 1.0), 1)))
    return fixes


def _cut_soma_bridges(G, pts, cy, cx, excise_r):
    """STEP 1 v3 (see SOMA_BRIDGE_CUT): delete long straight tangential edges near the soma; returns (G, [(u, v, length_px)])."""
    def d(n):
        return float(np.hypot(pts[n][0] - cy, pts[n][1] - cx))
    zone = BRIDGE_ZONE * excise_r
    cut = []
    for u, v in list(G.edges()):
        if d(u) > zone or d(v) > zone:
            continue
        e = np.array([pts[v][0] - pts[u][0], pts[v][1] - pts[u][1]], float)
        ne = float(np.hypot(*e))
        if ne < max(BRIDGE_MIN_PX, BRIDGE_MIN_EX * excise_r):
            continue
        r = np.array([(pts[u][0] + pts[v][0]) / 2.0 - cy, (pts[u][1] + pts[v][1]) / 2.0 - cx], float)
        nr = float(np.hypot(*r))
        if nr < 1e-6:
            continue
        if abs(float(e @ r) / (ne * nr)) < BRIDGE_TAN_COS:
            G.remove_edge(u, v)
            cut.append((u, v, round(ne, 1)))
    return G, cut


def _fix_inward_near_soma(forest, pts, hub, cy, cx, excise_r, vox, prob, excise_mask=None, gray_img=None):
    """STEP 3 (see FIX_INWARD_NEAR_SOMA). Edits `forest` in place; returns [(x_px, y_px, kind, net_inward_um)] of applied fixes.
    INWARD_MASK_AWARE (A): samples inside the excised soma region are INVALID for probability evidence (the tracer does not use the model there). Too few valid samples =
        probability evidence UNAVAILABLE (never counted as a pass); then the raw image must show a ridge along the segment (centre brighter than both sides) as well as
        the direction and topology checks. An end inside the region is not skipped: the chain is walked back to the node where the neurite leaves the region (the boundary crossing).
    INWARD_ALL_ARMS (B): every arm meeting at an interior/branch end gets its OWN soma candidate and its own join candidate (own heading, own radial approach), evaluated
        separately with every other condition; a candidate is never accepted on the heading alone.
    Every candidate decision is appended to the jsonl in env INWARD_LOG (if set)."""
    import os, json
    logf = os.environ.get('INWARD_LOG')
    def d(n):
        return float(np.hypot(pts[n][0] - cy, pts[n][1] - cx))
    zone_px = excise_r + INWARD_ZONE_UM / vox
    Hp, Wp = prob.shape
    use_mask = INWARD_MASK_AWARE and excise_mask is not None
    def inmask(r, c):
        return bool(excise_mask[min(max(int(round(r)), 0), Hp - 1), min(max(int(round(c)), 0), Wp - 1)]) if excise_mask is not None else False
    def prob_evidence(a, b, free_um=None):
        fu = INWARD_FREE_UM if free_um is None else free_um
        """-> ('pass'|'fail'|'unavailable'|'free', mean prob over valid samples, n valid)"""
        L = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        if L <= fu / vox:
            return 'free', 1.0, int(L)
        vals = []
        for t in np.linspace(0, 1, max(2, int(L))):
            r = a[0] + (b[0] - a[0]) * t; c = a[1] + (b[1] - a[1]) * t
            if use_mask and inmask(r, c):
                continue
            vals.append(float(prob[min(max(int(round(r)), 0), Hp - 1), min(max(int(round(c)), 0), Wp - 1)]))
        if use_mask and len(vals) <= fu / vox:
            return 'unavailable', 0.0, len(vals)
        m = float(np.mean(vals)) if vals else 0.0
        return ('pass' if m >= INWARD_PROB_MIN else 'fail'), m, len(vals)
    def ridge_evidence(a, b):
        """raw-image ridge along a->b: fraction of samples whose centre is brighter than BOTH sides (+-INWARD_RIDGE_LAT px)"""
        if gray_img is None:
            return None
        L = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        if L < 2:
            return 1.0
        u = np.array([b[0] - a[0], b[1] - a[1]]) / L; nrm = np.array([-u[1], u[0]]); k = 0; n = 0
        Hg, Wg = gray_img.shape
        def g(pt):
            return float(gray_img[min(max(int(round(pt[0])), 0), Hg - 1), min(max(int(round(pt[1])), 0), Wg - 1)])
        for t in np.linspace(0, 1, max(3, int(L))):
            pc = np.array(a, float) + (np.array(b, float) - np.array(a, float)) * t
            c0 = g(pc); l0 = g(pc + nrm * INWARD_RIDGE_LAT); r0 = g(pc - nrm * INWARD_RIDGE_LAT)
            n += 1; k += int(c0 > l0 and c0 > r0)
        return k / max(n, 1)
    def descendants(u):
        out = [u]; k = 0
        while k < len(out):
            out.extend(forest.successors(out[k])); k += 1
        return out
    def heading(path_nodes, e):
        """unit heading INTO e along path_nodes (list ending near e, nearest first walking away from e) over ~INWARD_DIR_PX px; also radial approach per px"""
        acc = 0.0; prev = e; z = e
        for n in path_nodes:
            acc += float(np.hypot(pts[n][0] - pts[prev][0], pts[n][1] - pts[prev][1])); prev = n; z = n
            if acc >= INWARD_DIR_PX:
                break
        t = np.array(pts[e], float) - np.array(pts[z], float); nt = float(np.hypot(*t))
        if nt < 1e-6:
            return None, 0.0
        return t / nt, (d(z) - d(e)) / max(acc, 1e-6)
    fixes = []; skip = set()
    def log(rec):
        if logf:
            rec.update(hub_x=int(cx), hub_y=int(cy))
            with open(logf, 'a') as fh:
                fh.write(json.dumps(rec) + '\n')
    for _ in range(INWARD_MAX_ITER):
        cands = []
        for p in list(forest.nodes):
            if p == hub:
                continue
            pp = next(iter(forest.predecessors(p)), None)
            if pp is None or pp == hub:
                continue
            for c0 in list(forest.successors(p)):
                if (p, c0) in skip or d(c0) >= d(p) - 0.5:
                    continue
                chain = [p, c0]; u = c0
                while True:
                    nxt = [x for x in forest.successors(u) if d(x) < d(u) + 0.5]
                    if not nxt:
                        break
                    u = min(nxt, key=d); chain.append(u)
                e = chain[-1]
                if (d(p) - d(e)) * vox >= INWARD_MIN_NET_UM and d(e) <= zone_px:
                    cands.append((d(e), p, c0, chain))
        if not cands:
            break
        _, p, c0, chain = min(cands, key=lambda t: t[0])
        e = chain[-1]
        rec = dict(ex=int(pts[e][1]), ey=int(pts[e][0]), px=int(pts[p][1]), py=int(pts[p][0]), net_um=round((d(p) - d(e)) * vox, 1), etype='leaf' if forest.out_degree(e) == 0 else ('branch' if forest.out_degree(e) >= 2 else 'interior'), arms=[], decision='none')
        # end inside the soma region: use the node where the neurite leaves it
        if use_mask and inmask(pts[e][0], pts[e][1]):
            k = len(chain) - 1
            while k > 1 and inmask(pts[chain[k]][0], pts[chain[k]][1]):
                k -= 1
            if k <= 1 and inmask(pts[chain[k]][0], pts[chain[k]][1]):
                rec['decision'] = 'reject:end_in_soma_region_no_boundary_node'; log(rec); skip.add((p, c0)); continue
            rec['trimmed_to_boundary_node'] = True; chain = chain[:k + 1]; e = chain[-1]
        pe = np.array(pts[e], float)
        sub = set(descendants(c0))
        v = np.array([cy - pe[0], cx - pe[1]]); nv = float(np.hypot(*v))
        # arms meeting at e
        arms = [('chain', list(reversed(chain[:-1])))]
        if INWARD_ALL_ARMS:
            for x in list(forest.successors(e)):
                if x in chain:
                    continue
                pth = []; z = x
                while True:
                    pth.append(z)
                    kids = list(forest.successors(z))
                    if not kids or len(pth) > 60:
                        break
                    z = max(kids, key=lambda k_: len(descendants(k_)))
                arms.append(('child', pth))
        opts = []
        for name, pth in arms:
            te, slope = heading(pth, e)
            a = dict(arm=name, soma=None, join=None)
            if te is None or nv < 1e-6:
                a['note'] = 'no heading'; rec['arms'].append(a); continue
            cos_s = float(te @ (v / nv))
            L1 = max(nv - excise_r, 0.0); tpt = pe + (v / nv) * L1
            dup = min([float(np.hypot(pts[k][0] - pe[0], pts[k][1] - pe[1])) * vox for k in forest.successors(hub) if k not in sub] or [999.0])
            pe_state, pm, nval = prob_evidence(pe, tpt, INWARD_SOMA_FREE_UM)
            rg = ridge_evidence(pe, tpt)
            fails = []
            if cos_s < INWARD_DIR_COS: fails.append('direction')
            if INWARD_ALL_ARMS and slope < INWARD_SLOPE_MIN: fails.append('no_radial_approach')
            if L1 * vox > INWARD_ZONE_UM: fails.append('too_far')
            if dup <= INWARD_DUP_UM: fails.append('duplicate_spoke')
            if pe_state == 'fail': fails.append('prob_low')
            if pe_state == 'unavailable' and not (rg is not None and rg >= INWARD_RIDGE_FRAC): fails.append('prob_unavailable_and_no_raw_ridge')
            a['soma'] = dict(cos=round(cos_s, 2), slope=round(slope, 2), gap_um=round(L1 * vox, 1), prob=pe_state, prob_mean=round(pm, 3), n_valid=nval, ridge=None if rg is None else round(rg, 2), dup_um=round(dup, 1), fails=fails)
            if not fails:
                opts.append((L1, 'soma', hub, name))
            best = None
            for q in forest.nodes:
                if q == hub or q in sub:
                    continue
                dq = float(np.hypot(pts[q][0] - pe[0], pts[q][1] - pe[1]))
                if dq > INWARD_JOIN_UM / vox:
                    continue
                if dq > 1e-6 and float(te @ ((np.array(pts[q], float) - pe) / dq)) < INWARD_JOIN_COS:
                    continue
                if best is None or dq < best[0]:
                    best = (dq, q)
            jf = []
            if best is None:
                jf.append('no_node_in_direction_within_%.0fum' % INWARD_JOIN_UM)
            else:
                st_, pm2, nv2 = prob_evidence(pe, np.array(pts[best[1]], float)); rg2 = ridge_evidence(pe, np.array(pts[best[1]], float))
                if INWARD_ALL_ARMS and slope < INWARD_SLOPE_MIN: jf.append('no_radial_approach')
                if st_ == 'fail': jf.append('prob_low')
                if st_ == 'unavailable' and not (rg2 is not None and rg2 >= INWARD_RIDGE_FRAC): jf.append('prob_unavailable_and_no_raw_ridge')
                a['join'] = dict(dist_um=round(best[0] * vox, 1), prob=st_, prob_mean=round(pm2, 3), ridge=None if rg2 is None else round(rg2, 2), qx=int(pts[best[1]][1]), qy=int(pts[best[1]][0]), fails=jf)
                if not jf:
                    opts.append((best[0], 'join', best[1], name))
            if best is None:
                a['join'] = dict(fails=jf)
            rec['arms'].append(a)
        if not opts:
            rec['decision'] = 'reject'; log(rec); skip.add((p, c0)); continue
        L, kind, tgt, armname = min(opts, key=lambda t: t[0])
        # topology check: the new parent must not be inside the detached piece; the old upstream must stay connected
        if tgt in sub:                                  # the new parent must not lie inside the piece being re-rooted (would make a loop)
            rec['decision'] = 'reject:topology'; log(rec); skip.add((p, c0)); continue
        forest.remove_edge(p, c0)
        for i in range(1, len(chain) - 1):
            forest.remove_edge(chain[i], chain[i + 1]); forest.add_edge(chain[i + 1], chain[i])
        forest.add_edge(tgt, e)
        rec['decision'] = 'apply:%s via %s arm' % (kind, armname); log(rec)
        fixes.append((int(pts[e][1]), int(pts[e][0]), kind, round((d(p) - d(e)) * vox, 1)))
    if fixes:
        print('INWARD_FIXES', fixes, flush=True)
    return fixes


def _diag_inward_runs(forest, pts, hub, cy, cx, excise_r, vox, prob, excise_mask):
    """READ-ONLY diagnosis (env INWARD_DIAG = jsonl path). For every inward run (net >= INWARD_MIN_NET_UM) of the FINAL tree, whatever the type of its near
    end (leaf / branch node / interior node) and wherever it lies, record the evidence the Step 3 fixer would see and the reasons each option fails."""
    import os, json
    from scipy.ndimage import distance_transform_edt as _edt
    path = os.environ.get('INWARD_DIAG')
    if not path:
        return
    def d(n):
        return float(np.hypot(pts[n][0] - cy, pts[n][1] - cx))
    zone_px = excise_r + INWARD_ZONE_UM / vox
    Hp, Wp = prob.shape
    have_mask = excise_mask is not None and bool(np.asarray(excise_mask).any())
    edt = _edt(~np.asarray(excise_mask, bool)) if have_mask else None
    def samp(a, b):
        L = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        pr, inm = [], []
        for t in np.linspace(0, 1, max(2, int(L))):
            r = min(max(int(round(a[0] + (b[0] - a[0]) * t)), 0), Hp - 1); c = min(max(int(round(a[1] + (b[1] - a[1]) * t)), 0), Wp - 1)
            pr.append(float(prob[r, c])); inm.append(bool(excise_mask[r, c]) if have_mask else False)
        return float(np.mean(pr)), float(np.min(pr)), float(np.mean(inm)), L
    def descendants(u):
        out = [u]; k = 0
        while k < len(out):
            out.extend(forest.successors(out[k])); k += 1
        return out
    seen = set(); recs = []
    for p in list(forest.nodes):
        if p == hub:
            continue
        pp = next(iter(forest.predecessors(p)), None)
        for c0 in list(forest.successors(p)):
            if c0 in seen or d(c0) >= d(p) - 0.5:
                continue
            chain = [p, c0]; u = c0
            while True:
                nxt = [x for x in forest.successors(u) if d(x) < d(u) + 0.5]
                if not nxt:
                    break
                u = min(nxt, key=d); chain.append(u)
            seen.update(chain)
            e = chain[-1]; net = (d(p) - d(e)) * vox
            if net < INWARD_MIN_NET_UM:
                continue
            pe = np.array(pts[e], float)
            acc = 0.0; a = e
            for i in range(len(chain) - 1, 0, -1):
                acc += float(np.hypot(pts[chain[i]][0] - pts[chain[i - 1]][0], pts[chain[i]][1] - pts[chain[i - 1]][1])); a = chain[i - 1]
                if acc >= INWARD_DIR_PX:
                    break
            te = pe - np.array(pts[a], float); nt = float(np.hypot(*te)); te = te / nt if nt > 1e-6 else te
            sub = set(descendants(c0))
            od = forest.out_degree(e)
            r = dict(hub_x=int(cx), hub_y=int(cy), ex=int(pts[e][1]), ey=int(pts[e][0]), etype='leaf' if od == 0 else ('branch' if od >= 2 else 'interior'), out_deg=int(od),
                     e_surface_um=round((d(e) - excise_r) * vox, 1), in_zone=bool(d(e) <= zone_px), px=int(pts[p][1]), py=int(pts[p][0]), p_parent_is_hub=bool(pp == hub),
                     c0x=int(pts[c0][1]), c0y=int(pts[c0][0]), net_um=round(net, 1), chain_um=round(sum(float(np.hypot(pts[chain[i]][0] - pts[chain[i + 1]][0], pts[chain[i]][1] - pts[chain[i + 1]][1])) for i in range(len(chain) - 1)) * vox, 1),
                     n_sub=len(sub))
            if have_mask:
                r['e_in_mask'] = bool(excise_mask[min(max(int(round(pe[0])), 0), Hp - 1), min(max(int(round(pe[1])), 0), Wp - 1)])
                r['e_to_mask_um'] = round(float(edt[min(max(int(round(pe[0])), 0), Hp - 1), min(max(int(round(pe[1])), 0), Wp - 1)]) * vox, 1)
            v = np.array([cy - pe[0], cx - pe[1]]); nv = float(np.hypot(*v)); cs = float(te @ (v / nv)) if nv > 1e-6 and nt > 1e-6 else -9.0
            L1 = max(nv - excise_r, 0.0); tpt = pe + (v / nv) * L1 if nv > 1e-6 else pe
            pm, pmin, fmask, _ = samp(pe, tpt)
            dup = min([float(np.hypot(pts[k][0] - pe[0], pts[k][1] - pe[1])) * vox for k in forest.successors(hub) if k not in sub] or [999.0])
            fails = []
            if cs < INWARD_DIR_COS: fails.append('direction')
            if L1 * vox > INWARD_ZONE_UM: fails.append('too_far')
            if dup <= INWARD_DUP_UM: fails.append('duplicate_spoke')
            if L1 * vox > INWARD_FREE_UM and pm < INWARD_PROB_MIN: fails.append('image_evidence')
            r['soma'] = dict(cos=round(cs, 2), gap_um=round(L1 * vox, 1), prob_mean=round(pm, 3), prob_min=round(pmin, 3), frac_in_mask=round(fmask, 2), dup_um=round(dup, 1), fails=fails,
                             tx=int(round(tpt[1])), ty=int(round(tpt[0])))
            near = []; blocked = 0
            for q in forest.nodes:
                if q == hub:
                    continue
                dq = float(np.hypot(pts[q][0] - pe[0], pts[q][1] - pe[1])) * vox
                if dq > 15.0:
                    continue
                if q in sub:
                    blocked += 1; continue
                cq = float(te @ ((np.array(pts[q], float) - pe) / max(dq / vox, 1e-6))) if dq > 1e-6 else 1.0
                pq = samp(pe, np.array(pts[q], float))[0]
                near.append((dq, cq, pq, q))
            near.sort(key=lambda t: t[0])
            j = dict(n_within_15um=len(near), subtree_nodes_within_15um=blocked, fails=[])
            if near:
                dq, cq, pq, q = near[0]
                j.update(nearest_um=round(dq, 1), nearest_cos=round(cq, 2), nearest_prob=round(pq, 3), qx=int(pts[q][1]), qy=int(pts[q][0]))
                good = [t for t in near if t[0] <= INWARD_JOIN_UM and t[1] >= INWARD_JOIN_COS]
                if dq > INWARD_JOIN_UM: j['fails'].append('no_node_within_%.0fum' % INWARD_JOIN_UM)
                elif not good: j['fails'].append('direction')
                elif good[0][2] < INWARD_PROB_MIN: j['fails'].append('image_evidence')
                dirs = [t for t in near if t[1] >= INWARD_JOIN_COS]
                if dirs: j['best_directional_um'] = round(dirs[0][0], 1)
            else:
                j['fails'].append('no_node_within_15um' + ('(only_own_subtree)' if blocked else ''))
            r['join'] = j
            r['would_apply'] = bool(r['in_zone'] and (not r['soma']['fails'] or not j['fails']))
            recs.append(r)
    with open(path, 'a') as fh:
        for r in recs:
            fh.write(json.dumps(r) + '\n')


def _skel_endpoints(sk):
    nb = _convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode='constant')
    return np.argwhere(sk & (nb == 2))       # a skeleton pixel with exactly 1 neighbour = a dangling tip


def _prob_heal(mask, prob, gap=None, floor=None):
    """Grow each dangling skeleton tip along the least-cost prob path to the nearest OTHER mask
    component; if the path's bottleneck prob stays >= floor (a continuous faint neurite the model
    predicted, below the mask threshold), paint it into the mask. Reconnecting at the PIXEL level means
    the re-skeletonized neurite is continuous: no false tip, and (for collinear breaks) no false branch."""
    try:
        from skimage.graph import route_through_array
    except Exception:
        return mask
    gap = gap or HEAL_GAP; floor = floor or HEAL_FLOOR
    sk = skeletonize(mask)
    eps = _skel_endpoints(sk)
    if len(eps) == 0:
        return mask
    lab, _ = cc_label(mask)
    ridge = _max_filter(prob, size=3)                       # neurites have width; the 1px path may wander
    cost = (1.0 - np.clip(prob, 0.0, 1.0)).astype(np.float64) + 1e-3
    H, W = mask.shape
    heal = np.zeros_like(mask)
    for ey, ex in eps:
        mycomp = lab[ey, ex]
        y0, y1 = max(0, ey - gap), min(H, ey + gap + 1)
        x0, x1 = max(0, ex - gap), min(W, ex + gap + 1)
        tgt = (lab[y0:y1, x0:x1] > 0) & (lab[y0:y1, x0:x1] != mycomp)   # a DIFFERENT neurite nearby
        if not tgt.any():
            continue
        ty, tx = ey - y0, ex - x0
        tys, txs = np.nonzero(tgt)
        dd = (tys - ty) ** 2 + (txs - tx) ** 2
        k = int(dd.argmin())
        if dd[k] > gap * gap:
            continue
        try:
            path, _ = route_through_array(cost[y0:y1, x0:x1], (ty, tx), (int(tys[k]), int(txs[k])),
                                          fully_connected=True, geometric=True)
        except Exception:
            continue
        rr = np.array([ridge[y0 + p[0], x0 + p[1]] for p in path])
        if len(path) <= HEAL_LEN and rr.min() >= floor:       # a continuous faint neurite the whole way
            for py, px in path:
                heal[y0 + py, x0 + px] = True
    if heal.any():
        mask = mask | binary_dilation(heal, disk(HEAL_DISK))
    return mask


def _soma_edge_point(center, stub, body):
    """Where the segment center->stub exits the soma body mask = the soma SURFACE. Neurites should
    attach here, not at the centre: path length is measured from the soma edge, and drawing a line to
    the centre both looks wrong (a star through the soma) and inflates length by ~one soma radius."""
    H, W = body.shape
    c = np.array(center, float); s = np.array(stub, float)
    d = s - c; L = float(np.hypot(*d))
    if L < 1:
        return tuple(s)
    steps = int(L) + 1

    def inside(p):
        rr = int(min(max(round(p[0]), 0), H - 1)); cc = int(min(max(round(p[1]), 0), W - 1))
        return body[rr, cc]

    if inside(c):                                        # normal case: centre is inside the soma
        last_in = c
        for i in range(steps + 1):
            p = c + d * (i / steps)
            if inside(p):
                last_in = p
            else:
                break
        return tuple(last_in)
    for i in range(steps, -1, -1):                       # crescent/hollow soma: centre outside the mask.
        p = c + d * (i / steps)                           # walk inward from the stub; first inside = surface
        if inside(p):
            return tuple(p)
    return tuple(s)                                       # segment never enters the soma -> don't clip


def trace(prob, vox, soma_center, soma_radius=90, gray=None, soma_mask=None, soma_body=None, gray_raw=None):
    """prob (H,W) in [0,1], vox um/px, soma_center (row,col) px, soma_radius px.
    gray (H,W) optional raw grayscale image, enables the CNN crossing prior.
    soma_mask (H,W) optional bool: the bright soma region to excise so each neurite attaches radially.
    soma_body (H,W) optional bool: the segmented soma; neurites attach to its SURFACE (length from the
    soma edge, not the centre) so no line is drawn or counted through the soma.
    Returns dict: swc text, metrics, Sholl arrays, tree edges, mask."""
    set_vox(vox)
    mask = apply_hysteresis_threshold(prob, LOW, HIGH)
    mask = binary_closing(mask, disk(CLOSE_DISK))
    if PROB_HEAL:
        mask = _prob_heal(mask, prob)       # fill faint prob gaps at the mask level (before skeleton)
    skel = skeletonize(mask)
    dt = distance_transform_edt(mask)               # local half-width (for crossing test + later)
    G, pts = _build_graph(skel)
    G = _drop_small(G, MIN_COMP_EARLY if DEFER_DROP else MIN_COMP)   # early: only kill tiny noise specks;
    if G.number_of_nodes() == 0:                    # keep short real fragments alive for the bridges below
        return None
    crossings = []
    if RESOLVE_CROSSINGS:
        G, crossings = _resolve_crossings(G, pts, dt, gray)   # X-crossing vs Y-branch (geometry + CNN prior)
    G = _evidence_bridge(G, pts, prob)
    if CONTINUITY_BRIDGE:
        G = _continuity_bridge(G, pts, prob)        # join fragments by good continuation (faint gaps)
    if FRAG_STITCH:
        G = _fragment_stitch(G, pts)                # attach floating fragments across short gaps (geometry)
    if DEFER_DROP:
        G = _drop_small(G, MIN_COMP)                # now drop what STAYED isolated after bridging (noise)
        if G.number_of_nodes() == 0:
            return None
    cy, cx = soma_center
    # ---- soma hub: neurites emanate from the soma, each one independently ----
    # The saturated soma blob produces a skeleton tangle that falsely links every proximal neurite
    # into one web; with an unweighted MST that web is rooted by a single arbitrary spoke and the
    # other neurites chain through each other instead of radiating from the soma. So: (1) excise the
    # bright soma region, severing the tangle; (2) weight edges by length; (3) give every proximal
    # stub in the attach zone a cheap edge to the hub, so the weighted MST roots each neurite at the
    # soma (2D crossings near the soma are false links -- radial-from-soma is the correct prior).
    def _d(n):
        return np.hypot(pts[n][0] - cy, pts[n][1] - cx)
    excise_mask = soma_mask
    if EXCISE_FULL_SOMA and soma_body is not None and soma_body.any():
        # excise the WHOLE soma body (bright core leaves a tangle just outside it that fuses neurites).
        excise_mask = binary_dilation(soma_body, disk(SOMA_EXCISE_PAD))
        if soma_mask is not None:
            excise_mask = excise_mask | soma_mask
    if SOMA_SEVER_DISK and (soma_body is not None and soma_body.any() or soma_mask is not None and soma_mask.any()):
        # cut a disk around the soma centre: severs near-soma cross-links between DIFFERENT neurites so
        # each attaches to the soma on its own (no chain-rooting, no dangling soma-end tips).
        base = soma_body if (soma_body is not None and soma_body.any()) else soma_mask
        r_body = float(np.sqrt(base.sum() / np.pi))
        Hd, Wd = base.shape
        yy, xx = np.ogrid[:Hd, :Wd]
        disk_m = (yy - cy) ** 2 + (xx - cx) ** 2 <= (SOMA_SEVER_MULT * r_body) ** 2
        excise_mask = disk_m if excise_mask is None else (excise_mask | disk_m)
    if excise_mask is not None and excise_mask.any():
        Hm, Wm = excise_mask.shape
        inside = [n for n in list(G.nodes)
                  if excise_mask[min(max(int(round(pts[n][0])), 0), Hm - 1),
                                 min(max(int(round(pts[n][1])), 0), Wm - 1)]]
        G.remove_nodes_from(inside)
        G = _drop_small(G, 3)
        if FRAG_STITCH:
            G = _fragment_stitch(G, pts)             # re-stitch neurite pieces the excision just severed
                                                     # (before excision they were linked via the soma tangle)
        excise_r = float(np.sqrt(excise_mask.sum() / np.pi))
    else:
        excise_r = (soma_radius or 90)
    if SOMA_RADIAL_CUT:
        # cut tangential skeleton edges near the soma (false 2D cross-links between radial neurites)
        NS2 = (SOMA_RADIAL_R_MULT * excise_r) ** 2
        for u, v in list(G.edges()):
            pu, pv = pts[u], pts[v]
            my, mx = (pu[0] + pv[0]) * 0.5, (pu[1] + pv[1]) * 0.5
            if (my - cy) ** 2 + (mx - cx) ** 2 > NS2:
                continue
            rv = np.array([my - cy, mx - cx], float); rn = np.hypot(*rv)
            ev = np.array([pv[0] - pu[0], pv[1] - pu[1]], float); en = np.hypot(*ev)
            if rn < 1e-6 or en < 1e-6:
                continue
            if abs(float(np.dot(rv / rn, ev / en))) < SOMA_RADIAL_COS:   # tangential -> a cross-link, cut it
                G.remove_edge(u, v)
        G = _drop_small(G, 3)
        if FRAG_STITCH:
            G = _fragment_stitch(G, pts)                 # re-stitch any radial neurite nicked by the cut
    G = _weigh(G, pts)
    if SOMA_BRIDGE_CUT:
        G, _br = _cut_soma_bridges(G, pts, cy, cx, excise_r)
        if _STASH is not None:
            _STASH['bridge_cuts'] = _br
    if SOMA_HAIRPIN_CUT:
        G, _n_hp = _cut_soma_hairpins(G, pts, cy, cx, excise_r)
        if _STASH is not None:
            _STASH['hairpin_cuts'] = _n_hp
    R = excise_r + SOMA_MARGIN
    hub = len(pts); pts.append((cy, cx)); G.add_node(hub)
    if SOMA_ONE_SPOKE:
        # ONE spoke per component: attach each in-zone component to the soma at its single best point
        # (its proximal-most degree-1 stub, else its nearest node). Attaching *every* proximal stub of a
        # component instead creates a hub cycle whose weighted MST then drops a real neurite edge -- a
        # break manufactured right at the soma. One spoke per component can't cycle -> no such break.
        for comp in list(nx.connected_components(G)):
            if hub in comp:
                continue
            innodes = [n for n in comp if _d(n) <= R]
            if not innodes:
                continue
            ends = [n for n in innodes if G.degree(n) == 1]
            pick = min(ends, key=_d) if ends else min(innodes, key=_d)
            G.add_edge(hub, pick, weight=SOMA_HUB_W)
    else:
        for n in [n for n in list(G.nodes) if n != hub and G.degree(n) == 1 and _d(n) <= R]:
            G.add_edge(hub, n, weight=SOMA_HUB_W)        # each proximal stub attaches directly to the soma
        for comp in list(nx.connected_components(G)):    # fallback: any in-zone component with no stub
            if hub in comp:
                continue
            nearest = min(comp, key=_d)
            if _d(nearest) <= R:
                G.add_edge(hub, nearest, weight=SOMA_HUB_W)
    if SOMA_NEAR_MULTISPOKE:                         # give every near-soma dangling stub its OWN radial spoke
        rr = SOMA_NEAR_MULT * excise_r
        for n in [n for n in list(G.nodes)
                  if n != hub and G.degree(n) == 1 and _d(n) <= rr and not G.has_edge(hub, n)]:
            G.add_edge(hub, n, weight=SOMA_HUB_W)   # MST keeps this short spoke, drops the long detour edge
    if SOMA_ANGLE_ATTACH:                            # attach near-soma dangling tips pointing AT the soma
        R2 = SOMA_ATTACH_R_MULT * excise_r           # (a neurite broken at the soma edge).
        hubset = set(nx.node_connected_component(G, hub))
        cands = [n for n in list(G.nodes) if n != hub and G.degree(n) == 1 and _d(n) <= R2]
        cands.sort(key=_d)                            # nearest the soma first
        for e in cands:
            if e in hubset:
                continue
            te = _endpoint_dir(G, pts, e)
            v = np.array([cy - pts[e][0], cx - pts[e][1]], float); nn = np.hypot(*v)
            if not (nn > 1e-6 and float(np.dot(te, v / nn)) >= SOMA_ATTACH_COS):
                continue
            comp_e = nx.node_connected_component(G, e)
            if hub not in comp_e:                     # not soma-connected -> plain bridge, no loop
                G.add_edge(hub, e, weight=SOMA_HUB_W)
                hubset |= nx.node_connected_component(G, e)
            elif SOMA_LOOP_BREAK:                     # already soma-connected: attach + drop the FARTHEST
                spokes = [n for n in G.neighbors(hub) if n in comp_e and n != e]  # existing redundant spokes
                if spokes:                            # break the worst-rooted one (a spoke, not a neurite)
                    G.add_edge(hub, e, weight=SOMA_HUB_W)
                    G.remove_edge(hub, max(spokes, key=_d))   # e stays connected -> mark fixed, no new tip
    roots = set()
    forest = nx.DiGraph()
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        root = hub if hub in comp else min(comp, key=_d)
        roots.add(root)
        T = nx.bfs_tree(nx.minimum_spanning_tree(sub, weight='weight'), root)
        forest.add_edges_from(T.edges()); forest.add_nodes_from(T.nodes())
    if SOMA_DETOUR_REROOT:
        _fx = _reroot_soma_detours(forest, pts, hub, cy, cx, excise_r, vox)
        if _STASH is not None:
            _STASH['detour_fixes'] = _fx
    if SOMA_SPOKE_MERGE:
        _sm = _merge_small_spokes(forest, pts, hub, vox)
        if _STASH is not None:
            _STASH['spoke_merges'] = _sm
    if FIX_INWARD_NEAR_SOMA:
        _fx3 = _fix_inward_near_soma(forest, pts, hub, cy, cx, excise_r, vox, prob, excise_mask, (gray_raw if gray_raw is not None else gray))
        if _STASH is not None:
            _STASH['inward_fixes'] = _fx3
    forest = _prune(forest, pts, roots, SPUR)
    forest = _collapse_junctions(forest, pts, roots, JUNCTION_MERGE)
    forest = _prune(forest, pts, roots, SPUR)
    forest = _remove_hairs(forest, pts, roots, HAIR_STUB, HAIR_MAIN)
    _tb = _turnback_near_soma(forest, pts, hub, roots, cy, cx, vox, prob)
    _pending = {e for e, _s, k, _d in _tb if k == 'pending'}
    if _tb:
        print('TURNBACK', [(int(pts[e][1]), int(pts[e][0]), int(pts[s_][1]), int(pts[s_][0]), k, d) for e, s_, k, d in _tb], flush=True)
    _pc = _piece_rules(forest, pts, hub, roots, vox)
    _pending = {e for e in _pending if e in forest and e in roots}
    if _pc['joined'] or _pc['dropped'] or _pc['candidates']:
        print('PIECES joined %d dropped %d candidates %d' % (len(_pc['joined']), len(_pc['dropped']), len(_pc['candidates'])), flush=True)
    if FIX_INWARD_NEAR_SOMA is not None:
        _diag_inward_runs(forest, pts, hub, cy, cx, excise_r, vox, prob, excise_mask)

    if _STASH is not None:
        _STASH.update(forest=forest, pts=pts, hub=hub, roots=roots, G=G)
    tips = [n for n in forest.nodes if forest.out_degree(n) == 0 and n not in roots]
    branches = [n for n in forest.nodes if forest.out_degree(n) >= 2 and n not in roots]

    def _edge_pts(u, v):
        """endpoints for drawing/length; a soma-hub edge is clipped to the soma surface (not the centre)."""
        pu, pv = pts[u], pts[v]
        if soma_body is not None:
            if u == hub:
                pu = _soma_edge_point((cy, cx), pv, soma_body)
            elif v == hub:
                pv = _soma_edge_point((cy, cx), pu, soma_body)
        return pu, pv

    edges_px = [_edge_pts(u, v) for u, v in forest.edges()]
    tot_len = sum(np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in edges_px) * vox

    # Sholl over the whole arbor (all edges), from soma, 10 um steps
    dist = {n: np.hypot(pts[n][0] - cy, pts[n][1] - cx) * vox for n in forest.nodes}
    rmax = max(dist.values()) if dist else 0
    radii = np.arange(0, rmax + 10, 10.0)
    inters = np.array([sum(1 for u, v in forest.edges() if (dist[u] < r) != (dist[v] < r)) for r in radii])

    dt = distance_transform_edt(mask)
    order = list(forest.nodes())
    if _tb or _pc['joined']:                   # reversed or re-joined stretches: write parents before children (hub first, then depth-first per tree)
        _rs = [hub] + [r for r in order if r in roots and r != hub]; order = []
        for _r in _rs:
            _st = [_r]
            while _st:
                _u = _st.pop(); order.append(_u); _st.extend(reversed(list(forest.successors(_u))))
    nid = {n: k + 1 for k, n in enumerate(order)}
    parent = {n: -1 for n in roots}
    for u, v in forest.edges():
        parent[v] = u
    H, W = dt.shape
    # neurite radius from the IMAGE (half of the FWHM across the neurite), not from the mask: the
    # mask is a constant-width ribbon by construction (centreline labels), so its distance
    # transform carries no thickness information. Validated on synthetic known widths: ranks
    # correctly, ~+0.2 um overestimate from optical blur. Nodes too dim to measure get the cell median.
    _g_fw = gray_raw if gray_raw is not None else gray     # raw 16-bit preferred; 8-bit fallback
    fw_r = _fwhm_radii_um(_g_fw, pts, order, parent, vox) if _g_fw is not None else {}
    med_r = float(np.median([v for v in fw_r.values() if v > 0])) if any(v > 0 for v in fw_r.values()) else 0.0
    lines = ["# neurite arbor from desktop app (tracer v1.4-fwhm); um; voxel %.4f; radius = FWHM/2 from image" % vox]
    if _pending:
        lines.append("# near the soma, connection pending (separate pieces, root is not a confirmed origin): node ids " + " ".join(str(nid[n]) for n in order if n in _pending))
    if _pc['candidates']:
        lines.append("# join candidates (piece root x y um -> main tree x y um), not applied: " + "; ".join(
            "%.1f %.1f -> %.1f %.1f" % (a[1] * vox, a[0] * vox, b[1] * vox, b[0] * vox) for a, b in _pc['candidates']))
    for n in order:
        y, x = pts[n]
        pa = parent.get(n, -1)
        rr = int(min(max(round(y), 0), H - 1)); cc = int(min(max(round(x), 0), W - 1))
        r_um = fw_r.get(n, 0.0)
        if r_um <= 0:
            r_um = med_r if med_r > 0 else max(dt[rr, cc] * vox, 0.1)
        lines.append("%d %d %.3f %.3f 0.0 %.3f %d" % (
            nid[n], 1 if (n in roots and n not in _pending) else 3, x * vox, y * vox, r_um, nid[pa] if pa != -1 else -1))
    tm = tree_metrics(forest, pts, roots, (cy, cx), vox, radii, inters)
    out = dict(swc="\n".join(lines) + "\n", total_length_um=tot_len,
               neurite_width_um=round(2 * med_r, 2) if med_r > 0 else 0.0,
               neurite_width_n=int(sum(1 for v in fw_r.values() if v > 0)),
               num_tips=len(tips), num_branch_points=len(branches),
               sholl_radii=radii, sholl_intersections=inters,
               branch_points_px=[tuple(pts[n]) for n in branches],   # (row,col) of each branch node
               tips_px=[tuple(pts[n]) for n in tips],                 # (row,col) of each terminal tip
               crossings_px=crossings,                                # (row,col) of each resolved X-crossing
               soma_rc=(cy, cx), tree_edges_px=edges_px, mask=mask,
               join_candidates_px=_pc['candidates'])                   # [(piece root (row,col), main-tree point (row,col))], for the editor
    out.update(tm)
    return out

# ---------------------------------------------------------------------------------------------------------------------
# Multi-soma images (sparse-labelled cultures, several filled cells per field; 2026-10-25). Each neurite is assigned to the soma
# it is reached from along the skeleton ("soma retrotracing"): multi-source shortest paths from every soma hub over the weighted
# skeleton graph. The existing single-soma tracer then runs once per soma on a probability map where the skeleton of the OTHER
# somas' neurites is blanked, so every rule (soma attachment, crossings, direction, pieces) works unchanged.
MULTI_MIN_SOMA_UM = 4.0        # detect_somas: a bright blob counts as a soma when its equivalent radius is at least this (um)
MULTI_MAX_SOMAS = 12
MULTI_BLANK_PAD_PX = 4         # pixels around another cell's skeleton that are blanked in this cell's probability map


def detect_somas(gray_u8, vox, max_n=MULTI_MAX_SOMAS):
    """(row, col) centres of the bright soma-like blobs in the image, brightest first. Same recipe as detect_soma (99.7th
    percentile, opening) but keeps every blob whose equivalent radius is >= MULTI_MIN_SOMA_UM."""
    set_vox(vox)
    g = gray_u8.astype(float)
    b = opening(g >= np.percentile(g, 99.7), disk(SOMA_DETECT_DISK))
    lab, n = cc_label(b)
    if n == 0:
        return []
    out = []
    for i in range(1, n + 1):
        m = lab == i; a = int(m.sum())
        if np.sqrt(a / np.pi) * vox < MULTI_MIN_SOMA_UM:
            continue
        ys, xs = np.nonzero(m); w = g[ys, xs]
        out.append((float(w.sum()), float((ys * w).sum() / w.sum()), float((xs * w).sum() / w.sum())))
    out.sort(reverse=True)
    return [(r, c) for _, r, c in out[:max_n]]


def assign_to_somas(prob, vox, soma_centers, soma_bodies=None):
    """node -> soma index by shortest skeleton path from the soma hubs. Returns (pts, assignment dict, skeleton graph).
    Nodes in no hub's reach are given to the Euclidean-nearest soma so every piece belongs to exactly one cell."""
    set_vox(vox)
    mask = apply_hysteresis_threshold(prob, LOW, HIGH)
    mask = binary_closing(mask, disk(CLOSE_DISK))
    if PROB_HEAL:
        mask = _prob_heal(mask, prob)
    skel = skeletonize(mask)
    G, pts = _build_graph(skel)
    G = _drop_small(G, MIN_COMP_EARLY if DEFER_DROP else MIN_COMP)
    if G.number_of_nodes() == 0:
        return pts, {}, G
    G = _weigh(G, pts)
    hubs = []
    for k, (cy, cx) in enumerate(soma_centers):
        body = soma_bodies[k] if soma_bodies is not None and soma_bodies[k] is not None else None
        r_ex = float(np.sqrt(body.sum() / np.pi)) if body is not None and body.any() else 90.0
        R = r_ex + SOMA_MARGIN
        hub = len(pts); pts.append((cy, cx)); G.add_node(hub); hubs.append(hub)
        for comp in nx.connected_components(G.subgraph([n for n in G.nodes if n not in hubs])):
            inn = [n for n in comp if np.hypot(pts[n][0] - cy, pts[n][1] - cx) <= R]
            if not inn:
                continue
            ends = [n for n in inn if G.degree(n) == 1]
            pick = min(ends or inn, key=lambda n: np.hypot(pts[n][0] - cy, pts[n][1] - cx))
            G.add_edge(hub, pick, weight=SOMA_HUB_W + np.hypot(pts[pick][0] - cy, pts[pick][1] - cx))
    assign = {}
    for k, hub in enumerate(hubs):
        for n, d in nx.single_source_dijkstra_path_length(G, hub, weight='weight').items():
            if n in hubs:
                continue
            if n not in assign or d < assign[n][1]:
                assign[n] = (k, d)
    C = np.array(soma_centers, float)
    for n in G.nodes:
        if n in hubs or n in assign:
            continue
        p = np.array(pts[n], float); assign[n] = (int(np.argmin(np.hypot(*(C - p).T))), float('inf'))
    return pts, {n: k for n, (k, _) in assign.items()}, G


def trace_multi(prob, vox, soma_centers, soma_radii=None, gray=None, soma_masks=None, soma_bodies=None, gray_raw=None):
    """trace() once per soma; neurites assigned to the other somas are blanked out of this soma's probability map first.
    Returns list of (result dict or None) in soma order, plus the assignment map (pts, node -> soma)."""
    pts, assign, G = assign_to_somas(prob, vox, soma_centers, soma_bodies)
    H, W = prob.shape
    results = []
    for k, (cy, cx) in enumerate(soma_centers):
        other = np.zeros((H, W), bool)
        for n, kk in assign.items():
            if kk != k:
                r, c = pts[n]; other[min(max(int(round(r)), 0), H - 1), min(max(int(round(c)), 0), W - 1)] = True
        if other.any():
            other = binary_dilation(other, disk(MULTI_BLANK_PAD_PX))
            # never blank this cell's own soma neighbourhood
            own = soma_bodies[k] if soma_bodies is not None and soma_bodies[k] is not None else None
            if own is not None:
                other &= ~binary_dilation(own, disk(MULTI_BLANK_PAD_PX))
        pk = prob.copy(); pk[other] = 0.0
        res = trace(pk, vox, (cy, cx), soma_radii[k] if soma_radii is not None else 90,
                    gray=gray, soma_mask=soma_masks[k] if soma_masks is not None else None,
                    soma_body=soma_bodies[k] if soma_bodies is not None else None, gray_raw=gray_raw)
        results.append(res)
    return results, (pts, assign)
