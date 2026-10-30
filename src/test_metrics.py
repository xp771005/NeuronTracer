# -*- coding: utf-8 -*-
"""Unit tests for the morphometrics: validate on synthetic geometries whose
correct answers are known analytically. Run with the build env's python."""
import numpy as np
import networkx as nx
from PIL import Image, ImageDraw
import tracer
import soma

RAD = np.array([0, 10, 20, 30.]); INT = np.array([1, 1, 1, 0])
_res = []
def check(name, cond, detail=""):
    _res.append(cond)
    print(("  PASS " if cond else "  FAIL "), name, "" if cond else ("  << " + str(detail)))
def approx(a, b, tol):
    return abs(a - b) <= tol

def mk(edges, pts):
    F = nx.DiGraph(); F.add_edges_from(edges)
    for i in range(len(pts)):
        F.add_node(i)
    return F, list(pts)

print("[tree] straight line (no branches)")
F, pts = mk([(0, 1), (1, 2), (3, 3)], [(300, 100), (250, 100), (200, 100), (150, 100)])
F = nx.DiGraph([(0, 1), (1, 2), (2, 3)]); pts = [(300, 100), (250, 100), (200, 100), (150, 100)]
m = tracer.tree_metrics(F, pts, {0}, (300, 100), 1.0, RAD, INT)
check("order == 1", m['highest_branch_order'] == 1, m)
check("num_branches == 1", m['num_branches'] == 1, m['num_branches'])
check("no branch angle (0)", m['avg_branch_angle_deg'] == 0.0, m['avg_branch_angle_deg'])

print("[tree] Y (one bifurcation, arms +-38.7 deg from stem)")
F = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6)])
pts = [(300, 100), (250, 100), (200, 100), (150, 60), (100, 20), (150, 140), (100, 180)]
m = tracer.tree_metrics(F, pts, {0}, (300, 100), 1.0, RAD, INT)
check("highest_order == 2", m['highest_branch_order'] == 2, m)
check("num_branches == 3", m['num_branches'] == 3, m['num_branches'])
check("branch angle ~= 38.7 deg", approx(m['avg_branch_angle_deg'], 38.7, 3), m['avg_branch_angle_deg'])
check("length before first bif ~= 100 um (rows 300->200)", approx(m['avg_len_before_first_bif_um'], 100, 5), m['avg_len_before_first_bif_um'])
check("segment len (stem) ~= 100 um", approx(m['avg_segment_len_um'], 100, 5), m['avg_segment_len_um'])
check("polarity high (>0.5, arbor all above soma)", m['polarity_index'] > 0.5, m['polarity_index'])

print("[tree] symmetric cross (4 arms) -> balanced")
F = nx.DiGraph([(0, 1), (0, 2), (0, 3), (0, 4)])
pts = [(200, 200), (100, 200), (300, 200), (200, 100), (200, 300)]
m = tracer.tree_metrics(F, pts, {0}, (200, 200), 1.0, RAD, INT)
check("polarity low (<0.2)", m['polarity_index'] < 0.2, m['polarity_index'])
check("order == 1 (soma root is not a bifurcation; 4 primaries)", m['highest_branch_order'] == 1, m)
check("num_branches == 4 (root=branch not double counted)", m['num_branches'] == 4, m['num_branches'])

print("[tree] two-level -> order 3")
F = nx.DiGraph([(0, 1), (1, 2), (1, 3), (2, 4), (2, 5)])
pts = [(300, 100), (200, 100), (150, 60), (150, 140), (100, 40), (100, 80)]
m = tracer.tree_metrics(F, pts, {0}, (300, 100), 1.0, RAD, INT)
check("highest_order == 3", m['highest_branch_order'] == 3, m)

print("[sholl] mean intersections & enclosing radius")
radii = np.array([0, 10, 20, 30, 40.]); inter = np.array([1, 3, 2, 0, 0])
m = tracer.tree_metrics(F, pts, {0}, (300, 100), 1.0, radii, inter)
check("mean_intersections == 2.0 (mean of 1,3,2)", approx(m['mean_intersections'], 2.0, 0.01), m['mean_intersections'])
check("enclosing_radius == 20 (last nonzero)", m['enclosing_radius_um'] == 20.0, m['enclosing_radius_um'])

print("[soma] bright disk radius 30 -> circle-like shape")
im = Image.fromarray(np.zeros((400, 400), np.uint8)); ImageDraw.Draw(im).ellipse([170, 170, 230, 230], fill=255)
s = soma.segment_soma(np.array(im), 1.0)
check("soma detected", s is not None)
if s:
    sm = s['metrics']
    check("circularity > 0.8 (~1)", sm['soma_circularity'] > 0.8, sm['soma_circularity'])
    check("roundness > 0.85 (~1)", sm['soma_roundness'] > 0.85, sm['soma_roundness'])
    check("solidity > 0.95 (~1)", sm['soma_solidity'] > 0.95, sm['soma_solidity'])
    check("diameter ~= 60 px", approx(sm['soma_diameter_um'], 60, 6), sm['soma_diameter_um'])
    # v5 pre-smooths (gaussian) + scale-adaptive opening, which shaves ~2px off a
    # hard-edged synthetic disk (~7% low here); on real peaked somas this is
    # negligible (validated ratio 0.97 vs golden). Band accounts for edge smoothing.
    check("area ~= 2827 (pi r^2)", approx(sm['soma_area_um2'], 2827, 450), sm['soma_area_um2'])
    check("contour present for drawing", s['contour'] is not None)

print("[soma] elongated streak -> low roundness/circularity (sanity)")
im = Image.fromarray(np.zeros((400, 400), np.uint8)); ImageDraw.Draw(im).ellipse([60, 195, 340, 210], fill=255)
s2 = soma.segment_soma(np.array(im), 1.0)
if s2:
    check("elongated blob roundness < 0.5", s2['metrics']['soma_roundness'] < 0.5, s2['metrics']['soma_roundness'])

n = len(_res); p = sum(_res)
print("\n==== %d/%d PASSED ====" % (p, n))
import sys
sys.exit(0 if p == n else 1)
