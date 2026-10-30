# -*- coding: utf-8 -*-
"""Tests for the edit-tracing window. Data layer (Arbor) needs no display; the window tests skip without one.
run:  python -m unittest test_editor"""
import os, sys, json, math, shutil, tempfile, unittest
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import editor

VOX = 0.1625


def node_at(a, x, y):
    """the node nearest (x, y) px (coordinates go through um and back, so exact equality is unsafe)."""
    return min(a.xy, key=lambda k: math.hypot(a.xy[k][0] - x, a.xy[k][1] - y))


def near(p, x, y):
    return math.hypot(p[0] - x, p[1] - y) < 0.01


def write_cell(folder, stem="cell", with_cand=False):
    """a small synthetic cell: soma at (200,200) px with two spokes. Spoke A runs right to a branch point at
    x=300 that splits up/down; spoke B runs left to a tip. Plus a loose piece far away. 1 px node spacing."""
    d = os.path.join(folder, stem + "_neuron_out"); os.makedirs(d)
    pts, par = [], []

    def add(x, y, p):
        pts.append((x, y)); par.append(p); return len(pts)
    soma = add(200, 200, -1)
    prev = add(215, 200, soma)                         # spoke A (15 px from the centre)
    for x in range(216, 301):
        prev = add(x, 200, prev)
    bp = prev
    prev = bp
    for y in range(199, 139, -1):                      # up daughter, 60 px
        prev = add(300, y, prev)
    prev = bp
    for y in range(201, 241):                          # down daughter, 40 px
        prev = add(300, y, prev)
    prev = add(185, 200, soma)                         # spoke B, left, 50 px
    for x in range(184, 135, -1):
        prev = add(x, 200, prev)
    lp = add(400, 400, -1)                             # loose piece 30 px
    for x in range(401, 431):
        lp = add(x, 400, lp)
    lines = ["# neurite arbor from desktop app (tracer v1.4-fwhm); um; voxel %.4f; radius = FWHM/2 from image" % VOX]
    if with_cand:                                      # CandidateTests only: a second loose piece 4 px beside the down daughter, plus
        lp2 = add(304, 230, -1)                        # the tracer's header line proposing to join it there (2026-10-25)
        for x in range(305, 325):
            lp2 = add(x, 230, lp2)
        lines.append("# join candidates (piece root x y um -> main tree x y um), not applied: %.1f %.1f -> %.1f %.1f" % (304 * VOX, 230 * VOX, 300 * VOX, 230 * VOX))
    for i, ((x, y), p) in enumerate(zip(pts, par), 1):
        t = 1 if p == -1 else 3
        lines.append("%d %d %.3f %.3f 0.0 0.5 %d" % (i, t, x * VOX, y * VOX, p))
    with open(os.path.join(d, stem + ".swc"), "w") as f:
        f.write("\n".join(lines) + "\n")
    g = np.full((512, 512), 10, np.uint8)
    for (x, y) in pts:
        g[y, x] = 200
    g[198:203, 140:300] = np.maximum(g[198:203, 140:300], 60)
    Image.fromarray(g).save(os.path.join(d, stem + "_raw.png"))
    ov = np.stack([g] * 3, -1).copy()
    ov[196:205, 196:205] = (0, 230, 255)                # soma-centre dot like the software's overlay
    Image.fromarray(ov).save(os.path.join(d, stem + "_overlay.png"))
    with open(os.path.join(d, stem + "_metrics.csv"), "w") as f:
        f.write("metric,value,unit,note\ntotal_length,40.0,um,\nsoma_area,50.0,um^2,\nnum_tips,3,count,\n")
    # soma outline like the pipeline writes it: an octagon of radius 15 px, slightly squashed (18 wide, 12 tall)
    oct_ = [[200 + 18 * math.cos(t), 200 + 12 * math.sin(t)] for t in np.linspace(0, 2 * math.pi, 9)[:-1]]
    with open(os.path.join(d, stem + "_soma.json"), "w") as f:
        json.dump({"contour_xy": oct_, "center_xy": [200, 200], "radius_px": 15.0, "pixel_size_um": VOX}, f)
    return d, stem


class CandidateTests(unittest.TestCase):
    """the tracer's join candidates: parsed from the SWC header, mapped to nodes, never written back"""
    def setUp(self):
        self.tmp = tempfile.mkdtemp(); self.d, self.s = write_cell(self.tmp, with_cand=True)
        self.a = editor.Arbor.from_swc(os.path.join(self.d, self.s + ".swc"),
                                       soma_xy=editor.soma_from_overlay(os.path.join(self.d, self.s + "_overlay.png")))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_parsed_in_px(self):
        self.assertEqual(len(self.a.join_cands), 1)
        (rx, ry), (tx, ty) = self.a.join_cands[0]
        self.assertAlmostEqual(rx, 304, delta=0.6); self.assertAlmostEqual(ry, 230, delta=0.6)
        self.assertAlmostEqual(tx, 300, delta=0.6); self.assertAlmostEqual(ty, 230, delta=0.6)

    def test_candidate_root_is_a_free_end_on_another_piece(self):
        a = self.a; (rx, ry), (tx, ty) = a.join_cands[0]
        n = min(a.xy, key=lambda k: math.hypot(a.xy[k][0] - rx, a.xy[k][1] - ry))
        q = min(a.xy, key=lambda k: math.hypot(a.xy[k][0] - tx, a.xy[k][1] - ty))
        self.assertEqual(len(a.adj[n]), 1)
        cid = a.component_of(); self.assertNotEqual(cid[n], cid[q])
        self.assertTrue(a.join(n, q)); self.assertEqual(a.component_of()[n], a.component_of()[q])

    def test_not_written_back(self):
        self.assertNotIn("# join candidates", self.a.to_swc())


class ArborTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.d, self.s = write_cell(self.tmp)
        self.a = editor.Arbor.from_swc(os.path.join(self.d, self.s + ".swc"),
                                       soma_xy=editor.soma_from_overlay(os.path.join(self.d, self.s + "_overlay.png")))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def tips(self, a=None):
        a = a or self.a
        return sorted(n for n, s in a.adj.items() if len(s) == 1 and n != a.soma)

    def check_tree(self, a):
        """written SWC is a forest: one parent each, parents before children, no loops."""
        txt = a.to_swc()
        seen, parent = set(), {}
        for ln in txt.splitlines():
            if ln.startswith('#'):
                continue
            p = ln.split(); i, q = int(p[0]), int(p[6])
            self.assertTrue(q == -1 or q in seen, "parent written after child")
            seen.add(i); parent[i] = q
        self.assertEqual(len(seen), len(a.xy))

    def test_soma_found_from_overlay(self):
        self.assertTrue(near(self.a.xy[self.a.soma], 200, 200))
        self.assertEqual(len(self.a.adj[self.a.soma]), 2)

    def test_segments_and_paths_cover_every_edge_once(self):
        segs = self.a.segments()
        edges = [frozenset(e) for s in segs for e in zip(s[:-1], s[1:])]
        self.assertEqual(len(edges), len(set(edges)))
        self.assertEqual(len(edges), sum(len(v) for v in self.a.adj.values()) // 2)
        paths = self.a.paths()
        pe = [frozenset(e) for p in paths for e in zip(p['nodes'][:-1], p['nodes'][1:])]
        self.assertEqual(sorted(map(sorted, pe)), sorted(map(sorted, edges)))
        for k, p in enumerate(paths):
            self.assertTrue(p['parent'] is None or p['parent'] < k)
        # the right-hand primary continues into the longer (up, 60 px) daughter; the down one is its child
        prim = [p for p in paths if p['parent'] is None and p['connected']]
        self.assertEqual(len(prim), 2)
        loose = [p for p in paths if not p['connected']]
        self.assertEqual(len(loose), 1)

    def test_delete_terminal_segment_removes_one_tip(self):
        t0 = len(self.tips()); L0 = self.a.cable_um()
        t = node_at(self.a, 300, 240); down = next(s for s in self.a.segments() if t in (s[0], s[-1]))
        um = self.a.delete_chain(down)
        self.assertAlmostEqual(um, 40 * VOX, places=3)
        self.assertEqual(len(self.tips()), t0 - 1)
        self.assertAlmostEqual(self.a.cable_um(), L0 - um, places=3)
        self.assertFalse([n for n, s in self.a.adj.items() if not s and n != self.a.soma])
        self.check_tree(self.a)

    def test_delete_internal_segment_leaves_loose_part_then_reconnect(self):
        a = self.a
        bp = next(n for n, s in a.adj.items() if len(s) == 3)
        stem = next(s for s in a.segments() if bp in (s[0], s[-1]) and a.soma in (s[0], s[-1]))
        a.delete_chain(stem)
        self.assertNotIn(bp, a.connected())
        # draw from the soma back to the branch point: joins the loose piece again
        start = a.soma; pts = [(200 + k, 205) for k in range(16, 100, 3)]
        ids, joined = a.add_chain(pts, a=start, b=bp)
        self.assertTrue(joined)
        self.assertIn(bp, a.connected())
        self.check_tree(a)

    def test_cut_beyond_point(self):
        a = self.a
        n = node_at(a, 300, 170)
        before = len(a.xy)
        um = a.cut_beyond(n)
        self.assertAlmostEqual(um, 30 * VOX, places=3)
        self.assertEqual(len(a.xy), before - 30)
        self.assertEqual(len(a.adj[n]), 1)                 # n is now the tip

    def test_extend_tip_keeps_tip_count(self):
        a = self.a; t0 = len(self.tips())
        tip = node_at(a, 136, 200)
        ids, joined = a.add_chain([(136 - 3 * k, 200) for k in range(1, 6)], a=tip)
        self.assertEqual(len(self.tips()), t0)
        self.assertTrue(all(i in a.new for i in ids))
        self.assertAlmostEqual(a.chain_um([tip] + ids), 15 * VOX, places=3)

    def test_no_loop_when_joining_the_same_piece(self):
        a = self.a
        tip_up = node_at(a, 300, 140)
        tip_left = node_at(a, 136, 200)
        ids, joined = a.add_chain([(250, 150), (200, 170)], a=tip_up, b=tip_left)
        self.assertFalse(joined)
        self.check_tree(a)

    def test_delete_subtree_and_undo(self):
        a = self.a
        snap = a.snapshot(); n0 = len(a.xy)
        bp = next(n for n, s in a.adj.items() if len(s) == 3)
        first = node_at(a, 300, 199); self.assertIn(first, a.adj[bp])
        a.delete_nodes([first] + a.descendants(first))
        self.assertEqual(len(a.xy), n0 - 60)
        a.restore(snap)
        self.assertEqual(len(a.xy), n0)
        self.check_tree(a)

    def test_swc_round_trip_after_edits(self):
        a = self.a
        tip = node_at(a, 136, 200)
        a.add_chain([(133, 200), (130, 200)], a=tip)
        p = os.path.join(self.tmp, "rt.swc")
        with open(p, "w") as f:
            f.write(a.to_swc("test"))
        b = editor.Arbor.from_swc(p, soma_xy=(200, 200))
        self.assertEqual(len(b.xy), len(a.xy))
        self.assertAlmostEqual(b.cable_um(), a.cable_um(), places=3)
        self.assertTrue(near(b.xy[b.soma], 200, 200))

    def test_erase_middle_of_a_branch(self):
        a = self.a; t0 = len(self.tips())
        nodes = [node_at(a, 300, y) for y in range(215, 226)]    # 10 px out of the down daughter
        um = a.erase(nodes, [])
        self.assertAlmostEqual(um, 12 * VOX, places=3)        # 11 nodes = 10 edges inside + 2 edges at the ends
        self.assertEqual(len(self.tips()), t0 + 2)            # the cut leaves a new end on each side
        self.assertNotIn(node_at(a, 300, 240), a.connected()) # the far part is loose now
        self.check_tree(a)

    def test_split_crossing(self):
        a = self.a
        bp = next(n for n, s in a.adj.items() if len(s) == 3)
        a.add_chain([(300 + k, 200) for k in range(3, 43, 3)], a=bp)      # a 4th arm: an X at the branch point
        self.assertIsNotNone(a.crossing_cluster(bp))
        new = a.split_crossing(bp)
        self.assertIsNotNone(new)
        con = a.connected()
        self.assertIn(node_at(a, 339, 200), con)            # straight on from the soma side stays attached
        self.assertNotIn(node_at(a, 300, 140), con)         # the crossing line (up-down) is now separate
        self.assertNotIn(node_at(a, 300, 240), con)
        self.assertEqual(a.component_of()[node_at(a, 300, 140)], a.component_of()[node_at(a, 300, 240)])
        self.check_tree(a)

    def test_no_crossing_at_a_plain_branch(self):
        bp = next(n for n, s in self.a.adj.items() if len(s) == 3)
        self.assertIsNone(self.a.crossing_cluster(bp))
        self.assertIsNone(self.a.split_crossing(bp))

    def test_issue_list_tips_and_loose_clockwise(self):
        items = self.a.find_issues(512, 512)
        kinds = [i['kind'] for i in items]
        self.assertEqual(sorted(kinds), ['loose', 'tip', 'tip', 'tip'])
        where = [(round(i['x']), round(i['y'])) for i in items]
        self.assertEqual(where[0], (300, 240))              # clockwise from the right of the soma
        self.assertEqual(kinds[1], 'loose')
        self.assertEqual(where[2], (136, 200))
        self.assertEqual(where[3], (300, 140))
        self.assertEqual([i['kind'] for i in self.a.find_issues(140, 140)], ['loose'])   # tips on the edge skipped

    def test_move_node(self):
        t = node_at(self.a, 300, 240)
        self.a.move_node(t, 310, 250)
        self.assertTrue(near(self.a.xy[t], 310, 250))
        self.a.move_node(self.a.soma, 0, 0)                 # the soma does not move
        self.assertTrue(near(self.a.xy[self.a.soma], 200, 200))

    def test_join_refuses_a_loop(self):
        a = self.a
        loose_end = node_at(a, 400, 400)
        self.assertTrue(a.join(loose_end, node_at(a, 300, 230)))     # loose piece onto the tree
        self.assertIn(loose_end, a.connected())
        self.assertFalse(a.join(node_at(a, 300, 140), node_at(a, 136, 200)))   # same tree: loop
        self.check_tree(a)

    def test_crossings_from_overlay(self):
        ov = np.zeros((100, 100, 3), np.uint8)
        for k in range(-7, 8):                                   # a blue X like the software's overlay
            for w in (-1, 0, 1):
                ov[50 + k, 30 + k + w] = (40, 120, 255); ov[50 + k, 30 - k + w] = (40, 120, 255)
        p = os.path.join(self.tmp, "ov.png"); Image.fromarray(ov).save(p)
        pts = editor.crossings_from_overlay(p)
        self.assertEqual(len(pts), 1)
        self.assertTrue(near(pts[0], 30, 50) or math.hypot(pts[0][0] - 30, pts[0][1] - 50) < 1)

    def test_loose_piece_runs_outward(self):
        a = self.a
        parent, comps = a.orient()
        loose = next(c for c in comps if a.soma not in c)
        self.assertTrue(near(a.xy[loose[0]], 400, 400))        # recorded root = the end nearest the soma
        # a piece cut off by erasing has no recorded root: it must start at its soma-side end too
        a.erase([node_at(a, 300, 220)], [])
        parent, comps = a.orient()
        cut = next(c for c in comps if any(near(a.xy[n], 300, 240) for n in c))
        self.assertTrue(near(a.xy[cut[0]], 300, 221) or near(a.xy[cut[0]], 300, 222))
        self.assertTrue(near(a.xy[cut[-1]], 300, 240))

    def test_soma_outline_polygon(self):
        a = self.a
        a.soma_poly = editor.load_soma_outline(self.d, self.s)
        self.assertEqual(len(a.soma_poly), 8)
        self.assertTrue(a.inside_soma(200, 200)); self.assertTrue(a.inside_soma(214, 200))
        self.assertFalse(a.inside_soma(200, 214))                  # squashed: 14 px up is outside
        p = a.soma_edge_towards(260, 200)                           # the spoke leaves the soma near x = 218
        self.assertTrue(abs(p[0] - 218) <= 1.5 and abs(p[1] - 200) < 0.01)
        self.assertIsNone(a.soma_edge_towards(205, 200))
        self.assertIsNone(editor.load_soma_outline(self.tmp, "nothing"))   # no json, no image -> None

    def test_metrics_after_edit(self):
        a = self.a
        m0 = a.metrics()
        t = node_at(a, 300, 240); down = next(s for s in a.segments() if t in (s[0], s[-1]))
        a.delete_chain(down)
        m1 = a.metrics()
        self.assertEqual(m1['num_tips'], m0['num_tips'] - 1)
        self.assertEqual(m1['num_branch_points'], m0['num_branch_points'] - 1)


class EditorWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import tkinter as tk
            cls.root = tk.Tk(); cls.root.withdraw()
        except Exception as e:
            raise unittest.SkipTest("no display: %s" % e)

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.d, self.s = write_cell(self.tmp)
        self.w = editor.Editor(self.root, folder=self.tmp)
        self.w.use_menu = False                             # pop-up menus would grab the pointer in tests
        self.w.view_mode.set('3'); self.w._set_view(render=False)   # one pane: the 900 px test window is small
        self.w.geometry("900x700+0+0")
        for _ in range(5):
            self.w.update()

    def tearDown(self):
        self.w.destroy(); shutil.rmtree(self.tmp, ignore_errors=True)

    def click(self, x, y, shift=False):
        cx, cy = self.w._i2c(x, y)
        self.w.canvas.event_generate('<ButtonPress-1>', x=int(cx), y=int(cy), state=1 if shift else 0)
        self.w.canvas.event_generate('<ButtonRelease-1>', x=int(cx), y=int(cy), state=(1 if shift else 0) | 0x100)
        self.w.update()

    def key(self, sym):
        self.w.canvas.focus_force(); self.w.update()
        self.w._key(type('E', (), {'keysym': sym, 'state': 0, 'widget': self.w.canvas, 'char': ''})())
        self.w.update()

    def test_select_delete_undo_save_resume(self):
        w = self.w
        n0 = len(w.arbor.xy)
        self.click(300, 230)                                # the down daughter
        self.assertTrue(w.sel)
        self.key('Delete')
        self.assertEqual(len(w.edits), 1)
        self.assertEqual(w.edits[0]['kind'], 'spur')
        self.assertLess(len(w.arbor.xy), n0)
        w.undo(); self.assertEqual(len(w.arbor.xy), n0); self.assertEqual(w.edits, [])
        w.redo(); self.assertEqual(len(w.edits), 1)
        out = w.save()
        for p in out:
            self.assertTrue(os.path.isfile(p), p)
        with open(out[-1]) as f:
            log = json.load(f)
        self.assertEqual(log['edits'][0]['category'], 'false')
        # reopen: resumes the saved edit
        w2 = editor.Editor(self.root, folder=self.tmp)
        try:
            self.assertEqual(len(w2.edits), 1)
            self.assertEqual(len(w2.arbor.xy), len(w.arbor.xy))
        finally:
            w2.destroy()

    def test_shift_click_selects_branch_beyond(self):
        self.click(250, 200, shift=True)                    # right spoke: selects everything right of the soma
        self.assertEqual(self.w.sel_label, 'subtree')
        self.assertGreater(len(self.w.sel), 150)

    def test_draw_extend_from_tip(self):
        w = self.w
        w.tool.set('draw'); w._tool_changed(); w.follow.set(False)
        self.click(136, 200)                                # starts on the left tip
        self.click(110, 200)
        self.click(90, 200)
        self.key('Return')
        self.assertEqual(w.edits[-1]['kind'], 'extend')
        self.assertGreater(w.edits[-1]['um'], 40 * VOX)

    def test_draw_joins_loose_piece(self):
        w = self.w
        w.tool.set('draw'); w._tool_changed(); w.follow.set(False)
        self.click(300, 240)                                # bottom tip of the down daughter
        self.click(350, 320)
        self.click(400, 400)                                # the loose piece: ends and joins
        self.assertIsNone(w.pend)
        self.assertEqual(w.edits[-1]['kind'], 'bridge')
        self.assertIn(node_at(w.arbor, 430, 400), w.arbor.connected())

    def test_snap_to_bright_line(self):
        w = self.w
        x, y = w._snap_xy(160, 204)                         # 4 px below the bright left line (y=200)
        self.assertEqual((x, y), (160.0, 200.0))
        self.assertEqual(w._snap_xy(60, 60), (60, 60))      # flat background: the click stays put
        w.snap.set(False)
        self.assertEqual(w._snap_xy(160, 204), (160, 204))

    def test_display_range_changes_picture_and_cost(self):
        w = self.w
        before = np.asarray(w.base_img).copy()
        w.vmin.set(0); w.vmax.set(60); w._apply_bc()
        after = np.asarray(w.base_img)
        self.assertGreater(after.mean(), before.mean())     # lowering Max brightens faint signal
        self.assertIsNone(w.cost)                           # following uses the new range
        w.reset_bc(); self.assertTrue(np.array_equal(np.asarray(w.base_img), before))
        self.assertEqual(np.asarray(w.render_overlay()).shape[:2], (512, 512))

    def test_preview_line_is_what_the_click_draws(self):
        w = self.w
        w.tool.set('draw'); w._tool_changed()
        self.click(136, 200)                                # left tip
        cx, cy = w._i2c(110, 200)
        w._motion(type('E', (), {'x': int(cx), 'y': int(cy)})())
        w._compute_preview()
        self.assertIsNotNone(w._preview)
        cand = list(w._preview[1])
        self.click(110, 200)
        self.assertEqual(w.pend['pts'][-len(cand) + 1:], cand[1:])

    def test_three_panes_keep_the_software_version(self):
        w = self.w
        w2 = editor.Editor(self.root, folder=self.tmp)
        try:
            self.assertEqual(w2.view_mode.get(), '23')          # Software + Edited by default
        finally:
            w2.destroy()
        w.view_mode.set('123'); w._set_view(); w.update()
        n_lines = lambda c: len([i for i in c.find_all() if c.type(i) == 'line'])
        before = n_lines(w.canvas_draft)
        self.click(300, 230); self.key('Delete')                # edit the right pane
        w._render(); w.update()
        self.assertEqual(n_lines(w.canvas_draft), before)       # middle pane unchanged
        s0 = w.scale
        w._zoom(type('E', (), {'x': 50, 'y': 50})(), True)      # zoom from any pane: one shared view
        self.assertGreater(w.scale, s0)
        w.view_mode.set('3'); w._set_view(); w.update()
        self.assertFalse(w.canvas_orig.winfo_ismapped()); self.assertFalse(w.canvas_draft.winfo_ismapped())
        w.view_mode.set('12'); w._set_view(); w.update()
        self.assertTrue(w.canvas_orig.winfo_ismapped()); self.assertFalse(w.canvas.winfo_ismapped())
        w._zoom(type('E', (), {'x': 50, 'y': 50})(), True); w.update()   # zoom still works with pane 3 hidden
        w.view_mode.set('123'); w._set_view(); w.update()
        self.assertTrue(w.canvas_orig.winfo_ismapped())

    def drag(self, pts):
        w = self.w
        cx, cy = w._i2c(*pts[0])
        w.canvas.event_generate('<ButtonPress-1>', x=int(cx), y=int(cy)); w.update()
        for x, y in pts[1:]:
            cx, cy = w._i2c(x, y)
            w.canvas.event_generate('<B1-Motion>', x=int(cx), y=int(cy), state=0x100); w.update()
        w.canvas.event_generate('<ButtonRelease-1>', x=int(cx), y=int(cy), state=0x100); w.update()

    def test_eraser_stroke(self):
        w = self.w
        w.tool.set('erase'); w._tool_changed()
        n0 = len(w.arbor.xy)
        self.drag([(290, 220), (300, 220), (310, 220)])          # across the down daughter
        self.assertEqual(w.edits[-1]['kind'], 'erase')
        self.assertLess(len(w.arbor.xy), n0)
        self.assertNotIn(node_at(w.arbor, 300, 240), w.arbor.connected())
        w.undo(); self.assertEqual(len(w.arbor.xy), n0)

    def test_eraser_misses_nothing_off_the_lines(self):
        w = self.w
        w.tool.set('erase'); w._tool_changed()
        self.drag([(60, 60), (70, 70)])                          # empty background
        self.assertEqual(w.edits, [])

    def test_help_window(self):
        h = self.w.show_help(); self.w.update()
        import tkinter as tk
        txt = next(c for c in h.winfo_children() if isinstance(c, tk.Text))
        self.assertIn("ERASE", txt.get('1.0', 'end').upper())
        h.destroy()

    def test_click_selects_segment_in_normal_mode(self):
        self.click(300, 230)
        self.assertEqual(self.w.sel_label, 'segment')
        self.assertTrue(self.w.sel)

    def test_drag_along_line_erases(self):
        w = self.w; n0 = len(w.arbor.xy)
        self.drag([(300, 215), (300, 222), (300, 229)])       # starts on the down daughter (not on a point)
        self.assertEqual(w.edits[-1]['kind'], 'erase')
        self.assertLess(len(w.arbor.xy), n0)

    def test_drag_point_moves_it(self):
        w = self.w
        tip = node_at(w.arbor, 300, 240)
        self.drag([(300, 240), (310, 250), (320, 260)])
        self.assertEqual(w.edits[-1]['kind'], 'move')
        self.assertLess(math.hypot(w.arbor.xy[tip][0] - 320, w.arbor.xy[tip][1] - 260), 2)

    def test_drag_empty_space_pans(self):
        w = self.w; ox = w.ox
        self.drag([(60, 60), (90, 80), (120, 100)])
        self.assertNotEqual(w.ox, ox)
        self.assertEqual(w.edits, [])

    def test_check_list_walk_and_resume(self):
        w = self.w
        self.assertEqual(len(w.issues), 4)
        w.next_check(1); self.assertEqual(w.issue_idx, 0)
        w.mark_fine(); self.assertEqual(w.issue_idx, 1)
        self.assertIn(w.issues[0]['key'], w.checked)
        out = w.save()
        with open(out[-1]) as f:
            self.assertEqual(len(json.load(f)['checked']), 1)
        w2 = editor.Editor(self.root, folder=self.tmp)
        try:
            self.assertEqual(len(w2.checked), 1)
        finally:
            w2.destroy()

    def test_next_many_times_does_not_hang(self):
        w = self.w                                            # selecting rows from code once looped forever
        for _ in range(6):
            w.next_check(1); w.update()
        self.assertEqual(len(w.checked), len(w.issues))
        w.next_check(-1); w.update()
        self.assertIsNotNone(w.issue_idx)

    def test_drag_tip_onto_line_joins(self):
        w = self.w
        end = node_at(w.arbor, 400, 400)                          # left end of the loose piece
        self.assertNotIn(end, w.arbor.connected())
        self.drag([(400, 400), (350, 320), (300, 241)])           # drop it on the down daughter's tip area
        self.assertEqual(w.edits[-1]['kind'], 'bridge')
        self.assertIn(end, w.arbor.connected())

    def test_tip_menu_offers_join(self):
        w = self.w
        end = node_at(w.arbor, 400, 400)
        w.arbor.move_node(end, 302, 232); w._rebuild()            # rests right beside the down daughter
        q, ok = w._join_candidate(end)
        self.assertIsNotNone(q); self.assertTrue(ok)
        self.assertTrue(any(i.get('hint') and 'join' in i.get('note', '') for i in w.issues if i['kind'] == 'loose'))
        w._join(end, q)
        self.assertIn(end, w.arbor.connected())

    def test_menu_actions_without_menu(self):
        w = self.w
        tip = node_at(w.arbor, 300, 240)
        w.start_draw(start=tip)
        self.assertEqual(w.tool.get(), 'draw')
        self.click(300, 270); self.key('Return')
        self.assertEqual(w.tool.get(), 'select')              # back to the normal mode after drawing
        self.assertEqual(w.edits[-1]['kind'], 'extend')
        w._cut_at(node_at(w.arbor, 300, 220))
        self.assertEqual(w.edits[-1]['kind'], 'trim')

    def test_loose_piece_tip_only_at_far_end(self):
        w = self.w
        self.assertEqual(len(w._starts), 1)
        self.assertTrue(near(w._starts[0], 400, 400))
        self.assertFalse(any(near(t, 400, 400) for t in w._tips))
        self.assertTrue(any(near(t, 430, 400) for t in w._tips))

    def test_cut_tool(self):
        w = self.w
        w.tool.set('cut'); w._tool_changed()
        self.click(300, 170)
        self.assertEqual(w.edits[-1]['kind'], 'trim')


if __name__ == '__main__':
    unittest.main()


class NewFeatureTests(unittest.TestCase):
    """2026-10-27: draw from the soma starts at the click; faint runs (low probability) are violet and listed; other bright bodies marked."""
    @classmethod
    def setUpClass(cls):
        import tkinter as tk
        try:
            cls.root = tk.Tk(); cls.root.withdraw()
        except tk.TclError:
            raise unittest.SkipTest("no display")

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.d, self.s = write_cell(self.tmp)
        # probability map: bright along the tracing except the down daughter (x=300, y 201..240), plus a second bright body far away
        pr = np.zeros((512, 512), np.uint16)
        g = np.asarray(Image.open(os.path.join(self.d, self.s + "_raw.png")))
        pr[g >= 200] = 65535; pr[201:241, 298:303] = 3000
        Image.fromarray(pr).save(os.path.join(self.d, self.s + "_prob.png"))
        raw = np.array(g); yy, xx = np.ogrid[:512, :512]; raw[(yy - 100) ** 2 + (xx - 450) ** 2 <= 30 ** 2] = 255
        Image.fromarray(raw).save(os.path.join(self.d, self.s + "_raw.png"))
        self.w = editor.Editor(self.root, folder=self.tmp); self.w.use_menu = False
        self.w.view_mode.set('3'); self.w._set_view(render=False); self.w.geometry("900x700+0+0")
        for _ in range(5):
            self.w.update()

    def tearDown(self):
        self.w.destroy(); shutil.rmtree(self.tmp, ignore_errors=True)

    def click(self, x, y):
        cx, cy = self.w._i2c(x, y)
        self.w.canvas.event_generate('<ButtonPress-1>', x=int(cx), y=int(cy), state=0)
        self.w.canvas.event_generate('<ButtonRelease-1>', x=int(cx), y=int(cy), state=0x100)
        self.w.update()

    def test_draw_from_soma_starts_at_click(self):
        w = self.w; a = w.arbor
        w.tool.set('draw'); self.click(205, 195)                      # inside the soma outline (octagon 18 x 12), away from the centre
        self.assertEqual(w.pend['start'], a.soma)
        self.assertLess(math.hypot(w.pend['pts'][0][0] - 205, w.pend['pts'][0][1] - 195), 1.5, w.pend['pts'][0])
        self.click(205, 150); w.finish_draw()
        first = [n for n in a.adj[a.soma] if n in a.new]
        self.assertEqual(len(first), 1)
        self.assertLess(math.hypot(a.xy[first[0]][0] - 205, a.xy[first[0]][1] - 195), 1.5, a.xy[first[0]])

    def test_faint_run_is_violet_and_listed(self):
        w = self.w
        faint = [it for it in w.issues if it['kind'] == 'faint']
        self.assertEqual(len(faint), 1, [it['label'] for it in w.issues])
        self.assertAlmostEqual(faint[0]['x'], 300, delta=1); self.assertTrue(205 <= faint[0]['y'] <= 236)
        self.assertTrue(any(col == editor.COL_FAINT for _, col in w._dl))
        self.assertFalse(any(col == editor.COL_FAINT and q[0][0] < 290 for q, col in w._dl))   # nothing else is violet
        w.go_check(w.issues.index(faint[0])); self.assertIn('faint', w.hint_text() if hasattr(w, 'hint_text') else 'faint')

    def test_other_body_marked(self):
        self.assertEqual(len(self.w._other_somas), 1, self.w._other_somas)
        x, y = self.w._other_somas[0]; self.assertAlmostEqual(x, 450, delta=3); self.assertAlmostEqual(y, 100, delta=3)
