# -*- coding: utf-8 -*-
"""Unit tests for annotator.py. Exercises the non-visual logic (coordinate transforms, zoom,
annotation model, visibility filtering, hit-testing, JSON round-trip, annotated-PNG rendering,
overlay discovery). Creates a hidden Tk root; skips cleanly if no display is available.

Run:  python -m unittest test_annotator -v      (from the src/ folder)
"""
import os, sys, json, tempfile, shutil, unittest
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import annotator as A

try:
    import tkinter as tk
    _ROOT = tk.Tk(); _ROOT.withdraw()
    _HAVE_TK = True
except Exception as _e:                       # pragma: no cover
    _HAVE_TK = False
    _ROOT = None
    print("no Tk display, skipping UI-object tests:", _e)


def _make_image(dirpath, stem="cellX", w=400, h=300, with_overlay=True):
    """Write a fake grayscale image and (optionally) a pipeline-style _neuron_out overlay."""
    g = (np.random.default_rng(0).random((h, w)) * 200 + 20).astype(np.uint8)
    img_path = os.path.join(dirpath, stem + ".png")
    Image.fromarray(g).save(img_path)
    if with_overlay:
        od = os.path.join(dirpath, stem + "_neuron_out"); os.makedirs(od, exist_ok=True)
        ov = np.stack([g, g, g], -1); ov[h // 2, :, 0] = 255      # a red line = "tracing"
        Image.fromarray(ov).save(os.path.join(od, stem + "_overlay.png"))
    return img_path


class _Evt:
    def __init__(self, x=0, y=0, delta=0, width=0, height=0, widget=None, keysym="", char=""):
        self.x, self.y, self.delta, self.width, self.height, self.widget = x, y, delta, width, height, widget
        self.keysym, self.char = keysym, char


def _pixels(im):
    return np.asarray(im.convert("RGB"))


class PureFuncTests(unittest.TestCase):
    def test_stretch_rgb(self):
        g = np.zeros((10, 12), np.uint16); g[5, 5] = 60000
        im = A._stretch_rgb(g)
        self.assertEqual(im.size, (12, 10))
        self.assertEqual(im.mode, "RGB")
        self.assertLessEqual(max(im.getextrema()[0]), 255)

    def test_find_overlay(self):
        d = tempfile.mkdtemp()
        try:
            p = _make_image(d, "abc", with_overlay=True)
            ov = A._find_overlay(p, None)
            self.assertIsNotNone(ov)
            self.assertTrue(ov.endswith("abc_overlay.png"))
            # no overlay case
            p2 = _make_image(d, "noov", with_overlay=False)
            self.assertIsNone(A._find_overlay(p2, None))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_categories_unique(self):
        keys = [c[0] for c in A.CATEGORIES]
        hotkeys = [c[4] for c in A.CATEGORIES]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(hotkeys), len(set(hotkeys)))


@unittest.skipUnless(_HAVE_TK, "no Tk display")
class AnnotatorTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.img = _make_image(self.d, "cell", w=400, h=300, with_overlay=True)
        self.win = A.Annotator(_ROOT, image_path=self.img)
        self.win.withdraw()
        # deterministic viewport for coordinate maths
        self.win.scale = 2.0; self.win.ox = 10.0; self.win.oy = 20.0

    def tearDown(self):
        try:
            self.win.destroy()
        except Exception:
            pass
        shutil.rmtree(self.d, ignore_errors=True)

    def test_load(self):
        self.assertEqual((self.win.W, self.win.H), (400, 300))
        self.assertIsNotNone(self.win.overlay)          # overlay auto-found

    def test_transform_roundtrip(self):
        for (x, y) in [(0, 0), (123, 45), (399, 299)]:
            cx, cy = self.win._i2c(x, y)
            ix, iy = self.win._c2i(cx, cy)
            self.assertAlmostEqual(ix, x, places=4)
            self.assertAlmostEqual(iy, y, places=4)

    def test_wheel_keeps_point_under_cursor(self):
        cx, cy = 150, 120
        ix0, iy0 = self.win._c2i(cx, cy)
        self.win._wheel(_Evt(x=cx, y=cy, delta=120))     # zoom in
        ix1, iy1 = self.win._c2i(cx, cy)
        self.assertAlmostEqual(ix0, ix1, places=3)
        self.assertAlmostEqual(iy0, iy1, places=3)
        self.assertGreater(self.win.scale, 2.0)

    def test_add_and_delete(self):
        self.win.active_cat.set("missing")
        self.win._add_anno(kind='box', box=[10, 10, 50, 50])
        self.assertEqual(len(self.win.annos), 1)
        self.assertEqual(self.win.annos[0]['category'], "missing")
        self.assertTrue(self.win.annos[0]['visible'])
        self.assertEqual(self.win.sel, 0)
        self.win._delete_sel()
        self.assertEqual(len(self.win.annos), 0)
        self.assertIsNone(self.win.sel)

    def test_ids_increment(self):
        self.win._add_anno(kind='box', box=[0, 0, 10, 10]); self.win._add_anno(kind='box', box=[0, 0, 10, 10])
        self.assertEqual([a['id'] for a in self.win.annos], [1, 2])

    def test_category_selects_default_tool(self):
        for cat, kind in [("crossbranch", "point"), ("missing", "line"), ("false", "box"), ("gap", "line")]:
            self.win.active_cat.set(cat)
            self.assertEqual(self.win.tool.get(), kind)

    def test_point_tool_click_places_point(self):
        self.win.active_cat.set("crossbranch")           # -> point tool
        self.assertEqual(self.win.tool.get(), "point")
        self.win._b1_press(_Evt(x=120, y=90))
        self.assertEqual(len(self.win.annos), 1)
        self.assertEqual(self.win._kind(self.win.annos[0]), "point")
        self.assertIsNotNone(self.win.annos[0]['point'])

    def test_undo(self):
        self.win._add_anno(kind='box', box=[0, 0, 5, 5]); self.win._add_anno(kind='box', box=[6, 6, 9, 9])
        self.win._undo()
        self.assertEqual(len(self.win.annos), 1)
        self.assertEqual(self.win.annos[0]['id'], 1)
        self.win._undo()
        self.assertEqual(len(self.win.annos), 0)
        self.win._undo()                                 # undo on empty is a no-op, not a crash

    def test_select_does_not_switch_tool(self):
        self.win.tool.set("box")
        self.win.active_cat.set("gap")                   # user pick -> would switch to 'line'...
        self.assertEqual(self.win.tool.get(), "line")
        # ...but selecting an annotation (via _sync_sel) must NOT switch the tool
        self.win.tool.set("box")
        self.win._add_anno(kind='point', point=[10, 10])
        self.win.annos[0]['category'] = "crossbranch"
        self.win.sel = 0; self.win._sync_sel()
        self.assertEqual(self.win.tool.get(), "box")     # unchanged despite crossbranch default being 'point'

    def test_kind_inference_legacy(self):
        self.assertEqual(self.win._kind({'box': [0, 0, 1, 1]}), "box")
        self.assertEqual(self.win._kind({'point': [1, 1]}), "point")
        self.assertEqual(self.win._kind({'path': [[0, 0], [1, 1]]}), "line")
        self.assertEqual(self.win._kind({'kind': 'point'}), "point")

    def test_hit_test_box(self):
        self.win._add_anno(kind='box', box=[100, 80, 160, 140])
        cx, cy = self.win._i2c(130, 110)                 # centre of the box
        self.assertEqual(self.win._hit_test(cx, cy), 0)
        cx2, cy2 = self.win._i2c(5, 5)                    # outside
        self.assertIsNone(self.win._hit_test(cx2, cy2))

    def test_hit_test_path_only(self):
        self.win._add_anno(kind='line')
        self.win.annos[0]['path'] = [(200, 100), (210, 105), (220, 110)]
        cx, cy = self.win._i2c(210, 105)
        self.assertEqual(self.win._hit_test(cx, cy), 0)

    def test_visibility_filter(self):
        self.win._add_anno(kind='box', box=[0, 0, 10, 10])           # missing (visible)
        a = self.win.annos[0]
        self.assertTrue(self.win._is_visible(a))
        a['visible'] = False
        self.assertFalse(self.win._is_visible(a))
        a['visible'] = True
        self.win.cat_visible['missing'].set(False)       # hide whole category
        self.assertFalse(self.win._is_visible(a))

    def test_toggle_and_hide_all(self):
        self.win._add_anno(kind='box', box=[0, 0, 10, 10]); self.win._add_anno(kind='box', box=[5, 5, 20, 20])
        self.win.sel = 0; self.win._toggle_visible()
        self.assertFalse(self.win.annos[0]['visible'])
        self.win._set_all_visible(False)
        self.assertTrue(all(not a['visible'] for a in self.win.annos))
        self.assertTrue(all(not v.get() for v in self.win.cat_visible.values()))
        self.win._set_all_visible(True)
        self.assertTrue(all(a['visible'] for a in self.win.annos))

    def test_hidden_hit_test_skipped(self):
        self.win._add_anno(kind='box', box=[100, 80, 160, 140])
        self.win.annos[0]['visible'] = False
        cx, cy = self.win._i2c(130, 110)
        self.assertIsNone(self.win._hit_test(cx, cy))    # hidden -> not selectable by click

    def test_category_hotkey(self):
        self.win._add_anno(kind='box', box=[0, 0, 10, 10])
        self.win.sel = 0
        self.win._on_key(_Evt(widget=self.win))          # widget != note
        # simulate pressing '3'
        e = _Evt(widget=self.win); e.keysym = "3"; e.char = "3"
        self.win._on_key(e)
        self.assertEqual(self.win.active_cat.get(), "crossbranch")
        self.assertEqual(self.win.annos[0]['category'], "crossbranch")

    def test_note_edit(self):
        self.win._add_anno(kind='box', box=[0, 0, 10, 10]); self.win.sel = 0
        self.win.note.delete('1.0', 'end'); self.win.note.insert('1.0', "wrong here")
        self.win._on_note_edit(_Evt())
        self.assertEqual(self.win.annos[0]['note'], "wrong here")

    def test_save_and_resume(self):
        self.win.active_cat.set("gap")
        self.win._add_anno(kind='box', box=[30, 40, 90, 120]); self.win.sel = 0
        self.win.annos[0]['note'] = "should connect"
        self.win.annos[0]['path'] = [(30, 40), (60, 80), (90, 120)]
        self.win.save(interactive=False)                 # -> annotations/ next to image (no dialog)
        jp = os.path.join(self.d, "annotations", "cell_annotations.json")
        pp = os.path.join(self.d, "annotations", "cell_annotated.png")
        self.assertTrue(os.path.isfile(jp))
        self.assertTrue(os.path.isfile(pp))
        data = json.load(open(jp, encoding='utf-8'))
        self.assertEqual(data['width'], 400)
        self.assertEqual(len(data['annotations']), 1)
        self.assertEqual(data['annotations'][0]['category'], "gap")
        self.assertEqual(data['annotations'][0]['note'], "should connect")
        # a fresh window on the same image should resume the saved annotation
        win2 = A.Annotator(_ROOT, image_path=self.img); win2.withdraw()
        self.assertEqual(len(win2.annos), 1)
        self.assertEqual(win2.annos[0]['note'], "should connect")
        self.assertEqual(win2._next_id, 2)               # next id past the resumed one
        win2.destroy()

    def _click(self, x, y):
        """simulate a click (press+release, no motion) through the real handlers."""
        self.win._b1_press(_Evt(x=x, y=y))
        self.win._b1_release(_Evt(x=x, y=y))

    def test_line_click_to_add_points(self):
        self.win.active_cat.set("gap")                   # -> line tool
        self.assertEqual(self.win.tool.get(), "line")
        self._click(40, 40); self._click(80, 60); self._click(120, 50)
        self.assertEqual(len(self.win.annos), 1)         # one line being drawn
        self.assertIsNotNone(self.win._line_idx)
        self.win._finish_line()
        self.assertIsNone(self.win._line_idx)
        self.assertEqual(self.win._kind(self.win.annos[0]), "line")
        self.assertGreaterEqual(len(self.win.annos[0]['path']), 3)

    def test_line_single_click_discarded_on_finish(self):
        self.win.active_cat.set("gap")
        self._click(50, 50)                              # only one point
        self.assertEqual(len(self.win.annos), 1)
        self.win._finish_line()                          # a 1-point line is not a real annotation
        self.assertEqual(len(self.win.annos), 0)

    def test_tool_change_finishes_line(self):
        self.win.active_cat.set("gap"); self._click(50, 50); self._click(90, 70)
        self.win.tool.set("point")                       # switching tool commits the line
        self.assertIsNone(self.win._line_idx)
        self.assertEqual(len(self.win.annos), 1)
        self.assertGreaterEqual(len(self.win.annos[0]['path']), 2)

    def test_line_cancel_discards(self):
        self.win.active_cat.set("gap"); self._click(50, 50); self._click(90, 70)
        self.assertEqual(len(self.win.annos), 1)
        self.win._cancel_line()                          # Esc discards the whole in-progress line
        self.assertEqual(len(self.win.annos), 0)
        self.assertIsNone(self.win._line_idx)

    def _make_batch(self, root, n=3):
        """fake a run's output folder: root/cellK_neuron_out/{cellK_overlay.png, cellK_raw.png}."""
        for k in range(n):
            od = os.path.join(root, "cell%d_neuron_out" % k); os.makedirs(od, exist_ok=True)
            g = (np.random.default_rng(k).random((100, 120)) * 200 + 20).astype(np.uint8)
            Image.fromarray(g).save(os.path.join(od, "cell%d_raw.png" % k))
            ov = np.stack([g, g, g], -1); ov[50, :, 0] = 255
            Image.fromarray(ov).save(os.path.join(od, "cell%d_overlay.png" % k))

    def test_load_batch_and_navigate(self):
        d = tempfile.mkdtemp()
        try:
            self._make_batch(d, 3)
            self.win.load_batch(d)
            self.assertEqual(len(self.win.cells), 3)
            self.assertEqual(self.win.cell_idx, 0)
            self.assertIsNotNone(self.win.base)          # first cell auto-loaded
            self.assertIsNotNone(self.win.overlay)       # tracing overlay loaded (the bug Liz hit)
            self.win._step_cell(1); self.assertEqual(self.win.cell_idx, 1)
            self.win._step_cell(1); self.assertEqual(self.win.cell_idx, 2)
            self.win._step_cell(1); self.assertEqual(self.win.cell_idx, 2)   # clamp at end
            self.win._step_cell(-5); self.assertEqual(self.win.cell_idx, 0)  # clamp at start
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_batch_autosave_one_folder_unique_names(self):
        d = tempfile.mkdtemp()
        try:
            self._make_batch(d, 3)
            self.win.load_batch(d)
            self.assertEqual(self.win.save_dir, os.path.join(d, "annotations"))   # ONE shared folder
            self.win.active_cat.set("false")
            self.win._add_anno(kind='box', box=[1, 1, 20, 20])              # annotate cell 0
            self.win._step_cell(1)                                          # switch -> auto-save cell 0
            self.win._add_anno(kind='box', box=[2, 2, 30, 30])              # annotate cell 1
            self.win._step_cell(1)                                          # -> auto-save cell 1
            self.win._add_anno(kind='point', point=[5, 5])                  # annotate cell 2
            self.win.write_annotations()
            ann = os.path.join(d, "annotations")
            import glob as _g
            jsons = sorted(os.path.basename(x) for x in _g.glob(os.path.join(ann, "*_annotations.json")))
            self.assertEqual(jsons, ["cell0_annotations.json", "cell1_annotations.json",
                                     "cell2_annotations.json"])            # 3 DISTINCT files (no stem collision)
            self.win._step_cell(-2)                                        # back to cell 0
            self.assertEqual(len(self.win.annos), 1)                       # resumes
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def _make_cell_out(self, d, folder, stem, raw=True, overlay=True):
        od = os.path.join(d, folder + "_neuron_out"); os.makedirs(od, exist_ok=True)
        g = (np.random.default_rng(0).random((100, 120)) * 200 + 20).astype(np.uint8)
        if raw:
            Image.fromarray(g).save(os.path.join(od, stem + "_raw.png"))
        if overlay:
            ov = np.stack([g, g, g], -1); ov[50, :, 0] = 255
            Image.fromarray(ov).save(os.path.join(od, stem + "_overlay.png"))
        return od

    def test_save_on_close(self):
        d = tempfile.mkdtemp()
        try:
            self._make_batch(d, 1)
            self.win.load_batch(d)
            self.win._add_anno(kind='box', box=[1, 1, 10, 10])
            self.win._on_close()                                        # emulate window-close
            self.assertTrue(os.path.isfile(os.path.join(d, "annotations", "cell0_annotations.json")))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_older_run_no_raw(self):
        d = tempfile.mkdtemp()
        try:
            self._make_cell_out(d, "old", "mor", raw=False, overlay=True)   # overlay but NO _raw
            self.win.load_batch(d)
            self.assertEqual(len(self.win.cells), 1)
            self.assertIsNotNone(self.win.base)
            self.assertIsNone(self.win.overlay)                         # no raw -> tracing baked into base
            self.win._add_anno(kind='box', box=[1, 1, 10, 10]); self.win.write_annotations()
            self.assertTrue(os.path.isfile(os.path.join(d, "annotations", "old_annotations.json")))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_failed_cell_skipped(self):
        d = tempfile.mkdtemp()
        try:
            self._make_batch(d, 1)
            os.makedirs(os.path.join(d, "failed_neuron_out"))           # empty (traced but no overlay)
            cells = A.Annotator._scan_cells(d)
            self.assertEqual(len(cells), 1)                             # failed cell skipped, no crash
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_spaces_in_path(self):
        d = tempfile.mkdtemp()
        try:
            self._make_cell_out(d, "cell line 1_n1", "mor")            # spaces (real data layout)
            self.win.load_batch(d)
            self.assertEqual(len(self.win.cells), 1)
            self.assertEqual(self.win.cells[0]['name'], "cell line 1_n1")
            self.win._add_anno(kind='point', point=[5, 5]); self.win.write_annotations()
            self.assertTrue(os.path.isfile(os.path.join(d, "annotations", "cell line 1_n1_annotations.json")))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_delete_all_persists(self):
        d = tempfile.mkdtemp()
        try:
            self._make_batch(d, 2)
            self.win.load_batch(d)
            self.win._add_anno(kind='box', box=[1, 1, 20, 20])          # cell 0: 1 anno
            self.win._step_cell(1)                                       # autosave cell 0
            self.win._step_cell(-1)                                      # back to cell 0 (resumes 1 anno)
            self.assertEqual(len(self.win.annos), 1)
            self.win._undo()                                            # remove it -> empty
            self.assertEqual(len(self.win.annos), 0)
            self.win._step_cell(1)                                       # leave -> must overwrite (not keep stale)
            jp = os.path.join(d, "annotations", "cell0_annotations.json")
            import json as _j
            self.assertEqual(_j.load(open(jp, encoding='utf-8'))['annotations'], [])   # cleared on disk
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_bracket_path(self):
        base = tempfile.mkdtemp()
        d = os.path.join(base, "run [1] out"); os.makedirs(d)             # bracket = glob metachar
        try:
            self._make_cell_out(d, "cellX", "mor")
            cells = A.Annotator._scan_cells(d)
            self.assertEqual(len(cells), 1)                             # glob.escape -> found despite [ ]
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_scan_cells_single_neuron_out(self):
        d = tempfile.mkdtemp()
        try:
            self._make_batch(d, 1)                                          # one cell0_neuron_out
            cells = A.Annotator._scan_cells(os.path.join(d, "cell0_neuron_out"))
            self.assertEqual(len(cells), 1)                                # a single _neuron_out is one cell
            self.assertTrue(cells[0]['overlay'].endswith("cell0_overlay.png"))
            self.assertTrue(cells[0]['raw'].endswith("cell0_raw.png"))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_help_dialog_opens(self):
        self.win._show_help()                            # must not raise
        tops = [w for w in self.win.winfo_children() if isinstance(w, tk.Toplevel)]
        self.assertTrue(tops)
        for w in tops:
            w.destroy()

    def test_pen_stroke_creates_path(self):
        self.win.tool.set("line"); self.win.sel = None
        self.win._b1_press(_Evt(x=100, y=100))
        self.win._b1_move(_Evt(x=140, y=100))
        self.win._b1_move(_Evt(x=180, y=130))
        self.win._b1_release(_Evt(x=180, y=130))
        self.assertEqual(len(self.win.annos), 1)
        self.assertGreaterEqual(len(self.win.annos[0]['path']), 2)

    def test_pen_click_no_stroke_discarded(self):
        self.win.tool.set("line"); self.win.sel = None
        self.win._b1_press(_Evt(x=100, y=100))
        self.win._b1_release(_Evt(x=100, y=100))         # one click starts a line (stays open)...
        self.win._finish_line()                          # ...and a 1-point line is dropped on finish
        self.assertEqual(len(self.win.annos), 0)

    def test_pen_click_preserves_existing_path(self):
        self.win._add_anno(kind='box', box=[10, 10, 40, 40]); self.win.sel = 0
        self.win.annos[0]['path'] = [(10, 10), (20, 20), (30, 30)]
        self.win.tool.set("line")
        self.win._b1_press(_Evt(x=200, y=200))
        self.win._b1_release(_Evt(x=200, y=200))         # stray click, no stroke
        self.assertEqual(self.win.annos[0]['path'], [(10, 10), (20, 20), (30, 30)])

    def test_zoom_clamp(self):
        for _ in range(80):
            self.win._wheel(_Evt(x=100, y=100, delta=-120))
        self.assertGreaterEqual(self.win.scale, self.win._min_scale() - 1e-6)
        for _ in range(80):
            self.win._wheel(_Evt(x=100, y=100, delta=120))
        self.assertLessEqual(self.win.scale, 80.0 + 1e-6)

    def test_unknown_category_visible(self):
        self.win._add_anno(kind='box', box=[0, 0, 5, 5])
        self.win.annos[0]['category'] = "legacy_unknown"
        self.assertTrue(self.win._is_visible(self.win.annos[0]))   # no KeyError, defaults visible

    def test_render_annotated_differs(self):
        self.win._add_anno(kind='box', box=[50, 50, 200, 200]); self.win.sel = 0
        out = self.win._render_annotated()
        self.assertEqual(out.size, (400, 300))
        # burning a box in must change pixels vs the plain overlay
        base = (self.win.overlay or self.win.base)
        self.assertFalse(np.array_equal(_pixels(out), _pixels(base)))

    def test_render_annotated_respects_visibility(self):
        self.win._add_anno(kind='box', box=[50, 50, 200, 200]); self.win.sel = 0
        shown = self.win._render_annotated()
        self.win.annos[0]['visible'] = False
        hidden = self.win._render_annotated()
        base = (self.win.overlay or self.win.base).copy()
        self.assertTrue(np.array_equal(_pixels(hidden), _pixels(base)))   # nothing burned
        self.assertFalse(np.array_equal(_pixels(shown), _pixels(hidden)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
