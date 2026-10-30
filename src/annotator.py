# -*- coding: utf-8 -*-
"""Error-review / annotation window for NeuronTracer.

Opens the original image with our tracing overlaid, lets a reviewer zoom/pan, draw a box around a
tracing error, tag it with an error category, optionally sketch the CORRECT neurite path with a pen,
add a note, and save everything to an annotations folder (one JSON record + a burned-in PNG per image).

Kept dependency-light (tkinter + PIL + numpy) so it packages with the rest of the app. Standalone-
runnable too: `python annotator.py [image] [tracing_overlay.png]`.
"""
import os, sys, json, datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk, ImageEnhance, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None
import theme
from theme import P, IMG_BG, px, paint_list, titlebar, on_theme_change

# error categories: (key, English label, 中文 label, RGB colour, hotkey, default primitive, gesture hint)
# the primitive is matched to the error's natural geometry: a missing branch / gap is a LINE, a
# mis-resolved crossing is a POINT (one junction), a blob of false signal is a BOX (region).
CATEGORIES = [
    ("missing",     "Missing neurite", "漏画的突起",    (80, 180, 255), "1", "line",  "draw the branch that's missing"),
    ("false",       "False neurite",   "多画的假突起",  (255, 140, 0),  "2", "box",   "circle the false trace"),
    ("crossbranch", "Crossing/branch", "交叉/分叉判错", (255, 60, 230), "3", "point", "click the wrong junction"),
    ("gap",         "Broken gap",      "断裂该连未连",  (60, 220, 120), "4", "line",  "draw a stroke joining the two ends"),
    ("other",       "Other",           "其他",          (230, 220, 60), "5", "box",   "box the area"),
]
CAT = {c[0]: c for c in CATEGORIES}
DEFAULT_KIND = {c[0]: c[5] for c in CATEGORIES}
GESTURE = {c[0]: c[6] for c in CATEGORIES}


def _stretch_rgb(gray):
    """percentile-stretched grayscale -> RGB PIL image (raw 16-bit fluorescence is otherwise blank)."""
    g = gray.astype(float)
    lo, hi = np.percentile(g, 0.5), np.percentile(g, 99.5)
    a = np.clip((g - lo) / max(hi - lo, 1), 0, 1) * 255
    return Image.fromarray(a.astype(np.uint8)).convert('RGB')


def _find_overlay(image_path, out_dir):
    """Locate the tracing overlay PNG produced by the pipeline for this image."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    cands = []
    if out_dir:                                            # single-run out_dir, or a batch root
        cands += [os.path.join(out_dir, stem + "_overlay.png"),
                  os.path.join(out_dir, stem + "_neuron_out", stem + "_overlay.png")]
    d = os.path.dirname(image_path)
    cands += [os.path.join(d, stem + "_neuron_out", stem + "_overlay.png"),
              os.path.join(d, stem + "_overlay.png")]
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


class Annotator(tk.Toplevel):
    def __init__(self, master=None, image_path=None, out_dir=None, folder=None):
        super().__init__(master)
        self.title("Review & annotate tracing errors")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry("%dx%d" % (min(px(1360), sw - px(60)), min(px(860), sh - px(100))))
        if not ttk.Style(self).lookup('Card.TFrame', 'background'):   # opened on its own (--annotate, tests)
            theme.apply_style(self._root(), theme.CURRENT)
        self.configure(bg=P['bg'])

        self.image_path = None
        self.out_dir = out_dir          # where the pipeline wrote outputs (used to find the overlay)
        self.save_dir = None            # where annotations get saved (chosen or default)
        self.base = None                # stretched raw, RGB PIL, full res
        self.overlay = None             # tracing overlay, RGB PIL, full res (may be None)
        self.W = self.H = 0

        # viewport: image coord at canvas (0,0) = (ox,oy); scale = canvas_px per image_px
        self.scale = 1.0
        self.ox = self.oy = 0.0
        self.alpha = 0.85               # tracing overlay opacity
        self.brightness = 1.0
        self.tool = tk.StringVar(value="line")    # point | line | box | pan
        self.active_cat = tk.StringVar(value="missing")
        self.cat_visible = {c[0]: tk.BooleanVar(value=True) for c in CATEGORIES}  # show/hide per category

        self.annos = []                 # list of dicts {id,category,box,path,note,time,visible}
        self.sel = None                 # index of selected annotation
        self._next_id = 1
        self._photo = None
        self._drag = None               # transient drag state
        self._line_idx = None           # index of the polyline currently being drawn by clicks (or None)
        self._fitted = False            # auto-fit once the canvas gets a real size
        self._syncing = False           # guard against list-select recursion
        self.cells = []                 # traced cells when reviewing a whole batch
        self.cell_idx = None
        self.cell_key = None            # UNIQUE name for the current cell's annotation file (avoids stem collisions)

        self._build_ui()
        self.bind("<Key>", self._on_key)
        self.bind("<Control-s>", lambda e: self.save())
        self.bind("<Control-z>", lambda e: self._undo())
        self.protocol("WM_DELETE_WINDOW", self._on_close)   # don't lose the current cell's work on close
        if folder:
            self.load_batch(folder)
        elif image_path:
            self.load_image(image_path, out_dir)

    # ---------- UI ----------
    def _build_ui(self):
        # ---- top toolbar ----
        bar = ttk.Frame(self, padding=(px(12), px(10), px(12), px(8))); bar.pack(side='top', fill='x')
        ttk.Label(bar, text="Tool", style='Sub.TLabel').pack(side='left', padx=(0, px(8)))
        for val, txt in (("point", "Point"), ("line", "Line"), ("box", "Box"), ("pan", "Pan")):
            ttk.Radiobutton(bar, text=txt, value=val, variable=self.tool, style='Seg.Toolbutton').pack(side='left', padx=(0, 1))
        ttk.Separator(bar, orient='vertical').pack(side='left', fill='y', padx=px(14))

        self.show_tracing = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Show tracing", variable=self.show_tracing,
                        command=self._toggle_tracing).pack(side='left')
        self.alpha_s = ttk.Scale(bar, from_=0, to=1, value=self.alpha, length=px(90),
                                 command=lambda v: self._set_alpha(float(v)))
        self.alpha_s.pack(side='left', padx=(px(6), 0))
        ttk.Label(bar, text="Brightness", style='Sub.TLabel').pack(side='left', padx=(px(16), px(6)))
        self.bri_s = ttk.Scale(bar, from_=0.4, to=2.5, value=1.0, length=px(90),
                               command=lambda v: self._set_bright(float(v)))
        self.bri_s.pack(side='left')
        ttk.Button(bar, text="Fit", style='Small.TButton', command=self.fit).pack(side='left', padx=(px(16), 0))
        ttk.Button(bar, text="? Help", style='Small.TButton', command=self._show_help).pack(side='right')

        # ---- bottom: help line + status bar ----
        sbar = ttk.Frame(self, style='Status.TFrame', padding=(px(14), px(6))); sbar.pack(side='bottom', fill='x')
        self.status = ttk.Label(sbar, text="Open an image to begin.", anchor='w', style='Status.TLabel')
        self.status.pack(side='left', fill='x', expand=True)
        help_txt = ("Pick an error type (1-5) and its tool is chosen for you.     "
                    "Wheel  zoom     middle/right-drag  pan     v  show/hide     Del  remove     Ctrl+S  save")
        ttk.Label(self, text=help_txt, style='Hint.TLabel', wraplength=px(1200),
                  padding=(px(14), px(4))).pack(side='bottom', fill='x')

        # ---- main split: cells card (left) + canvas (centre) + side card (right) ----
        body = ttk.Frame(self, padding=(px(12), px(2), px(12), px(6))); body.pack(side='top', fill='both', expand=True)

        cells = ttk.Frame(body, style='Card.TFrame', width=px(210), padding=px(10))
        cells.pack(side='left', fill='y'); cells.pack_propagate(False)
        chdr = ttk.Frame(cells, style='Card.TFrame', borderwidth=0); chdr.pack(fill='x')
        ttk.Label(chdr, text="Cells", style='CardHead.TLabel').pack(side='left')
        self.cell_count_lbl = ttk.Label(chdr, text="", style='CardMuted.TLabel'); self.cell_count_lbl.pack(side='right')
        ttk.Button(cells, text="Open folder…", command=self._open_folder).pack(fill='x', pady=(px(8), px(6)))
        ttk.Label(cells, text="✓ annotated   • traced", style='CardMuted.TLabel').pack(anchor='w', pady=(0, px(4)))
        self.cell_list = tk.Listbox(cells, activestyle='none', font=(theme.MONO, 9), exportselection=False)
        self.cell_list.pack(fill='both', expand=True, pady=px(2))
        self.cell_list.bind('<<ListboxSelect>>', self._on_cell_select)
        nav = ttk.Frame(cells, style='Card.TFrame', borderwidth=0); nav.pack(fill='x', pady=(px(6), 0))
        nav.columnconfigure((0, 1), weight=1, uniform='n')
        ttk.Button(nav, text="◀ Prev", command=lambda: self._step_cell(-1)).grid(row=0, column=0, sticky='ew', padx=(0, px(3)))
        ttk.Button(nav, text="Next ▶", command=lambda: self._step_cell(1)).grid(row=0, column=1, sticky='ew', padx=(px(3), 0))

        side = ttk.Frame(body, style='Card.TFrame', width=px(320), padding=px(12))
        side.pack(side='right', fill='y'); side.pack_propagate(False)

        self.canvas = tk.Canvas(body, bg=IMG_BG, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side='left', fill='both', expand=True, padx=px(10))

        hdr = ttk.Frame(side, style='Card.TFrame', borderwidth=0); hdr.pack(fill='x', pady=(0, px(6)))
        ttk.Label(hdr, text="Error category", style='CardHead.TLabel').pack(side='left')
        ttk.Label(hdr, text="active / show", style='CardMuted.TLabel').pack(side='right')
        self._catbtns = {}
        for key, en, zh, rgb, hk, kind, hint in CATEGORIES:
            hexc = "#%02x%02x%02x" % rgb
            f = tk.Frame(side, bd=0, highlightthickness=0); f.pack(fill='x', pady=1)
            tk.Frame(f, bg=hexc, width=px(4)).pack(side='left', fill='y')
            # per-category show/hide "eye"
            ttk.Checkbutton(f, variable=self.cat_visible[key], style='Card.TCheckbutton',
                            command=self._render).pack(side='right', padx=(px(6), 0))
            b = ttk.Radiobutton(f, text="%s     %s · %s" % (en, hk, kind), value=key, variable=self.active_cat,
                                style='Cat.Toolbutton')
            b.pack(side='left', fill='x', expand=True)
            self._catbtns[key] = f
        self.active_cat.trace_add('write', lambda *_: self._on_cat_change())

        ttk.Separator(side).pack(fill='x', pady=px(12))
        arow = ttk.Frame(side, style='Card.TFrame', borderwidth=0); arow.pack(fill='x')
        ttk.Label(arow, text="Annotations", style='CardHead.TLabel').pack(side='left')
        ttk.Button(arow, text="Show all", style='Small.TButton', command=lambda: self._set_all_visible(True)).pack(side='right')
        ttk.Button(arow, text="Hide all", style='Small.TButton', command=lambda: self._set_all_visible(False)).pack(side='right', padx=px(4))
        ttk.Label(side, text="● shown   ○ hidden   (double-click or v)",
                  style='CardMuted.TLabel').pack(anchor='w', pady=(px(4), px(4)))
        self.listbox = tk.Listbox(side, height=10, activestyle='none', font=(theme.MONO, 9))
        self.listbox.pack(fill='x')
        self.listbox.bind('<<ListboxSelect>>', self._on_list_select)
        self.listbox.bind('<Double-Button-1>', lambda e: self._toggle_visible())
        lrow = ttk.Frame(side, style='Card.TFrame', borderwidth=0); lrow.pack(fill='x', pady=px(6))
        lrow.columnconfigure((0, 1, 2), weight=1, uniform='l')
        ttk.Button(lrow, text="Delete", style='Small.TButton', command=self._delete_sel).grid(row=0, column=0, sticky='ew', padx=(0, px(3)))
        ttk.Button(lrow, text="Show / hide", style='Small.TButton', command=self._toggle_visible).grid(row=0, column=1, sticky='ew', padx=px(3))
        ttk.Button(lrow, text="Clear path", style='Small.TButton', command=self._clear_path).grid(row=0, column=2, sticky='ew', padx=(px(3), 0))

        ttk.Label(side, text="Note", style='CardHead.TLabel').pack(anchor='w', pady=(px(8), px(4)))
        self.note = tk.Text(side, height=4, width=34, font=(theme.FONT, 9), padx=px(6), pady=px(4))
        self.note.pack(fill='x')
        self.note.bind('<KeyRelease>', self._on_note_edit)

        ttk.Separator(side).pack(fill='x', pady=px(12))
        ttk.Button(side, text="Save annotations", style='Accent.TButton', command=self.save).pack(side='bottom', fill='x')
        self.save_lbl = ttk.Label(side, text="save: annotations/ next to image",
                                  style='CardMuted.TLabel', wraplength=px(290))
        self.save_lbl.pack(side='bottom', fill='x', pady=(px(4), px(8)))
        ttk.Button(side, text="Choose save folder…", command=self._choose_savedir).pack(side='bottom', fill='x')

        def _paint():
            if not self.winfo_exists():
                return
            self.configure(bg=P['bg'])
            for w in (self.cell_list, self.listbox, self.note):
                paint_list(w)
            for f in self._catbtns.values():
                f.configure(bg=P['card'])
            self._hilite_cat()
            self.save_lbl.configure(foreground=(P['ok'] if self.save_dir else P['muted']))
            titlebar(self)
        on_theme_change(_paint)

        # canvas bindings
        self.canvas.bind("<ButtonPress-1>", self._b1_press)
        self.canvas.bind("<B1-Motion>", self._b1_move)
        self.canvas.bind("<ButtonRelease-1>", self._b1_release)
        self.canvas.bind("<Double-Button-1>", self._finish_line)   # double-click ends a click-drawn line
        self.canvas.bind("<ButtonPress-2>", self._pan_press)
        self.canvas.bind("<B2-Motion>", self._pan_move)
        self.canvas.bind("<ButtonPress-3>", self._pan_press)     # right-drag pans too
        self.canvas.bind("<B3-Motion>", self._pan_move)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Motion>", self._hover)
        self.canvas.bind("<Configure>", self._on_configure)
        self.bind("<Return>", self._finish_line_key)             # Enter finishes a click-drawn line
        self.bind("<Escape>", self._cancel_line_key)             # Esc cancels (discards) it
        self.tool.trace_add('write', lambda *_: self._finish_line())  # changing tool commits any open line

    def _finish_line_key(self, e):
        if self.focus_get() is not self.note:       # don't hijack Enter while typing a note
            self._finish_line()

    def _cancel_line_key(self, e):
        self._cancel_line()

    def _cancel_line(self):
        """discard the in-progress click-drawn polyline entirely."""
        if self._line_idx is None:
            return
        idx = self._line_idx; self._line_idx = None
        if 0 <= idx < len(self.annos):
            self.annos.pop(idx)
            if self.sel == idx:
                self.sel = None
        self._refresh_list(); self._render()

    def _on_configure(self, e):
        # once the canvas has a real size, do the initial fit; afterwards just re-render
        if self.base is not None and not self._fitted and e.width > 50 and e.height > 50:
            self._fitted = True; self.fit()
        else:
            self._render()

    def _hilite_cat(self):
        for key, f in self._catbtns.items():
            hexc = "#%02x%02x%02x" % CAT[key][3]
            f.config(highlightbackground=hexc, highlightcolor=hexc,
                     highlightthickness=1 if key == self.active_cat.get() else 0)

    def _on_cat_change(self):
        """picking an error category auto-selects its natural drawing tool (still overridable).
        Only when the user actively picks a category -- not when selecting an existing annotation."""
        self._hilite_cat()
        cat = self.active_cat.get()
        if not self._syncing and self.tool.get() != 'pan':    # don't switch on mere selection / don't leave pan
            self.tool.set(DEFAULT_KIND.get(cat, 'box'))
        if self.base is not None:
            self.status.config(text="%s → %s" % (CAT[cat][1], GESTURE.get(cat, "")))

    # ---------- batch review (read straight from the run's *_neuron_out folders) ----------
    def _open_folder(self):
        d = filedialog.askdirectory(title="Select the run's output folder (contains the *_neuron_out cells)")
        if d:
            self.load_batch(d)

    @staticmethod
    def _scan_cells(root):
        """Find traced cells = *_neuron_out folders (each holds <stem>_overlay.png + <stem>_raw.png).
        Accepts the batch output root, or a single cell's _neuron_out folder."""
        import glob
        dirs = []
        if root.rstrip("/\\").lower().endswith("_neuron_out"):
            dirs = [root]
        else:                                            # glob.escape: paths with [ ] * ? (e.g. cell[1]) are literal
            dirs = sorted(glob.glob(os.path.join(glob.escape(root), "**", "*_neuron_out"), recursive=True))
        cells = []
        for d in dirs:
            ov = glob.glob(os.path.join(glob.escape(d), "*_overlay.png"))
            if not ov:
                continue
            stem = os.path.basename(ov[0])[:-len("_overlay.png")]
            raw = glob.glob(os.path.join(glob.escape(d), "*_raw.png"))
            name = os.path.basename(d.rstrip("/\\"))[:-len("_neuron_out")]
            cells.append({'dir': d, 'overlay': ov[0], 'raw': raw[0] if raw else None,
                          'stem': stem, 'name': name or stem})
        return cells

    def load_batch(self, root):
        """Populate the cell list from a run's output folder. Everything (raw + tracing) is read from
        the *_neuron_out folders. All annotations save to ONE folder: <root>/annotations."""
        try:
            cells = self._scan_cells(root)
        except Exception as e:
            messagebox.showerror("Could not read folder", "Failed to scan:\n%s\n\n%s" % (root, e)); return
        if not cells:
            messagebox.showinfo("No traced cells",
                                "No traced cells found in:\n%s\n\nRun tracing on a folder first "
                                "(main window: Open folder (batch) → Run), then this opens automatically." % root)
            return
        self.cells = cells
        self.save_dir = os.path.join(root, "annotations")     # ONE folder for every cell's annotations
        self.save_lbl.config(text="saving to: " + self.save_dir, foreground=P['ok'])
        self.cell_count_lbl.config(text="%d cells" % len(cells))
        self.cell_idx = None
        self._refresh_cell_list()
        self._select_cell(0)

    def _cell_status(self, cell):
        savedir = self.save_dir or os.path.join(cell['dir'], "annotations")
        if os.path.isfile(os.path.join(savedir, cell['name'] + "_annotations.json")):
            return "✓"                                   # already annotated
        return "•"                                       # traced (every listed cell has an overlay)

    def _refresh_cell_list(self):
        self.cell_list.delete(0, 'end')
        for c in self.cells:
            self.cell_list.insert('end', "%s %s" % (self._cell_status(c), c['name']))
        if self.cell_idx is not None and 0 <= self.cell_idx < len(self.cells):
            self.cell_list.selection_clear(0, 'end'); self.cell_list.selection_set(self.cell_idx)
            self.cell_list.see(self.cell_idx)

    def _on_cell_select(self, e):
        s = self.cell_list.curselection()
        if s and s[0] != self.cell_idx:
            self._select_cell(s[0])

    def _autosave_current(self):
        """save the current cell if it has annotations OR already has a saved file (so deleting all
        annotations and leaving actually clears the file, instead of leaving a stale ✓)."""
        if self.base is None or not self.cell_key:
            return
        existing = os.path.join(self._default_savedir(), self._anno_key() + "_annotations.json")
        if self.annos or os.path.isfile(existing):
            try:
                self.write_annotations()
            except Exception:
                pass

    def _select_cell(self, i):
        if not (0 <= i < len(self.cells)):
            return
        self._autosave_current()                         # persist the cell we're leaving (incl. emptied ones)
        self.cell_idx = i
        self.load_cell(self.cells[i])
        self._refresh_cell_list()

    def load_cell(self, cell):
        """Load one traced cell: raw grayscale base + tracing overlay, both from its _neuron_out folder.
        Annotation file is keyed by the UNIQUE cell name and saved to the shared batch annotations folder."""
        try:
            ov = Image.open(cell['overlay']).convert('RGB')
            if cell['raw']:
                base = _stretch_rgb(np.asarray(Image.open(cell['raw']).convert('L')))
                overlay = ov.resize(base.size)
            else:                                        # older run without a raw preview: overlay is the base
                base = ov.copy(); overlay = None
        except Exception as e:
            messagebox.showerror("Could not load cell", "%s\n\n%s" % (cell['name'], e)); return
        self._set_display(base, overlay, cell['name'], cell['overlay'])
        self.status.config(text="Cell %d/%d: %s   ·   from %s"
                           % (self.cell_idx + 1, len(self.cells), cell['name'], cell['dir']))

    def _step_cell(self, d):
        if self.cells and self.cell_idx is not None:
            self._select_cell(min(max(self.cell_idx + d, 0), len(self.cells) - 1))

    def _on_close(self):
        """auto-save the current cell before closing so the last cell's work is never lost."""
        self._autosave_current()
        self.destroy()

    # ---------- loading ----------
    def _set_display(self, base_rgb, overlay_rgb, cell_key, image_ref):
        """Common loader: set the raw base + tracing overlay, reset/resume annotations, fit the view.
        cell_key = unique annotation-file name; image_ref = recorded in the JSON. save_dir is managed
        by the caller (one shared folder in batch mode)."""
        self.base = base_rgb; self.overlay = overlay_rgb
        self.W, self.H = self.base.size
        self.cell_key = cell_key
        self.image_path = image_ref
        self.annos = []; self.sel = None; self._next_id = 1
        self._load_existing()               # resume prior annotations if present
        self._refresh_list()
        self._fitted = False; self.fit()

    def load_image(self, path, out_dir):
        try:
            import neuron_pipeline as pipe
            gray = pipe.load_gray(path)
        except Exception:
            gray = np.asarray(Image.open(path).convert('L'))
        self.out_dir = out_dir or self.out_dir
        base = _stretch_rgb(gray)
        ovp = _find_overlay(path, self.out_dir)
        ov = Image.open(ovp).convert('RGB').resize(base.size) if ovp else None
        self.save_dir = os.path.join(os.path.dirname(path), "annotations")
        self._set_display(base, ov, os.path.splitext(os.path.basename(path))[0], path)
        msg = "Loaded %s (%dx%d)" % (os.path.basename(path), self.W, self.H)
        msg += " · tracing loaded" if ov is not None else " · NO tracing found for this image"
        self.status.config(text=msg)

    def _default_savedir(self):
        return self.save_dir or os.path.join(os.path.dirname(self.image_path or "."), "annotations")

    def _anno_key(self):
        """unique base name for this cell's annotation file (cell folder name in batch; stem otherwise)."""
        return self.cell_key or os.path.splitext(os.path.basename(self.image_path or "cell"))[0]

    def _load_existing(self):
        p = os.path.join(self._default_savedir(), self._anno_key() + "_annotations.json")
        if os.path.isfile(p):
            try:
                with open(p, encoding='utf-8') as f:
                    data = json.load(f)
                self.annos = data.get('annotations', [])
                for a in self.annos:
                    a.setdefault('visible', True)     # tolerate records saved before visibility existed
                self._next_id = max([a.get('id', 0) for a in self.annos], default=0) + 1
                self.status.config(text="Resumed %d existing annotations." % len(self.annos))
            except Exception:
                pass

    # ---------- coordinate transforms ----------
    def _i2c(self, x, y):
        return (x - self.ox) * self.scale, (y - self.oy) * self.scale

    def _c2i(self, cx, cy):
        return cx / self.scale + self.ox, cy / self.scale + self.oy

    def fit(self):
        if self.base is None:
            return
        cw = max(self.canvas.winfo_width(), 50); ch = max(self.canvas.winfo_height(), 50)
        self.scale = min(cw / self.W, ch / self.H) * 0.98
        self.ox = self.W / 2 - cw / (2 * self.scale)
        self.oy = self.H / 2 - ch / (2 * self.scale)
        self._render()

    # ---------- rendering ----------
    def _set_alpha(self, v): self.alpha = v; self._render()

    def _set_bright(self, v): self.brightness = v; self._render()

    def _toggle_tracing(self):
        if self.overlay is None and not self.show_tracing.get():
            # this cell has no separate raw layer -> the tracing is baked into the image, can't hide it
            self.status.config(text="This cell has no separate raw layer (older run) — tracing can't be hidden. Re-run tracing to enable.")
        self._render()

    def _render(self):
        if self.base is None:
            return
        cw = max(self.canvas.winfo_width(), 1); ch = max(self.canvas.winfo_height(), 1)
        ix0 = max(0, int(self.ox)); iy0 = max(0, int(self.oy))
        ix1 = min(self.W, int(self.ox + cw / self.scale) + 1)
        iy1 = min(self.H, int(self.oy + ch / self.scale) + 1)
        self.canvas.delete('img')
        if ix1 <= ix0 or iy1 <= iy0:
            self._draw_annos(); return
        base = self.base.crop((ix0, iy0, ix1, iy1))
        if abs(self.brightness - 1.0) > 1e-3:
            base = ImageEnhance.Brightness(base).enhance(self.brightness)
        show = getattr(self, 'show_tracing', None) is None or self.show_tracing.get()
        if self.overlay is not None and self.alpha > 0.01 and show:
            ov = self.overlay.crop((ix0, iy0, ix1, iy1))
            comp = Image.blend(base, ov, self.alpha)
        else:
            comp = base
        tw = max(1, int((ix1 - ix0) * self.scale)); th = max(1, int((iy1 - iy0) * self.scale))
        comp = comp.resize((tw, th), Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(comp)
        px, py = self._i2c(ix0, iy0)
        self.canvas.create_image(px, py, anchor='nw', image=self._photo, tags='img')
        self.canvas.tag_lower('img')
        self._draw_annos()

    def _is_visible(self, a):
        cv = self.cat_visible.get(a.get('category'))          # unknown category -> treat as visible
        return a.get('visible', True) and (cv.get() if cv is not None else True)

    def _draw_annos(self):
        self.canvas.delete('anno')
        for i, a in enumerate(self.annos):
            if not self._is_visible(a):
                continue
            rgb = CAT.get(a.get('category', 'other'), CAT['other'])[3]
            hexc = "#%02x%02x%02x" % rgb
            sel = (i == self.sel)
            w = 3 if sel else 2
            label_xy = None
            if a.get('box'):
                x0, y0, x1, y1 = a['box']
                cx0, cy0 = self._i2c(x0, y0); cx1, cy1 = self._i2c(x1, y1)
                self.canvas.create_rectangle(cx0, cy0, cx1, cy1, outline=hexc, width=w,
                                             dash=() if sel else (4, 2), tags='anno')
                label_xy = (cx0 + 2, cy0 - 8)
            if a.get('point'):
                cx, cy = self._i2c(*a['point']); r = 7 if sel else 5
                self.canvas.create_line(cx - r - 3, cy, cx + r + 3, cy, fill=hexc, width=w, tags='anno')
                self.canvas.create_line(cx, cy - r - 3, cx, cy + r + 3, fill=hexc, width=w, tags='anno')
                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=hexc, width=w, tags='anno')
                label_xy = (cx + r + 4, cy - r)
            if a.get('path') and len(a['path']) > 1:
                pts = []
                for x, y in a['path']:
                    cx, cy = self._i2c(x, y); pts += [cx, cy]
                self.canvas.create_line(*pts, fill=hexc, width=w + 1, tags='anno', smooth=True)
                if label_xy is None:
                    label_xy = (pts[0] + 4, pts[1] - 10)
            if label_xy is not None:
                self.canvas.create_text(label_xy[0], label_xy[1], anchor='w', fill=hexc,
                                        text="#%d %s" % (a['id'], CAT.get(a.get('category', 'other'), CAT['other'])[1]),
                                        font=(theme.FONT, 9, 'bold'), tags='anno')

    # ---------- pointer: pan / zoom ----------
    def _pan_press(self, e):
        self._drag = ('pan', e.x, e.y, self.ox, self.oy)

    def _pan_move(self, e):
        if not self._drag or self._drag[0] != 'pan':
            return
        _, sx, sy, ox0, oy0 = self._drag
        self.ox = ox0 - (e.x - sx) / self.scale
        self.oy = oy0 - (e.y - sy) / self.scale
        self._render()

    def _wheel(self, e):
        if self.base is None:
            return
        factor = 1.2 if e.delta > 0 else 1 / 1.2
        new_scale = min(max(self.scale * factor, self._min_scale()), 80.0)   # clamp: can't zoom past sane limits
        if new_scale == self.scale:
            return
        ix, iy = self._c2i(e.x, e.y)                 # image point under cursor stays put
        self.scale = new_scale
        self.ox = ix - e.x / self.scale
        self.oy = iy - e.y / self.scale
        self._render()

    def _min_scale(self):
        """don't let the user zoom out below ~1/4 of the whole image filling the canvas."""
        cw = max(self.canvas.winfo_width(), 50); ch = max(self.canvas.winfo_height(), 50)
        return min(cw / max(self.W, 1), ch / max(self.H, 1)) * 0.25

    def _hover(self, e):
        if self.base is None:
            return
        ix, iy = self._c2i(e.x, e.y)
        self.status.config(text="x=%d y=%d   zoom=%.2fx   [%s]   %s"
                           % (ix, iy, self.scale, self.tool.get(), GESTURE.get(self.active_cat.get(), "")))

    # ---------- pointer: left button (tool-dependent) ----------
    # Point : one click drops a marker.
    # Box   : press-drag-release rubber-bands a rectangle.
    # Line  : click to drop points one-by-one (double-click / Enter finishes), OR press-drag for freehand.
    def _b1_press(self, e):
        if self.base is None:
            return
        tool = self.tool.get()
        if tool == 'pan':
            self._finish_line()
            hit = self._hit_test(e.x, e.y)
            self.sel = hit; self._sync_sel(); self._pan_press(e); self._render(); return
        if tool == 'point':
            self._finish_line()
            ix, iy = self._c2i(e.x, e.y)
            self._add_anno(kind='point', point=[round(ix), round(iy)])
            self._drag = None; self._refresh_list(); self._render(); return
        if tool == 'box':
            self._finish_line()
            self._drag = ('box', e.x, e.y)
            return
        if tool == 'line':
            self._drag = ('line', e.x, e.y, False)     # (kind, press_x, press_y, became_drag)

    def _b1_move(self, e):
        if not self._drag:
            return
        kind = self._drag[0]
        if kind == 'pan':
            self._pan_move(e)
        elif kind == 'box':
            self.canvas.delete('rubber')
            x0, y0 = self._drag[1], self._drag[2]
            hexc = "#%02x%02x%02x" % CAT.get(self.active_cat.get(), CAT['other'])[3]
            # dark backing + category-coloured outline so it's visible live on ANY background
            self.canvas.create_rectangle(x0, y0, e.x, e.y, outline="#000", width=3, tags='rubber')
            self.canvas.create_rectangle(x0, y0, e.x, e.y, outline=hexc, width=1, tags='rubber')
            ix0, iy0 = self._c2i(x0, y0); ix1, iy1 = self._c2i(e.x, e.y)
            self.status.config(text="box  %d x %d px" % (abs(ix1 - ix0), abs(iy1 - iy0)))
        elif kind == 'line':
            px, py, dragged = self._drag[1], self._drag[2], self._drag[3]
            ix, iy = self._c2i(e.x, e.y)
            if not dragged:                            # crossed the drag threshold -> freehand stroke
                if (e.x - px) ** 2 + (e.y - py) ** 2 <= 9:
                    return
                if self._line_idx is None:
                    self._add_anno(kind='line'); self._line_idx = self.sel
                    self.annos[self._line_idx]['path'] = [self._c2i(px, py)]
                self._drag = ('line', px, py, True)
            pts = self.annos[self._line_idx]['path']
            if (ix - pts[-1][0]) ** 2 + (iy - pts[-1][1]) ** 2 > (3 / self.scale) ** 2:
                pts.append((ix, iy)); self._draw_annos()

    def _b1_release(self, e):
        if not self._drag:
            return
        kind = self._drag[0]
        if kind == 'box':
            self.canvas.delete('rubber')
            x0, y0 = self._c2i(self._drag[1], self._drag[2]); x1, y1 = self._c2i(e.x, e.y)
            if abs(x1 - x0) > 4 and abs(y1 - y0) > 4:
                self._add_anno(kind='box', box=[round(min(x0, x1)), round(min(y0, y1)),
                                                round(max(x0, x1)), round(max(y0, y1))])
            else:                                    # a click, not a drag: select instead
                self.sel = self._hit_test(e.x, e.y); self._sync_sel()
        elif kind == 'line':
            if self._drag[3]:                        # freehand stroke complete
                self._finish_line()
            else:                                    # a click: add one polyline vertex (start if none)
                ix, iy = self._c2i(e.x, e.y)
                if self._line_idx is None:
                    self._add_anno(kind='line'); self._line_idx = self.sel
                    self.annos[self._line_idx]['path'] = [(ix, iy)]
                else:
                    self.annos[self._line_idx]['path'].append((ix, iy))
                self._draw_annos()
                self.status.config(text="drawing line — click to add points, double-click / Enter to finish (%d)"
                                   % len(self.annos[self._line_idx]['path']))
        self._drag = None
        self._refresh_list()
        self._render()                                   # redraw so a finished box actually appears

    def _finish_line(self, *_):
        """commit the in-progress click-drawn polyline; drop it if it has fewer than 2 points."""
        if self._line_idx is None:
            return
        idx = self._line_idx; self._line_idx = None
        if 0 <= idx < len(self.annos):
            path = self.annos[idx].get('path') or []
            deduped = []
            for p in path:                            # a finishing double-click repeats the last point
                if not deduped or abs(p[0] - deduped[-1][0]) > 1e-6 or abs(p[1] - deduped[-1][1]) > 1e-6:
                    deduped.append(p)
            self.annos[idx]['path'] = deduped
            if len(deduped) < 2:
                self.annos.pop(idx)
                if self.sel == idx:
                    self.sel = None
        self._refresh_list(); self._render()

    def _hit_test(self, cx, cy):
        ix, iy = self._c2i(cx, cy); tol = 8 / max(self.scale, 1e-6)
        for i in range(len(self.annos) - 1, -1, -1):
            a = self.annos[i]
            if not self._is_visible(a):
                continue
            b = a.get('box')
            if b and b[0] - 3 <= ix <= b[2] + 3 and b[1] - 3 <= iy <= b[3] + 3:
                return i
            p = a.get('point')
            if p and abs(p[0] - ix) <= tol + 4 and abs(p[1] - iy) <= tol + 4:
                return i
            for px, py in (a.get('path') or []):           # lines are clickable near any vertex
                if abs(px - ix) <= tol and abs(py - iy) <= tol:
                    return i
        return None

    # ---------- annotation model ----------
    def _add_anno(self, kind='box', box=None, point=None):
        a = {'id': self._next_id, 'category': self.active_cat.get(), 'kind': kind,
             'box': box, 'point': point, 'path': [], 'note': '', 'visible': True,
             'time': datetime.datetime.now().isoformat(timespec='seconds')}
        self._next_id += 1
        self.annos.append(a); self.sel = len(self.annos) - 1
        self._refresh_list(); self._sync_sel()

    def _delete_sel(self):
        if self.sel is not None and 0 <= self.sel < len(self.annos):
            del self.annos[self.sel]; self.sel = None
            self._refresh_list(); self._sync_sel(); self._render()

    def _undo(self):
        """remove the most recently added annotation."""
        if self.annos:
            self.annos.pop(); self.sel = None
            self._refresh_list(); self._sync_sel(); self._render()

    def _clear_path(self):
        if self.sel is not None:
            self.annos[self.sel]['path'] = []; self._render()

    def _toggle_visible(self):
        if self.sel is not None and 0 <= self.sel < len(self.annos):
            a = self.annos[self.sel]; a['visible'] = not a.get('visible', True)
            self._refresh_list(); self._render()

    def _set_all_visible(self, vis):
        for a in self.annos:
            a['visible'] = vis
        for v in self.cat_visible.values():
            v.set(vis)
        self._refresh_list(); self._render()

    def _refresh_list(self):
        keep = self.sel
        self.listbox.delete(0, 'end')
        for a in self.annos:
            zh = CAT.get(a.get('category', 'other'), CAT['other'])[1]
            tag = self._kind(a)
            note = (a.get('note') or '').replace('\n', ' ')[:14]
            eye = "●" if a.get('visible', True) else "○"
            self.listbox.insert('end', "%s #%-2d %s %-5s %s" % (eye, a['id'], zh, tag, note))
        if keep is not None and 0 <= keep < len(self.annos):
            self.listbox.selection_clear(0, 'end'); self.listbox.selection_set(keep)

    @staticmethod
    def _kind(a):
        """the annotation's primitive; infer from geometry for legacy records that predate 'kind'."""
        if a.get('kind'):
            return a['kind']
        if a.get('point'):
            return 'point'
        if a.get('box'):
            return 'box'
        return 'line'

    def _on_list_select(self, e):
        if self._syncing:
            return
        s = self.listbox.curselection()
        if s:
            self.sel = s[0]; self._sync_sel(); self._render()

    def _sync_sel(self):
        self._syncing = True
        try:
            self.note.delete('1.0', 'end')
            if self.sel is not None and 0 <= self.sel < len(self.annos):
                a = self.annos[self.sel]
                self.note.insert('1.0', a.get('note', ''))
                self.active_cat.set(a.get('category', 'other'))
            self._refresh_list()
        finally:
            self._syncing = False

    def _on_note_edit(self, e):
        if self.sel is not None and 0 <= self.sel < len(self.annos):
            self.annos[self.sel]['note'] = self.note.get('1.0', 'end').strip()
            self._refresh_list()      # live-update the note snippet in the list

    def _on_key(self, e):
        if e.widget is self.note:            # don't hijack typing in the note box
            return
        if e.keysym in ('Delete', 'BackSpace'):
            self._delete_sel()
        elif e.char in ('v', 'V'):
            self._toggle_visible()
        elif e.char in [c[4] for c in CATEGORIES]:
            key = next(c[0] for c in CATEGORIES if c[4] == e.char)
            self.active_cat.set(key)
            if self.sel is not None and 0 <= self.sel < len(self.annos):
                self.annos[self.sel]['category'] = key; self._refresh_list(); self._render()

    # ---------- help ----------
    def _show_help(self):
        win = tk.Toplevel(self); win.title("How to use"); win.geometry("640x560")
        win.transient(self); win.configure(bg=P['bg']); titlebar(win)
        ttk.Button(win, text="Close", command=win.destroy).pack(side='bottom', pady=px(10))
        txt = tk.Text(win, wrap='word', font=(theme.FONT, 10), padx=px(16), pady=px(14))
        paint_list(txt); txt.configure(bg=P['card'], highlightthickness=0)
        sb = ttk.Scrollbar(win, command=txt.yview); sb.pack(side='right', fill='y')
        txt.pack(fill='both', expand=True)
        txt.config(yscrollcommand=sb.set)
        H = """REVIEW & ANNOTATE — how to use

WORKFLOW
  1. The original image loads with our tracing drawn on top (red lines, markers).
  2. Pick the error type (buttons on the right, or number keys 1-5). The best drawing
     tool is selected automatically — you can still change it in the top toolbar.
  3. Mark the error on the image (see each tool below).
  4. Type a note if useful, then click "Save annotations" (or Ctrl+S).
     Saves <name>_annotations.json + <name>_annotated.png into an "annotations" folder.

ERROR TYPES → the tool picked for you
  1  Missing neurite    → LINE   (draw where the branch should be)
  2  False neurite      → BOX    (circle the wrong trace)
  3  Crossing/branch    → POINT  (click the wrong junction)
  4  Broken gap         → LINE   (draw a stroke joining the two ends)
  5  Other              → BOX

TOOLS
  • POINT — click once to drop a marker (a crosshair + ring).
  • LINE — CLICK to drop points one after another; DOUBLE-CLICK or press Enter
           to finish (Esc cancels). Or press and DRAG to draw freehand. This is how
           you mark a missing branch (#1) or a gap (#4).
  • BOX — press and DRAG a rectangle around the area.
  • PAN — drag to move; click an annotation to select it.

NAVIGATION
  • Mouse wheel = zoom (to the cursor).  Middle-drag or right-drag = pan.  Fit = reset view.
  • "Tracing" slider = fade our tracing in/out.  "Bright" slider = brighten the raw image.

MANAGING ANNOTATIONS
  • Right-side list shows every annotation.  ● = shown, ○ = hidden.
  • Double-click a row (or press v) to show/hide it.  "Show all" / "Hide all" toggle everything.
  • Each error type also has a checkbox to hide that whole category.
  • Click an annotation (Pan tool) or a row to select it, then edit its note or Delete it.
  • Ctrl+Z = undo last.  Del = delete selected.  Ctrl+S = save.
  Hidden annotations are left out of the saved PNG too.
"""
        txt.insert('1.0', H); txt.config(state='disabled')

    # ---------- choose save dir / save ----------
    def _choose_savedir(self):
        d = filedialog.askdirectory(title="Folder to save annotations into")
        if d:
            self.save_dir = d
            self.save_lbl.config(text="save: " + d, foreground=P['ok'])

    def write_annotations(self, outdir=None):
        """Pure I/O: write the JSON record + burned-in PNG to ONE folder, named by the unique cell key
        (so cells that share a basename like 'mor' don't overwrite each other). Returns (json, png)."""
        outdir = outdir or self._default_savedir()
        os.makedirs(outdir, exist_ok=True)
        key = self._anno_key()
        data = {
            'cell': key,
            'image': os.path.basename(self.image_path or ""),
            'image_path': self.image_path,
            'width': self.W, 'height': self.H,
            'saved': datetime.datetime.now().isoformat(timespec='seconds'),
            'categories': {c[0]: c[1] for c in CATEGORIES},
            'annotations': self.annos,
        }
        jp = os.path.join(outdir, key + "_annotations.json")
        with open(jp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        pp = os.path.join(outdir, key + "_annotated.png")
        self._render_annotated().save(pp)
        return jp, pp

    def save(self, interactive=True):
        if self.base is None:
            return None
        if interactive and not self.annos and not messagebox.askyesno(
                "No annotations", "No annotations yet. Save an empty record anyway?"):
            return None
        try:
            jp, pp = self.write_annotations()
        except Exception as e:                       # never lose a reviewer's work to an unexpected error
            if interactive:
                messagebox.showerror("Save failed", str(e))
            return None
        outdir = os.path.dirname(jp)
        self.status.config(text="Saved %d annotations -> %s" % (len(self.annos), outdir))
        if interactive:
            messagebox.showinfo("Saved", "Saved to:\n%s\n\n%s\n%s"
                                % (outdir, os.path.basename(jp), os.path.basename(pp)))
        return jp, pp

    def _render_annotated(self):
        """Burn boxes + points + paths + labels onto the tracing overlay (or raw) for a visual record."""
        img = (self.overlay or self.base).copy()
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("msyh.ttc", max(14, self.W // 90))
        except Exception:
            font = ImageFont.load_default()
        lw = max(2, self.W // 500)
        for a in self.annos:
            if not self._is_visible(a):
                continue
            rgb = CAT.get(a.get('category', 'other'), CAT['other'])[3]
            lbl = "#%d %s" % (a['id'], CAT.get(a.get('category', 'other'), CAT['other'])[1])
            label_xy = None
            if a.get('box'):
                d.rectangle(a['box'], outline=rgb, width=lw)
                label_xy = (a['box'][0] + 3, a['box'][1] + 3)
            if a.get('point'):
                x, y = a['point']; r = max(8, self.W // 180)
                d.line([x - r - 4, y, x + r + 4, y], fill=rgb, width=lw)
                d.line([x, y - r - 4, x, y + r + 4], fill=rgb, width=lw)
                d.ellipse([x - r, y - r, x + r, y + r], outline=rgb, width=lw)
                label_xy = (x + r + 5, y - r)
            if a.get('path') and len(a['path']) > 1:
                d.line([tuple(p) for p in a['path']], fill=rgb, width=max(3, self.W // 400), joint='curve')
                if label_xy is None:
                    label_xy = (a['path'][0][0] + 4, a['path'][0][1] - self.W // 90)
            if label_xy is not None:
                d.text(label_xy, lbl, fill=rgb, font=font)
        return img


def open_annotator(master=None, image_path=None, out_dir=None, folder=None):
    return Annotator(master, image_path=image_path, out_dir=out_dir, folder=folder)


if __name__ == '__main__':
    root = tk.Tk(); root.withdraw()
    img = sys.argv[1] if len(sys.argv) > 1 else None
    win = Annotator(root, image_path=img)
    if len(sys.argv) > 2 and win.base is not None:
        try:
            win.overlay = Image.open(sys.argv[2]).convert('RGB').resize((win.W, win.H)); win._render()
        except Exception:
            pass
    win.protocol("WM_DELETE_WINDOW", lambda: (win._on_close(), root.destroy()))
    root.mainloop()
