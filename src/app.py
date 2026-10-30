# -*- coding: utf-8 -*-
"""NeuronTracer desktop app.
Single image or whole-folder batch -> segmentation, SWC tree + soma, morphometrics.
CPU only. Tkinter UI. Packaged to a folder app with PyInstaller.
"""
import os, sys, threading, queue, traceback

# frozen bootstrap: put wheel-vendored DLL dirs on the search path before scipy/onnx
if getattr(sys, "frozen", False):
    _base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    sys.path.insert(0, _base)
    for _sub in ("", "scipy.libs", "numpy.libs", "scikit_image.libs",
                 "skimage.libs", "pillow.libs", os.path.join("onnxruntime", "capi")):
        _d = os.path.join(_base, _sub)
        if os.path.isdir(_d):
            try:
                os.add_dll_directory(_d)
            except Exception:
                pass

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
Image.MAX_IMAGE_PIXELS = None
import theme as th
from theme import P, FONT, IMG_BG, LEGEND, px, apply_style, Tooltip
try:
    import neuron_pipeline as pipe
except Exception:
    _err = os.path.join(os.path.dirname(sys.executable), "import_error.txt")
    try:
        with open(_err, "w") as _f:
            traceback.print_exc(file=_f)
    except Exception:
        pass
    traceback.print_exc()
    sys.exit(3)

APP_TITLE = "NeuronTracer"
PREVIEW = 560

class ImageViewer(ttk.Frame):
    """Preview area: Original / Result / Side by side, fills the window.
    Ctrl+wheel zooms toward the cursor, left-drag pans, double-click fits.
    In side-by-side mode both panes share one view, so they stay aligned."""
    MAX_ZOOM = 16.0

    def __init__(self, master):
        super().__init__(master)
        self.orig = None; self.res = None
        self.mode = tk.StringVar(value='orig')
        self.zoom = 1.0; self.cx = self.cy = 0.0; self.fit = True
        self._drag = None; self._tk = []; self._msg = "Open an image, or a folder for batch"

        bar = ttk.Frame(self); bar.pack(fill='x', pady=(0, px(8)))
        self.mode_btns = {}
        for val, txt in (('orig', "Original"), ('res', "Result"), ('both', "Side by side")):
            b = ttk.Radiobutton(bar, text=txt, value=val, variable=self.mode, style='Seg.Toolbutton',
                                command=self._mode_changed)
            b.pack(side='left', padx=(0, 1)); self.mode_btns[val] = b
        ttk.Button(bar, text="Fit", style='Small.TButton', command=self.fit_view).pack(side='left', padx=(px(12), 0))
        self.zoom_lbl = ttk.Label(bar, text="", width=6, style='Sub.TLabel'); self.zoom_lbl.pack(side='left', padx=px(8))
        ttk.Label(bar, text="wheel  zoom     drag  pan     double-click  fit",
                  style='Hint.TLabel').pack(side='right')

        self.legend = tk.Frame(self, bg=IMG_BG, padx=px(10), pady=px(6))
        for kind, col, txt in LEGEND:
            sw = tk.Canvas(self.legend, width=px(16), height=px(16), bg=IMG_BG, highlightthickness=0)
            self._swatch(sw, kind, col); sw.pack(side='left', padx=(0, px(4)))
            tk.Label(self.legend, text=txt, bg=IMG_BG, fg='#c3c8cf', font=(FONT, 9)).pack(side='left', padx=(0, px(14)))

        self.canvas = tk.Canvas(self, width=PREVIEW, height=PREVIEW, bg=IMG_BG, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        c = self.canvas
        c.bind('<Configure>', lambda e: self.render())
        c.bind('<MouseWheel>', lambda e: self._zoom_at(e, e.delta > 0))        # same as the edit window
        c.bind('<Button-4>', lambda e: self._zoom_at(e, True))                 # Linux
        c.bind('<Button-5>', lambda e: self._zoom_at(e, False))
        c.bind('<ButtonPress-1>', self._press)
        c.bind('<B1-Motion>', self._move)
        c.bind('<ButtonRelease-1>', self._release)
        c.bind('<Double-Button-1>', lambda e: self.fit_view())
        self._update_buttons()

    @staticmethod
    def _swatch(c, kind, col):
        s = px(16); m = s // 2
        if kind == 'line':
            c.create_line(px(1), m, s - px(1), m, fill=col, width=max(2, px(2)))
        elif kind == 'ring':
            c.create_oval(px(2), px(2), s - px(2), s - px(2), outline=col, width=max(2, px(2)))
        elif kind == 'dot':
            c.create_oval(m - px(4), m - px(4), m + px(4), m + px(4), fill=col, outline='')
        else:
            c.create_line(px(3), px(3), s - px(3), s - px(3), fill=col, width=max(2, px(2)))
            c.create_line(px(3), s - px(3), s - px(3), px(3), fill=col, width=max(2, px(2)))

    # ---------- public ----------
    def set_original(self, img):
        self.orig = img.convert('RGB'); self.res = None; self._msg = None
        self.mode.set('orig'); self._update_buttons(); self.fit_view()

    def set_result(self, img):
        img = img.convert('RGB')
        if self.orig is not None and img.size != self.orig.size:
            img = img.resize(self.orig.size, Image.BILINEAR)
        self.res = img
        if self.orig is None:
            self.orig = img
        self.mode.set('both'); self._update_buttons(); self.fit_view()

    def show_message(self, text):
        self.orig = self.res = None; self._msg = text
        self._update_buttons(); self.render()

    def fit_view(self):
        self.fit = True; self.render()

    # ---------- internals ----------
    def _update_buttons(self):
        for val, b in self.mode_btns.items():
            ok = (self.orig is not None) if val == 'orig' else (self.res is not None)
            b.config(state=('normal' if ok else 'disabled'))
        self._sync_legend()

    def _sync_legend(self):
        want = self.res is not None and self.mode.get() in ('res', 'both')
        if want and not self.legend.winfo_manager():
            self.legend.pack(side='bottom', fill='x', before=self.canvas)
        elif not want and self.legend.winfo_manager():
            self.legend.pack_forget()

    def _mode_changed(self):
        self._sync_legend(); self.render()

    def _panes(self):
        """list of (image, title) for the current mode."""
        m = self.mode.get()
        if m == 'both' and self.res is not None:
            return [(self.orig, "Original"), (self.res, "Result")]
        if m == 'res' and self.res is not None:
            return [(self.res, None)]
        return [(self.orig, None)]

    def _pane_size(self, n):
        w = max(self.canvas.winfo_width(), 50); h = max(self.canvas.winfo_height(), 50)
        gap = px(4) if n > 1 else 0
        return (w - gap * (n - 1)) // n, h, gap

    def _fit_zoom(self, pw, ph):
        iw, ih = self.orig.size
        return min(pw / iw, ph / ih)

    def _clamp(self, pw, ph):
        iw, ih = self.orig.size
        fz = self._fit_zoom(pw, ph)
        if self.fit or self.zoom <= fz * 1.0001:
            self.zoom = fz; self.fit = True
        self.zoom = min(self.zoom, self.MAX_ZOOM)
        for attr, size, view in (('cx', iw, pw), ('cy', ih, ph)):
            half = view / (2 * self.zoom)
            v = getattr(self, attr)
            v = size / 2 if half * 2 >= size else min(max(v, half), size - half)
            setattr(self, attr, v)

    def render(self):
        c = self.canvas; c.delete('all'); self._tk = []
        if self.orig is None:
            w = max(c.winfo_width(), 50); h = max(c.winfo_height(), 50)
            if self._msg:
                c.create_text(w // 2, h // 2, text=self._msg, fill='#7d838b', font=(FONT, 12), justify='center')
            self.zoom_lbl.config(text=""); return
        panes = self._panes()
        pw, ph, gap = self._pane_size(len(panes))
        self._clamp(pw, ph)
        z = self.zoom; iw, ih = self.orig.size
        x0 = self.cx - pw / (2 * z); y0 = self.cy - ph / (2 * z)
        ix0, iy0 = max(0.0, x0), max(0.0, y0)
        ix1, iy1 = min(float(iw), x0 + pw / z), min(float(ih), y0 + ph / z)
        dw, dh = int(round((ix1 - ix0) * z)), int(round((iy1 - iy0) * z))
        ox, oy = int(round((ix0 - x0) * z)), int(round((iy0 - y0) * z))
        resample = Image.NEAREST if z >= 2 else Image.BILINEAR
        for k, (img, title) in enumerate(panes):
            px0 = k * (pw + gap)
            if dw > 0 and dh > 0:
                tile = img.resize((dw, dh), resample, box=(ix0, iy0, ix1, iy1))
                t = ImageTk.PhotoImage(tile, master=c); self._tk.append(t)
                c.create_image(px0 + ox, oy, image=t, anchor='nw')
            if title:
                tid = c.create_text(px0 + px(16), px(18), text=title, anchor='w', fill='#f1f3f5',
                                    font=(FONT, 9, 'bold'))
                bx0, by0, bx1, by1 = c.bbox(tid)
                c.tag_lower(c.create_rectangle(bx0 - px(8), by0 - px(3), bx1 + px(8), by1 + px(3),
                                               fill='#000000', outline='', stipple='gray50'), tid)
            if k:
                c.create_rectangle(px0 - gap, 0, px0, ph, fill='#2a2d33', outline='')
        self.zoom_lbl.config(text="%d%%" % round(z * 100))

    def _local(self, e):
        """canvas event -> (pane width, pane height, x inside its pane)."""
        n = len(self._panes()); pw, ph, gap = self._pane_size(n)
        return pw, ph, e.x - (e.x // (pw + gap)) * (pw + gap)

    def _zoom_at(self, e, zoom_in):
        if self.orig is None:
            return
        pw, ph, lx = self._local(e)
        z0 = self.zoom
        z1 = min(z0 * (1.25 if zoom_in else 1 / 1.25), self.MAX_ZOOM)
        if z1 == z0:
            return
        ix = self.cx + (lx - pw / 2) / z0; iy = self.cy + (e.y - ph / 2) / z0   # image point under cursor
        self.zoom = z1; self.fit = False
        self.cx = ix - (lx - pw / 2) / z1; self.cy = iy - (e.y - ph / 2) / z1
        self.render()

    def _press(self, e):
        if self.orig is not None:
            self._drag = (e.x, e.y, self.cx, self.cy); self.canvas.config(cursor='fleur')

    def _move(self, e):
        if not self._drag or self.fit:
            return
        sx, sy, cx0, cy0 = self._drag
        self.cx = cx0 - (e.x - sx) / self.zoom; self.cy = cy0 - (e.y - sy) / self.zoom
        self.render()

    def _release(self, e):
        self._drag = None; self.canvas.config(cursor='')


# results table: (group, [(label, metric key, format, unit)]); fmt None = count
RESULT_ROWS = [
    ("Size & shape", [("Total length", 'total_length_um', "%.1f", "µm"), ("Branch points", 'num_branch_points', None, ""),
                      ("Tips", 'num_tips', None, ""), ("Branches", 'num_branches', None, ""),
                      ("Mean intersections", 'mean_intersections', "%.2f", ""),
                      ("Enclosing radius", 'enclosing_radius_um', "%.1f", "µm"),
                      ("Polarity index", 'polarity_index', "%.3f", "")]),
    ("Soma", [("Area", 'soma_area_um2', "%.1f", "µm²"), ("Perimeter", 'soma_perimeter_um', "%.1f", "µm"),
              ("Circularity", 'soma_circularity', "%.3f", ""), ("Roundness", 'soma_roundness', "%.3f", ""),
              ("Solidity", 'soma_solidity', "%.3f", ""), ("Diameter", 'soma_diameter_um', "%.1f", "µm")]),
    ("Neurite thickness", [("Median width (FWHM)", 'neurite_width_um', "%.2f", "µm")]),
    ("Branch structure · preliminary", [("Highest path order", 'highest_branch_order', None, ""),
                                       ("Avg bifurcation angle", 'avg_branch_angle_deg', "%.1f", "°"),
                                       ("Length before 1st bif.", 'avg_len_before_first_bif_um', "%.1f", "µm"),
                                       ("Avg segment length", 'avg_segment_len_um', "%.1f", "µm")]),
]


class App:
    def __init__(self, root):
        self.root = root
        root.title(APP_TITLE)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry("%dx%d" % (min(px(1360), sw - px(60)), min(px(860), sh - px(100))))
        root.minsize(px(900), px(600))
        self.image_path = None
        self.batch_folder = None
        self.result = None
        self.out_dir = None
        self.q = queue.Queue()
        self.theme = 'dark'
        apply_style(root, self.theme)

        # ---- bottom status bar: status text + progress ----
        sbar = ttk.Frame(root, style='Status.TFrame', padding=(px(14), px(6))); sbar.pack(side='bottom', fill='x')
        self.status = ttk.Label(sbar, text="Ready.", style='Status.TLabel'); self.status.pack(side='left')
        self.theme_btn = ttk.Button(sbar, text="Light theme", style='Small.TButton', command=self.toggle_theme)
        self.theme_btn.pack(side='right')
        self.pct_lbl = ttk.Label(sbar, text="", width=5, style='Status.TLabel', anchor='e')
        self.pct_lbl.pack(side='right', padx=(0, px(14)))
        self.prog = ttk.Progressbar(sbar, mode='determinate', maximum=1.0, length=px(220))
        self.prog.pack(side='right', padx=(0, px(6)))

        main = ttk.Frame(root, padding=(px(14), px(12))); main.pack(fill='both', expand=True)
        right = ttk.Frame(main, width=px(360)); right.pack(side='right', fill='y'); right.pack_propagate(False)
        self.viewer = ImageViewer(main); self.viewer.pack(side='left', fill='both', expand=True, padx=(0, px(14)))
        root.bind('<Control-0>', lambda e: self.viewer.fit_view())

        ttk.Label(right, text="NeuronTracer", style='Title.TLabel').pack(anchor='w')
        ttk.Label(right, text="Fluorescence image  →  SWC tree + morphometrics",
                  style='Sub.TLabel').pack(anchor='w', pady=(0, px(12)))

        # ---- card 1: input ----
        card = self._card(right, "Input")
        brow = ttk.Frame(card, style='Card.TFrame', borderwidth=0); brow.pack(fill='x')
        brow.columnconfigure((0, 1), weight=1, uniform='b')
        self.open_img_btn = ttk.Button(brow, text="Open image…", command=self.open_image)
        self.open_img_btn.grid(row=0, column=0, sticky='ew', padx=(0, px(4)))
        self.open_folder_btn = ttk.Button(brow, text="Open folder…", command=self.open_folder)
        self.open_folder_btn.grid(row=0, column=1, sticky='ew', padx=(px(4), 0))
        Tooltip(self.open_folder_btn, "Batch mode: trace every image in a folder (subfolders too) and "
                "write one summary.csv with a row per cell.")
        self.file_lbl = ttk.Label(card, text="No input selected", style='CardMuted.TLabel', wraplength=px(320))
        self.file_lbl.pack(fill='x', pady=(px(8), 0))

        # ---- card 2: settings + run ----
        card = self._card(right, "Settings")
        g = ttk.Frame(card, style='Card.TFrame', borderwidth=0); g.pack(fill='x')
        g.columnconfigure(1, weight=1)
        ttk.Label(g, text="Pixel size", style='Card.TLabel').grid(row=0, column=0, sticky='w')
        self.vox_var = tk.StringVar(value="0.1625")
        ve = ttk.Entry(g, textvariable=self.vox_var, width=8, justify='right'); ve.grid(row=0, column=2, sticky='e')
        ttk.Label(g, text="µm/px", style='CardMuted.TLabel').grid(row=0, column=3, sticky='w', padx=(px(6), 0))
        Tooltip(ve, "Microns per pixel of your camera. 0.1625 is the lab's default.")
        self.hq_var = tk.BooleanVar(value=False)
        hq = ttk.Checkbutton(g, text="High quality (slower)", variable=self.hq_var, style='Card.TCheckbutton')
        hq.grid(row=1, column=0, columnspan=4, sticky='w', pady=(px(10), 0))
        Tooltip(hq, "Mirror test-time augmentation: about 2 min per image instead of about 30 s.")

        ttk.Label(card, text="Save to", style='Card.TLabel').pack(anchor='w', pady=(px(12), px(4)))
        orow = ttk.Frame(card, style='Card.TFrame', borderwidth=0); orow.pack(fill='x')
        self.out_var = tk.StringVar(value="")        # the user's choice ('' = auto)
        self.out_disp = tk.StringVar(value="Auto: next to the input")
        oe = ttk.Entry(orow, textvariable=self.out_disp, state='readonly'); oe.pack(side='left', fill='x', expand=True)
        Tooltip(oe, lambda: "Results are saved here:\n%s\n\nSet this before you Run. To save elsewhere, "
                "change it and Run again." % (self._effective_out_root() or "a *_neuron_out folder next to the input"))
        b = ttk.Button(orow, text="Auto", style='Small.TButton', command=self.reset_out); b.pack(side='right')
        Tooltip(b, "Save into a *_neuron_out folder next to the input.")
        ttk.Button(orow, text="…", width=3, style='Small.TButton', command=self.choose_out).pack(side='right', padx=px(4))

        self.run_btn = ttk.Button(card, text="Choose an input first", style='Accent.TButton', command=self.run,
                                  state='disabled')
        self.run_btn.pack(fill='x', pady=(px(14), 0))

        # ---- card 3: results ----
        card = self._card(right, "Results", expand=True)
        abar = ttk.Frame(card, style='Card.TFrame', borderwidth=0); abar.pack(side='bottom', fill='x', pady=(px(10), 0))
        abar.columnconfigure((0, 1), weight=1, uniform='a')
        self.open_out_btn = ttk.Button(abar, text="Open output", command=self.open_out, state='disabled')
        self.open_out_btn.grid(row=0, column=0, sticky='ew', padx=(0, px(4)))
        Tooltip(self.open_out_btn, "Per cell: mask, overlay, raw, .swc, metrics.csv, sholl.csv. "
                "Batch also writes summary.csv.")
        self.edit_btn = ttk.Button(abar, text="Edit tracing", command=self.open_editor, state='disabled')
        self.edit_btn.grid(row=0, column=1, sticky='ew', padx=(px(4), 0))
        # the older flag-only review window stays reachable, but out of the main flow
        self.annot_btn = ttk.Button(abar, text="Mark errors only (old review window)", style='Small.TButton',
                                    command=self.open_annotator, state='disabled')
        self.annot_btn.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(px(8), 0))
        Tooltip(self.edit_btn, "Correct the tracing by hand: delete wrong lines, cut overshooting tips, draw "
                "missed branches. Saves <name>_edited.swc and recomputed metrics next to the software's files, "
                "never over them.")
        Tooltip(self.annot_btn, "Open the review window to mark tracing errors. Annotations are saved to an "
                "'annotations' folder inside the output.")

        self.res_holder = ttk.Frame(card, style='Card.TFrame', borderwidth=0)
        self.res_holder.pack(fill='both', expand=True)
        self.res_empty = ttk.Label(self.res_holder, text="Run tracing to see the measurements here.",
                                   style='CardMuted.TLabel', wraplength=px(300))
        self.tree = ttk.Treeview(self.res_holder, columns=('value',), show='tree', style='Results.Treeview',
                                 selectmode='browse')
        self.tree.column('#0', stretch=True, minwidth=px(120))
        self.tree.column('value', width=px(120), stretch=False, anchor='e')
        self.tree_sb = ttk.Scrollbar(self.res_holder, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.tree_sb.set)
        self._style_tree()
        self._set_results(None)

    # ---------- small helpers ----------
    def _card(self, parent, title, expand=False):
        f = ttk.Frame(parent, style='Card.TFrame', padding=px(14))
        f.pack(fill='both' if expand else 'x', expand=expand, pady=(0, px(12)))
        ttk.Label(f, text=title.upper(), style='Section.TLabel').pack(anchor='w', pady=(0, px(10)))
        return f

    def _style_tree(self):
        self.tree.tag_configure('group', foreground=P['muted'], font=(FONT, 9, 'bold'))
        self.tree.tag_configure('prelim', foreground=P['muted'])
        self.tree.tag_configure('bad', foreground=P['err'])

    def toggle_theme(self):
        self.theme = 'light' if self.theme == 'dark' else 'dark'
        apply_style(self.root, self.theme)
        self._style_tree()
        self.theme_btn.config(text=("Light theme" if self.theme == 'dark' else "Dark theme"))
        self.viewer.render()

    def _set_status(self, text, kind=None, frac=None):
        self.status.config(text=text, style={'ok': 'StatusOk.TLabel', 'err': 'StatusErr.TLabel'}.get(kind, 'Status.TLabel'))
        if frac is not None:
            self.prog['value'] = frac; self.pct_lbl.config(text="%d%%" % round(frac * 100))

    def _effective_out_root(self):
        """the folder that will hold THIS run's outputs, given the current input + output choice."""
        out = self.out_var.get().strip()
        if self.batch_folder:
            return out or (self.batch_folder.rstrip('/\\') + "_neuron_out")
        if self.image_path:
            stem = os.path.splitext(os.path.basename(self.image_path))[0]
            base = out or os.path.dirname(self.image_path)
            return os.path.join(base, stem + "_neuron_out")
        return out or None

    def _refresh_flow(self):
        """keep the UI honest: enable Run once an input is chosen, and show where results will go."""
        has_input = bool(self.image_path or self.batch_folder)
        self.run_btn.config(state=('normal' if has_input else 'disabled'),
                            text=("▶  Run tracing" if has_input else "Choose an input first"))
        root = self._effective_out_root()
        if root:
            self.out_disp.set(root)
        else:
            self.out_disp.set("Auto: next to the input")

    def open_image(self):
        p = filedialog.askopenfilename(title="Select a fluorescence image",
            filetypes=[("Images", "*.png *.tif *.tiff *.jpg *.jpeg"), ("All files", "*.*")])
        if not p:
            return
        self.image_path = p; self.batch_folder = None
        self.file_lbl.config(text=os.path.basename(p), style='Card.TLabel')
        try:
            self.viewer.set_original(self._preview(p))
        except Exception:
            self.viewer.set_original(Image.open(p).convert('RGB'))
        self._set_results(None)
        self.open_out_btn.config(state='disabled'); self.annot_btn.config(state='disabled'); self.edit_btn.config(state='disabled')
        self._set_status("Image loaded. Click Run tracing.", frac=0)
        self._refresh_flow()

    def _preview(self, path):
        """Normalized grayscale preview (raw 16-bit fluorescence would look blank/white
        without this)."""
        g = pipe.load_gray(path).astype(float)
        lo, hi = np.percentile(g, 0.5), np.percentile(g, 99.5)
        a = np.clip((g - lo) / max(hi - lo, 1), 0, 1) * 255
        return Image.fromarray(a.astype('uint8')).convert('RGB')

    def open_folder(self):
        d = filedialog.askdirectory(title="Select a folder of images (batch)")
        if not d:
            return
        imgs = pipe.find_images(d)
        n = len(imgs)
        self.batch_folder = d; self.image_path = None
        if n == 0:
            self.file_lbl.config(text="Folder: %s  ·  no images found" % (os.path.basename(d) or d),
                                 style='CardMuted.TLabel')
            messagebox.showwarning("No images", "No images found in that folder (searched subfolders too).\n"
                                   "Supported: .png .tif .tiff .jpg. Files named like *traced*, *sholl*, "
                                   "*mask*, patching, profile are skipped as non-raw.")
            return
        self.file_lbl.config(text="Folder: %s  ·  %d images" % (os.path.basename(d) or d, n), style='Card.TLabel')
        self.viewer.show_message("Batch mode: %d images\nclick Run tracing" % n)
        self._set_results(None)
        self.open_out_btn.config(state='disabled'); self.annot_btn.config(state='disabled'); self.edit_btn.config(state='disabled')
        self._set_status("Folder loaded. Click Run tracing.", frac=0)
        self._refresh_flow()

    def choose_out(self):
        d = filedialog.askdirectory(title="Select an output folder for the results")
        if not d:
            return
        self.out_var.set(d)
        self._refresh_flow()

    def reset_out(self):
        self.out_var.set("")
        self._refresh_flow()

    def run(self):
        if not self.image_path and not self.batch_folder:
            messagebox.showwarning("No input", "Open an image or a folder first.")
            return
        try:
            vox = float(self.vox_var.get())
        except ValueError:
            messagebox.showerror("Bad value", "Pixel size must be a number (um/px).")
            return
        # lock the inputs so changing them mid-run can't mismatch what's actually being traced
        self.run_btn.config(state='disabled', text="Running…"); self._set_inputs_enabled(False)
        self._set_status("Starting...", frac=0)
        t = threading.Thread(target=self._worker,
                             args=(vox, self.hq_var.get(), self.out_var.get().strip(),
                                   self.image_path, self.batch_folder), daemon=True)
        t.start(); self.root.after(100, self._poll)

    def _set_inputs_enabled(self, on):
        st = 'normal' if on else 'disabled'
        for b in (self.open_img_btn, self.open_folder_btn):
            b.config(state=st)

    def _worker(self, vox, hq, out_choice, image_path, batch_folder):
        try:
            def prog(msg, frac):
                self.q.put(("progress", msg, frac))
            if batch_folder:
                out_root = out_choice or (batch_folder.rstrip('/\\') + "_neuron_out")
                out = pipe.run_batch(batch_folder, out_root, vox=vox, mirror=hq, progress=prog)
                self.q.put(("done_batch", out))
            else:
                out_dir = None
                if out_choice:
                    stem = os.path.splitext(os.path.basename(image_path))[0]
                    out_dir = os.path.join(out_choice, stem + "_neuron_out")
                res = pipe.run(image_path, vox=vox, mirror=hq, out_dir=out_dir, progress=prog)
                self.q.put(("done", res))
        except Exception as e:
            self.q.put(("error", "%s\n%s" % (e, traceback.format_exc())))

    def _poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                if item[0] == "progress":
                    self._set_status(item[1], frac=item[2])
                elif item[0] == "done":
                    self._finish_single(item[1]); return
                elif item[0] == "done_batch":
                    self._finish_batch(item[1]); return
                elif item[0] == "error":
                    self._set_inputs_enabled(True); self._refresh_flow(); self._set_status("Error.", 'err')
                    messagebox.showerror("Pipeline error", item[1]); return
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _finish_single(self, res):
        self.result = res; self.out_dir = res['out_dir']
        self.viewer.set_result(res['overlay'])
        m = res['all_metrics']

        def g(k, f):
            v = m.get(k, None)
            try:
                return ("%d" % v) if f is None else (f % v)
            except (TypeError, ValueError):
                return str(v)
        groups = []
        for title, rows in RESULT_ROWS:
            groups.append((title, [(lab, (g(k, f) + ("  " + u if u else ""))) for lab, k, f, u in rows],
                           'prelim' if 'preliminary' in title else None))
        self._set_results(groups)
        self._set_status("Done.  Saved to " + res['out_dir'], 'ok', 1.0)
        self._set_inputs_enabled(True); self._refresh_flow(); self.out_disp.set(res['out_dir'])
        self.open_out_btn.config(state='normal'); self.annot_btn.config(state='normal'); self.edit_btn.config(state='normal')

    def _finish_batch(self, out):
        self.out_dir = out['folder']
        ok = sum(1 for r in out['rows'] if not r.get('error'))
        failed = out['n'] - ok
        groups = [("Batch complete", [("Images processed", str(out['n'])), ("Succeeded", str(ok)),
                                      ("Failed", str(failed))], None),
                  ("Files", [("Combined table", "summary.csv"), ("Per-cell folders", "<name>_neuron_out")], None)]
        if failed:
            groups.append(("Failed cells", [(r['cell'], r.get('error', '')) for r in out['rows']
                                            if r.get('error')][:8], 'bad'))
        self._set_results(groups)
        self._set_status("Batch done.  Saved to " + out['folder'], 'ok' if not failed else 'err', 1.0)
        self._set_inputs_enabled(True); self._refresh_flow(); self.out_disp.set(out['folder'])
        self.open_out_btn.config(state='normal'); self.annot_btn.config(state='normal'); self.edit_btn.config(state='normal')

    def _set_results(self, groups):
        """groups: [(title, [(label, value)], row_tag)] or None for the empty state."""
        self.tree.delete(*self.tree.get_children())
        if not groups:
            self.tree.pack_forget(); self.tree_sb.pack_forget()
            self.res_empty.pack(anchor='w'); return
        self.res_empty.pack_forget()
        self.tree_sb.pack(side='right', fill='y'); self.tree.pack(side='left', fill='both', expand=True)
        for title, rows, tag in groups:
            gid = self.tree.insert('', 'end', text=title.upper(), open=True, tags=('group',))
            for lab, val in rows:
                self.tree.insert(gid, 'end', text=lab, values=(val,), tags=((tag,) if tag else ()))

    def open_out(self):
        if self.out_dir and os.path.isdir(self.out_dir):
            th.open_folder(self.out_dir)

    def open_editor(self):
        """Open the edit-tracing window on this run's output (one cell or a whole batch)."""
        try:
            import editor
        except Exception as e:
            messagebox.showerror("Edit tracing", "Could not load the editor:\n%s" % e); return
        if not (self.out_dir and os.path.isdir(self.out_dir)):
            messagebox.showinfo("Edit tracing", "Run tracing first, then edit the result."); return
        try:
            editor.open_editor(self.root, folder=self.out_dir)
        except Exception as e:
            messagebox.showerror("Edit tracing", "Could not open the editor:\n%s\n\n%s" % (e, traceback.format_exc()))

    def open_annotator(self):
        """Open the error-review window on the run's output. After a single OR batch run, self.out_dir
        points at the output (a *_neuron_out folder, or the batch root of them) — the reviewer picks
        cells from there. If nothing has been traced yet, prompt to run tracing first."""
        try:
            import annotator
        except Exception as e:
            messagebox.showerror("Annotator", "Could not load the annotator:\n%s" % e); return
        try:
            if self.out_dir and os.path.isdir(self.out_dir):
                annotator.open_annotator(self.root, folder=self.out_dir)
            else:
                annotator.open_annotator(self.root)  # empty: shows the "run tracing first" guidance
                messagebox.showinfo("Review & annotate",
                                    "Run tracing first: open an image or a folder, click Run, then "
                                    "Review & annotate.\n\nOr use “Open folder…” in the review window to "
                                    "pick a previous run's output folder (the one holding *_neuron_out).")
        except Exception as e:
            messagebox.showerror("Annotator", "Could not open the review window:\n%s\n\n%s"
                                 % (e, traceback.format_exc()))


def main():
    if '--check-annotator' in sys.argv:          # headless smoke test that the annotator bundles & runs
        import tempfile, annotator as _an
        import numpy as _np
        from PIL import Image as _Im
        d = tempfile.mkdtemp(); ip = os.path.join(d, "t.png")
        _Im.fromarray((_np.random.rand(80, 100) * 255).astype('uint8')).save(ip)
        r = tk.Tk(); r.withdraw()
        w = _an.Annotator(r, image_path=ip); w.withdraw()
        w._add_anno(kind='point', point=[10, 10]); w._add_anno(kind='box', box=[5, 5, 40, 40])
        jp, pp = w.write_annotations(os.path.join(d, "ann"))
        msg = "OK annotator: wrote %s %s" % (os.path.basename(jp), os.path.basename(pp))
        print(msg)
        try:                                       # marker file so a windowed (no-stdout) build can be verified
            with open(os.path.join(os.path.dirname(sys.executable), "annotator_selfcheck_ok.txt"), "w") as _f:
                _f.write(msg)
        except Exception:
            pass
        w.destroy(); r.destroy(); return
    if '--check-batch' in sys.argv:               # headless: exercise the batch-annotator on a real output folder
        args = [a for a in sys.argv[1:] if not a.startswith('-')]
        import annotator as _an
        r = tk.Tk(); r.withdraw()
        w = _an.Annotator(r, folder=args[0]); w.withdraw()
        n = len(w.cells)
        loaded = sum(1 for i in range(n) if (w._select_cell(i) or w.overlay is not None))
        w.active_cat.set("false"); w._add_anno(kind='box', box=[5, 5, 40, 40])
        jp, pp = w.write_annotations()
        msg = "OK batch-annotator: %d cells, overlays loaded, saved -> %s" % (n, os.path.dirname(jp))
        print(msg)
        try:
            with open(os.path.join(os.path.dirname(sys.executable), "batch_selfcheck_ok.txt"), "w") as _f:
                _f.write(msg)
        except Exception:
            pass
        w.destroy(); r.destroy(); return
    if '--check-editor' in sys.argv:              # headless: the editor bundles, edits, recomputes metrics, saves
        import tempfile, shutil, test_editor as _te, editor as _ed
        d = tempfile.mkdtemp(); _te.write_cell(d)
        r = tk.Tk(); r.withdraw(); apply_style(r, 'dark')
        w = _ed.Editor(r, folder=d); w.withdraw()
        seg = next(sg for sg in w.arbor.segments() if len(w.arbor.adj[sg[-1]]) == 1)
        w.sel = set(seg); w.sel_label = 'segment'; w._sel_chain = w.arbor.oriented(seg); w.delete_selection()
        out = w.save()
        msg = "OK editor: %d edit(s), wrote %s" % (len(w.edits), " ".join(os.path.basename(p) for p in out))
        print(msg)
        try:
            with open(os.path.join(os.path.dirname(sys.executable), "editor_selfcheck_ok.txt"), "w") as _f:
                _f.write(msg)
        except Exception:
            pass
        w.destroy(); r.destroy(); shutil.rmtree(d, ignore_errors=True); return
    if '--edit' in sys.argv:                      # open the edit window directly on an output folder
        args = [a for a in sys.argv[1:] if not a.startswith('-')]
        import editor
        th.hidpi()
        root = tk.Tk(); root.withdraw(); th.SCALE = max(1.0, root.winfo_fpixels('1i') / 96.0)
        apply_style(root, 'dark')
        win = editor.open_editor(root, folder=(args[0] if args else None))

        def _close():
            if win._ask_save():
                win.destroy(); root.destroy()
        win.protocol("WM_DELETE_WINDOW", _close)
        root.mainloop(); return
    if '--annotate' in sys.argv:                  # open the review/annotation window directly on an image
        args = [a for a in sys.argv[1:] if not a.startswith('-')]
        import annotator
        root = tk.Tk(); root.withdraw()
        win = annotator.open_annotator(root, image_path=(args[0] if args else None))
        win.protocol("WM_DELETE_WINDOW", root.destroy); root.mainloop(); return
    if '--cli' in sys.argv:
        def _say(s):
            try:
                if sys.stdout:
                    print(s, flush=True)
            except Exception:
                pass
        # positional args: <input> [output_folder]   (output folder is optional)
        args = [a for a in sys.argv[1:] if not a.startswith('-')]
        out_choice = args[1] if len(args) > 1 else ""
        if '--batch' in sys.argv:
            out_root = out_choice or (args[0].rstrip('/\\') + "_neuron_out")
            r = pipe.run_batch(args[0], out_root, mirror=('--hq' in sys.argv),
                               progress=lambda m, f: _say("[%3d%%] %s" % (int(f * 100), m)))
            _say("OK batch %d cells -> %s" % (r['n'], r['summary']))
        else:
            out_dir = None
            if out_choice:
                stem = os.path.splitext(os.path.basename(args[0]))[0]
                out_dir = os.path.join(out_choice, stem + "_neuron_out")
            r = pipe.run(args[0], out_dir=out_dir, mirror=('--hq' in sys.argv),
                         progress=lambda m, f: _say("[%3d%%] %s" % (int(f * 100), m)))
            _say("OK length=%.1f branches=%s soma_area=%s -> %s" % (
                r['total_length_um'], r.get('num_branches'), r['soma_metrics'].get('soma_area_um2'), r['out_dir']))
        return
    th.hidpi()
    root = tk.Tk()
    th.SCALE = max(1.0, root.winfo_fpixels('1i') / 96.0)
    App(root); root.mainloop()


if __name__ == '__main__':
    main()
