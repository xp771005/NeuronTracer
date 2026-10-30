# -*- coding: utf-8 -*-
"""Shared look & feel for NeuronTracer (main window + review window).
Hand-tuned ttk 'clam' theme: no extra packages to bundle. Two palettes, dark (default) and light.
UI only: nothing here touches tracing."""
import sys
import tkinter as tk
from tkinter import ttk

FONT = 'Segoe UI' if sys.platform == 'win32' else ('Helvetica Neue' if sys.platform == 'darwin' else 'DejaVu Sans')
MONO = 'Consolas' if sys.platform == 'win32' else ('Menlo' if sys.platform == 'darwin' else 'DejaVu Sans Mono')
CURRENT = 'dark'

# ---------- look & feel: hand-tuned 'clam' theme, no extra packages ----------
PALETTES = {
    'dark': dict(bg='#1b1d21', card='#25282d', raised='#33373d', hover='#3d424a', border='#383c43',
                 fg='#e7e9ec', muted='#9aa0a8', dim='#62686f', accent='#3b82f6', accent_hi='#5b97f7',
                 accent_lo='#2563eb', field='#1b1d21', sel='#2d4a73', ok='#4ade80', err='#f87171'),
    'light': dict(bg='#eef0f3', card='#ffffff', raised='#dfe3e8', hover='#d2d7dd', border='#d5d9de',
                  fg='#1f2328', muted='#5f6670', dim='#a3a9b1', accent='#2563eb', accent_hi='#3b82f6',
                  accent_lo='#1d4ed8', field='#ffffff', sel='#dbe7fd', ok='#15803d', err='#b91c1c'),
}
IMG_BG = '#0e0f11'          # image area stays dark in both themes (microscopy)
P = dict(PALETTES['dark'])  # current palette; updated in place by apply_style
SCALE = 1.0                 # screen DPI / 96, set once the Tk root exists

# overlay colours, copied from neuron_pipeline.overlay_rgb so the legend matches the image
LEGEND = (('line', '#ff2828', "Neurite / soma outline"), ('ring', '#ffeb00', "Branch point"),
          ('dot', '#32ff5a', "Tip"), ('x', '#2878ff', "Crossing"), ('dot', '#00e6ff', "Soma centre"))


def px(n):
    return int(round(n * SCALE))


def apply_style(root, name):
    global CURRENT
    CURRENT = name
    P.clear(); P.update(PALETTES[name])
    st = ttk.Style(root)
    st.theme_use('clam')
    root.configure(bg=P['bg'])
    flat = lambda c: dict(background=c, lightcolor=c, darkcolor=c, bordercolor=c)
    st.configure('.', background=P['bg'], foreground=P['fg'], font=(FONT, 10), bordercolor=P['border'],
                 lightcolor=P['bg'], darkcolor=P['bg'], troughcolor=P['field'], focuscolor=P['bg'],
                 selectbackground=P['sel'], selectforeground=P['fg'], fieldbackground=P['field'],
                 insertcolor=P['fg'])
    st.configure('TFrame', background=P['bg'])
    st.configure('Card.TFrame', background=P['card'], bordercolor=P['border'], relief='solid', borderwidth=1)
    st.configure('Status.TFrame', background=P['card'])
    st.configure('TLabel', background=P['bg'], foreground=P['fg'])
    st.configure('Title.TLabel', font=(FONT, 16, 'bold'))
    st.configure('Sub.TLabel', foreground=P['muted'], font=(FONT, 9))
    st.configure('Hint.TLabel', foreground=P['dim'], font=(FONT, 8))
    st.configure('Card.TLabel', background=P['card'], foreground=P['fg'])
    st.configure('CardMuted.TLabel', background=P['card'], foreground=P['muted'], font=(FONT, 9))
    st.configure('Section.TLabel', background=P['card'], foreground=P['muted'], font=(FONT, 9, 'bold'))
    st.configure('Status.TLabel', background=P['card'], foreground=P['muted'], font=(FONT, 9))
    st.configure('StatusOk.TLabel', background=P['card'], foreground=P['ok'], font=(FONT, 9, 'bold'))
    st.configure('StatusErr.TLabel', background=P['card'], foreground=P['err'], font=(FONT, 9, 'bold'))

    st.configure('TButton', foreground=P['fg'], relief='flat', padding=(px(10), px(6)), anchor='center',
                 **flat(P['raised']))
    st.map('TButton',
           background=[('disabled', P['raised']), ('pressed', P['border']), ('active', P['hover'])],
           lightcolor=[('disabled', P['raised']), ('pressed', P['border']), ('active', P['hover'])],
           darkcolor=[('disabled', P['raised']), ('pressed', P['border']), ('active', P['hover'])],
           bordercolor=[('disabled', P['raised']), ('pressed', P['border']), ('active', P['hover'])],
           foreground=[('disabled', P['dim'])])
    st.configure('Small.TButton', padding=(px(8), px(4)), font=(FONT, 9))
    st.configure('Accent.TButton', foreground='#ffffff', font=(FONT, 11, 'bold'), padding=(px(12), px(9)),
                 **flat(P['accent']))
    acc = [('disabled', P['raised']), ('pressed', P['accent_lo']), ('active', P['accent_hi'])]
    st.map('Accent.TButton', background=acc, lightcolor=acc, darkcolor=acc, bordercolor=acc,
           foreground=[('disabled', P['dim'])])

    # segmented Original / Result / Side by side switch
    st.configure('Seg.Toolbutton', foreground=P['muted'], font=(FONT, 9, 'bold'), padding=(px(12), px(5)),
                 relief='flat', borderwidth=0, **flat(P['raised']))
    seg = [('disabled', P['bg']), ('selected', P['accent']), ('active', P['hover'])]
    st.map('Seg.Toolbutton', background=seg, lightcolor=seg, darkcolor=seg, bordercolor=seg,
           foreground=[('disabled', P['dim']), ('selected', '#ffffff'), ('active', P['fg'])])

    st.configure('Card.TCheckbutton', background=P['card'], foreground=P['fg'], focuscolor=P['card'],
                 indicatorbackground=P['field'], indicatorforeground='#ffffff',
                 upperbordercolor=P['border'], lowerbordercolor=P['border'], indicatormargin=(0, 0, px(8), 0))
    st.map('Card.TCheckbutton', background=[('active', P['card'])],
           indicatorbackground=[('selected', P['accent']), ('active', P['hover'])],
           upperbordercolor=[('selected', P['accent'])], lowerbordercolor=[('selected', P['accent'])])

    st.configure('TEntry', foreground=P['fg'], fieldbackground=P['field'], bordercolor=P['border'],
                 lightcolor=P['field'], darkcolor=P['field'], insertcolor=P['fg'], padding=(px(6), px(4)))
    st.map('TEntry', bordercolor=[('focus', P['accent'])], lightcolor=[('focus', P['accent'])],
           fieldbackground=[('readonly', P['field'])], foreground=[('readonly', P['muted'])])

    st.configure('Horizontal.TProgressbar', troughcolor=P['field'], background=P['accent'],
                 bordercolor=P['card'], lightcolor=P['accent'], darkcolor=P['accent'], thickness=px(6))

    st.configure('Results.Treeview', background=P['card'], fieldbackground=P['card'], foreground=P['fg'],
                 bordercolor=P['card'], lightcolor=P['card'], darkcolor=P['card'], borderwidth=0,
                 rowheight=px(24), font=(FONT, 10))
    st.map('Results.Treeview', background=[('selected', P['sel'])], foreground=[('selected', P['fg'])])
    st.layout('Results.Treeview', [('Treeview.treearea', {'sticky': 'nswe'})])
    st.configure('Results.Treeview.Item', indicatorsize=px(9))

    st.configure('Vertical.TScrollbar', troughcolor=P['card'], arrowcolor=P['muted'], gripcount=0,
                 arrowsize=px(12), **flat(P['raised']))
    st.map('Vertical.TScrollbar', background=[('active', P['hover'])])
    st.configure('TSeparator', background=P['border'])

    # extras used by the review window
    st.configure('Head.TLabel', font=(FONT, 10, 'bold'))
    st.configure('CardHead.TLabel', background=P['card'], foreground=P['fg'], font=(FONT, 10, 'bold'))
    for sty, bg in (('TCheckbutton', P['bg']), ('TRadiobutton', P['bg'])):
        st.configure(sty, background=bg, foreground=P['fg'], focuscolor=bg, indicatorbackground=P['field'],
                     indicatorforeground='#ffffff', upperbordercolor=P['border'], lowerbordercolor=P['border'])
        st.map(sty, background=[('active', bg)], indicatorbackground=[('selected', P['accent']), ('active', P['hover'])])
    st.configure('Horizontal.TScale', troughcolor=P['raised'], **flat(P['raised']))
    _check_indicator(st, name)
    st.configure('Cat.Toolbutton', foreground=P['fg'], anchor='w', padding=(px(10), px(6)), relief='flat',
                 borderwidth=0, font=(FONT, 9), **flat(P['card']))
    cat = [('selected', P['sel']), ('active', P['hover'])]
    st.map('Cat.Toolbutton', background=cat, lightcolor=cat, darkcolor=cat, bordercolor=cat,
           foreground=[('selected', P['fg'])], font=[('selected', (FONT, 9, 'bold'))])
    for w in _painted:
        try:
            w()
        except Exception:
            pass
    titlebar(root, name == 'dark')


_check_imgs = {}   # keep PhotoImages alive (Tk drops images that Python garbage-collects)


def _check_indicator(st, name):
    """clam draws a cross in its checkbox, which reads as 'close'. Draw a real check mark instead."""
    from PIL import Image, ImageDraw, ImageTk
    key = (name, id(st.master.tk))                  # images and elements belong to one Tk interpreter
    if key in _check_imgs:
        pass
    else:
        n = px(16); gap = px(7); k = 4; N = n * k       # draw 4x, then shrink = smooth edges
        def box(fill, outline):                          # transparent strip on the right = gap before the label
            im = Image.new('RGBA', (N + gap * k, N), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
            d.rounded_rectangle([k, k, N - k - 1, N - k - 1], radius=3 * k, fill=fill, outline=outline, width=max(k, N // 14))
            return im, d
        off_im, _ = box(P['field'], P['muted'])
        on_im, d = box(P['accent'], P['accent'])
        d.line([(N * 0.26, N * 0.52), (N * 0.43, N * 0.68), (N * 0.75, N * 0.33)], fill='white',
               width=max(2 * k, N // 8), joint='curve')
        off = ImageTk.PhotoImage(off_im.resize((n + gap, n), Image.LANCZOS), master=st.master)
        on = ImageTk.PhotoImage(on_im.resize((n + gap, n), Image.LANCZOS), master=st.master)
        # slider thumb: clam paints the trough edge in the thumb colour, so draw the thumb as an image
        w, h = px(10), px(16)
        def thumb(col):
            im = Image.new('RGBA', (w * k, h * k), (0, 0, 0, 0))
            ImageDraw.Draw(im).rounded_rectangle([0, 0, w * k - 1, h * k - 1], radius=3 * k, fill=col)
            return ImageTk.PhotoImage(im.resize((w, h), Image.LANCZOS), master=st.master)
        t_on, t_hi = thumb(P['accent']), thumb(P['accent_hi'])
        _check_imgs[key] = (on, off, t_on, t_hi)
        st.element_create('Check%s.indicator' % name, 'image', off, ('selected', on), ('alternate', off),
                          border=0, sticky='')
        st.element_create('Thumb%s.slider' % name, 'image', t_on, ('active', t_hi), ('pressed', t_hi), sticky='')
    lay = [('Checkbutton.padding', {'sticky': 'nswe', 'children': [
        ('Check%s.indicator' % name, {'side': 'left', 'sticky': ''}),
        ('Checkbutton.focus', {'side': 'left', 'sticky': 'w', 'children': [('Checkbutton.label', {'sticky': 'nswe'})]})]})]
    for sty in ('TCheckbutton', 'Card.TCheckbutton'):
        st.layout(sty, lay)
        st.configure(sty, indicatormargin=0, padding=(0, 0, px(4), 0))
    st.configure('TCheckbutton', padding=(0, 0, px(4), 0))
    st.layout('Horizontal.TScale', [('Horizontal.Scale.padding', {'sticky': 'nswe', 'children': [
        ('Horizontal.Scale.trough', {'sticky': 'nswe', 'children': [
            ('Thumb%s.slider' % name, {'side': 'left', 'sticky': ''})]})]})])


_painted = []      # callbacks that recolour plain tk widgets (Listbox, Text, Canvas) on a theme switch


def on_theme_change(fn):
    _painted.append(fn); fn()


def paint_list(w, mono=True):
    """Listbox / Text in the current palette (they are plain tk, so ttk styles don't reach them)."""
    w.configure(bg=P['field'], fg=P['fg'], relief='flat', borderwidth=0, highlightthickness=1,
                highlightbackground=P['border'], highlightcolor=P['accent'])
    try:
        w.configure(selectbackground=P['sel'], selectforeground=P['fg'])
    except tk.TclError:
        pass
    try:
        w.configure(insertbackground=P['fg'])
    except tk.TclError:
        pass


def titlebar(win, dark=None):
    """Windows 10/11: dark title bar to match (no-op elsewhere)."""
    if dark is None:
        dark = CURRENT == 'dark'
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        win.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(win.winfo_id())
        v = ctypes.c_int(1 if dark else 0)
        for attr in (20, 19):
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(v), ctypes.sizeof(v)) == 0:
                break
    except Exception:
        pass


def open_folder(path):
    """show a folder in the file manager (Explorer / Finder / xdg-open)."""
    import os, subprocess
    if sys.platform == 'win32':
        os.startfile(path)
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', path])
    else:
        subprocess.Popen(['xdg-open', path])


def hidpi():
    """sharp text on scaled Windows displays (125%, 150%) instead of a blurry bitmap stretch."""
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


class Tooltip:
    """small hover note; replaces the grey help text that used to crowd the panel."""
    def __init__(self, widget, text):
        self.w, self.text, self.tip, self._job = widget, text, None, None
        widget.bind('<Enter>', self._schedule, add='+')
        widget.bind('<Leave>', self._hide, add='+')
        widget.bind('<ButtonPress>', self._hide, add='+')

    def _schedule(self, e=None):
        self._job = self.w.after(450, self._show)

    def _show(self):
        text = self.text() if callable(self.text) else self.text
        if not text:
            return
        self.tip = tk.Toplevel(self.w); self.tip.wm_overrideredirect(True)
        tk.Label(self.tip, text=text, justify='left', wraplength=px(320), bg=P['raised'], fg=P['fg'],
                 font=(FONT, 9), padx=px(8), pady=px(5), highlightthickness=1,
                 highlightbackground=P['border']).pack()
        x = self.w.winfo_rootx() + px(8); y = self.w.winfo_rooty() + self.w.winfo_height() + px(4)
        self.tip.wm_geometry("+%d+%d" % (x, y))

    def _hide(self, e=None):
        if self._job:
            self.w.after_cancel(self._job); self._job = None
        if self.tip:
            self.tip.destroy(); self.tip = None


