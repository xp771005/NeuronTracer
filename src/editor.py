# -*- coding: utf-8 -*-
"""Edit-tracing window for NeuronTracer: the software draws a first version, a person corrects it.

Designed from what Liz actually changed on 20 corrected cells (yuqi/edit_research/RESULTS.md):
  delete a whole side branch, trim an overshooting tip, delete a wrong connection,
  extend a tip, add a missed branch, bridge a gap.
Tools (SNT-like, see Cecilia's 2026-10-15 notes), plus an eraser (user's request 2026-10-18):
  Erase (E)   hold the left button and drag: lines under the circle turn pink and go on release
  Select (S)  click a line = one segment (branch point to branch point/tip); Shift+click = that branch
              and everything beyond it; Delete removes the selection
  Cut (C)     click a point on a line: everything beyond that point (away from the soma) is removed
  Draw (D)    click to add points; each new stretch follows the bright neurite in the raw image
              ("Follow image", F) or is straight; start on a line or the soma to attach, click another
              piece to join it, Enter / double-click finishes, Backspace removes the last point, Esc cancels.
              Like SNT: clicks snap to the brightest pixel within 8 px, and a cyan candidate line previews
              where the next click would draw.
  Display Min / Max: the brightness range, used for the picture AND for following the image (as in SNT,
              where the B&C range sets the Min-Max of the cost function). Cecilia adjusts it all the time.
  Paths panel: the tree as SNT-style parent / daughter paths. Edits panel: every change, with its error type.
View: three panes side by side (user's request 2026-10-17): Original image | Software tracing (read-only) |
Edited (the person's version). One shared view: zoom or pan in any pane and the others follow. Only the right
pane takes edits; left-drag pans the other two. "Edit only" switches to one large pane for fine work.
Hold Space to hide the tracing and look at the image. Ctrl+Z / Ctrl+Y undo / redo, Ctrl+S save.

Saving never overwrites the software's files: it writes <stem>_edited.swc, _edited_metrics.csv,
_edited_sholl.csv, _edited_overlay.png and <stem>_edits.json next to them. Re-opening resumes the saved edits.
Metrics are recomputed with tracer.tree_metrics (read-only use), so their definitions match the draft's.
UI only: nothing here changes how the software traces.
"""
import os, re, json, math, datetime, glob
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
from scipy.spatial import cKDTree
import theme
from theme import P, FONT, IMG_BG, px, paint_list, titlebar, on_theme_change, Tooltip
Image.MAX_IMAGE_PIXELS = None

# edit kinds -> label shown to the user, and the matching category of the error-review window
KINDS = {
    'extend':   ("Extend tip",       'missing'),
    'branch':   ("Missing branch",   'missing'),
    'primary':  ("Missing primary",  'missing'),
    'bridge':   ("Gap joined",       'gap'),
    'fragment': ("New piece",        'other'),
    'spur':     ("False branch",     'false'),
    'internal': ("Wrong connection", 'crossbranch'),
    'piece':    ("False piece",      'false'),
    'trim':     ("Tip trimmed",      'false'),
    'erase':    ("Erased",           'false'),
    'move':     ("Point moved",      'crossbranch'),
    'crossing': ("Crossing, not a branch", 'crossbranch'),
    'subtree':  ("Branch removed",   'false'),
}
COL_TRACE = '#ff2828'        # same red as the software's overlay
COL_NEW = '#3dff7a'          # what the person drew
COL_FLOAT = '#ffa63d'        # pieces not connected to the soma
COL_SEL = '#00e5ff'
COL_PEND = '#f5f5f5'          # the line being drawn (white: yellow now marks branch points, as in the overlay)
COL_JUNC = '#ffeb00'
COL_CROSS = '#2878ff'        # crossings: blue X, as in the overlay         # branch points: yellow rings, like the software's overlay
COL_CAND = '#00e5ff'         # candidate line before the click (SNT shows it cyan too)
SNAP_PX = 8                  # SNT default: snap within 8 px
COL_CUT = '#ff4fd8'
COL_FAINT = '#c77dff'        # the image is faint under this part of the line (low probability): check it is real (2026-10-27)
FAINT_PROB = 0.5             # smoothed probability below the tracer's own HIGH threshold = the line was bridged, not seen (measured on cell 318: lines sit at 0.99 median)
FAINT_MIN_UM = 1.5           # shorter runs are not listed (on 318: 24 runs at 0.5, 5 of them >= 1.5 um)
OTHER_SOMA_MIN_UM = 20.0     # a bright body closer than this to the soma is the soma (or its glare), not another cell
STEP_PX = 3                  # spacing of nodes on hand-drawn lines


# =====================================================================================================
# data layer (no Tk): an undirected tree that can be edited and written back as SWC
# =====================================================================================================
class Arbor:
    def __init__(self):
        self.xy, self.r, self.typ, self.adj = {}, {}, {}, {}
        self.new = set()
        self.soma = None
        self.roots0 = []
        self.vox = 0.1625
        self.header = []
        self.next_id = 1
        self.soma_r = 0.0           # px; equal-area soma radius, used to clip soma spokes for length
        self.soma_poly = None       # Nx2 (x, y) px: the segmented soma outline, when known
        self.join_cands = []        # [((rx, ry), (tx, ty))] px: pieces the tracer found within 5 um of a line but did not join (2026-10-25)
        self._orient = None

    # ---------- io ----------
    @classmethod
    def from_swc(cls, path, soma_area_um2=None, soma_xy=None):
        a = cls(); par = {}
        with open(path, encoding='utf-8', errors='ignore') as fh:
            text = fh.read()
        for ln in text.splitlines():
            s = ln.strip()
            if not s:
                continue
            if s.startswith('#'):
                a.header.append(s)
                m = re.search(r'voxel\s+([\d.]+)', s)
                if m:
                    a.vox = float(m.group(1))
                if s.startswith('# join candidates'):
                    for q in re.findall(r'([\d.]+)\s+([\d.]+)\s*->\s*([\d.]+)\s+([\d.]+)', s):
                        rx, ry, tx, ty = (float(v) for v in q)
                        a.join_cands.append(((rx, ry), (tx, ty)))        # um, converted to px after vox is known
                continue
            p = s.split()
            if len(p) < 7:
                continue
            i = int(float(p[0]))
            a.typ[i] = int(float(p[1])); a.r[i] = float(p[5]); par[i] = int(float(p[6]))
            a.xy[i] = (float(p[2]) / a.vox, float(p[3]) / a.vox)
            a.adj[i] = set()
        for i, q in par.items():
            if q in a.adj:
                a.adj[i].add(q); a.adj[q].add(i)
        a.join_cands = [((rx / a.vox, ry / a.vox), (tx / a.vox, ty / a.vox)) for (rx, ry), (tx, ty) in a.join_cands]
        a.roots0 = [i for i, q in par.items() if q not in a.adj]
        a.soma = a._find_soma(soma_xy)
        a.next_id = (max(a.xy) + 1) if a.xy else 1
        if soma_area_um2:
            a.soma_r = math.sqrt(float(soma_area_um2) / math.pi) / a.vox
        return a

    def _find_soma(self, soma_xy=None):
        """soma_xy (from the overlay's cyan soma-centre dot) decides when given. Otherwise: every tree root is written with type 1, and with one spoke per piece the soma need not have the most
        children. The soma hub is the root whose spoke is longest: spokes run from the soma centre to its edge,
        ordinary nodes are 1 to 1.4 px apart."""
        if not self.roots0:
            return None
        if soma_xy is not None:
            r = min(self.roots0, key=lambda k: math.hypot(self.xy[k][0] - soma_xy[0], self.xy[k][1] - soma_xy[1]))
            if math.hypot(self.xy[r][0] - soma_xy[0], self.xy[r][1] - soma_xy[1]) <= 6:
                return r
        def reach(r):
            x, y = self.xy[r]
            return max((math.hypot(self.xy[k][0] - x, self.xy[k][1] - y) for k in self.adj[r]), default=0.0)
        best = max(self.roots0, key=lambda r: (reach(r), len(self.adj[r])))
        return best if reach(best) > 3.0 else max(self.roots0, key=lambda r: len(self.adj[r]))

    def inside_soma(self, x, y):
        if self.soma_poly is not None and len(self.soma_poly) >= 3:
            from skimage.measure import points_in_poly
            return bool(points_in_poly(np.array([[x, y]], float), self.soma_poly)[0])
        if self.soma in self.xy and self.soma_r:
            sx, sy = self.xy[self.soma]
            return math.hypot(x - sx, y - sy) <= self.soma_r
        return False

    def soma_edge_towards(self, x, y):
        """where the straight line from the soma centre to (x, y) leaves the soma; None if (x, y) is inside."""
        sx, sy = self.xy[self.soma]
        d = math.hypot(x - sx, y - sy)
        if self.soma_poly is not None and len(self.soma_poly) >= 3:
            from skimage.measure import points_in_poly
            n = max(2, int(d))
            ts = np.linspace(0, 1, n + 1)
            pts = np.c_[sx + (x - sx) * ts, sy + (y - sy) * ts]
            inside = points_in_poly(pts, self.soma_poly)
            if inside[-1]:
                return None
            k = int(np.argmin(inside)) if not inside[0] else int(np.argmax(~inside))
            k = max(k, 1)
            return (float(pts[k][0]), float(pts[k][1]))
        if d <= self.soma_r:
            return None
        return (sx + (x - sx) * self.soma_r / d, sy + (y - sy) * self.soma_r / d)

    def to_swc(self, note=None, with_ids=False):
        parent, comps = self.orient()
        order = [n for c in comps for n in c]
        nid = {n: k + 1 for k, n in enumerate(order)}
        rad = [v for v in self.r.values() if v > 0]
        med = float(np.median(rad)) if rad else 0.5
        lines = [h for h in self.header if not h.startswith(("# near the soma, connection pending", "# join candidates"))]
        if note:
            lines.append("# " + note)
        for n in order:
            x, y = self.xy[n]; pa = parent[n]
            t = 1 if n == self.soma else (self.typ.get(n, 3) if pa == -1 else (3 if n in self.new else self.typ.get(n, 3)))
            lines.append("%d %d %.3f %.3f 0.0 %.3f %d" % (nid[n], t, x * self.vox, y * self.vox,
                                                            self.r.get(n, med) or med, nid[pa] if pa != -1 else -1))
        text = "\n".join(lines) + "\n"
        return (text, nid) if with_ids else text

    # ---------- structure ----------
    def touch(self):
        self._orient = None

    def orient(self):
        """parent map + components (soma piece first, then the others by size); parents precede children."""
        if self._orient is not None:
            return self._orient
        parent, comps, seen = {}, [], set()

        def walk(s):
            comp = [s]; parent[s] = -1; seen.add(s); st = [s]
            while st:
                u = st.pop()
                for v in sorted(self.adj[u]):
                    if v not in seen:
                        seen.add(v); parent[v] = u; st.append(v); comp.append(v)
            return comp

        starts = ([self.soma] if self.soma in self.adj else []) + \
                 [r for r in self.roots0 if r in self.adj and r != self.soma]
        for s in starts:
            if s not in seen:
                comps.append(walk(s))
        sx, sy = self.xy[self.soma] if self.soma in self.xy else (0.0, 0.0)
        for n in sorted(self.adj):
            if n in seen:
                continue
            c = walk(n)                                   # find the piece, then restart it from the end nearest
            ends = [m for m in c if len(self.adj[m]) <= 1]   # the soma, so that it runs outward (user rule)
            for m in c:
                seen.discard(m); parent.pop(m, None)
            start = min(ends, key=lambda m: (self.xy[m][0] - sx) ** 2 + (self.xy[m][1] - sy) ** 2) if ends else n
            comps.append(walk(start))
        main = comps[:1] if (comps and comps[0][0] == self.soma) else []
        rest = sorted(comps[len(main):], key=len, reverse=True)
        self._orient = (parent, main + rest)
        return self._orient

    def component_of(self):
        parent, comps = self.orient()
        if getattr(self, '_comp_for', None) is not self._orient:   # cached until the next edit
            self._comp = {n: k for k, c in enumerate(comps) for n in c}; self._comp_for = self._orient
        return self._comp

    def connected(self):
        """nodes attached to the soma."""
        parent, comps = self.orient()
        return set(comps[0]) if comps and comps[0][0] == self.soma else set()

    def children(self):
        parent, _ = self.orient()
        ch = {n: [] for n in parent}
        for n, p in parent.items():
            if p != -1:
                ch[p].append(n)
        return ch

    def descendants(self, n):
        ch = self.children(); out = []; st = list(ch.get(n, []))
        while st:
            u = st.pop(); out.append(u); st.extend(ch[u])
        return out

    def segments(self):
        """chains between nodes of degree != 2 (and the soma)."""
        stop = {n for n, s in self.adj.items() if len(s) != 2}
        if self.soma in self.adj:
            stop.add(self.soma)
        segs, used = [], set()
        for s in sorted(stop):
            for v in sorted(self.adj[s]):
                if (s, v) in used:
                    continue
                chain = [s, v]; used.add((s, v)); used.add((v, s))
                while chain[-1] not in stop:
                    u = chain[-1]
                    nxt = next(w for w in self.adj[u] if w != chain[-2])
                    used.add((u, nxt)); used.add((nxt, u)); chain.append(nxt)
                segs.append(tuple(chain))
        return segs

    def oriented(self, chain):
        """the chain ordered proximal -> distal (towards the soma first)."""
        parent, _ = self.orient()
        a, b = chain[0], chain[-1]
        return tuple(chain) if parent.get(b) == chain[-2] else tuple(reversed(chain))

    def paths(self):
        """SNT-style paths: each soma spoke starts a primary path that keeps going along the daughter with
        the most cable beyond it; other daughters start child paths at that branch point."""
        parent, comps = self.orient()
        ch = self.children()
        cable = {}
        for c in comps:
            for n in reversed(c):
                cable[n] = sum(cable[k] + self.edge_px(n, k) for k in ch[n])
        out = []
        comp0 = set(comps[0]) if comps and comps[0][0] == self.soma else set()
        for c in comps:
            root = c[0]
            todo = [(root, k, None) for k in sorted(ch[root], key=lambda k: -cable[k])]
            while todo:
                start, first, parent_path = todo.pop(0)
                nodes = [start, first]
                while ch[nodes[-1]]:
                    u = nodes[-1]
                    kids = sorted(ch[u], key=lambda k: cable[k] + self.edge_px(u, k), reverse=True)
                    for k in kids[1:]:
                        todo.append((u, k, len(out)))       # len(out) = index this path is about to get
                    nodes.append(kids[0])
                out.append(dict(nodes=nodes, parent=parent_path, connected=start in comp0))
        for p in out:
            p['um'] = sum(self.edge_px(a, b) for a, b in zip(p['nodes'][:-1], p['nodes'][1:])) * self.vox
        return out

    # ---------- geometry ----------
    def edge_px(self, a, b):
        (x0, y0), (x1, y1) = self.xy[a], self.xy[b]
        d = math.hypot(x1 - x0, y1 - y0)
        if self.soma_r and self.soma in (a, b):
            d = max(0.0, d - self.soma_r)
        return d

    def cable_um(self):
        return sum(self.edge_px(a, b) for a in self.adj for b in self.adj[a] if a < b) * self.vox

    def chain_um(self, nodes):
        nodes = list(nodes)
        return sum(self.edge_px(a, b) for a, b in zip(nodes[:-1], nodes[1:]) if b in self.adj.get(a, ())) * self.vox

    def subset_um(self, nodes):
        s = set(nodes)
        return sum(self.edge_px(a, b) for a in s for b in self.adj[a] if b in s and a < b) * self.vox

    # ---------- edits ----------
    def snapshot(self):
        return ({k: v for k, v in self.xy.items()}, dict(self.r), dict(self.typ), set(self.new),
                {k: set(v) for k, v in self.adj.items()}, self.next_id, self.soma, list(self.roots0))

    def restore(self, s):
        self.xy, self.r, self.typ, self.new, adj, self.next_id, self.soma, self.roots0 = \
            dict(s[0]), dict(s[1]), dict(s[2]), set(s[3]), s[4], s[5], s[6], list(s[7])
        self.adj = {k: set(v) for k, v in adj.items()}
        self.touch()

    def _drop_isolated(self):
        for n in [n for n, s in self.adj.items() if not s and n != self.soma]:
            self._remove(n)

    def _remove(self, n):
        for m in self.adj.pop(n, ()):
            self.adj[m].discard(n)
        self.xy.pop(n, None); self.r.pop(n, None); self.typ.pop(n, None); self.new.discard(n)

    def delete_chain(self, chain):
        """remove a segment's edges; its inner nodes and a free end go with it, branch points stay."""
        um = self.chain_um(chain)
        for a, b in zip(chain[:-1], chain[1:]):
            self.adj[a].discard(b); self.adj[b].discard(a)
        self._drop_isolated(); self.touch()
        return um

    def delete_nodes(self, nodes):
        nodes = set(nodes) - {self.soma}
        um = self.subset_um(nodes) + sum(self.edge_px(a, b) for a in nodes for b in self.adj[a] if b not in nodes) * self.vox
        for n in nodes:
            self._remove(n)
        self._drop_isolated(); self.touch()
        return um

    def erase(self, nodes, edges):
        """remove the given nodes and edges (what an eraser stroke covered); returns the cable removed."""
        nodes = set(nodes) - {self.soma}
        gone = {tuple(sorted(e)) for e in edges if e[0] in self.adj and e[1] in self.adj.get(e[0], ())}
        gone |= {tuple(sorted((u, v))) for u in nodes for v in self.adj.get(u, ())}
        um = sum(self.edge_px(u, v) for u, v in gone) * self.vox
        for u, v in gone:
            self.adj[u].discard(v); self.adj[v].discard(u)
        for n in nodes:
            self._remove(n)
        self._drop_isolated(); self.touch()
        return um

    # ---------- the "To check" list ----------
    # Measured on Liz's 20 corrected cells (yuqi/edit_research/sweep_issue_rules.py): most of her changes are at
    # free ends (extend, trim, remove a side branch). Visiting every tip that is not on the image edge, plus
    # every loose piece, covers 82% of her edits (about 16 items per cell). Ranking by suspicion did not beat a
    # random order, so items are ordered clockwise around the soma (one walk round the cell). Gap / crossing /
    # short-branch detectors were tested and dropped: only 7 to 18% of their items were real problems.
    EDGE_PX = 15
    CROSS_UM = 8.0         # split_crossing: two branch points this close count as one crossing (double T)

    def find_issues(self, W=None, H=None):
        """places to review, each dict(kind, x, y, label, key, node): every free end (not on the image edge)
        and every piece not connected to the soma, clockwise around the soma."""
        out = []
        if not self.adj:
            return out
        parent, comps = self.orient()
        sx, sy = self.xy[self.soma] if self.soma in self.xy else (0.0, 0.0)
        con = self.connected()

        def add(kind, n, label, x=None, y=None):
            x = self.xy[n][0] if x is None else x; y = self.xy[n][1] if y is None else y
            out.append(dict(kind=kind, x=float(x), y=float(y), label=label, node=n,
                            key="%s:%d:%d" % (kind, round(x / 10), round(y / 10)),
                            angle=math.atan2(y - sy, x - sx)))
        for c in comps:
            if self.soma in c or len(c) < 2:
                continue
            add('loose', c[len(c) // 2], "Loose %.0f µm" % self.subset_um(c))
        for n, nb in self.adj.items():
            if len(nb) != 1 or n == self.soma or n not in con:
                continue
            x, y = self.xy[n]
            if W is not None and min(x, y, W - 1 - x, H - 1 - y) < self.EDGE_PX:
                continue                              # ends at the image border: nothing to decide
            add('tip', n, "Tip")
        out.sort(key=lambda o: (o['angle'] + 2 * math.pi) % (2 * math.pi))
        for o in out:                                 # where it is, in words a person can match to the picture
            clock = int(round(((o['angle'] + math.pi / 2) % (2 * math.pi)) / (2 * math.pi) * 12)) or 12
            o['clock'] = "%d o'clock" % clock
            o['where'] = "%.0f µm" % (math.hypot(o['x'] - sx, o['y'] - sy) * self.vox)
        return out

    def _reach(self, s, limit):
        """tree distance (px) from s to every node within limit."""
        dist = {s: 0.0}; st = [s]
        while st:
            u = st.pop()
            for v in self.adj[u]:
                dv = dist[u] + math.hypot(self.xy[u][0] - self.xy[v][0], self.xy[u][1] - self.xy[v][1])
                if dv <= limit and dv < dist.get(v, 1e18):
                    dist[v] = dv; st.append(v)
        return dist

    def move_node(self, n, x, y):
        if n in self.xy and n != self.soma:
            self.xy[n] = (float(x), float(y)); self.touch()

    def crossing_cluster(self, n):
        """(cluster nodes, four arms) if n is a crossing candidate: a 4-way node, or a 3-way node with a second
        3-way node within CROSS_UM (a crossing traced as a 'double T'). None otherwise."""
        if n not in self.adj or n == self.soma:
            return None
        cluster = {n}
        if len(self.adj[n]) == 3:
            for nb in self.adj[n]:
                chain, prev, cur, um = [], n, nb, 0.0
                while True:
                    um += self.edge_px(prev, cur) * self.vox
                    if um > self.CROSS_UM:
                        break
                    if len(self.adj[cur]) == 3 and cur != self.soma:
                        cluster |= set(chain) | {cur}; break
                    if len(self.adj[cur]) != 2:
                        break
                    chain.append(cur)
                    prev, cur = cur, next(w for w in self.adj[cur] if w != prev)
                if len(cluster) > 1:
                    break
            if len(cluster) == 1:
                return None
        elif len(self.adj[n]) < 4:
            return None
        if self.soma in cluster:
            return None
        arms = [(c, o) for c in cluster for o in self.adj[c] if o not in cluster]
        return (cluster, arms) if len(arms) == 4 else None

    def split_crossing(self, n):
        """make the junction at n a crossing: its four arms are paired straight through, the pair that leads
        to the soma stays attached, the other pair becomes its own line (not joined here). Handles a 4-way
        node and a 'double T' (two 3-way nodes within CROSS_UM). Returns the new node of the detached line,
        or None when n is not a crossing candidate."""
        found = self.crossing_cluster(n)
        if found is None:
            return None
        cluster, arms = found
        cx = float(np.mean([self.xy[c][0] for c in cluster])); cy = float(np.mean([self.xy[c][1] for c in cluster]))

        def direction(c, o):
            prev, cur = c, o
            for _ in range(10):
                nxt = [w for w in self.adj[cur] if w != prev]
                if len(nxt) != 1:
                    break
                prev, cur = cur, nxt[0]
            dx, dy = self.xy[cur][0] - cx, self.xy[cur][1] - cy
            d = math.hypot(dx, dy) or 1.0
            return dx / d, dy / d
        v = [direction(c, o) for c, o in arms]
        pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
        straight = lambda i, j: -(v[i][0] * v[j][0] + v[i][1] * v[j][1])     # 1 = straight through
        best = max(pairings, key=lambda pr: straight(*pr[0]) + straight(*pr[1]))
        parent, _ = self.orient()
        up = next((k for k, (c, o) in enumerate(arms) if parent.get(c) == o), None)
        keep, other = (best if up is None or up in best[0] else (best[1], best[0]))
        rad = [x for x in self.r.values() if x > 0]
        med = float(np.median(rad)) if rad else 0.5
        outs = [o for c, o in arms]
        for c in cluster:
            self._remove(c)
        new = []
        for pair in (keep, other):
            i = self.next_id; self.next_id += 1
            self.xy[i] = (cx, cy); self.r[i] = med; self.typ[i] = 3; self.adj[i] = set()
            for k in pair:
                self.adj[i].add(outs[k]); self.adj[outs[k]].add(i)
            new.append(i)
        self.touch()
        return new[1]

    def cut_beyond(self, n):
        d = self.descendants(n)
        if not d:
            return 0.0
        um = self.subset_um(d + [n])
        for m in d:
            self._remove(m)
        self.touch()
        return um

    def join(self, t, q):
        """connect free end t to node q. Refused (False) when they are on the same piece: that would be a loop."""
        if t not in self.adj or q not in self.adj or t == q or q in self.adj[t]:
            return False
        comp = self.component_of()
        if comp.get(t) == comp.get(q):
            return False
        self.adj[t].add(q); self.adj[q].add(t); self.new.add(t); self.touch()
        return True

    def add_chain(self, pts, a=None, b=None):
        """new nodes along pts (image px); joined to node a at the start and node b at the end.
        b is ignored when it would close a loop (same piece as the start). Returns (ids, joined_end)."""
        rad = [v for v in self.r.values() if v > 0]
        med = float(np.median(rad)) if rad else 0.5
        ids = []
        for x, y in pts:
            i = self.next_id; self.next_id += 1
            self.xy[i] = (float(x), float(y)); self.r[i] = med; self.typ[i] = 3; self.adj[i] = set()
            self.new.add(i); ids.append(i)
        for u, v in zip(ids[:-1], ids[1:]):
            self.adj[u].add(v); self.adj[v].add(u)
        if a is not None and a in self.adj and ids:
            self.adj[a].add(ids[0]); self.adj[ids[0]].add(a)
        self.touch()
        joined = False
        if b is not None and b in self.adj and ids:
            comp = self.component_of()
            if comp.get(b) != comp.get(ids[0]):
                self.adj[b].add(ids[-1]); self.adj[ids[-1]].add(b); joined = True
                self.touch()
        return ids, joined

    # ---------- metrics (same definitions as the draft) ----------
    def metrics(self):
        import networkx as nx
        import tracer
        parent, comps = self.orient()
        F = nx.DiGraph(); F.add_nodes_from(sorted(parent))
        for v in sorted(parent):                   # file order: the tracer's metrics walk children in insertion order
            if parent[v] != -1:
                F.add_edge(parent[v], v)
        roots = [n for n, u in parent.items() if u == -1]
        pts = {n: (self.xy[n][1], self.xy[n][0]) for n in parent}       # (row, col) like the tracer
        cy, cx = pts[self.soma] if self.soma in pts else (0.0, 0.0)
        e = np.array(list(F.edges())) if F.number_of_edges() else np.zeros((0, 2), int)
        dist = {n: math.hypot(pts[n][0] - cy, pts[n][1] - cx) * self.vox for n in F.nodes}
        rmax = max(dist.values()) if dist else 0.0
        radii = np.arange(0, rmax + 10, 10.0)
        if len(e):
            du = np.array([dist[u] for u in e[:, 0]]); dv = np.array([dist[v] for v in e[:, 1]])
            inters = np.array([int(np.sum((du < r) != (dv < r))) for r in radii])
        else:
            inters = np.zeros(len(radii), int)
        tm = tracer.tree_metrics(F, pts, roots, (cy, cx), self.vox, radii, inters)
        tm['num_tips'] = sum(1 for n in F if F.out_degree(n) == 0 and n not in roots)
        tm['num_branch_points'] = sum(1 for n in F if F.out_degree(n) >= 2 and n not in roots)
        tm['sholl_radii'] = radii; tm['sholl_intersections'] = inters
        return tm


# =====================================================================================================
# window
# =====================================================================================================
def crossings_from_overlay(path):
    """positions (x, y) of the blue X marks the software draws where it decided two neurites only cross
    (overlay colour (40, 120, 255)). The SWC does not record crossings, so this is where they come from."""
    try:
        from scipy import ndimage
        ov = np.asarray(Image.open(path).convert('RGB')).astype(np.int16)
    except Exception:
        return np.zeros((0, 2))
    m = (np.abs(ov[..., 0] - 40) < 25) & (np.abs(ov[..., 1] - 120) < 25) & (ov[..., 2] > 230)
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros((0, 2))
    idx = range(1, n + 1)
    sizes = ndimage.sum(m, lab, idx)
    cents = ndimage.center_of_mass(m, lab, idx)
    pts = [(x, y) for (y, x), sz in zip(cents, sizes) if 40 <= sz <= 400]
    return np.array(pts, float).reshape(-1, 2)


def load_soma_outline(d, stem, raw_img=None, vox=0.1625):
    """Nx2 (x, y) px soma outline. Runs from 2026-10-24 on save <stem>_soma.json; older runs get the outline
    recomputed from the normalised image with the same soma code (close to, not exactly, the run's own)."""
    p = os.path.join(d, stem + "_soma.json")
    try:
        with open(p, encoding='utf-8') as fh:
            q = np.array(json.load(fh)["contour_xy"], float).reshape(-1, 2)
            if len(q) >= 3:
                return q
    except Exception:
        pass
    if raw_img is None:
        return None
    try:
        import soma as _soma
        g = np.asarray(raw_img.convert('L'), np.uint8)
        res = _soma.segment_soma(g, vox)
        if res and res.get("contour") is not None and len(res["contour"]) >= 3:
            return np.array([(x, y) for y, x in res["contour"]], float)
    except Exception:
        pass
    return None


def load_prob(d, stem):
    """the run's probability map (<stem>_prob.png, 16 bit) as float 0..1, or None."""
    try:
        return np.asarray(Image.open(os.path.join(d, stem + "_prob.png")), np.float32) / 65535.0
    except Exception:
        return None


def find_other_somas(img, vox, soma_xy):
    """(x, y) px of other bright cell bodies in the picture (tracer.detect_somas), farther than OTHER_SOMA_MIN_UM from the
    traced soma. Shown as a hint only: the software never assigns neurites to them."""
    try:
        import tracer
        g = np.asarray(img.convert('L'), np.uint8)
        out = []
        for r, c in tracer.detect_somas(g, vox):
            if soma_xy is None or math.hypot(c - soma_xy[0], r - soma_xy[1]) * vox >= OTHER_SOMA_MIN_UM:
                out.append((float(c), float(r)))
        return out
    except Exception:
        return []


def soma_from_overlay(path):
    """the software's overlay marks the soma centre with a cyan (0, 230, 255) dot, a colour used for nothing
    else. Returns (x, y) px or None. This is the only place the soma centre is recorded besides the SWC."""
    try:
        ov = np.asarray(Image.open(path).convert('RGB')).astype(np.int16)
    except Exception:
        return None
    m = (ov[..., 0] < 20) & (ov[..., 1] > 215) & (ov[..., 1] < 245) & (ov[..., 2] > 240)
    ys, xs = np.nonzero(m)
    return (float(xs.mean()), float(ys.mean())) if 20 <= len(xs) <= 200 else None


def _read_metrics_csv(path):
    out = {}
    try:
        import csv
        with open(path, encoding='utf-8') as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            try:
                out[row['metric']] = float(row['value'])
            except (TypeError, ValueError):
                out[row['metric']] = row['value']
    except Exception:
        pass
    return out


def scan_cells(root):
    """traced cells = *_neuron_out folders holding <stem>.swc + <stem>_raw.png (a batch root, or one cell)."""
    root = root.rstrip("/\\")
    dirs = [root] if root.lower().endswith("_neuron_out") else \
        sorted(glob.glob(os.path.join(glob.escape(root), "**", "*_neuron_out"), recursive=True))
    cells = []
    for d in dirs:
        raws = glob.glob(os.path.join(glob.escape(d), "*_raw.png"))
        if not raws:
            continue
        stem = os.path.basename(raws[0])[:-len("_raw.png")]
        if not os.path.isfile(os.path.join(d, stem + ".swc")):
            continue
        name = os.path.basename(d)[:-len("_neuron_out")] or stem
        cells.append({'dir': d, 'stem': stem, 'name': name})
    return cells


class Editor(tk.Toplevel):
    HIT = 9                                       # screen px: how close a click must be to a line

    def __init__(self, master=None, folder=None):
        super().__init__(master)
        self.title("Edit tracing")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry("%dx%d" % (min(px(1800), sw - px(40)), min(px(960), sh - px(80))))
        if not ttk.Style(self).lookup('Card.TFrame', 'background'):
            theme.apply_style(self._root(), theme.CURRENT)
        self.configure(bg=P['bg'])

        self.cells, self.cell_idx, self.cell = [], None, None
        self.arbor = None; self.base_img = None; self.cost = None; self.prob = None; self._other_somas = []
        self.base_cable = 0.0; self.base_metrics = {}
        self.edits = []; self.undo_stack = []; self.redo_stack = []
        self.dirty = False
        self.scale = 1.0; self.ox = self.oy = 0.0; self._fitted = False
        self.tool = tk.StringVar(value='select')
        self.follow = tk.BooleanVar(value=True)
        self.snap = tk.BooleanVar(value=True)
        self.vmin = tk.DoubleVar(value=0.0); self.vmax = tk.DoubleVar(value=255.0)
        self._bc_job = None; self._prev_job = None; self._preview = None; self._target = None
        self.disp = None; self.raw_img = None; self._cost2 = None
        self.panel_on = True
        self.hide_trace = False
        self.sel = set(); self.sel_label = ""
        self.pend = None                          # drawing: dict(start=node|None, clicks=[(x,y)], pts=[(x,y)])
        self._cursor_img = None; self._drag = None; self._hover_cut = set(); self._hover_anchor = None
        self._sel_anchor = None; self._sel_chain = ()
        self._segs = []; self._kd = None; self._kd_seg = None; self._kd_nodes = None; self._kd_ids = None
        self._photo = None; self._side_photos = {}; self._draft_lines = []
        self._draft_tips = np.zeros((0, 2)); self._draft_soma = None
        self._starts = np.zeros((0, 2)); self._draft_starts = np.zeros((0, 2))
        self.view_mode = tk.StringVar(value='23')     # panes shown: '12' result, '23' edit, '123' all, '3' yours only
        self.issues = []; self.checked = set(); self.issue_idx = None; self._check_sync = False
        self.use_menu = True                          # tests switch the pop-up menus off
        self._press = None; self._hover = None; self._mini_box = None; self._mini_key = None
        self._junc = np.zeros((0, 2)); self._draft_junc = np.zeros((0, 2))
        self._cross_sw = np.zeros((0, 2)); self._cross_now = np.zeros((0, 2))
        self._kd_handles = None; self._handle_ids = np.zeros(0, int)
        self.erase_r = tk.DoubleVar(value=14.0)       # eraser radius in screen px
        self._rr_job = None; self._settle_job = None; self._drag_last = None
        self._tile_cache = None; self._pyr = {}; self._dl = []; self._tips = np.zeros((0, 2))
        self._erase = None; self._brush = None; self._max_edge = 0.0

        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        if folder:
            self.load_folder(folder)

    # ---------- UI ----------
    def _build(self):
        bar = ttk.Frame(self, padding=(px(12), px(10), px(12), px(8))); bar.pack(side='top', fill='x')
        self.draw_btn = ttk.Button(bar, text="Draw a line", command=self.start_draw)
        self.draw_btn.pack(side='left')
        Tooltip(self.draw_btn, "Then click on a line or on the soma to start, and keep clicking along the neurite. "
                "Enter or double-click finishes. Or click any line and choose 'Draw from here'.")
        ttk.Separator(bar, orient='vertical').pack(side='left', fill='y', padx=px(12))
        ttk.Button(bar, text="Undo", style='Small.TButton', command=self.undo).pack(side='left')
        ttk.Button(bar, text="Redo", style='Small.TButton', command=self.redo).pack(side='left', padx=(px(4), 0))
        ttk.Button(bar, text="Fit", style='Small.TButton', command=self.fit).pack(side='left', padx=(px(12), 0))
        ttk.Label(bar, text="View", style='Sub.TLabel').pack(side='left', padx=(px(18), px(8)))
        for val, txt in (('12', "1+2  Result"), ('23', "2+3  Edit"), ('123', "1+2+3  All"), ('3', "3 only")):
            ttk.Radiobutton(bar, text=txt, value=val, variable=self.view_mode, style='Seg.Toolbutton',
                            command=self._set_view).pack(side='left', padx=(0, 1))
        self.panel_btn = ttk.Button(bar, text="Hide panel ▸", style='Small.TButton', command=self.toggle_panel)
        self.panel_btn.pack(side='right')
        ttk.Button(bar, text="Save", style='Accent.TButton', command=self.save).pack(side='right', padx=(0, px(10)))
        ttk.Button(bar, text="Revert", style='Small.TButton', command=self.revert).pack(side='right', padx=(0, px(10)))
        ttk.Button(bar, text="? Help", style='Small.TButton', command=self.show_help).pack(side='right', padx=(0, px(10)))

        # second row: image look, drawing options (plain words, no numbers)
        vbar = ttk.Frame(self, padding=(px(12), 0, px(12), px(8))); vbar.pack(side='top', fill='x')
        ttk.Label(vbar, text="Image", style='Sub.TLabel').pack(side='left', padx=(0, px(10)))
        ttk.Label(vbar, text="Dark cut-off", style='Sub.TLabel').pack(side='left')
        s_min = ttk.Scale(vbar, from_=0, to=254, variable=self.vmin, length=px(130),
                          command=lambda v: self._schedule_bc())
        s_min.pack(side='left', padx=(px(6), px(12)))
        ttk.Label(vbar, text="Bright cut-off", style='Sub.TLabel').pack(side='left')
        s_max = ttk.Scale(vbar, from_=1, to=255, variable=self.vmax, length=px(130),
                          command=lambda v: self._schedule_bc())
        s_max.pack(side='left', padx=(px(6), px(6)))
        self.vmin_lbl = ttk.Label(vbar, text="", width=0); self.vmax_lbl = ttk.Label(vbar, text="", width=0)
        ttk.Button(vbar, text="Reset", style='Small.TButton', command=self.reset_bc).pack(side='left')
        Tooltip(s_min, "Raise it to see neurites inside the bright soma glow (everything darker than this turns black).")
        Tooltip(s_max, "Lower it to see faint branches (everything brighter than this turns white). "
                "Drawing follows what you see.")
        ttk.Separator(vbar, orient='vertical').pack(side='left', fill='y', padx=px(16))
        ttk.Label(vbar, text="Drawing", style='Sub.TLabel').pack(side='left', padx=(0, px(10)))
        fl = ttk.Checkbutton(vbar, text="Follow the neurite", variable=self.follow)
        fl.pack(side='left')
        Tooltip(fl, "On: each new stretch follows the bright neurite between your clicks (F toggles). "
                "Off: straight lines. Click about every 10 µm on faint or crossing neurites.")
        sn = ttk.Checkbutton(vbar, text="Snap to the bright pixel", variable=self.snap)
        sn.pack(side='left', padx=(px(12), 0))
        Tooltip(sn, "Clicks jump to the brightest pixel within 8 px, like SNT's cursor auto-snapping. "
                "A click on an existing line always joins that line.")
        self.erase_lbl = ttk.Label(vbar, text="", style='Sub.TLabel'); self.erase_lbl.pack(side='left', padx=(px(16), 0))
        ttk.Label(vbar, text="hold Space: image only     wheel: zoom     [ ]: eraser size",
                  style='Hint.TLabel').pack(side='right')

        sbar = ttk.Frame(self, style='Status.TFrame', padding=(px(14), px(6))); sbar.pack(side='bottom', fill='x')
        self.status = ttk.Label(sbar, text="", style='Status.TLabel'); self.status.pack(side='left')
        self.len_lbl = ttk.Label(sbar, text="", style='Status.TLabel'); self.len_lbl.pack(side='right')
        self.pos_lbl = ttk.Label(sbar, text="", style='Status.TLabel', width=16, anchor='e')
        self.pos_lbl.pack(side='right', padx=(0, px(18)))

        self.body = ttk.Frame(self, padding=(px(12), px(2), px(12), px(8))); self.body.pack(fill='both', expand=True)
        self.side = ttk.Frame(self.body, style='Card.TFrame', width=px(340), padding=px(10))
        self.side.pack(side='right', fill='y', padx=(px(10), 0)); self.side.pack_propagate(False)
        self.views = ttk.Frame(self.body); self.views.pack(side='left', fill='both', expand=True)
        self.views.rowconfigure(0, weight=1)
        self.canvas_orig = tk.Canvas(self.views, bg=IMG_BG, highlightthickness=0, cursor='fleur')
        self.canvas_draft = tk.Canvas(self.views, bg=IMG_BG, highlightthickness=0, cursor='fleur')
        self.canvas = tk.Canvas(self.views, bg=IMG_BG, highlightthickness=0, cursor='crosshair')
        for k, cv in enumerate((self.canvas_orig, self.canvas_draft, self.canvas)):
            cv.grid(row=0, column=k, sticky='nsew', padx=(0 if k == 0 else px(4), 0))
        leg = tk.Frame(self.views, bg=IMG_BG, padx=px(10), pady=px(6))
        leg.grid(row=1, column=0, columnspan=3, sticky='ew', pady=(px(4), 0))
        for kind, col, txt in (('line', COL_TRACE, "Software"), ('dash', COL_FLOAT, "Loose piece (not connected)"),
                               ('line', COL_NEW, "Drawn by you"), ('line', COL_FAINT, "Faint image"), ('line', COL_PEND, "Being drawn"),
                               ('dot', '#00e6ff', "Tip"), ('ring', COL_TRACE, "Soma"),
                               ('ring', COL_JUNC, "Branch point"), ('x', COL_CROSS, "Crossing")):
            sw = tk.Canvas(leg, width=px(22), height=px(14), bg=IMG_BG, highlightthickness=0)
            m = px(7)
            if kind == 'line':
                sw.create_line(px(2), m, px(20), m, fill=col, width=2)
            elif kind == 'dash':
                sw.create_line(px(2), m, px(20), m, fill=col, width=2, dash=(4, 2))
            elif kind == 'dot':
                sw.create_oval(px(8), m - px(3), px(14), m + px(3), fill=col, outline='')
            elif kind == 'x':
                sw.create_line(px(6), m - px(5), px(16), m + px(5), fill=col, width=2)
                sw.create_line(px(6), m + px(5), px(16), m - px(5), fill=col, width=2)
            else:
                sw.create_oval(px(5), m - px(6), px(17), m + px(6), outline=col, width=1)
            sw.pack(side='left', padx=(0, px(4)))
            tk.Label(leg, text=txt, bg=IMG_BG, fg='#c3c8cf', font=(FONT, 9)).pack(side='left', padx=(0, px(16)))
        tk.Label(leg, text="(last two: when zoomed in)", bg=IMG_BG, fg='#7d838b', font=(FONT, 8)).pack(side='left')
        for cv in (self.canvas_orig, self.canvas_draft):
            cv.bind('<Configure>', lambda e: self._render_side())
            cv.bind('<MouseWheel>', lambda e: self._zoom(e, e.delta > 0))
            cv.bind('<Button-4>', lambda e: self._zoom(e, True))
            cv.bind('<Button-5>', lambda e: self._zoom(e, False))
            for b in ('1', '2', '3'):
                cv.bind('<ButtonPress-%s>' % b, self._pan_press)
                cv.bind('<B%s-Motion>' % b, self._pan_move)
                cv.bind('<ButtonRelease-%s>' % b, self._pan_release)
        self._set_view(render=False)

        # side panel: Paths / Edits / Cells
        tabs = ttk.Frame(self.side, style='Card.TFrame', borderwidth=0); tabs.pack(fill='x', pady=(0, px(8)))
        self.tab = tk.StringVar(value='check')
        for val, txt in (('check', "To check"), ('edits', "Edits"), ('paths', "Paths"), ('cells', "Cells")):
            ttk.Radiobutton(tabs, text=txt, value=val, variable=self.tab, style='Seg.Toolbutton',
                            command=self._show_tab).pack(side='left', fill='x', expand=True, padx=(0, 1))
        self.pages = {}
        for key in ('check', 'paths', 'edits', 'cells'):
            f = ttk.Frame(self.side, style='Card.TFrame', borderwidth=0); self.pages[key] = f

        f = self.pages['check']
        self.check_lbl = ttk.Label(f, text="", style='CardMuted.TLabel', wraplength=px(290))
        self.check_lbl.pack(anchor='w')
        self.check_bar = ttk.Progressbar(f, mode='determinate', maximum=1.0)
        self.check_bar.pack(fill='x', pady=(px(4), px(8)))
        nav = ttk.Frame(f, style='Card.TFrame', borderwidth=0); nav.pack(side='bottom', fill='x', pady=(px(6), 0))
        nav.columnconfigure((0, 1, 2), weight=1, uniform='c')
        ttk.Button(nav, text="◀ Prev", style='Small.TButton', command=lambda: self.next_check(-1)).grid(row=0, column=0, sticky='ew', padx=(0, px(3)))
        ttk.Button(nav, text="Looks fine", style='Small.TButton', command=self.mark_fine).grid(row=0, column=1, sticky='ew', padx=px(3))
        ttk.Button(nav, text="Next ▶", style='Small.TButton', command=lambda: self.next_check(1)).grid(row=0, column=2, sticky='ew', padx=(px(3), 0))
        self.ctree = self._tree(f, ('where',), 62)
        self.ctree.bind('<<TreeviewSelect>>', self._on_check_select)

        f = self.pages['paths']
        ttk.Label(f, text="Click a path to select it. Daughters are nested under their parent.",
                  style='CardMuted.TLabel', wraplength=px(290)).pack(anchor='w', pady=(0, px(6)))
        self.ptree = self._tree(f, ('um',), 90)
        self.ptree.bind('<<TreeviewSelect>>', self._on_path_select)
        r = ttk.Frame(f, style='Card.TFrame', borderwidth=0); r.pack(fill='x', pady=(px(6), 0))
        ttk.Button(r, text="Delete path + daughters", style='Small.TButton',
                   command=self.delete_selection).pack(fill='x')

        f = self.pages['edits']
        ttk.Label(f, text="Every change, with the error type it fixes. Click one to go there.",
                  style='CardMuted.TLabel', wraplength=px(290)).pack(anchor='w', pady=(0, px(6)))
        self.etree = self._tree(f, ('um',), 100)
        self.etree.bind('<<TreeviewSelect>>', self._on_edit_select)

        f = self.pages['cells']
        ttk.Button(f, text="Open folder…", command=self._open_folder).pack(fill='x', pady=(0, px(6)))
        self.cell_list = tk.Listbox(f, activestyle='none', font=(theme.MONO, 9), exportselection=False)
        self.cell_list.pack(fill='both', expand=True)
        self.cell_list.bind('<<ListboxSelect>>', self._on_cell_select)
        nav = ttk.Frame(f, style='Card.TFrame', borderwidth=0); nav.pack(fill='x', pady=(px(6), 0))
        nav.columnconfigure((0, 1), weight=1, uniform='n')
        ttk.Button(nav, text="◀ Prev", command=lambda: self._step_cell(-1)).grid(row=0, column=0, sticky='ew', padx=(0, px(3)))
        ttk.Button(nav, text="Next ▶", command=lambda: self._step_cell(1)).grid(row=0, column=1, sticky='ew', padx=(px(3), 0))
        self._show_tab()

        c = self.canvas
        c.bind('<Configure>', self._on_configure)
        c.bind('<ButtonPress-1>', self._b1)
        c.bind('<B1-Motion>', self._b1_drag)
        c.bind('<ButtonRelease-1>', self._b1_up)
        c.bind('<Leave>', lambda e: self._move_brush(None))
        c.bind('<Double-Button-1>', lambda e: self.finish_draw())
        c.bind('<Motion>', self._motion)
        c.bind('<MouseWheel>', lambda e: self._zoom(e, e.delta > 0))
        c.bind('<Button-4>', lambda e: self._zoom(e, True))
        c.bind('<Button-5>', lambda e: self._zoom(e, False))
        for b in ('2', '3'):
            c.bind('<ButtonPress-%s>' % b, self._pan_press)
            c.bind('<B%s-Motion>' % b, self._pan_move)
            c.bind('<ButtonRelease-%s>' % b, self._pan_release)
        self.bind('<KeyPress>', self._key)
        self.bind('<KeyPress-space>', lambda e: self._set_hide(True))
        self.bind('<KeyRelease-space>', lambda e: self._set_hide(False))
        self.bind('<Control-z>', lambda e: self.undo())
        self.bind('<Control-y>', lambda e: self.redo())
        self.bind('<Control-s>', lambda e: self.save())

        def _paint():
            if not self.winfo_exists():
                return
            self.configure(bg=P['bg']); paint_list(self.cell_list)
            for t in (self.ptree, self.etree):
                t.tag_configure('float', foreground=P['muted']); t.tag_configure('fix', foreground=P['ok'])
                t.tag_configure('del', foreground=P['err'])
            self.ctree.tag_configure('done', foreground=P['dim'])
            self.ctree.tag_configure('hint', foreground='#fbbf24')
            titlebar(self)
        on_theme_change(_paint)
        self._hint()

    def _tree(self, parent, cols, w):
        holder = ttk.Frame(parent, style='Card.TFrame', borderwidth=0); holder.pack(fill='both', expand=True)
        t = ttk.Treeview(holder, columns=cols, show='tree', style='Results.Treeview', selectmode='browse')
        t.column('#0', width=px(150), stretch=True, minwidth=px(110)); t.column(cols[0], width=px(w), stretch=False, anchor='e')
        sb = ttk.Scrollbar(holder, orient='vertical', command=t.yview); t.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y'); t.pack(side='left', fill='both', expand=True)
        return t

    def _show_tab(self):
        for k, f in self.pages.items():
            f.pack_forget()
        self.pages[self.tab.get()].pack(fill='both', expand=True)

    def _set_view(self, render=True):
        """which panes are shown: 1 = Original, 2 = Software, 3 = Edited (yours). '12' shows what the software just
        produced, '23' is for editing, '123' is the final side-by-side, '3' is one large pane for fine work."""
        mode = self.view_mode.get()
        n = len(mode)
        for k, (cv, num) in enumerate(((self.canvas_orig, '1'), (self.canvas_draft, '2'), (self.canvas, '3'))):
            on = num in mode
            if on:
                cv.grid()
            else:
                cv.grid_remove()
            self.views.columnconfigure(k, weight=1 if on else 0, uniform='v' if (on and n > 1) else '', minsize=0)
        if render:
            self.update_idletasks(); self.fit()

    def _ref(self):
        """a visible canvas: all panes share one view, so any of them gives the size."""
        return self._canvases()[0]

    def toggle_panel(self):
        self.panel_on = not self.panel_on
        if self.panel_on:
            self.side.pack(side='right', fill='y', padx=(px(10), 0), before=self.views)
            self.panel_btn.config(text="Hide panel ▸")
        else:
            self.side.pack_forget(); self.panel_btn.config(text="◂ Show panel")

    def _hint(self, msg=None):
        if msg is None:
            msg = {'select': "Click a line for options. Drag a point to move it, drag along a line to erase, drag "
                             "elsewhere to move the picture. N = next spot to check.",
                   'cut': "Cut: click a point on a line. The pink part beyond it will be removed. Esc: back.",
                   'erase': "Erase: hold the left button and drag over lines. [ and ] change the size. Esc: back.",
                   'draw': "Draw: click along the neurite (start on a line or the soma). The cyan line shows what the "
                           "next click draws. Enter or double-click finishes, Backspace undoes a point, Esc cancels."}[self.tool.get()]
        self.status.config(text=msg)

    # ---------- loading ----------
    HELP_TEXT = 'HOW TO CORRECT A TRACING\n\nStart with the "To check" list on the right. It walks you round the cell, one tip or\nloose piece at a time. On Liz\'s corrected cells, most of the changes were at these spots.\n  Next (N) goes to the next spot. "Looks fine" ticks it off. Fix anything wrong on the way.\n\nTHE MOUSE (no tools to switch)\n  Click a line            a menu: delete this piece, delete the whole branch beyond,\n                          cut here, draw a new branch from here, "this is a crossing".\n  Drag a yellow ring or   move that branch point or tip.\n  a cyan tip\n  Drag along a line       erase it, like an eraser. It turns pink, and goes when you let go.\n                          (Esc while dragging cancels.) Eraser size: the [ and ] keys.\n  Drag a tip onto a line  join it to that line (a tip resting on a line is not joined).\n  Drag on empty space     move the picture. The wheel zooms.\n  Shift+click a line      select the whole branch beyond that point. Delete removes it.\n\nADD A MISSING LINE\n  Click a line, choose "Draw a new branch from here" (or "Extend from here" on a tip),\n  or press "Draw a line" and click where it starts.\n  Keep clicking along the neurite. The cyan line shows what the next click will draw.\n  Enter or double-click finishes (the new line is green). Backspace removes the last\n  point. Esc cancels. Clicking on another line ends your line there and joins it.\n\nCROSSINGS\n  Where two neurites only cross, the software sometimes joins them into a branch.\n  Click the junction and choose "This is a crossing, not a branch". The two straightest\n  pairs of arms become two lines that pass each other. The one not joined there turns\n  orange: connect it where it really starts, or erase it.\n\nORANGE DASHED PIECES\n  Found by the software but not connected to the soma. Real: draw from a red line to it.\n  Wrong: erase it. A cyan dashed link means the software found a line within 5 \u00b5m of the\n  piece but was not sure: press J to join it there, or \"Looks fine\" to leave it apart.\n\nSAVE\n  Save writes <name>_edited.swc, recomputed metrics, a Sholl table, an overlay picture and\n  the list of your edits, next to the software\'s files. Those are never changed.\n  Opening the cell again continues where you stopped, ticks included.\n  Ctrl+Z undo. Ctrl+Y redo. Revert goes back to the software\'s tracing.\n\nHELPERS\n  Hold Space: hide the lines to see the image.\n  Image sliders: lower the bright cut-off for faint branches, raise the dark cut-off inside the\n  soma glow. Follow the neurite (F): new lines follow the bright neurite. Snap: clicks jump\n  to the brightest pixel nearby. Corner map (when zoomed in): click it to jump.\n  Branch rings and crossing marks appear once you zoom in; tips are always shown.\n  View: 1+2 = original and the software\'s result, 2+3 = the software\'s result and yours (editing),\n  1+2+3 = all three, "3 only" = one large pane. Only pane 3 takes edits.\n\nCOLOURS\n  Red: the software\'s tracing.  Orange dashed: not connected to the soma.  Green: drawn by you.\n  White: the line being drawn.  Yellow rings: branch points.  Cyan dots: tips.  Red circle: soma.\n  Blue X: crossing, two lines pass without joining (from the software, or made by you).\n  Violet: the image is faint under this line (low probability). Check that it is a real neurite.\n  Yellow dashed circle: another bright cell body in the picture. Its neurites may have been\n  traced as this cell\'s: erase what belongs to it.\n'

    def show_help(self):
        win = tk.Toplevel(self); win.title("How to edit a tracing"); win.transient(self)
        win.geometry("%dx%d" % (px(720), px(640))); win.configure(bg=P['bg']); titlebar(win)
        ttk.Button(win, text="Close", command=win.destroy).pack(side='bottom', pady=px(10))
        txt = tk.Text(win, wrap='word', font=(theme.MONO, 10), padx=px(16), pady=px(14))
        paint_list(txt); txt.configure(bg=P['card'], highlightthickness=0)
        sb = ttk.Scrollbar(win, command=txt.yview); sb.pack(side='right', fill='y')
        txt.configure(yscrollcommand=sb.set); txt.pack(fill='both', expand=True)
        txt.insert('1.0', self.HELP_TEXT); txt.configure(state='disabled')
        return win

    def _open_folder(self):
        d = filedialog.askdirectory(title="Select a run's output folder (holds the *_neuron_out cells)")
        if d:
            self.load_folder(d)

    def load_folder(self, folder):
        cells = scan_cells(folder)
        if not cells:
            messagebox.showinfo("No traced cells", "No traced cells found in:\n%s\n\nRun tracing first." % folder,
                                parent=self)
            return
        self.cells = cells
        self._refresh_cells()
        if len(cells) > 1:
            self.tab.set('cells'); self._show_tab()
        self.select_cell(0)

    def _refresh_cells(self):
        self.cell_list.delete(0, 'end')
        for c in self.cells:
            mark = "✓" if os.path.isfile(os.path.join(c['dir'], c['stem'] + "_edited.swc")) else "•"
            self.cell_list.insert('end', "%s %s" % (mark, c['name']))
        if self.cell_idx is not None:
            self.cell_list.selection_clear(0, 'end'); self.cell_list.selection_set(self.cell_idx)

    def _on_cell_select(self, e):
        s = self.cell_list.curselection()
        if s and s[0] != self.cell_idx:
            self.select_cell(s[0])

    def _step_cell(self, d):
        if self.cells:
            self.select_cell(min(max((self.cell_idx or 0) + d, 0), len(self.cells) - 1))

    def _ask_save(self):
        """True = go on, False = stay."""
        if not self.dirty:
            return True
        r = messagebox.askyesnocancel("Unsaved edits", "Save your edits to %s first?" % self.cell['name'], parent=self)
        if r is None:
            return False
        if r:
            self.save(quiet=True)
        return True

    def select_cell(self, i):
        if self.cell is not None and not self._ask_save():
            self._refresh_cells(); return
        self.cell_idx = i; self.cell = c = self.cells[i]
        d, s = c['dir'], c['stem']
        m = _read_metrics_csv(os.path.join(d, s + "_metrics.csv"))
        self.base_metrics = m
        src = os.path.join(d, s + ".swc")
        self.soma_xy = soma_from_overlay(os.path.join(d, s + "_overlay.png"))
        self._cross_sw = crossings_from_overlay(os.path.join(d, s + "_overlay.png"))
        img = Image.open(os.path.join(d, s + "_raw.png")).convert('L')
        self.gray = np.asarray(img, np.float32) / 255.0
        self.raw_img = img.convert('RGB'); self.W, self.H = img.size
        self.soma_poly = load_soma_outline(d, s, img, m.get('pixel_size', 0.1625))
        self.prob = load_prob(d, s)
        self.arbor = Arbor.from_swc(src, m.get('soma_area'), self.soma_xy)
        self._other_somas = find_other_somas(img, m.get('pixel_size', 0.1625), self.arbor.xy.get(self.arbor.soma))
        self.arbor.soma_poly = self.soma_poly
        self.base_cable = self.arbor.cable_um()
        # the middle pane: the software's tracing, frozen as coordinates (the arbor itself gets edited)
        con = self.arbor.connected()
        a0 = self.arbor
        d_starts = {c[0] for c in a0.orient()[1] if a0.soma not in c}
        self._draft_tips = np.array([a0.xy[n] for n, nb in a0.adj.items() if len(nb) == 1 and n != a0.soma
                                     and n not in d_starts] or np.zeros((0, 2)), float).reshape(-1, 2)
        self._draft_starts = np.array([a0.xy[n] for n in d_starts] or np.zeros((0, 2)), float).reshape(-1, 2)
        self._draft_junc = np.array([a0.xy[n] for n, nb in a0.adj.items() if len(nb) >= 3 and n != a0.soma] or
                                    np.zeros((0, 2)), float).reshape(-1, 2)
        self._draft_soma = (a0.xy[a0.soma][0], a0.xy[a0.soma][1], a0.soma_r) if a0.soma in a0.xy and a0.soma_r else None
        self._draft_lines = []
        for sg in self.arbor.segments():
            q = self._chain_xy(sg)
            if len(q) >= 2:
                self._draft_lines.append((np.array(q), sg[0] in con))
        self.edits = []; self.undo_stack = []; self.redo_stack = []
        resumed = os.path.join(d, s + "_edited.swc")
        if os.path.isfile(resumed):
            self.arbor = Arbor.from_swc(resumed, m.get('soma_area'), self.soma_xy)
            self.arbor.soma_poly = self.soma_poly
            try:
                with open(os.path.join(d, s + "_edits.json"), encoding='utf-8') as fh:
                    log = json.load(fh)
                self.edits = log.get('edits', [])
                self.checked = set(log.get('checked', []))
                self.arbor.new = {int(k) for k in log.get('new_nodes', [])} & set(self.arbor.xy)
            except Exception:
                pass
        self.vmin.set(0.0); self.vmax.set(255.0); self._apply_bc(render=False)
        self.sel = set(); self.pend = None; self.dirty = False; self._fitted = False
        self.issue_idx = None
        if not os.path.isfile(resumed):
            self.checked = set()
        self.title("Edit tracing  ·  %s" % c['name'])
        self._refresh_cells(); self._rebuild(); self.fit()
        self._hint("Resumed your saved edits for %s." % c['name'] if os.path.isfile(resumed) else None)

    # ---------- derived data after every edit ----------
    def _rebuild(self):
        a = self.arbor
        self._segs = a.segments()
        mids, segi = [], []
        for k, s in enumerate(self._segs):
            q = np.array([a.xy[n] for n in s])
            inner = q[1:-1] if len(q) > 2 else q[:0]           # junction nodes belong to several segments
            pts = np.vstack([inner, (q[:-1] + q[1:]) / 2, q[:-1] * 0.75 + q[1:] * 0.25, q[:-1] * 0.25 + q[1:] * 0.75])
            mids.append(pts); segi.append(np.full(len(pts), k))
        self._kd = cKDTree(np.vstack(mids)) if mids else None
        self._kd_seg = np.concatenate(segi) if segi else np.zeros(0, int)
        ids = list(a.xy); self._kd_ids = np.array(ids)
        self._kd_nodes = cKDTree(np.array([a.xy[i] for i in ids])) if ids else None
        self._connected = a.connected()
        self._max_edge = min(80.0, max((math.hypot(a.xy[u][0] - a.xy[v][0], a.xy[u][1] - a.xy[v][1])
                                        for u in a.adj for v in a.adj[u] if u < v and a.soma not in (u, v)), default=0.0))
        # what gets drawn, computed once per edit (rendering then only shifts and scales arrays)
        self._dl = []
        self._faint_edges, faint_runs = self._faint(a)
        for sg in self._segs:
            runs, cur, col = [], [sg[0]], None
            for u, v in zip(sg[:-1], sg[1:]):
                ecol = COL_NEW if (u in a.new or v in a.new) else (COL_TRACE if u in self._connected else COL_FLOAT)
                if (u, v) in self._faint_edges and ecol == COL_TRACE:
                    ecol = COL_FAINT
                if col is not None and ecol != col:
                    runs.append((cur, col)); cur = [u]
                cur.append(v); col = ecol
            runs.append((cur, col or COL_TRACE))
            for nodes, col in runs:
                q = self._chain_xy(nodes)
                if len(q) >= 2:
                    self._dl.append((np.array(q, float), col))
        starts = {c[0] for c in a.orient()[1] if a.soma not in c}      # the inner end of every loose piece
        self._tips = np.array([a.xy[n] for n, nb in a.adj.items() if len(nb) == 1 and n != a.soma and n not in starts]
                              or np.zeros((0, 2)), float).reshape(-1, 2)
        self._starts = np.array([a.xy[n] for n in starts] or np.zeros((0, 2)), float).reshape(-1, 2)
        self._junc = np.array([a.xy[n] for n, nb in a.adj.items() if len(nb) >= 3 and n != a.soma] or
                              np.zeros((0, 2)), float).reshape(-1, 2)
        hids = [n for n, nb in a.adj.items() if n != a.soma and (len(nb) == 1 or len(nb) >= 3)]
        self._handle_ids = np.array(hids, int)
        self._kd_handles = cKDTree(np.array([a.xy[n] for n in hids])) if hids else None
        self._hover = None
        self.issues = a.find_issues(self.W, self.H)
        if faint_runs:
            sx, sy = a.xy[a.soma] if a.soma in a.xy else (0.0, 0.0)
            for nodes, um in faint_runs:
                m = nodes[len(nodes) // 2]; x, y = a.xy[m]; ang = math.atan2(y - sy, x - sx)
                clock = int(round(((ang + math.pi / 2) % (2 * math.pi)) / (2 * math.pi) * 12)) or 12
                self.issues.append(dict(kind='faint', x=float(x), y=float(y), label="Faint %.0f µm" % um, node=m, hint=True,
                                        key="faint:%d:%d" % (round(x / 10), round(y / 10)), angle=ang, clock="%d o'clock" % clock,
                                        where="%.0f µm" % (math.hypot(x - sx, y - sy) * a.vox),
                                        note="the image is faint under this line (violet): check that it is a real neurite"))
            self.issues.sort(key=lambda o: (o['angle'] + 2 * math.pi) % (2 * math.pi))
        marks = list(self._cross_sw) + [tuple(ed['at']) for ed in self.edits if ed['kind'] == 'crossing']
        self._cross_now = np.array([m for m in marks if self._kd_nodes is not None and
                                    self._kd_nodes.query(m)[0] <= 20], float).reshape(-1, 2)   # marks sit at the crossing centre, up to ~17 px off the lines
        comp_nodes = {}
        for n, k in a.component_of().items():
            comp_nodes.setdefault(k, []).append(n)
        cid = a.component_of()
        self._cand_links = []                                   # [(root node, target node)] still open tracer candidates
        for it in self.issues:
            if it['kind'] == 'loose':
                ends = [n for n in comp_nodes.get(cid.get(it['node']), []) if len(a.adj[n]) == 1]
                tc = self._tracer_candidate(ends, cid)
                if tc is not None:
                    it['hint'] = True; it['cand'] = tc; self._cand_links.append(tc)
                    it['note'] = "the tracer found a line %.1f µm from this piece (dashed): J joins it, 'Looks fine' leaves it" % (
                        math.hypot(a.xy[tc[0]][0] - a.xy[tc[1]][0], a.xy[tc[0]][1] - a.xy[tc[1]][1]) * a.vox)
                elif any(self._join_candidate(n, radius=4.0)[1] for n in ends):
                    it['hint'] = True; it['note'] = "an end of it rests on a line: join it?"
            elif it['kind'] == 'tip' and self._join_candidate(it['node'], radius=4.0)[1]:
                it['label'] = "Tip on a line"; it['hint'] = True; it['note'] = "it rests on another line: join it?"
            elif it['kind'] == 'tip' and self._continues(it['node']):
                it['label'] = "Tip: the image continues beyond it"; it['hint'] = True
        if self.issue_idx is not None and self.issue_idx >= len(self.issues):
            self.issue_idx = None
        self._fill_check()
        self._fill_paths(); self._fill_edits(); self._update_len(); self._render()

    def _faint(self, a):
        """(set of (u, v) edges, [(nodes, um)] runs) where the smoothed probability along the software's line is below FAINT_PROB.
        Hand-drawn nodes and loose pieces are not judged (the person drew them / they are already orange)."""
        if self.prob is None or a.soma not in a.xy:
            return set(), []
        H, W = self.prob.shape; edges = set(); runs = []
        for sg in self._segs:
            if len(sg) < 3 or sg[0] not in self._connected or any(n in a.new for n in sg):
                continue
            q = np.array([a.xy[n] for n in sg], float)
            xs = np.clip(np.round(q[:, 0]).astype(int), 0, W - 1); ys = np.clip(np.round(q[:, 1]).astype(int), 0, H - 1)
            p = self.prob[ys, xs]
            k = min(11, len(p)); sm = np.convolve(np.pad(p, (k // 2, k // 2), mode='edge'), np.ones(k) / k, mode='valid')[:len(p)]
            low = sm < FAINT_PROB
            i = 0
            while i < len(sg) - 1:
                if not (low[i] and low[i + 1]) or a.soma in (sg[i], sg[i + 1]):
                    i += 1; continue
                j = i
                while j < len(sg) - 1 and low[j] and low[j + 1] and sg[j + 1] != a.soma:
                    j += 1
                um = a.chain_um(sg[i:j + 1])
                if um >= FAINT_MIN_UM:
                    runs.append((sg[i:j + 1], um)); edges.update(zip(sg[i:j], sg[i + 1:j + 1]))
                i = j + 1
        return edges, runs

    def _other_rings(self, c, off):
        """dashed yellow circle on every other bright cell body: its neurites may have been traced as this cell's."""
        a = self.arbor
        if a is None or not getattr(self, '_other_somas', None):
            return
        r = max(6.0, 8.0 / a.vox * self.scale)
        for x, y in self._other_somas:
            cx, cy = (x - off[0]) * self.scale, (y - off[1]) * self.scale
            c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=COL_JUNC, width=2, dash=(6, 4))
            c.create_text(cx, cy - r - px(6), text="other cell?", fill=COL_JUNC, font=(FONT, 8))

    def _tracer_candidate(self, ends, cid):
        """(piece end node, target node) when the tracer listed a join candidate for this piece: the candidate's root point must sit
        on one of the piece's free ends (within 3 px) and the target must be on ANOTHER piece (joining inside one piece makes a loop)."""
        a = self.arbor
        if not a.join_cands or self._kd_nodes is None or not ends:
            return None
        for (rx, ry), (tx, ty) in a.join_cands:
            n = min(ends, key=lambda k: math.hypot(a.xy[k][0] - rx, a.xy[k][1] - ry))
            if math.hypot(a.xy[n][0] - rx, a.xy[n][1] - ry) > 3.0:
                continue
            d, k = self._kd_nodes.query((tx, ty)); q = int(self._kd_ids[k])
            if d <= 3.0 and cid.get(q) != cid.get(n):
                return (n, q)
        return None

    def _draw_cand_links(self, c, off):
        """dashed cyan link for every open tracer candidate (piece end -> line it may belong to)."""
        for n, q in getattr(self, '_cand_links', []):
            a = self.arbor
            if n not in a.xy or q not in a.xy:
                continue
            (x0, y0), (x1, y1) = a.xy[n], a.xy[q]
            c.create_line((x0 - off[0]) * self.scale, (y0 - off[1]) * self.scale, (x1 - off[0]) * self.scale, (y1 - off[1]) * self.scale,
                          fill=COL_CAND, width=2, dash=(5, 4), tags='candlink')

    def join_current_candidate(self):
        """J: apply the tracer's join candidate of the To-check item being looked at."""
        if self.issue_idx is None or self.issue_idx >= len(self.issues):
            self._hint("No item selected. Walk the To-check list with N, then press J on a dashed candidate."); return
        it = self.issues[self.issue_idx]
        if not it.get('cand'):
            self._hint("This item has no join candidate. Drag the end onto a line to join it by hand."); return
        n, q = it['cand']
        self._join(n, q)
        self.checked.add(it['key']); self.dirty = True

    def _fill_paths(self):
        t = self.ptree; t.delete(*t.get_children()); self._path_nodes = {}
        paths = self.arbor.paths()
        iid = {}
        for k, p in enumerate(paths):
            parent = iid.get(p['parent'], '')
            name = "Path %d" % (k + 1) + ("" if p['connected'] else "  (not connected)")
            iid[k] = t.insert(parent, 'end', text=name, values=("%.1f µm" % p['um'],), open=False,
                              tags=(() if p['connected'] else ('float',)))
            self._path_nodes[iid[k]] = p['nodes']

    def _fill_edits(self):
        t = self.etree; t.delete(*t.get_children())
        for k, e in enumerate(self.edits):
            t.insert('', 'end', iid=str(k), text="#%d  %s" % (k + 1, KINDS.get(e['kind'], (e['kind'],))[0]),
                     values=("" if e['op'] in ('move', 'crossing') else
                             "%s%.1f µm" % ('+' if e['op'] == 'draw' else '−', e['um']),),
                     tags=(('fix',) if e['op'] == 'draw' else ('del',)))

    def _update_len(self):
        if self.arbor is None:
            return
        now = self.arbor.cable_um()
        base = self.base_metrics.get('total_length')
        if isinstance(base, float):
            now = base + (now - self.base_cable)
            self.len_lbl.config(text="Total length  %.1f  →  %.1f µm     %d edits%s" %
                                (base, now, len(self.edits), "   · unsaved" if self.dirty else ""))
        else:
            self.len_lbl.config(text="Total length  %.1f µm     %d edits" % (now, len(self.edits)))

    # ---------- view ----------
    def _on_configure(self, e):
        if self.arbor is not None and not self._fitted and e.width > 50:
            self._fitted = True; self.fit()
        else:
            self._render()

    def fit(self):
        if self.base_img is None:
            return
        cw = max(self._ref().winfo_width(), 50); ch = max(self._ref().winfo_height(), 50)
        self.scale = min(cw / self.W, ch / self.H)
        self.ox = (self.W - cw / self.scale) / 2; self.oy = (self.H - ch / self.scale) / 2
        self._render()

    def _i2c(self, x, y):
        return (x - self.ox) * self.scale, (y - self.oy) * self.scale

    def _c2i(self, cx, cy):
        return cx / self.scale + self.ox, cy / self.scale + self.oy

    def _zoom(self, e, zin):
        if self.base_img is None:
            return
        cw = max(self._ref().winfo_width(), 50); ch = max(self._ref().winfo_height(), 50)
        fit = min(cw / self.W, ch / self.H)
        ns = min(max(self.scale * (1.25 if zin else 0.8), fit * 0.5), 24.0)
        ix, iy = self._c2i(e.x, e.y)
        self.scale = ns; self.ox = ix - e.x / ns; self.oy = iy - e.y / ns
        self._request_render()

    def _canvases(self):
        mode = self.view_mode.get()
        return tuple(cv for cv, num in ((self.canvas_orig, '1'), (self.canvas_draft, '2'), (self.canvas, '3'))
                     if num in mode) or (self.canvas,)

    def _request_render(self, delay=10):
        """many mouse events in a row -> one redraw."""
        if self._rr_job is None:
            self._rr_job = self.after(delay, self._do_request)

    def _do_request(self):
        self._rr_job = None; self._render()

    def _pan_press(self, e):
        self._drag = (e.x, e.y, self.ox, self.oy); self._drag_last = (e.x, e.y)

    def _pan_move(self, e):
        """while dragging, slide what is already on screen (instant); redraw properly once the mouse rests."""
        if not self._drag:
            return
        sx, sy, ox, oy = self._drag
        lx, ly = self._drag_last
        dx, dy = e.x - lx, e.y - ly
        self._drag_last = (e.x, e.y)
        self.ox = ox - (e.x - sx) / self.scale; self.oy = oy - (e.y - sy) / self.scale
        for c in self._canvases():
            c.move('all', dx, dy)
            c.move('title', -dx, -dy)                 # pane titles and the mini map stay put
            c.move('mini', -dx, -dy)
        if self._settle_job:
            self.after_cancel(self._settle_job)
        self._settle_job = self.after(120, self._settle)

    def _pan_release(self, e):
        if self._drag:
            self._drag = None; self._settle()

    def _settle(self):
        if self._settle_job:
            self.after_cancel(self._settle_job)
        self._settle_job = None; self._render()

    def _set_hide(self, on):
        if self.hide_trace != on and self.focus_get() is not None:
            self.hide_trace = on; self._render()

    def _tile(self):
        """the visible image part, made once per redraw and shared by all panes (they show the same view).
        Zoomed out, it is cut from a 1/2 or 1/4 size copy: resizing the full 2048 px image each time was slow."""
        c = self._ref()
        cw = max(c.winfo_width(), 50); ch = max(c.winfo_height(), 50)
        key = (round(self.scale, 6), round(self.ox, 3), round(self.oy, 3), cw, ch, id(self.base_img))
        if self._tile_cache is not None and self._tile_cache[0] == key:
            return self._tile_cache[1]
        x0, y0 = self._c2i(0, 0); x1, y1 = self._c2i(cw, ch)
        ix0, iy0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
        ix1, iy1 = min(self.W, int(math.ceil(x1))), min(self.H, int(math.ceil(y1)))
        out = None
        if ix1 > ix0 and iy1 > iy0:
            dw = max(1, int(round((ix1 - ix0) * self.scale))); dh = max(1, int(round((iy1 - iy0) * self.scale)))
            f = 4 if self.scale <= 0.25 else (2 if self.scale <= 0.5 else 1)
            src = self._pyr.get(f, self.base_img)
            box = (ix0 / f, iy0 / f, ix1 / f, iy1 / f) if f > 1 else (ix0, iy0, ix1, iy1)
            tile = src.resize((dw, dh), Image.NEAREST if self.scale >= 2 else Image.BILINEAR, box=box)
            px0, py0 = self._i2c(ix0, iy0)
            out = (ImageTk.PhotoImage(tile, master=self.canvas), px0, py0)
        self._tile_cache = (key, out)
        return out

    def _draw_image(self, c):
        t = self._tile()
        if t is not None:
            c.create_image(t[1], t[2], image=t[0], anchor='nw')

    def _step(self):
        """zoomed out, every k-th point is enough (the eye cannot see the difference, drawing gets much faster)."""
        return max(1, int(0.7 / self.scale))

    def _lines(self, c, items, w, off):
        k = self._step()
        for q, col in items:
            if k > 1 and len(q) > 2 * k:
                q = np.vstack([q[::k], q[-1:]])
            coords = ((q - off) * self.scale).ravel().tolist()
            if len(coords) >= 4:
                c.create_line(*coords, fill=col, width=w, dash=(6, 3) if col == COL_FLOAT else None,
                              capstyle='round', joinstyle='round')

    MARK_ZOOM = 0.6          # branch rings and crossing marks appear from this zoom on (overview stays clean)

    def _rings(self, c, pts, off):
        if self.scale < self.MARK_ZOOM:
            return
        rt = 3 if self.scale < 1.2 else 5
        for x, y in ((pts - off) * self.scale if len(pts) else []):
            c.create_oval(x - rt, y - rt, x + rt, y + rt, outline=COL_JUNC, width=1)

    def _xmarks(self, c, pts, off):
        if self.scale < self.MARK_ZOOM:
            return
        r = 4 if self.scale < 1.2 else 6
        for x, y in ((pts - off) * self.scale if len(pts) else []):
            c.create_line(x - r, y - r, x + r, y + r, fill=COL_CROSS, width=2)
            c.create_line(x - r, y + r, x + r, y - r, fill=COL_CROSS, width=2)

    def _draw_focus(self):
        """white ring around the To-check item being looked at."""
        c = self.canvas; c.delete('focus')
        if self.issue_idx is None or self.issue_idx >= len(self.issues):
            return
        it = self.issues[self.issue_idx]
        x, y = self._i2c(it['x'], it['y']); r = px(18)
        c.create_oval(x - r, y - r, x + r, y + r, outline='#ffffff', width=2, dash=(5, 3), tags='focus')

    def _draw_minimap(self):
        """small whole-cell map in the corner when zoomed in; click it to jump."""
        c = self.canvas; c.delete('mini'); self._mini_box = None
        cw = max(c.winfo_width(), 50); ch = max(c.winfo_height(), 50)
        if self.base_img is None or self.scale <= min(cw / self.W, ch / self.H) * 1.3:
            return
        mw = px(150); mh = max(1, int(mw * self.H / self.W)); x0 = px(10); y0 = ch - mh - px(10)
        key = (id(self.base_img), mw, mh)
        if self._mini_key != key:
            self._mini = ImageTk.PhotoImage(self._pyr.get(4, self.base_img).resize((mw, mh), Image.BILINEAR), master=c)
            self._mini_key = key
        c.create_rectangle(x0 - 2, y0 - 2, x0 + mw + 2, y0 + mh + 2, fill='#000000', outline='#55595f', tags='mini')
        c.create_image(x0, y0, image=self._mini, anchor='nw', tags='mini')
        vx0, vy0 = self._c2i(0, 0); vx1, vy1 = self._c2i(cw, ch)
        f = mw / self.W
        c.create_rectangle(x0 + max(0, vx0) * f, y0 + max(0, vy0) * f, x0 + min(self.W, vx1) * f,
                           y0 + min(self.H, vy1) * f, outline='#ffffff', width=1, tags='mini')
        self._mini_box = (x0, y0, mw, mh)

    def _start_marks(self, c, pts, off):
        """the start (soma-side end) of a loose piece: it runs outward from here."""
        r = 3 if self.scale < 1.2 else 4
        for x, y in ((pts - off) * self.scale if len(pts) else []):
            c.create_oval(x - r, y - r, x + r, y + r, outline=COL_FLOAT, width=1)

    def _dots(self, c, pts, off):
        rt = 2 if self.scale < 1.2 else 3
        for x, y in ((pts - off) * self.scale if len(pts) else []):
            c.create_oval(x - rt, y - rt, x + rt, y + rt, fill='#00e6ff', outline='')

    def _lw(self):
        """line width: thin at overview zoom, a little bolder when zoomed in."""
        return 1 if self.scale < 1.2 else 2

    def _chain_xy(self, nodes):
        """image points of a node chain, with a soma end moved to the soma edge (the SWC joins neurites to the
        soma CENTRE by straight spokes; drawn that way the soma looks like a star)."""
        a = self.arbor
        pts = [a.xy[n] for n in nodes]
        if not (a.soma_r or a.soma_poly is not None) or a.soma not in nodes:
            return pts
        for end in (0, -1):
            if len(pts) < 2 or nodes[end] != a.soma:
                continue
            nx_, ny_ = pts[1] if end == 0 else pts[-2]
            p = a.soma_edge_towards(nx_, ny_)
            if p is None:                               # the whole edge lies inside the soma
                pts = pts[1:] if end == 0 else pts[:-1]
            else:
                pts = [p] + pts[1:] if end == 0 else pts[:-1] + [p]
        return pts

    def _soma_outline(self, c, off):
        """the segmented soma outline when the run saved one (<stem>_soma.json), else the equal-area circle."""
        a = self.arbor
        if a is None or a.soma not in a.xy:
            return
        if a.soma_poly is not None and len(a.soma_poly) >= 3:
            q = a.soma_poly
            k = max(1, int(0.7 / self.scale))
            if k > 1 and len(q) > 2 * k:
                q = q[::k]
            coords = ((q - off) * self.scale).ravel().tolist()
            if len(coords) >= 6:
                c.create_polygon(*coords, outline=COL_TRACE, fill='', width=max(1, self._lw()))
        elif a.soma_r:
            sx, sy = self._i2c(*a.xy[a.soma]); rr = a.soma_r * self.scale
            c.create_oval(sx - rr, sy - rr, sx + rr, sy + rr, outline=COL_TRACE, width=1, dash=(3, 3))

    def _pane_title(self, c, text):
        tid = c.create_text(px(12), px(14), text=text, anchor='w', fill='#f1f3f5', font=(FONT, 9, 'bold'),
                            tags='title')
        bx0, by0, bx1, by1 = c.bbox(tid)
        c.tag_lower(c.create_rectangle(bx0 - px(6), by0 - px(3), bx1 + px(6), by1 + px(3), fill='#000000',
                                       outline='', stipple='gray50', tags='title'), tid)

    def _render_side(self):
        mode = self.view_mode.get()
        for c, kind in ((self.canvas_orig, 'orig'), (self.canvas_draft, 'draft')):
            if ('1' if kind == 'orig' else '2') not in mode:
                continue
            c.delete('all')
            if self.base_img is None:
                continue
            self._draw_image(c)
            if kind == 'draft' and not self.hide_trace:
                off = np.array([self.ox, self.oy])
                self._lines(c, [(q, COL_TRACE if con else COL_FLOAT) for q, con in self._draft_lines], self._lw(), off)
                self._rings(c, self._draft_junc, off)
                self._xmarks(c, self._cross_sw, off)
                self._dots(c, self._draft_tips, off)
                self._soma_outline(c, off)
                self._other_rings(c, off)
            self._pane_title(c, "1  Original" if kind == 'orig' else "2  Software (read-only)")

    def _render(self):
        self._render_side()
        c = self.canvas; c.delete('all')
        if '3' not in self.view_mode.get():
            return
        if self.base_img is None:
            c.create_text(max(c.winfo_width(), 50) // 2, max(c.winfo_height(), 50) // 2, fill='#7d838b',
                          font=(FONT, 12), text="Open a run's output folder (Cells tab)")
            return
        self._draw_image(c)
        if len(self.view_mode.get()) > 1:
            self._pane_title(c, "3  Edited (yours)")
        if self.hide_trace:
            return
        a = self.arbor
        w = self._lw()
        off = np.array([self.ox, self.oy])
        # soma outline (neurites start at its edge; the SWC's centre spokes are not drawn)
        self._soma_outline(c, off)
        self._other_rings(c, off)
        self._lines(c, self._dl, w, off)
        self._draw_cand_links(c, off)
        # branch points (yellow rings, as in the overlay) and tips: at every zoom
        self._rings(c, self._junc, off)
        self._xmarks(c, self._cross_now, off)
        self._dots(c, self._tips, off)
        self._draw_focus(); self._draw_minimap()
        if self.tool.get() == 'select':
            self._draw_hover()
        # selection / cut preview
        for edges, col in ((self._sel_edges(), COL_SEL), (self._edges_of(self._hover_cut, self._hover_anchor), COL_CUT)):
            for u, v in edges:
                q = self._chain_xy((u, v))
                if len(q) == 2:
                    c.create_line(*self._i2c(*q[0]), *self._i2c(*q[1]), fill=col, width=w + 3, capstyle='round')
        if self.tool.get() == 'erase' and self._brush is not None:
            self._move_brush(None)
        # pending drawing
        if self.pend and self.pend['pts']:
            coords = [q for p in self.pend['pts'] for q in self._i2c(*p)]
            if len(coords) >= 4:
                c.create_line(*coords, fill=COL_PEND, width=w + 1, capstyle='round', joinstyle='round')
            for p in self.pend['clicks']:
                x, y = self._i2c(*p)
                c.create_oval(x - 4, y - 4, x + 4, y + 4, outline=COL_PEND, width=2)
        self._draw_candidate()

    def _draw_candidate(self):
        """the cyan 'next click draws this' line; redrawn alone on every mouse move."""
        c = self.canvas; c.delete('cand')
        if not (self.pend and self.pend['pts']) or self._target is None:
            return
        w = self._lw()
        tx, ty = self._i2c(*self._target)
        if self._preview is not None and self._preview[0] == self._prev_key():
            q = np.array(self._preview[1], float)
            coords = ((q - np.array([self.ox, self.oy])) * self.scale).ravel().tolist()
            if len(coords) >= 4:
                c.create_line(*coords, fill=COL_CAND, width=w, capstyle='round', joinstyle='round', tags='cand')
        else:
            lx, ly = self._i2c(*self.pend['pts'][-1])
            c.create_line(lx, ly, tx, ty, fill=COL_CAND, dash=(4, 4), tags='cand')
        c.create_oval(tx - 5, ty - 5, tx + 5, ty + 5, outline=COL_CAND, width=2, tags='cand')

    def _edges_of(self, nodes, anchor=None):
        a = self.arbor
        out = [(u, v) for u in nodes for v in a.adj.get(u, ()) if v in nodes and u < v]
        if anchor is not None:
            out += [(anchor, v) for v in a.adj.get(anchor, ()) if v in nodes]
        return out

    def _sel_edges(self):
        if not self.sel:
            return []
        if self.sel_label == 'segment':
            ch = self._sel_chain
            return [(u, v) for u, v in zip(ch[:-1], ch[1:]) if v in self.arbor.adj.get(u, ())]
        return self._edges_of(self.sel, self._sel_anchor)

    # ---------- hit testing ----------
    def _hit_seg(self, e):
        if self._kd is None:
            return None
        ix, iy = self._c2i(e.x, e.y)
        d, k = self._kd.query((ix, iy))
        return int(self._kd_seg[k]) if d * self.scale <= self.HIT else None

    def _hit_node(self, e, radius=None):
        if self._kd_nodes is None:
            return None
        ix, iy = self._c2i(e.x, e.y)
        d, k = self._kd_nodes.query((ix, iy))
        return int(self._kd_ids[k]) if d * self.scale <= (radius or self.HIT) else None

    def _in_soma(self, ix, iy):
        return self.arbor.inside_soma(ix, iy)

    # ---------- pointer ----------
    def _tool_changed(self):
        if self.tool.get() != 'draw' and self.pend:
            self.finish_draw()
        self.sel = set(); self._hover_cut = set(); self._brush = None; self._hover = None
        self.canvas.config(cursor={'draw': 'crosshair', 'cut': 'crosshair', 'erase': 'circle'}.get(self.tool.get(), 'arrow'))
        self._hint(); self._render()

    def _key(self, e):
        if isinstance(e.widget, (tk.Entry, ttk.Entry, tk.Text)):
            return
        k = e.keysym.lower()
        if e.state & 0x4:                         # Ctrl shortcuts are bound separately
            return
        if k == 'd':                              # draw, starting from the line under the mouse if any
            self.start_draw(start=self._hover[1] if self._hover and self._hover[0] == 'h' else None)
        elif k in ('c', 'e'):                     # older single-purpose modes, kept as shortcuts
            self.tool.set({'c': 'cut', 'e': 'erase'}[k]); self._tool_changed()
        elif k == 's':
            self.tool.set('select'); self._tool_changed()
        elif k in ('n', 'p'):
            self.next_check(1 if k == 'n' else -1)
        elif k == 'j':
            self.join_current_candidate()
        elif k in ('bracketleft', 'bracketright'):
            self.erase_r.set(min(60, max(4, self.erase_r.get() * (1.25 if k == 'bracketright' else 0.8))))
            self._move_brush(None)
            self.erase_lbl.config(text="eraser %d px" % self.erase_r.get()); self.after(1500, lambda: self.erase_lbl.config(text=""))
        elif k == 'f':
            self.follow.set(not self.follow.get())
        elif k in ('delete',):
            self.delete_selection()
        elif k == 'backspace':
            if self.pend:
                self.undo_point()
            else:
                self.delete_selection()
        elif k == 'escape':
            if self._erase is not None:           # cancel an eraser stroke before it is applied
                self._erase = None; self._press = None
            self.pend = None; self.sel = set(); self.canvas.delete('moving')
            self.tool.set('select'); self._tool_changed()
        elif k in ('return', 'kp_enter'):
            self.finish_draw()

    def _click_mark(self, e):
        """a small cross where the click landed, for 0.7 s (reviewers reported clicks landing 'off')."""
        c = self.canvas; c.delete('clickmark'); r = 6
        c.create_line(e.x - r, e.y, e.x + r, e.y, fill='#ffffff', width=1, tags='clickmark')
        c.create_line(e.x, e.y - r, e.x, e.y + r, fill='#ffffff', width=1, tags='clickmark')
        self.after(700, lambda: c.delete('clickmark'))

    def _b1(self, e):
        if self.arbor is None:
            return
        self.canvas.focus_set(); self._click_mark(e)
        t = self.tool.get()
        if self._mini_box is not None:            # the corner map: jump there
            x0, y0, mw, mh = self._mini_box
            if x0 <= e.x <= x0 + mw and y0 <= e.y <= y0 + mh:
                self._center_on(((e.x - x0) / mw * self.W, (e.y - y0) / mh * self.H), keep_zoom=True); return
        if t == 'select':
            # one mode: what happens depends on where the button goes down and whether the mouse then moves
            h = self._hit_handle(e)
            k = self._hit_seg(e) if h is None else None
            self._press = dict(x=e.x, y=e.y, handle=h, seg=k, mode=None, shift=bool(e.state & 0x1))
        elif t == 'cut':
            n = self._hit_node(e)
            if n is None or n == self.arbor.soma:
                return
            if not self.arbor.descendants(n):
                self._hint("That is already the end of the line."); return
            self._push()
            um = self.arbor.cut_beyond(n)
            self._log('cut', 'trim', um, self.arbor.xy[n])
            self._hover_cut = set(); self._rebuild()
            self._hint("Removed %.1f µm beyond that point. Ctrl+Z to undo." % um)
        elif t == 'draw':
            self._draw_click(e)
        elif t == 'erase':
            self._erase = dict(nodes=set(), edges=set())
            self._erase_at(e)

    # ---------- one-mode interaction ----------
    def _hit_handle(self, e):
        """a branch point or tip under the mouse (these can be dragged)."""
        if self._kd_handles is None:
            return None
        ix, iy = self._c2i(e.x, e.y)
        d, k = self._kd_handles.query((ix, iy))
        return int(self._handle_ids[k]) if d * self.scale <= self.HIT else None

    def _draw_hover(self):
        c = self.canvas; c.delete('hover')
        if not self._hover or self.arbor is None:
            return
        a = self.arbor; w = self._lw()
        if self._hover[0] == 'h':
            x, y = self._i2c(*a.xy[self._hover[1]]); r = 8
            c.create_oval(x - r, y - r, x + r, y + r, outline=COL_SEL, width=2, tags='hover')
        else:
            q = np.array(self._chain_xy(self._segs[self._hover[1]]), float)
            coords = ((q - np.array([self.ox, self.oy])) * self.scale).ravel().tolist()
            if len(coords) >= 4:
                c.create_line(*coords, fill=COL_SEL, width=w + 1, capstyle='round', joinstyle='round', tags='hover')

    def _nearest_on(self, e, chain):
        ix, iy = self._c2i(e.x, e.y)
        return min(chain, key=lambda n: (self.arbor.xy[n][0] - ix) ** 2 + (self.arbor.xy[n][1] - iy) ** 2)

    def _click(self, e, p):
        """a click without dragging: select what is under the mouse and offer what can be done there."""
        a = self.arbor
        if p['handle'] is not None:
            n = p['handle']
            self.sel = set(); self._render()
            items = []
            if len(a.adj[n]) == 1:
                q, ok = self._join_candidate(n)
                if q is not None and ok:
                    items += [("Join to the line next to it", lambda: self._join(n, q))]
                elif q is not None:
                    items += [("(Touches its own branch: joining would make a loop)", None)]
                items += [("Extend from here", lambda: self.start_draw(start=n)),
                          ("Delete this piece", lambda: self._delete_seg_of(n))]
            else:
                items += [("Draw a new branch from here", lambda: self.start_draw(start=n))]
            if a.descendants(n):
                items += [("Cut here: remove everything beyond this point", lambda: self._cut_at(n))]
            j = self._crossing_near(n)
            if j is not None:
                items += [None, ("This is a crossing, not a branch", lambda: self._do_crossing(j))]
            items += [None, ("Tip: drag the point to move it", None)]
            self._hint("Point selected. Drag it to move it, or choose from the menu.")
            self._menu(e, items)
            return
        if p['seg'] is None:
            self.sel = set(); self._render(); self._hint(); return
        seg = a.oriented(self._segs[p['seg']])
        n = self._nearest_on(e, seg)
        if p['shift']:
            self._select_subtree(seg)
        else:
            self.sel = set(seg); self.sel_label = 'segment'; self._sel_chain = seg
        self._hint("Selected %.1f µm. Delete removes it. Shift+click selects the whole branch beyond." %
                   a.subset_um(self.sel))
        self._render()
        items = [("Delete this piece", self.delete_selection),
                 ("Delete this branch and everything beyond it", lambda: (self._select_subtree(seg), self.delete_selection())),
                 ("Cut here: remove everything beyond this point", lambda: self._cut_at(n)),
                 ("Draw a new branch from here", lambda: self.start_draw(start=n))]
        j = self._crossing_near(n)
        if j is not None:
            items += [None, ("This is a crossing, not a branch", lambda: self._do_crossing(j))]
        self._menu(e, items)

    def _menu(self, e, items):
        if not self.use_menu:
            return
        m = tk.Menu(self, tearoff=0)
        for it in items:
            if it is None:
                m.add_separator()
            elif it[1] is None:
                m.add_command(label=it[0], state='disabled')
            else:
                m.add_command(label=it[0], command=it[1])
        self._last_menu = m
        x, y = getattr(e, 'x_root', None), getattr(e, 'y_root', None)
        if x is None:
            x, y = self.canvas.winfo_rootx() + e.x, self.canvas.winfo_rooty() + e.y
        self.after_idle(lambda: m.tk_popup(x, y))

    def _select_subtree(self, seg):
        start = seg[1]
        self.sel = set([start] + self.arbor.descendants(start))
        self.sel_label = 'subtree'; self._sel_anchor = seg[0]

    def _delete_seg_of(self, n):
        seg = next((sg for sg in self._segs if n in (sg[0], sg[-1])), None)
        if seg is not None:
            seg = self.arbor.oriented(seg)
            self.sel = set(seg); self.sel_label = 'segment'; self._sel_chain = seg
            self.delete_selection()

    def _cut_at(self, n):
        if n == self.arbor.soma or not self.arbor.descendants(n):
            self._hint("Nothing beyond that point."); return
        self._push()
        at = self.arbor.xy[n]
        um = self.arbor.cut_beyond(n)
        self._log('cut', 'trim', um, at)
        self._rebuild()
        self._hint("Removed %.1f µm beyond that point. Ctrl+Z to undo." % um)

    def _crossing_near(self, n):
        """a branch point near n that looks like a crossing (4-way, or a double T)."""
        a = self.arbor
        if self._kd_nodes is None:
            return None
        cands = [int(self._kd_ids[k]) for k in self._kd_nodes.query_ball_point(a.xy[n], 12)]
        for m in sorted(cands, key=lambda m: (a.xy[m][0] - a.xy[n][0]) ** 2 + (a.xy[m][1] - a.xy[n][1]) ** 2):
            if len(a.adj.get(m, ())) >= 3 and a.crossing_cluster(m) is not None:
                return m
        return None

    def _do_crossing(self, j):
        self._push()
        at = self.arbor.xy[j]
        new = self.arbor.split_crossing(j)
        if new is None:
            self.undo_stack.pop(); self._hint("That point does not look like a crossing."); return
        self._log('crossing', 'crossing', 0.0, at)
        self._rebuild()
        self._hint("Made it a crossing. The line that is not joined here is now orange: connect it where it "
                   "really starts (draw from a red line to it), or erase it. Ctrl+Z to undo.")

    def _move_preview(self, n, e):
        c = self.canvas; c.delete('moving')
        a = self.arbor
        x, y = e.x, e.y
        for v in a.adj.get(n, ()):
            c.create_line(*self._i2c(*a.xy[v]), x, y, fill=COL_SEL, width=self._lw() + 1, tags='moving')
        c.create_oval(x - 6, y - 6, x + 6, y + 6, outline=COL_SEL, width=2, tags='moving')

    def _join_candidate(self, t, at=None, radius=None):
        """(node, joinable) of another line next to free end t (or next to point `at`): not t's own
        segment, and, on the same piece, only where going round by the tree is much longer (a real gap).
        joinable is False when it is the same piece (joining would make a loop). (None, False) if nothing."""
        a = self.arbor
        if self._kd_nodes is None or t not in a.adj:
            return None, False
        p = a.xy[t] if at is None else at
        r = radius if radius is not None else max(4.0, (self.HIT + 6) / self.scale)
        own = set(next((sg for sg in self._segs if t in (sg[0], sg[-1])), (t,)))
        comp = a.component_of()
        reach = None; best = None
        for k in self._kd_nodes.query_ball_point(p, r):
            q = int(self._kd_ids[k])
            if q in own or q == t:
                continue
            d = math.hypot(a.xy[q][0] - p[0], a.xy[q][1] - p[1])
            same = comp.get(q) == comp.get(t)
            if same:
                if reach is None:
                    reach = a._reach(t, 4 * r + 40)
                if reach.get(q, 1e18) <= 4 * d + 40:
                    continue                          # just further along the same branch
            score = (same, d)                         # prefer another piece, then the nearest
            if best is None or score < best[0]:
                best = (score, q)
        if best is None:
            return None, False
        return best[1], not best[0][0]

    def _join(self, t, q):
        a = self.arbor
        self._push()
        um = math.hypot(a.xy[t][0] - a.xy[q][0], a.xy[t][1] - a.xy[q][1]) * a.vox
        if not a.join(t, q):
            self.undo_stack.pop()
            self._hint("Not joined: this end and that line are already on the same piece, so joining them "
                       "would make a loop."); return
        self._log('draw', 'bridge', um, a.xy[t])
        self._rebuild()
        self._hint("Joined: that end is now connected to the line. Ctrl+Z to undo.")

    def _finish_move(self, n, e):
        self.canvas.delete('moving')
        ix, iy = self._c2i(e.x, e.y)
        a = self.arbor
        q = None
        if len(a.adj.get(n, ())) == 1:            # a free end dropped onto another line: join it there
            q, ok = self._join_candidate(n, at=(ix, iy), radius=self.HIT / self.scale)
            if q is not None and ok:
                self._push()
                a.move_node(n, ix, iy)
                um = math.hypot(ix - a.xy[q][0], iy - a.xy[q][1]) * a.vox
                a.join(n, q)
                self._log('draw', 'bridge', um, (ix, iy))
                self._rebuild()
                self._hint("Dropped on a line: joined. Ctrl+Z to undo."); return
            if q is not None:
                self._hint("Moved, but not joined: that line is on the same piece (it would make a loop).")
        self._push()
        a.move_node(n, ix, iy)
        self._log('move', 'move', 0.0, (ix, iy))
        self._rebuild()
        if q is None:
            self._hint("Point moved. Ctrl+Z to undo.")

    def start_draw(self, start=None):
        """draw mode: from a given point, or wait for the first click."""
        if self.arbor is None:
            return
        self.sel = set(); self.tool.set('draw'); self._hover = None; self.canvas.delete('hover')
        self.canvas.config(cursor='crosshair')
        if start is not None:
            p0 = self.arbor.xy[start]
            self.pend = dict(start=start, clicks=[p0], pts=[p0], marks=[1])
            self._target = None; self._preview = None
            self._hint("Drawing: click along the neurite. The cyan line shows what the next click draws. "
                       "Enter or double-click finishes, Backspace removes a point, Esc cancels.")
        else:
            self._hint("Click on a line or on the soma to start the new line (or anywhere for a separate piece).")
        self._render()

    # ---------- the To-check list ----------
    def _continues(self, n, thr=0.10):
        """the image stays bright beyond tip n (on Liz's cells this flagged real problems 80% of the time)."""
        a = self.arbor
        back, prev = n, None
        for _ in range(8):
            nxt = [v for v in a.adj[back] if v != prev]
            if len(nxt) != 1:
                break
            prev, back = back, nxt[0]
        tx, ty = a.xy[n]; vx, vy = tx - a.xy[back][0], ty - a.xy[back][1]; vn = math.hypot(vx, vy) or 1.0
        ux, uy = vx / vn, vy / vn
        g = self.gray

        def mean_along(dx, dy):
            vals = [g[int(round(ty + dy * s_)), int(round(tx + dx * s_))] for s_ in range(4, 31, 2)
                    if 0 <= int(round(tx + dx * s_)) < self.W and 0 <= int(round(ty + dy * s_)) < self.H]
            return float(np.mean(vals)) if vals else 0.0
        side = 0.5 * (mean_along(ux * 0.5 - uy * 0.866, uy * 0.5 + ux * 0.866) +
                      mean_along(ux * 0.5 + uy * 0.866, uy * 0.5 - ux * 0.866))
        return mean_along(ux, uy) - side >= thr

    def _fill_check(self):
        t = self.ctree; self._check_sync = True
        t.delete(*t.get_children())
        for i, it in enumerate(self.issues):
            done = it['key'] in self.checked
            t.insert('', 'end', iid=str(i), text="%s %d. %s · %s" % ("•" if done else "  ", i + 1, it['label'], it.get('clock', '')),
                     values=(it.get('where', ''),), tags=(('done',) if done else (('hint',) if it.get('hint') else ())))
        if self.issue_idx is not None and self.issue_idx < len(self.issues):
            t.selection_set(str(self.issue_idx)); t.see(str(self.issue_idx))
        self.after_idle(lambda: setattr(self, '_check_sync', False))   # the select event arrives later
        n = len(self.issues); k = sum(1 for it in self.issues if it['key'] in self.checked)
        self.check_lbl.config(text="%d of %d looked at.  One walk round the cell, tip by tip.  N = next." % (k, n))
        self.check_bar['value'] = (k / n) if n else 0

    def _on_check_select(self, e):
        if self._check_sync:
            return
        s = self.ctree.selection()
        if s and int(s[0]) != self.issue_idx:       # setting the selection from code fires this event too
            self.go_check(int(s[0]))

    def go_check(self, i):
        if not self.issues:
            return
        self.issue_idx = i % len(self.issues)
        it = self.issues[self.issue_idx]
        cw = max(self._ref().winfo_width(), 50)
        self.scale = max(self.scale, min(2.0, cw / (60 / self.arbor.vox)))     # about 60 um across the view
        self._center_on((it['x'], it['y']), keep_zoom=True)
        if self.ctree.selection() != (str(self.issue_idx),):
            self.ctree.selection_set(str(self.issue_idx))
        self.ctree.see(str(self.issue_idx))
        self._hint("%d/%d  %s%s.  Fix it, or press 'Looks fine'. N = next, P = previous." %
                   (self.issue_idx + 1, len(self.issues), it['label'], (": " + it['note']) if it.get('note') else ""))

    def next_check(self, d=1):
        if not self.issues:
            self._hint("Nothing to check in this cell."); return
        if self.issue_idx is not None and d > 0:
            self.checked.add(self.issues[self.issue_idx]['key'])
        start = -1 if self.issue_idx is None else self.issue_idx
        n = len(self.issues)
        for step in range(1, n + 1):
            j = (start + d * step) % n
            if self.issues[j]['key'] not in self.checked or d < 0:
                self._fill_check(); self.go_check(j); return
        self._fill_check(); self._hint("Every spot in the list is checked. Save when you are done.")

    def mark_fine(self):
        if self.issue_idx is None:
            self.next_check(1); return
        self.checked.add(self.issues[self.issue_idx]['key']); self.dirty = True
        self.next_check(1)

    # ---------- eraser ----------
    def _move_brush(self, e):
        """the eraser circle follows the mouse without redrawing everything."""
        c = self.canvas
        if e is not None:
            self._brush = (e.x, e.y)
        if self.tool.get() != 'erase' or self._brush is None:
            c.delete('brush'); return
        r = float(self.erase_r.get()); x, y = self._brush
        if c.find_withtag('brush'):
            c.coords('brush', x - r, y - r, x + r, y + r)
        else:
            c.create_oval(x - r, y - r, x + r, y + r, outline=COL_CUT, width=1, dash=(3, 2), tags='brush')

    def _erase_at(self, e):
        """mark what lies under the eraser (applied on release, so dragging stays fast)."""
        a = self.arbor
        if self._erase is None or self._kd_nodes is None:
            return
        ix, iy = self._c2i(e.x, e.y); r = float(self.erase_r.get()) / self.scale
        hit = {int(self._kd_ids[k]) for k in self._kd_nodes.query_ball_point((ix, iy), r)} - {a.soma}
        new_edges = []
        for k in self._kd_nodes.query_ball_point((ix, iy), r + self._max_edge):
            u = int(self._kd_ids[k])
            for v in a.adj.get(u, ()):
                if u < v and (u in hit or v in hit or self._seg_dist((ix, iy), a.xy[u], a.xy[v]) <= r):
                    if (u, v) not in self._erase['edges']:
                        new_edges.append((u, v))
        self._erase['nodes'] |= hit
        self._erase['edges'] |= set(new_edges)
        w = self._lw() + 3
        for u, v in new_edges:                         # paint just the new pink bits
            q = self._chain_xy((u, v))
            if len(q) == 2:
                self.canvas.create_line(*self._i2c(*q[0]), *self._i2c(*q[1]), fill=COL_CUT, width=w,
                                        capstyle='round', tags='erasing')
        self.canvas.tag_raise('brush')

    @staticmethod
    def _seg_dist(p, a, b):
        (px_, py_), (ax, ay), (bx, by) = p, a, b
        dx, dy = bx - ax, by - ay; L = dx * dx + dy * dy
        t = 0.0 if L == 0 else max(0.0, min(1.0, ((px_ - ax) * dx + (py_ - ay) * dy) / L))
        return math.hypot(px_ - ax - t * dx, py_ - ay - t * dy)

    def _b1_drag(self, e):
        if self.tool.get() == 'erase' and self._erase is not None:
            self._move_brush(e); self._erase_at(e); return
        p = self._press
        if p is None or self.tool.get() != 'select':
            return
        if p['mode'] is None:
            if abs(e.x - p['x']) + abs(e.y - p['y']) < 5:
                return                            # still a click
            E = type('E', (), {})
            start = E(); start.x, start.y = p['x'], p['y']
            if p['handle'] is not None:
                p['mode'] = 'move'
            elif p['seg'] is not None:            # dragging along a line rubs it out
                p['mode'] = 'erase'; self._erase = dict(nodes=set(), edges=set()); self._erase_at(start)
            else:
                p['mode'] = 'pan'; self._pan_press(start)
        if p['mode'] == 'move':
            self._move_preview(p['handle'], e)
        elif p['mode'] == 'erase':
            self._erase_at(e)
        else:
            self._pan_move(e)

    def _b1_up(self, e):
        if self.tool.get() == 'select':
            p, self._press = self._press, None
            if p is None:
                return
            if p['mode'] is None:
                self._click(e, p)
            elif p['mode'] == 'move':
                self._finish_move(p['handle'], e)
            elif p['mode'] == 'erase':
                self._erase_finish()
            else:
                self._pan_release(e)
            return
        if self.tool.get() == 'erase' and self._erase is not None:
            self._erase_finish()

    def _erase_finish(self):
        er, self._erase = self._erase, None
        if er is None:
            return
        if not er['edges'] and not er['nodes']:
            self._render(); return
        self._push()
        a = self.arbor
        at = a.xy[next(iter(er['nodes']))] if er['nodes'] else a.xy[next(iter(er['edges']))[0]]
        um = a.erase(er['nodes'], er['edges'])
        self._log('erase', 'erase', um, at)
        self._rebuild()
        self._hint("Erased %.1f µm. Ctrl+Z to undo." % um +
                   ("  Pieces cut off from the soma are now orange: erase them too, or draw to reconnect."
                    if any(n not in self._connected for n in a.adj) else ""))

    def _motion(self, e):
        if self.arbor is None:
            return
        ix, iy = self._c2i(e.x, e.y)
        self.pos_lbl.config(text="x %d  y %d px" % (ix, iy))       # helps a reviewer report where clicks land
        t = self.tool.get()
        if t == 'select':
            h = self._hit_handle(e)
            k = None if h is not None else self._hit_seg(e)
            key = ('h', h) if h is not None else (('s', k) if k is not None else None)
            if key != self._hover:
                self._hover = key; self._draw_hover()
                self.canvas.config(cursor='fleur' if h is not None else ('hand2' if k is not None else 'arrow'))
            return
        if t == 'erase':
            self._move_brush(e); return
        if t == 'cut':
            n = self._hit_node(e)
            new = set(self.arbor.descendants(n)) if n not in (None, self.arbor.soma) else set()
            if new != self._hover_cut:
                self._hover_cut = new; self._hover_anchor = n if new else None; self._render()
        elif t == 'draw' and self.pend:
            self._target = self._click_target(e)[0]
            if self._prev_job:
                self.after_cancel(self._prev_job)
            self._prev_job = self.after(90, self._compute_preview)
            self._draw_candidate()

    # ---------- drawing ----------
    # ---------- display range (Fiji B&C) ----------
    def _schedule_bc(self):
        if self._bc_job:
            self.after_cancel(self._bc_job)
        self._bc_job = self.after(60, self._apply_bc)

    def reset_bc(self):
        self.vmin.set(0.0); self.vmax.set(255.0); self._apply_bc()

    def _apply_bc(self, render=True):
        self._bc_job = None
        if getattr(self, 'gray', None) is None:
            return
        lo, hi = float(self.vmin.get()), float(self.vmax.get())
        if hi <= lo + 1:
            hi = lo + 1
        self.disp = np.clip((self.gray * 255.0 - lo) / (hi - lo), 0, 1).astype(np.float32)
        self.base_img = Image.fromarray((self.disp * 255).astype(np.uint8)).convert('RGB')
        self._pyr = {2: self.base_img.reduce(2), 4: self.base_img.reduce(4)}
        self._tile_cache = None
        self.cost = None; self._cost2 = None; self._preview = None
        if render:
            self._render()

    def _snap_xy(self, ix, iy):
        """brightest displayed pixel within SNAP_PX (SNT cursor auto-snapping)."""
        if not self.snap.get() or self.disp is None:
            return ix, iy
        x0, x1 = max(0, int(ix) - SNAP_PX), min(self.W, int(ix) + SNAP_PX + 1)
        y0, y1 = max(0, int(iy) - SNAP_PX), min(self.H, int(iy) + SNAP_PX + 1)
        if x1 <= x0 or y1 <= y0:
            return ix, iy
        win = self.disp[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = (xx - ix) ** 2 + (yy - iy) ** 2
        win = np.where(d2 <= SNAP_PX ** 2, win, -1)
        here = self.disp[min(max(int(round(iy)), 0), self.H - 1), min(max(int(round(ix)), 0), self.W - 1)]
        top = float(win.max())
        if top - here < 0.02:                     # nothing clearly brighter nearby: stay where clicked
            return ix, iy
        cand = np.where(win >= top - 1e-6, d2, np.inf)   # among the brightest, the closest
        k = int(np.argmin(cand))
        return float(x0 + k % win.shape[1]), float(y0 + k // win.shape[1])

    def _click_target(self, e):
        """(point, node): a click near an existing line targets that node, otherwise the snapped pixel."""
        a = self.arbor
        node = self._hit_node(e, radius=self.HIT + 3)
        if node is not None and self.pend and node != self.pend['start']:
            return a.xy[node], node
        ix, iy = self._c2i(e.x, e.y)
        return self._snap_xy(ix, iy), None

    def _prev_key(self):
        if not self.pend or self._target is None:
            return None
        (ax, ay), (bx, by) = self.pend['pts'][-1], self._target
        return (round(ax, 1), round(ay, 1), round(bx, 1), round(by, 1), self.follow.get())

    def _compute_preview(self):
        self._prev_job = None
        key = self._prev_key()
        if key is None:
            return
        self._preview = (key, self._route(self.pend['pts'][-1], self._target, fast=True))
        self._draw_candidate()

    def _route(self, a, b, fast=False):
        """image path from a to b (x, y px): least-cost through bright displayed pixels, or straight.
        fast: long previews are routed at half resolution so the candidate keeps up with the mouse."""
        (x0, y0), (x1, y1) = a, b
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        straight = [(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t) for t in np.linspace(0, 1, max(n, 2))]
        if not self.follow.get():
            return straight
        try:
            if self.cost is None:
                self.cost = 1.0 / (self.disp + 0.05)           # SNT 'Reciprocal' on the displayed range
            dist = math.hypot(x1 - x0, y1 - y0)
            if fast and dist > 300:
                if self._cost2 is None:
                    self._cost2 = self.cost[::2, ::2].copy()
                half = self._route_on(self._cost2, (x0 / 2, y0 / 2), (x1 / 2, y1 / 2))
                return [(x * 2, y * 2) for x, y in half]
            return self._route_on(self.cost, a, b)
        except Exception:
            return straight

    @staticmethod
    def _route_on(cost, a, b):
        """least-cost path on a crop around a and b (margin grows with the distance)."""
        from skimage.graph import route_through_array
        (x0, y0), (x1, y1) = a, b
        H, W = cost.shape
        m = int(max(30, 0.5 * math.hypot(x1 - x0, y1 - y0)))
        c0, c1 = max(0, int(min(x0, x1)) - m), min(W, int(max(x0, x1)) + m + 1)
        r0, r1 = max(0, int(min(y0, y1)) - m), min(H, int(max(y0, y1)) + m + 1)
        sub = cost[r0:r1, c0:c1]
        s = (min(max(int(round(y0)) - r0, 0), sub.shape[0] - 1), min(max(int(round(x0)) - c0, 0), sub.shape[1] - 1))
        t = (min(max(int(round(y1)) - r0, 0), sub.shape[0] - 1), min(max(int(round(x1)) - c0, 0), sub.shape[1] - 1))
        idx, _ = route_through_array(sub, s, t, fully_connected=True, geometric=True)
        return [(float(c + c0), float(r + r0)) for r, c in idx]

    def _draw_click(self, e):
        a = self.arbor
        ix, iy = self._c2i(e.x, e.y)
        snap = self._hit_node(e, radius=self.HIT + 3)
        if self.pend is None:
            if snap is not None and snap != a.soma:
                start, p0 = snap, a.xy[snap]
            elif snap == a.soma or self._in_soma(ix, iy):
                start, p0 = a.soma, (ix, iy)                     # attached to the soma, but starts at the click (10/26: not at the centre)
            else:
                start, p0 = None, self._snap_xy(ix, iy)
            self.pend = dict(start=start, clicks=[p0], pts=[p0], marks=[1])
            self._target = None; self._preview = None
            self._hint("Drawing: keep clicking along the neurite. Click another line to join it, Enter or "
                       "double-click to finish." + ("" if start is not None else "  (Not attached: started away "
                                                                                     "from the tracing.)"))
            self._render(); return
        # a click on another piece (not the one we started from) ends the line there and joins it
        if snap is not None and snap != self.pend['start']:
            comp = a.component_of()
            if self.pend['start'] is None or comp.get(snap) != comp.get(self.pend['start']):
                seg = self._route(self.pend['pts'][-1], a.xy[snap], fast=True)
                self.pend['pts'] += seg[1:]; self.pend['clicks'].append(a.xy[snap])
                self.pend['marks'].append(len(self.pend['pts']))
                self.finish_draw(end=snap); return
        tgt = self._snap_xy(ix, iy)
        self._target = tgt
        key = self._prev_key()
        if self._preview is not None and self._preview[0] == key:
            seg = self._preview[1]                          # exactly the candidate the user just saw
        else:                                               # long jumps at half resolution: full res can take seconds
            seg = self._route(self.pend['pts'][-1], tgt, fast=True)
        self.pend['pts'] += seg[1:]; self.pend['clicks'].append(tgt); self.pend['marks'].append(len(self.pend['pts']))
        self._preview = None; self._target = None
        self._render()

    def undo_point(self):
        if not self.pend:
            return
        if len(self.pend['clicks']) <= 1:
            self.pend = None
        else:
            self.pend['clicks'].pop(); self.pend['marks'].pop()
            self.pend['pts'] = self.pend['pts'][:self.pend['marks'][-1]]
        self._render()

    def finish_draw(self, end=None):
        """end the line being drawn, then go back to the normal (one-mode) behaviour."""
        self._finish_draw(end)
        if self.tool.get() == 'draw':
            self.tool.set('select'); self.canvas.config(cursor='arrow')

    def _finish_draw(self, end=None):
        pend, self.pend = self.pend, None; self._cursor_img = None; self._target = None; self._preview = None
        if not pend or len(pend['clicks']) < 2:
            self._render(); return
        a = self.arbor
        pts = pend['pts']
        keep = [pts[0]]
        for p in pts[1:]:
            if math.hypot(p[0] - keep[-1][0], p[1] - keep[-1][1]) >= STEP_PX:
                keep.append(p)
        if keep[-1] != pts[-1]:
            keep.append(pts[-1])
        start = pend['start']
        body = keep if (start is None or start == a.soma) else keep[1:]   # from the soma the clicked point is the first node
        if end is not None:
            body = body[:-1] if len(body) > 1 else body
        if not body:
            self._render(); return
        was_tip = start is not None and len(a.adj.get(start, ())) == 1 and start != a.soma
        self._push()
        ids, joined = a.add_chain(body, a=start, b=end)
        um = a.chain_um(([start] if start is not None else []) + ids + ([end] if joined else []))
        if start is None:
            kind = 'bridge' if joined else 'fragment'
        elif joined:
            kind = 'bridge'
        elif start == a.soma:
            kind = 'primary'
        else:
            kind = 'extend' if was_tip else 'branch'
        self._log('draw', kind, um, pts[len(pts) // 2], nodes=ids)
        self._rebuild()
        msg = "Added %.1f µm (%s)." % (um, KINDS[kind][0])
        if end is not None and not joined:
            msg += "  Not joined at the end: that would close a loop."
        self._hint(msg + "  Ctrl+Z to undo.")

    # ---------- deleting ----------
    def delete_selection(self):
        if not self.sel or self.arbor is None:
            return
        a = self.arbor
        self._push()
        nodes = set(self.sel)
        connected_before = a.connected()
        at = a.xy[next(iter(nodes))]
        if self.sel_label == 'segment':
            seg = self._sel_chain
            free_end = len(a.adj[seg[-1]]) == 1 or len(a.adj[seg[0]]) == 1
            kind = 'piece' if seg[0] not in connected_before else ('spur' if free_end else 'internal')
            um = a.delete_chain(seg)
        else:
            um = a.delete_nodes(nodes)
            kind = 'subtree' if next(iter(nodes)) in connected_before else 'piece'
        self._log('delete', kind, um, at)
        self.sel = set(); self._rebuild()
        self._hint("Removed %.1f µm (%s). Ctrl+Z to undo." % (um, KINDS[kind][0]) +
                   ("  The part beyond is now loose (orange): draw a line to reconnect it, or delete it."
                    if kind == 'internal' else ""))

    def _on_path_select(self, e):
        s = self.ptree.selection()
        if not s:
            return
        nodes = self._path_nodes.get(s[0], [])
        if len(nodes) < 2:
            return
        first = nodes[1]
        self.sel = set([first] + self.arbor.descendants(first)); self.sel_label = 'subtree'; self._sel_anchor = nodes[0]
        self.tool.set('select'); self._hint("Path selected with its daughters (%.1f µm). Delete removes them." %
                                            self.arbor.subset_um(self.sel))
        self._center_on(self.arbor.xy[nodes[len(nodes) // 2]]); self._render()

    def _on_edit_select(self, e):
        s = self.etree.selection()
        if s:
            ed = self.edits[int(s[0])]
            self._center_on(ed['at'])

    def _center_on(self, p, keep_zoom=False):
        cw = max(self._ref().winfo_width(), 50); ch = max(self._ref().winfo_height(), 50)
        if self.scale < 1.0 and not keep_zoom:
            self.scale = 1.0
        self.ox = p[0] - cw / 2 / self.scale; self.oy = p[1] - ch / 2 / self.scale
        self._render()

    # ---------- history ----------
    def _push(self):
        self.undo_stack.append((self.arbor.snapshot(), [dict(e) for e in self.edits]))
        self.redo_stack.clear(); self.dirty = True

    def _log(self, op, kind, um, at, nodes=None):
        self.edits.append(dict(op=op, kind=kind, label=KINDS[kind][0], category=KINDS[kind][1], um=round(um, 2),
                               at=[round(float(at[0]), 1), round(float(at[1]), 1)],
                               time=datetime.datetime.now().isoformat(timespec='seconds')))

    def undo(self):
        if self.pend:
            self.undo_point(); return
        if not self.undo_stack:
            return
        self.redo_stack.append((self.arbor.snapshot(), [dict(e) for e in self.edits]))
        snap, self.edits = self.undo_stack.pop()
        self.arbor.restore(snap); self.sel = set(); self.dirty = True; self._rebuild(); self._hint("Undone.")

    def redo(self):
        if not self.redo_stack:
            return
        self.undo_stack.append((self.arbor.snapshot(), [dict(e) for e in self.edits]))
        snap, self.edits = self.redo_stack.pop()
        self.arbor.restore(snap); self.sel = set(); self.dirty = True; self._rebuild(); self._hint("Redone.")

    def revert(self):
        if self.cell is None:
            return
        if not messagebox.askyesno("Revert", "Go back to the software's tracing for %s? Your edits on screen are "
                                   "discarded (Ctrl+Z still brings them back)." % self.cell['name'], parent=self):
            return
        self._push()
        d, s = self.cell['dir'], self.cell['stem']
        fresh = Arbor.from_swc(os.path.join(d, s + ".swc"), self.base_metrics.get('soma_area'), self.soma_xy)
        self.arbor.restore(fresh.snapshot()); self.arbor.header = fresh.header; self.arbor.soma_poly = self.soma_poly
        self.edits = []; self._rebuild(); self._hint("Back to the software's tracing. Save to keep it that way.")

    # ---------- saving ----------
    def write_outputs(self):
        """write the edited SWC + metrics + Sholl + overlay + edit log next to the draft. Returns the paths."""
        a = self.arbor; d, s = self.cell['dir'], self.cell['stem']
        stamp = datetime.datetime.now().isoformat(timespec='seconds')
        swc = os.path.join(d, s + "_edited.swc")
        text, nid = a.to_swc("edited by hand in NeuronTracer (Edit tracing), %s, %d edits" % (stamp, len(self.edits)),
                             with_ids=True)
        with open(swc, 'w', encoding='utf-8') as f:
            f.write(text)
        m = a.metrics()
        base = self.base_metrics.get('total_length')
        total = (base + a.cable_um() - self.base_cable) if isinstance(base, float) else a.cable_um()
        vals = dict(self.base_metrics)                       # soma, width, pixel size: unchanged by editing
        mm = {'total_length': total, 'num_branch_points': m['num_branch_points'], 'num_tips': m['num_tips'],
              'num_branches': m['num_branches'], 'mean_intersections': m['mean_intersections'],
              'enclosing_radius': m['enclosing_radius_um'], 'polarity_index': m['polarity_index'],
              'highest_branch_order': m['highest_branch_order'],
              'highest_branch_order_centrifugal': m['highest_branch_order_centrifugal'],
              'avg_branch_angle': m['avg_branch_angle_deg'], 'avg_parent_daughter_angle': m['avg_parent_daughter_angle_deg'],
              'avg_length_before_first_bifurcation': m['avg_len_before_first_bif_um'],
              'avg_segment_length_before_bifurcation': m['avg_segment_len_um']}
        vals.update(mm)
        import csv
        rows = []
        try:
            with open(os.path.join(d, s + "_metrics.csv"), encoding='utf-8') as fh:
                rows = list(csv.DictReader(fh))
        except Exception:
            pass
        names = [r['metric'] for r in rows] or list(mm)
        met = os.path.join(d, s + "_edited_metrics.csv")
        with open(met, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f); w.writerow(["metric", "value", "unit", "note"])
            for nm in names:
                r = next((x for x in rows if x['metric'] == nm), {})
                v = vals.get(nm, r.get('value', ''))
                w.writerow([nm, round(v, 4) if isinstance(v, float) else v, r.get('unit', ''),
                            (r.get('note', '') + ("; " if r.get('note') else "") + "after hand edits")
                            if nm in mm else r.get('note', '')])
        sh = os.path.join(d, s + "_edited_sholl.csv")
        with open(sh, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f); w.writerow(["radius_um", "intersections"])
            for r_, n_ in zip(m['sholl_radii'], m['sholl_intersections']):
                w.writerow(["%.1f" % r_, int(n_)])
        ov = os.path.join(d, s + "_edited_overlay.png")
        self.render_overlay().save(ov)
        log = os.path.join(d, s + "_edits.json")
        with open(log, 'w', encoding='utf-8') as f:
            json.dump(dict(cell=self.cell['name'], source_swc=s + ".swc", saved=stamp, pixel_size_um=a.vox,
                           total_length_before_um=base, total_length_after_um=round(total, 2),
                           edits=self.edits, new_nodes=sorted(nid[n] for n in a.new if n in nid),
                           checked=sorted(self.checked)), f, indent=1, ensure_ascii=False)
        return swc, met, sh, ov, log

    def render_overlay(self):
        a = self.arbor
        img = (self.raw_img or self.base_img).copy(); d = ImageDraw.Draw(img)
        con = a.connected()
        for u in a.adj:
            for v in a.adj[u]:
                if u < v:
                    col = (61, 255, 122) if (u in a.new and v in a.new) else ((255, 40, 40) if u in con else (255, 166, 61))
                    d.line([a.xy[u], a.xy[v]], fill=col, width=2)
        return img

    def save(self, quiet=False):
        if self.arbor is None:
            return None
        try:
            out = self.write_outputs()
        except Exception as e:
            messagebox.showerror("Save failed", str(e), parent=self); return None
        self.dirty = False; self._refresh_cells(); self._update_len()
        self._hint("Saved: %s and the edited metrics, Sholl, overlay and edit log." % os.path.basename(out[0]))
        return out

    def _on_close(self):
        if self._ask_save():
            self.destroy()


def open_editor(master=None, folder=None):
    return Editor(master, folder=folder)
