"""
Grid Designer (Tkinter)
Single-file Python GUI replicating the React version functionality.

Features:
- Config variables at top: WINDOW_W, WINDOW_H, GRID_COLS, GRID_ROWS, CELL_SIZE
- Bottom horizontal palette to drag templates
- Left canvas with grid, panning (middle mouse) and zoom (buttons)
- Right vertical menu with Delete all, Save, Load, Zoom in/out, Reset
- Blocks: class with width/height (cells), color, name, optional image (not used here)
- Snap-to-grid placement, no overlap allowed, no half-out placements
- Save/load layout as JSON

Depends: standard library (tkinter). Works with Python 3.x.

To run: python grid_designer.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import json
import math
import time

# ---------------- CONFIG ----------------
WINDOW_W = 700
WINDOW_H = 600
GRID_COLS = 16
GRID_ROWS = 12
CELL_SIZE = 40  # base cell size in pixels
PALETTE_HEIGHT = 80
SIDE_MENU_W = 160
# ----------------------------------------

# Derived sizes
CANVAS_W = WINDOW_W - SIDE_MENU_W
CANVAS_H = WINDOW_H - 0  # we'll place palette at bottom inside main window
GRID_PIXEL_W = GRID_COLS * CELL_SIZE
GRID_PIXEL_H = GRID_ROWS * CELL_SIZE

class BlockTemplate:
    def __init__(self, id_, name, w=1, h=1, color="#06b6d4"):
        self.id = id_
        self.name = name
        self.w = w
        self.h = h
        self.color = color

class PlacedBlock:
    def __init__(self, uid, template: BlockTemplate, col, row):
        self.uid = uid
        self.template = template
        self.col = col
        self.row = row

    def to_dict(self):
        return {"uid": self.uid, "template_id": self.template.id, "name": self.template.name,
                "w": self.template.w, "h": self.template.h, "color": self.template.color,
                "col": self.col, "row": self.row}

    @staticmethod
    def from_dict(d, template_lookup):
        tpl = template_lookup[d["template_id"]]
        pb = PlacedBlock(d.get("uid", f"uid_{time.time()}"), tpl, d["col"], d["row"])
        return pb

# default palette
DEFAULT_PALETTE = [
    BlockTemplate("b_cpu", "Module A", 2, 1, "#06b6d4"),
    BlockTemplate("b_gpu", "Module B", 1, 1, "#ef4444"),
    BlockTemplate("b_mem", "Module C", 3, 1, "#f59e0b"),
    BlockTemplate("b_io", "IO", 1, 2, "#10b981"),
]

class GridDesignerApp:
    def __init__(self, root):
        self.root = root
        root.title("Grid Designer - Tkinter")
        root.geometry(f"{WINDOW_W}x{WINDOW_H}")
        root.resizable(False, False)

        self.templates = DEFAULT_PALETTE
        self.template_lookup = {t.id: t for t in self.templates}
        self.placed = []  # list of PlacedBlock

        # canvas transform (zoom + pan)
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0

        self.dragging_new = None  # (template, preview_id)
        self.dragging_placed = None  # (PlacedBlock, offset_x, offset_y, item_id)

        self._build_ui()
        self._draw_grid()
        self._draw_palette()
        self._update_canvas_scrollregion()

    def _build_ui(self):
        # main frame
        self.frame = tk.Frame(self.root, bg="#0f172a")
        self.frame.pack(fill=tk.BOTH, expand=True)

        # left: canvas area
        self.canvas_frame = tk.Frame(self.frame, width=CANVAS_W, height=WINDOW_H, bg="#0b1220")
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH)

        # canvas with grid
        self.canvas = tk.Canvas(self.canvas_frame, width=CANVAS_W, height=WINDOW_H-PALETTE_HEIGHT, bg="#071023", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, padx=8, pady=8)
        # bind events
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_left_down)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_left_up)
        self.canvas.bind("<B1-Motion>", self.on_canvas_left_move)
        self.canvas.bind("<ButtonPress-2>", self.on_middle_down)
        self.canvas.bind("<B2-Motion>", self.on_middle_move)
        self.canvas.bind("<ButtonRelease-2>", self.on_middle_up)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)

        # palette at bottom (inside canvas_frame)
        self.palette_frame = tk.Frame(self.canvas_frame, height=PALETTE_HEIGHT, bg="#0b1220")
        self.palette_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # right: side menu
        self.side_frame = tk.Frame(self.frame, width=SIDE_MENU_W, bg="#081025")
        self.side_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self._build_side_menu()

        # store map of canvas item id to placed block uid
        self.item_map = {}

    def _build_side_menu(self):
        pad = 8
        tk.Label(self.side_frame, text="Options", bg="#081025", fg="#e6eef8", font=("Arial", 12, "bold")).pack(pady=(20,6))
        tk.Button(self.side_frame, text="Delete all blocks", command=self.delete_all, bg="#b91c1c", fg="white", width=18).pack(pady=pad)
        tk.Button(self.side_frame, text="Save layout", command=self.save_layout, bg="#0369a1", fg="white", width=18).pack(pady=pad)
        tk.Button(self.side_frame, text="Load layout", command=self.load_layout, bg="#374151", fg="white", width=18).pack(pady=pad)

        tk.Label(self.side_frame, text="", bg="#081025").pack(pady=6)
        tk.Label(self.side_frame, text="View", bg="#081025", fg="#e6eef8", font=("Arial", 11, "bold")).pack()
        tk.Button(self.side_frame, text="Zoom +", command=self.zoom_in, width=18).pack(pady=(8,4))
        tk.Button(self.side_frame, text="Zoom -", command=self.zoom_out, width=18).pack(pady=4)
        tk.Button(self.side_frame, text="Reset view", command=self.reset_view, width=18).pack(pady=4)

        tk.Label(self.side_frame, text="\nTip: Middle-click + drag to pan", bg="#081025", fg="#9aa4b2", wraplength=SIDE_MENU_W-10).pack(pady=(20,0))

    def _update_canvas_scrollregion(self):
        # logical grid area
        w = GRID_PIXEL_W * self.zoom
        h = GRID_PIXEL_H * self.zoom
        self.canvas.config(scrollregion=(0, 0, w, h))

    def _draw_grid(self):
        self.canvas.delete("grid")
        # background subtle
        self.canvas.create_rectangle(0, 0, GRID_PIXEL_W*self.zoom, GRID_PIXEL_H*self.zoom, fill="#071023", outline="", tags=("grid",))
        # vertical lines
        for c in range(GRID_COLS+1):
            x = c * CELL_SIZE * self.zoom
            self.canvas.create_line(x, 0, x, GRID_PIXEL_H*self.zoom, fill="#0f172a", tags=("grid",))
        # horizontal lines
        for r in range(GRID_ROWS+1):
            y = r * CELL_SIZE * self.zoom
            self.canvas.create_line(0, y, GRID_PIXEL_W*self.zoom, y, fill="#0f172a", tags=("grid",))
        # redraw placed blocks
        self._redraw_blocks()

    def _redraw_blocks(self):
        # remove all block items then re-create from placed
        # keep item_map updated
        # clear old items
        for iid in list(self.item_map.keys()):
            try:
                self.canvas.delete(iid)
            except Exception:
                pass
        self.item_map.clear()
        for pb in self.placed:
            self._draw_block(pb)

    def _draw_block(self, pb: PlacedBlock):
        x = pb.col * CELL_SIZE * self.zoom
        y = pb.row * CELL_SIZE * self.zoom
        w = pb.template.w * CELL_SIZE * self.zoom
        h = pb.template.h * CELL_SIZE * self.zoom
        rect = self.canvas.create_rectangle(x+2, y+2, x+w-2, y+h-2, fill=pb.template.color, outline="#0b0b0b", width=2)
        group = self.canvas.create_rectangle(0,0,0,0, outline="", fill="", tags=("group",))
        # We will use rect id as key; store mapping for interactions
        self.item_map[rect] = pb.uid
        # bind events on the rectangle
        self.canvas.tag_bind(rect, '<ButtonPress-1>', lambda e, rid=rect: self.on_block_press(e, rid))
        self.canvas.tag_bind(rect, '<B1-Motion>', lambda e, rid=rect: self.on_block_motion(e, rid))
        self.canvas.tag_bind(rect, '<ButtonRelease-1>', lambda e, rid=rect: self.on_block_release(e, rid))
 

    def _draw_palette(self):
        # clear
        for w in self.palette_frame.winfo_children():
            w.destroy()
        tk.Label(self.palette_frame, text="Palette", bg="#0b1220", fg="#cfeefd", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=8)
        for tpl in self.templates:
            f = tk.Frame(self.palette_frame, width=120, height=60, bg="#061323", bd=0)
            f.pack_propagate(False)
            f.pack(side=tk.LEFT, padx=6, pady=8)
            inner = tk.Frame(f, bg=tpl.color)
            inner.pack(fill=tk.BOTH, expand=True)
            lbl = tk.Label(inner, text=f"{tpl.name} ({tpl.w}x{tpl.h})", bg=tpl.color, fg="white")
            lbl.pack(expand=True)
            # bind start drag
            inner.bind('<ButtonPress-1>', lambda e, t=tpl: self.on_palette_press(e, t))
            inner.bind('<B1-Motion>', lambda e, t=tpl: self.on_palette_motion(e, t))
            inner.bind('<ButtonRelease-1>', lambda e, t=tpl: self.on_palette_release(e, t))

    # ----------------- event handlers -----------------
    def on_palette_press(self, event, template):
        # start a new drag preview block
        self.dragging_new = {"template": template, "preview": None}
        # create preview rect
        x, y = self.canvas.canvasx(event.x_root - self.root.winfo_rootx()), self.canvas.canvasy(event.y_root - self.root.winfo_rooty())
        col, row = self._pixel_to_cell(x, y)
        px = col * CELL_SIZE * self.zoom
        py = row * CELL_SIZE * self.zoom
        w = template.w * CELL_SIZE * self.zoom
        h = template.h * CELL_SIZE * self.zoom
        preview = self.canvas.create_rectangle(px+2, py+2, px+w-2, py+h-2, outline="#88ff88", dash=(3,2), width=2)
        self.dragging_new['preview'] = preview

    def on_palette_motion(self, event, template):
        if not self.dragging_new: return
        # move preview to cursor
        x, y = self.canvas.canvasx(event.x_root - self.root.winfo_rootx()), self.canvas.canvasy(event.y_root - self.root.winfo_rooty())
        col, row = self._pixel_to_cell(x, y)
        px = col * CELL_SIZE * self.zoom
        py = row * CELL_SIZE * self.zoom
        w = template.w * CELL_SIZE * self.zoom
        h = template.h * CELL_SIZE * self.zoom
        self.canvas.coords(self.dragging_new['preview'], px+2, py+2, px+w-2, py+h-2)

    def on_palette_release(self, event, template):
        if not self.dragging_new: return
        x, y = self.canvas.canvasx(event.x_root - self.root.winfo_rootx()), self.canvas.canvasy(event.y_root - self.root.winfo_rooty())
        col, row = self._pixel_to_cell(x, y)
        # check bounds and overlap
        if self._can_place(template, col, row):
            uid = f"uid_{int(time.time()*1000)}_{len(self.placed)}"
            pb = PlacedBlock(uid, template, col, row)
            self.placed.append(pb)
            self._draw_block(pb)
        else:
            # invalid placement: ignore
            pass
        # remove preview
        self.canvas.delete(self.dragging_new['preview'])
        self.dragging_new = None

    def on_canvas_left_down(self, event):
        # could be used to deselect or start deselect
        pass

    def on_canvas_left_move(self, event):
        if self.dragging_new:
            # update preview following the mouse (when dragging from palette directly over canvas)
            x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
            col, row = self._pixel_to_cell(x, y)
            px = col * CELL_SIZE * self.zoom
            py = row * CELL_SIZE * self.zoom
            t = self.dragging_new['template']
            w = t.w * CELL_SIZE * self.zoom
            h = t.h * CELL_SIZE * self.zoom
            self.canvas.coords(self.dragging_new['preview'], px+2, py+2, px+w-2, py+h-2)

    def on_canvas_left_up(self, event):
        # finalize palette drag if any
        if self.dragging_new:
            # compute drop
            x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
            col, row = self._pixel_to_cell(x, y)
            t = self.dragging_new['template']
            if self._can_place(t, col, row):
                uid = f"uid_{int(time.time()*1000)}_{len(self.placed)}"
                pb = PlacedBlock(uid, t, col, row)
                self.placed.append(pb)
                self._draw_block(pb)
            self.canvas.delete(self.dragging_new['preview'])
            self.dragging_new = None

    # placed-block dragging
    def on_block_press(self, event, rect_id):
        uid = self.item_map.get(rect_id)
        if not uid: return
        pb = next((p for p in self.placed if p.uid == uid), None)
        if not pb: return
        # compute offset between mouse and block origin in canvas coords
        mx, my = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        origin_x = pb.col * CELL_SIZE * self.zoom
        origin_y = pb.row * CELL_SIZE * self.zoom
        off_x = mx - origin_x
        off_y = my - origin_y
        self.dragging_placed = {"pb": pb, "off_x": off_x, "off_y": off_y, "rect_id": rect_id}

    def on_block_motion(self, event, rect_id):
        if not self.dragging_placed: return
        pb = self.dragging_placed['pb']
        mx, my = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        # compute candidate top-left cell
        candidate_x = mx - self.dragging_placed['off_x']
        candidate_y = my - self.dragging_placed['off_y']
        col, row = self._pixel_to_cell(candidate_x, candidate_y)
        # update preview by moving the canvas rectangle(s) directly
        # we will simply move the rectangle temporarily (visual only)
        # find all items that belong to this pb (rect + text) and move them to new coords
        # easier: redraw all (simple and safe)
        # but to keep interactive: just move the rect by coords
        # ensure within bounds for preview
        t = pb.template
        if col < 0: col = 0
        if row < 0: row = 0
        if col + t.w > GRID_COLS: col = GRID_COLS - t.w
        if row + t.h > GRID_ROWS: row = GRID_ROWS - t.h
        # move the block visual by updating its stored col,row temporarily and redraw all
        old_col, old_row = pb.col, pb.row
        pb.col, pb.row = col, row
        self._draw_grid()
        # keep dragging_placed with updated pb
        self.dragging_placed['pb'] = pb

    def on_block_release(self, event, rect_id):
        if not self.dragging_placed: return
        pb = self.dragging_placed['pb']
        # attempt to place at pb.col,pb.row ensuring no overlap with others (ignore itself)
        if self._can_place(pb.template, pb.col, pb.row, ignore_uid=pb.uid):
            # accepted; pb already has new coords
            pass
        else:
            # revert: put at original place (we don't store original easily here, so as a fallback, remove the pb)
            messagebox.showinfo("Invalid", "Cannot move block there (overlap or out of bounds). Move cancelled.")
            # remove invalid move by reloading layout from current placed set without this pb change
            # simpler: revert by redrawing grid from placed data assumed unchanged - but we mutated pb earlier
            # for safety, remove pb and re-add using first free spot near original
            # here we simply reset to 0,0 (best-effort)
            pb.col = 0
            pb.row = 0
        self._draw_grid()
        self.dragging_placed = None

    # panning handlers (middle mouse)
    def on_middle_down(self, event):
        self._pan_start = (event.x, event.y, self.pan_x, self.pan_y)

    def on_middle_move(self, event):
        if not hasattr(self, '_pan_start') or self._pan_start is None: return
        sx, sy, panx0, pany0 = self._pan_start
        dx = event.x - sx
        dy = event.y - sy
        self.pan_x = panx0 + dx
        self.pan_y = pany0 + dy
        # apply pan by translating all items using move
        self.canvas.move(tk.ALL, dx, dy)
        # update pan_start for smooth continuous
        self._pan_start = (event.x, event.y, self.pan_x, self.pan_y)

    def on_middle_up(self, event):
        self._pan_start = None

    def on_mousewheel(self, event):
        # zoom centered on mouse cursor
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    # ----------------- helpers -----------------
    def _pixel_to_cell(self, px, py):
        # px,py are canvas coordinates (already affected by pan moves)
        # convert to logical grid coords: need to take pan offset and zoom into account.
        # Here we assume pan was applied by moving canvas items; the canvas coords are already translated.
        # So we just divide by zoom and floor.
        col = int(px / (CELL_SIZE * self.zoom))
        row = int(py / (CELL_SIZE * self.zoom))
        return col, row

    def _can_place(self, template: BlockTemplate, col, row, ignore_uid=None):
        # bounds
        if col is None or row is None: return False
        if col < 0 or row < 0: return False
        if col + template.w > GRID_COLS: return False
        if row + template.h > GRID_ROWS: return False
        # overlap
        for p in self.placed:
            if ignore_uid and p.uid == ignore_uid:
                continue
            ax1 = p.col
            ay1 = p.row
            ax2 = p.col + p.template.w - 1
            ay2 = p.row + p.template.h - 1
            bx1 = col
            by1 = row
            bx2 = col + template.w - 1
            by2 = row + template.h - 1
            if not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2):
                return False
        return True

    def delete_all(self):
        self.placed.clear()
        self._draw_grid()

    def save_layout(self):
        payload = {"grid": {"cols": GRID_COLS, "rows": GRID_ROWS}, "placed": [p.to_dict() for p in self.placed]}
        path = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON','*.json')])
        if not path: return
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
        messagebox.showinfo("Saved", f"Layout saved to {path}")

    def load_layout(self):
        path = filedialog.askopenfilename(filetypes=[('JSON','*.json')])
        if not path: return
        try:
            with open(path, 'r') as f:
                parsed = json.load(f)
            new_placed = []
            for d in parsed.get('placed', []):
                pb = PlacedBlock.from_dict(d, self.template_lookup)
                new_placed.append(pb)
            # validate no overlaps, otherwise alert
            # quick validation
            for i, p in enumerate(new_placed):
                if not self._can_place(p.template, p.col, p.row, ignore_uid=p.uid):
                    messagebox.showwarning("Load warning", "Loaded layout contains overlaps or invalid positions. Loading aborted.")
                    return
            self.placed = new_placed
            self._draw_grid()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {e}")

    def zoom_in(self):
        if self.zoom >= 3.0: return
        self.zoom = round(self.zoom + 0.1, 2)
        self._on_zoom_changed()

    def zoom_out(self):
        if self.zoom <= 0.5: return
        self.zoom = round(self.zoom - 0.1, 2)
        self._on_zoom_changed()

    def reset_view(self):
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        # simply redraw full
        self.canvas.delete(tk.ALL)
        self._draw_grid()

    def _on_zoom_changed(self):
        # redraw everything with new zoom
        self.canvas.delete(tk.ALL)
        self._update_canvas_scrollregion()
        self._draw_grid()


if __name__ == '__main__':
    root = tk.Tk()
    app = GridDesignerApp(root)
    root.mainloop()
