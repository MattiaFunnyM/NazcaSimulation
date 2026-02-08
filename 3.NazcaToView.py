# TODO: Visualize refractive index block 
    #: Make the plot 3D
    #: Make the plot looking good
 
# TODO: Simulation (entire block together)
# TODO: Present results
# TODO: Simulation (divide the simulation)
# GOALS: 
# - Waveguide HOR 10 um under 5 min
# - Waveguide VER 15 um under 4 min
# - Coupler 2x2 50 um under 10 min
# - Ring resonator 100 um under 10 min
import sys
import json
import socket
import threading
import numpy as np
import pyvista as pv
from shapely.ops import unary_union, triangulate
from shapely.geometry import Polygon

#############################
### PLOTTING INFORMATIONS ###
#############################

# Open the Technology File
with open("TechnologyExample.json", "r") as f:
    tech = json.load(f)

# Take the information of the colors
tech_layers = tech["layers"]

def merge_polygons_by_layer(polygons):
    layer_groups = {}

    for poly in polygons:
        layer = poly["layer"]
        layer_key = f"{layer['layer']}/{layer['datatype']}"

        pts = poly["points"]
        shp = Polygon(pts)

        if layer_key not in layer_groups:
            layer_groups[layer_key] = []
        layer_groups[layer_key].append(shp)

    # Merge touching polygons
    merged = {}
    for layer_key, polys in layer_groups.items():
        merged[layer_key] = unary_union(polys)

    return merged

def shapely_to_mesh(shp, bottom, height):
    meshes = []

    if shp.geom_type == "Polygon":
        meshes.append(extrude_polygon(list(shp.exterior.coords), bottom, height))

    elif shp.geom_type == "MultiPolygon":
        for geom in shp.geoms:
            meshes.append(extrude_polygon(list(geom.exterior.coords), bottom, height))

    return meshes


def extrude_polygon(pts, bottom, height):
    pts = np.array(pts[:-1])
    n = len(pts)

    # 3D points
    bottom_pts = np.c_[pts, np.full(n, bottom)]
    top_pts    = np.c_[pts, np.full(n, bottom + height)]

    all_pts = np.vstack([bottom_pts, top_pts])

    faces = []

    # ---- SIDE FACES (unchanged) ----
    for i in range(n):
        j = (i + 1) % n
        faces.append([
            4,
            i,
            j,
            n + j,
            n + i
        ])

    # ---- TRIANGULATED BOTTOM & TOP ----
    poly2d = Polygon(pts)
    tris = triangulate(poly2d)
    
    for tri in tris:
        if not poly2d.contains(tri.centroid):
            continue

        tri_pts = np.array(tri.exterior.coords)[:-1]

        idx = []
        for p in tri_pts:
            idx.append(np.where((pts == p).all(axis=1))[0][0])

        # bottom (CCW)
        faces.append([3, idx[0], idx[1], idx[2]])

        # top (reverse winding)
        faces.append([3, n + idx[2], n + idx[1], n + idx[0]])

    return pv.PolyData(all_pts, np.hstack(faces))


def plot_data(polygons):
    plotter = pv.Plotter(window_size=(1200, 800))
    used_layers = {}

    # Merge polygons per layer
    merged = merge_polygons_by_layer(polygons)

    for layer_key, shp in merged.items():
        if layer_key not in tech_layers:
            continue

        info = tech_layers[layer_key]
        color = info["color"]
        bottom = info["bottom"]
        height = info["height"]
        alpha = info.get("alpha", 0.6)

        # Convert merged Shapely polygon(s) to PyVista meshes
        meshes = shapely_to_mesh(shp, bottom, height)

        for mesh in meshes:
            plotter.add_mesh(mesh, color=color, opacity=alpha, show_edges=False)

        used_layers[layer_key] = info

    # Legend
    legend_entries = []
    for key, info in used_layers.items():
        name = info.get("name", key)
        legend_entries.append([name, info["color"]])

    if legend_entries:
        plotter.add_legend(legend_entries, bcolor="white")

    plotter.show()

# -----------------------------
# Helper functions
# -----------------------------
def keyboard_listener():
    global running
    print("Press 'quit' + Enter to stop the server.")
    for line in sys.stdin:
        if line.strip().lower() == "quit":
            print("Shutdown command received.")
            running = False
            break

def recv_exact(conn, n):
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Connection closed before receiving full message")
        data += chunk
    return data

# -----------------------------
# Server setup
# -----------------------------
running = True
threading.Thread(target=keyboard_listener, daemon=True).start()

HOST = "127.0.0.1"
PORT = 50007

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)
print("Server ready")

# -----------------------------
# Main server loop
# -----------------------------
try:
    while running:
        try:
            sock.settimeout(1.0)
            conn, addr = sock.accept()

            # ---- Read 4‑byte length prefix ----
            length_bytes = recv_exact(conn, 4)
            msg_len = int.from_bytes(length_bytes, "big")

            # ---- Read the full JSON payload ----
            data = recv_exact(conn, msg_len).decode()
            conn.close()

            if not data:
                continue

            # Parse JSON
            payload = json.loads(data)
            polygons = payload.get("polygons", [])
            if not polygons:
                continue

            # Plot
            plot_data(polygons)
            print("Data plotted.")

        except socket.timeout:
            continue
        except Exception as e:
            print("Error:", e)

finally:
    sock.close()
    print("Server stopped.")

"""
# This is the script from Klayout part
import pya
import socket
import json

# ------------------------------------------------------------
# Send data to external Python plotter
# ------------------------------------------------------------
def send_to_plotter(payload_dict):
    try:
        payload = json.dumps(payload_dict).encode()

        # Prefix with 4‑byte length
        msg = len(payload).to_bytes(4, "big") + payload

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("127.0.0.1", 50007))
        s.sendall(msg)
        s.close()

        pya.Logger.info("Data sent to external plotter.")
    except Exception as e:
        pya.Logger.error(f"Failed to send data: {e}")


# ------------------------------------------------------------
# Collect ruler region (user-drawn box)
# ------------------------------------------------------------
def get_ruler_region_in_dbu(view, layout):
    region = pya.Region()
    dbu = layout.dbu  # microns per DBU

    for ann in view.each_annotation():
        try:
            dbox = ann.box()  # pya.DBox in microns
        except Exception:
            dbox = None

        if dbox is not None and not dbox.empty():
            box = pya.Box(
                int(dbox.left / dbu),
                int(dbox.bottom / dbu),
                int(dbox.right / dbu),
                int(dbox.top / dbu)
            )
            region.insert(box)

    return region


# ------------------------------------------------------------
# Main logic
# ------------------------------------------------------------
def main():
    view = pya.LayoutView.current()
    if view is None:
        pya.Logger.info("No layout open.")
        return

    cellview = view.active_cellview()
    if cellview is None:
        pya.Logger.info("No active cell view.")
        return

    layout = cellview.layout()
    top_cell = cellview.cell
    if top_cell is None:
        pya.Logger.info("No top cell found.")
        return

    dbu = layout.dbu

    # Get ruler box region
    selection_region = get_ruler_region_in_dbu(view, layout)
    if selection_region.is_empty():
        pya.Logger.info("No ruler box found.")
        return

    bbox_dbu = selection_region.bbox()
    bbox_micron = pya.DBox(bbox_dbu) * dbu
    pya.Logger.info(f"Selection box (microns): {bbox_micron}")

    all_polygons = []

    # Iterate through all layers
    for layer_index in layout.layer_indexes():
        layer_info = layout.get_info(layer_index)

        it = top_cell.begin_shapes_rec_overlapping(layer_index, bbox_dbu)

        while not it.at_end():
            shape = it.shape()
            trans = it.trans()

            shape_region_local = pya.Region()

            if shape.is_box():
                shape_region_local.insert(shape.box)
            elif shape.is_polygon():
                shape_region_local.insert(shape.polygon)
            else:
                it.next()
                continue

            # Transform to top-level coordinates
            shape_region_top = shape_region_local.transformed(trans)

            # Intersect with ruler region
            shape_intersection = shape_region_top & selection_region

            if not shape_intersection.is_empty():
              for poly in shape_intersection.each():
                pts = []
            
                # Convert Region polygon → SimplePolygon
                polygon = poly.to_simple_polygon()
            
                for p in polygon.each_point():
                    pts.append([p.x * dbu, p.y * dbu])
            
                all_polygons.append({
                    "layer": {
                        "layer": layer_info.layer,
                        "datatype": layer_info.datatype
                    },
                    "points": pts
                })

            it.next()

    # Send to external plotter
    if all_polygons:
        send_to_plotter({"polygons": all_polygons})
    else:
        pya.Logger.info("No polygons found in selection.")


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
"""