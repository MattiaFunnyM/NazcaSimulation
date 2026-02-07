# TODO: Select a rectangle in a GDS
# TODO: Transform the selection region in refractive index block
# TODO: Visualize refractive index block
# TODO: Simulation (entire block together)
# TODO: Present results
# TODO: Simulation (divide the simulation)
# GOALS: 
# - Waveguide HOR 10 um under 5 min
# - Waveguide VER 15 um under 4 min
# - Coupler 2x2 50 um under 10 min
# - Ring resonator 100 um under 10 min
import json
import socket
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Assign a random color to each layer
layer_colors = {}

def get_color_for_layer(layer_key):
    if layer_key not in layer_colors:
        layer_colors[layer_key] = (
            random.random(),
            random.random(),
            random.random()
        )
    return layer_colors[layer_key]

# Prepare the server
HOST = "127.0.0.1"
PORT = 50007
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)
print("Server ready")

# Server is listening for a new package of data
while True:
    conn, addr = sock.accept()
    data = conn.recv(100000).decode()
    conn.close()

    if not data:
        continue

    print("Received data block")

    # Parse JSON
    try:
        payload = json.loads(data)
    except Exception as e:
        print("JSON error:", e)
        continue

    polygons = payload.get("polygons", [])
    if not polygons:
        print("No polygons received")
        continue

    # Create a new figure
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # Plot each polygon
    for poly in polygons:
        layer = poly["layer"]
        pts = poly["points"]
 
        layer_key = f"{layer['layer']}/{layer['datatype']}"
        color = get_color_for_layer(layer_key)

        patch = Polygon(pts, closed=True, facecolor=color, edgecolor="black", alpha=0.6)
        ax.add_patch(patch)

    # Rescale property
    all_x = []
    all_y = []
    for poly in polygons:
        for x, y in poly["points"]:
            all_x.append(x)
            all_y.append(y)

    # Add a small margin
    margin = 1.0

    xmin, xmax = min(all_x) - margin, max(all_x) + margin
    ymin, ymax = min(all_y) - margin, max(all_y) + margin

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    plt.title("KLayout Selection Plot")
    plt.xlabel("Microns")
    plt.ylabel("Microns")
    plt.grid(True)

    plt.show()

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
        payload = json.dumps(payload_dict)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("127.0.0.1", 50007))
        s.send(payload.encode())
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