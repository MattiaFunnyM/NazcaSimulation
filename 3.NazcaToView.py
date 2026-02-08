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
import random
import threading
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

#############################
### PLOTTING INFORMATIONS ###
#############################

# Open the Technology File
with open("TechnologyExample.json", "r") as f:
    tech = json.load(f)

# Take the information of the colors
tech_layers = tech["layers"]

# Function used for plotting
def plot_data(polygons):
   
    # Configure the plotting
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # Plot each polygon independently
    for poly in polygons:
        # Get the layer of the polygon
        layer = poly["layer"]
        layer_key = f"{layer['layer']}/{layer['datatype']}"
        
        # Skip layers not in the technology
        if layer_key not in tech_layers:
            continue
        
        # Take information from the technology file
        layer_info = tech_layers[layer_key]
        color = layer_info["color"]
        height = layer_info["height"]  

        # Plot the polygons
        pts = poly["points"]
        patch = Polygon(pts, closed=True, facecolor=color,
                        edgecolor="black", alpha=0.6)
        ax.add_patch(patch)

    margin = 1
    all_x = [x for poly in polygons for x, _ in poly["points"]]
    all_y = [y for poly in polygons for _, y in poly["points"]]
    if all_x or all_y:
        ax.set_xlim(min(all_x)-margin, max(all_x)+margin)
        ax.set_ylim(min(all_y)-margin, max(all_y)+margin)

    plt.title("2D Selection Plot")
    plt.xlabel("µm")
    plt.ylabel("µm")
    plt.grid(True)
    plt.show()
    return

###########################
### SERVER INFORMATIONS ###
###########################

# Status of the server
running = True

# Function to call for shutting down the server
def keyboard_listener():
    global running
    print("Press 'quit' + Enter to stop the server.")
    for line in sys.stdin:
        if line.strip().lower() == "quit":
            print("Shutdown command received.")
            running = False
            break

# Start the thread
threading.Thread(target=keyboard_listener, daemon=True).start()

# Server setup
HOST = "127.0.0.1"
PORT = 50007
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)
print("Server ready")

# Loop run for the server
try:
    while running:
        try:
            # Information for connections
            sock.settimeout(1.0)  
            conn, addr = sock.accept()
            data = conn.recv(100000).decode()
            conn.close()

            # If no data is received continue the loop
            if not data:
                continue
            
            # If there are data load them
            payload = json.loads(data)
            polygons = payload.get("polygons", [])
            if not polygons:
                continue
            
            # And plot them
            plot_data(polygons)
            print("Data plotted.")
          
        # If there are exceptions, they do not stop the server
        except socket.timeout:
            continue
        except Exception as e:
            print("Error:", e)

# Exit if the quit combination is pressed
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