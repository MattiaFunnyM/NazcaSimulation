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

import pya

def get_ruler_region_in_dbu(view, layout):
    """Collect box rulers and return them as a pya.Region in DBU."""
    region = pya.Region()

    dbu = layout.dbu  # microns per database unit (µm/DBU)

    for ann in view.each_annotation():
        try:
            dbox = ann.box()  # This is a pya.DBox (in micron units)
        except Exception:
            dbox = None

        if dbox is not None and not dbox.empty():
            # Convert DBox (µm) → Box (DBU)
            box = pya.Box(
                int(dbox.left / dbu),
                int(dbox.bottom / dbu),
                int(dbox.right / dbu),
                int(dbox.top / dbu)
            )
            region.insert(box)

    return region


def main():
    # Get the current layout view
    view = pya.LayoutView.current()
    if view is None:
        pya.Logger.info("No layout open.")
        return

    # Get the active cell view
    cellview = view.active_cellview()
    if cellview is None:
        pya.Logger.info("No active cell view.")
        return

    # Get the layout and the top cell (the root of the hierarchy)
    layout = cellview.layout()
    top_cell = cellview.cell

    if top_cell is None:
        pya.Logger.info("No top cell found.")
        return

    # Get the ruler-defined region in DBU
    selection_region = get_ruler_region_in_dbu(view, layout)

    if selection_region.is_empty():
        pya.Logger.info("No ruler box found. Draw a box using the ruler tool first.")

    # Get the bounding box of the selection region in DBU
    bbox_dbu = selection_region.bbox()

    # Print the ruler selection box in microns for clarity
    dbu = layout.dbu  # microns per DBU
    bbox_micron = pya.DBox(bbox_dbu) * dbu
    pya.Logger.info(f"Ruler selection bounding box (microns): {bbox_micron}")

    # Iterate through shapes overlapping the bounding box in DBU
    for layer_index in layout.layer_indexes():
        layer_info = layout.get_info(layer_index)

        # Create an iterator for shapes overlapping the ruler-defined region
        it = top_cell.begin_shapes_rec_overlapping(layer_index, bbox_dbu)

        found = False
        while not it.at_end():
            shape = it.shape()
            trans = it.trans()  # Get the transformation to the top-level coordinate space

            # Convert the shape into a region that represents its actual geometry
            shape_region_local = pya.Region()

            if shape.is_box():
                shape_region_local.insert(shape.box)
            elif shape.is_polygon():
                shape_region_local.insert(shape.polygon)
            else:
                it.next()
                continue 

            # Apply the transformation to bring the shape region into top-level coordinates
            shape_region_top = shape_region_local.transformed(trans)

            # Compute the intersection between the shape's full region and the ruler selection box (DBU)
            shape_intersection = shape_region_top & selection_region

            # Only consider shapes that actually overlap (non-empty intersection)
            if not shape_intersection.is_empty():
                found = True

                # Print the intersection (the "cut" portion) in microns
                for poly in shape_intersection.each():
                    pya.Logger.info(f"Layer {layer_info}: Intersection polygon (microns) {poly}")

            it.next()

        if not found:
            pya.Logger.info(f"Layer {layer_info}: No overlapping shapes or intersections")


if __name__ == "__main__":
    main()