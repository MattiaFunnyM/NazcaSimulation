import meep as mp

# Define Simulation Size
cell_width = 16  # X direction
cell_height = 8  # Y direction
cell_depth = 8   # Z direction
cell_size = mp.Vector3(cell_width, cell_height, cell_depth) 