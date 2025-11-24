import meep as mp

# Reference wavelength or center wavelength
c_wvl = 1.55 # in microns

# Define Simulation Size
cell_width = 8 * c_wvl   # X direction (-4, 4)
cell_height = 8 * c_wvl  # Y direction (-4, 4)
cell_depth = 8 * c_wvl   # Z direction (-4, 4)
cell_size = mp.Vector3(cell_width, cell_height, cell_depth) 

# Define the Simulation Source: gaussian source
source_frequency = 1 / c_wvl  # Frequency corresponding to the reference wavelength
source = mp.Source(
    mp.GaussianSource(frequency=source_frequency, fwidth=0.1),  # Gaussian pulse
    component=mp.Ex,           # x-polarized electric field
    center=mp.Vector3(0, 0, 0) # position at the center of the cell
)