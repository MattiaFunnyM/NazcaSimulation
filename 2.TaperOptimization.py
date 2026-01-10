import numpy as np
import meep as mp
import SimLibrary as SL
import matplotlib.pyplot as plt

# =========================
# USER PARAMETERS
# =========================
document_url = "https://www.mdpi.com/2304-6732/10/5/510"
n_Si   = 3.48
n_SiO2 = 1.44
n_SiN  = 1.97
n_air  = 1.0
cld_width = 6
wvg_widths = [0.28]
wvg_height = 0.4

SiO2_width = 6
SiO2_height = 7.6
Si_bottom_height = 0.4
Si_wvg_distance = 4

sim_width  = 8.3
sim_height = 8.3
sim_length = 2
sim_resolution = 16
sim_bnd_thickness = 0.25

cross_section_size = 7

wavelength = 1.55
frequency = 1/wavelength

# CROSS SECTION DEFINITION
cross_section = mp.Volume(
    center=mp.Vector3(),
    size=mp.Vector3(cross_section_size,
                    cross_section_size)
)

# Define fiber geometry and parameters
radius_core = 1.37
n_core = 1.49836
n_clad = 1.4447
geometry_fiber = [
    mp.Block(
        size=mp.Vector3(sim_width, sim_height, sim_length),
        center=mp.Vector3(),
        material=mp.Medium(epsilon=n_clad**2)
    ),

    mp.Cylinder(radius=radius_core, 
                height=mp.inf, 
                center=mp.Vector3(),
                material=mp.Medium(epsilon=n_core**2))
]
FiberTE0 = SL.find_mode_from_cross_section(geometry = geometry_fiber, 
                                           cross_section = cross_section, 
                                           mode_order=1, 
                                           frequency=frequency, 
                                           sim_resolution=sim_resolution)

for wvg_width in wvg_widths:
    geometry_waveguide = [

    # --- Air background substrate ---
    mp.Block(
        size=mp.Vector3(sim_width, sim_height, sim_length),
        center=mp.Vector3(),
        material=mp.Medium(epsilon=n_air**2)
    ),

    # --- Cladding background substrate ---
    mp.Block(
        size=mp.Vector3(SiO2_width, SiO2_height, sim_length),
        center=mp.Vector3(),
        material=mp.Medium(epsilon=n_SiO2**2)
    ),

    # --- Silicon Nitride waveguide ---
    mp.Block(
        size=mp.Vector3(wvg_width, wvg_height, sim_length),
        center=mp.Vector3(0, wvg_height/2),
        material=mp.Medium(epsilon=n_SiN**2)
    ),

    # --- Silicon bottom substrate ---
    mp.Block(
        size=mp.Vector3(sim_width, Si_bottom_height, sim_length),
        center=mp.Vector3(0, +Si_bottom_height/2 - sim_height / 2),
        material=mp.Medium(epsilon=n_Si**2)
    )]

    # MODE CALCULATION
    TE0 = SL.find_mode_from_cross_section(
            geometry = geometry_waveguide, 
            cross_section = cross_section, 
            mode_order=1, 
            frequency=frequency, 
            sim_resolution=sim_resolution)

plt.imshow(abs(TE0['Ey']), extent=[-cross_section_size/2, cross_section_size/2, -cross_section_size/2, cross_section_size/2])
plt.show()

#SL.visualize_geometry(geometry, resolution=sim_resolution)
