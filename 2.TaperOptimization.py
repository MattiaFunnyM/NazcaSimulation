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
wvg_width  = [0.1, 0.2, 0.3, 0.4, 0.5]
wvg_height = 0.4

SiO2_width = 6
SiO2_height = 7.6

sim_width  = 12
sim_height = 12
sim_resolution = 32

geometry = [

    # --- Air background substrate ---
    mp.Block(
        size=mp.Vector3(sim_width, sim_height, 0),
        center=mp.Vector3(),
        material=mp.Medium(epsilon=n_air**2)
    ),

    # --- Cladding background substrate ---
    mp.Block(
        size=mp.Vector3(SiO2_width, SiO2_height, 0),
        center=mp.Vector3(),
        material=mp.Medium(epsilon=n_SiO2**2)
    )
]

SL.visualize_geometry(geometry, resolution=sim_resolution)

"""
for wvg_width in wvg_widths[2:3]:
    # GEOMETRY DEFINITION
    geometry = [
        mp.Block(
            size=mp.Vector3(sim_width, sim_height, sim_length),
            center=mp.Vector3(),
            material=mp.Medium(epsilon=cld_neff**2)
        ),
        mp.Block(
            size=mp.Vector3(wvg_width, wvg_height, sim_length),
            center=mp.Vector3(),
            material=mp.Medium(epsilon=wvg_neff**2)
        )
    ]

    # CROSS SECTION DEFINITION
    cross_section = mp.Volume(
        center=mp.Vector3(0, 0, 0),
        size=mp.Vector3(
            sim_width - 2*bnd_thickness,
            sim_height - 2*bnd_thickness,
            0
        )
    )

    # MODE CALCULATION
    TE0 = SL.find_mode_from_cross_section(
            geometry = geometry, 
            cross_section = cross_section, 
            mode_order=1, 
            frequency=frequency, 
            sim_resolution=32)

plt.imshow(abs(TE0['Ex']))
plt.show()
"""