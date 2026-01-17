import numpy as np
import meep as mp
import SimLibrary as SL
import matplotlib.pyplot as plt

def calculate_overlap(E1, E2):
    """
    Calculate normalized modal overlap between two field E1 and E2.

    E1: complex arrays (Nx, Ny)
    E2: complex arrays (Nx, Ny)
    
    Returns
    -------
    overlap_z : float value 
        Normalized overlap result between E1 and E2
    """

    # Integrate over x,y 
    overlap_int = np.sum((E1 * np.conj(E2)))

    # Norm of E1
    norm1 = np.sum(np.abs(E1)**2)

    # Norm of E2
    norm2 = np.sum(np.abs(E2)**2)

    # Normalized overlap vs z
    overlap_z = np.abs(overlap_int)**2 / (norm1 * norm2)

    return np.round(overlap_z * 100, 2)

# =========================
# USER PARAMETERS
# =========================
document_url = "https://www.mdpi.com/2304-6732/10/5/510"

# Waveguide parameters
n_Si   = 3.48
n_SiO2 = 1.44
n_SiN  = 1.97
n_air  = 1.0
cld_width = 6
wvg_widths = np.linspace(0.1, 0.5, 8)
wvg_height = 0.4
wvg_length = 1
wvg_length_mode = 1
SiO2_width = 6
SiO2_height = 7.6
Si_wvg_distance = 4

# Fiber parameters
radius_core = 1.2
n_core = 1.502
n_clad = 1.4447
fbr_length = 1

# Overall simulation parameters
sim_width  = 12
sim_height = 12
sim_resolution = 16
sim_bnd_thickness = 0.2

# Frequency parameters
wavelength = 1.55
frequency = 1/wavelength

# =========================
# CALCULATE FIBER MODE LOOP
# =========================
geometry_fiber_mode = [
        mp.Block(
            size=mp.Vector3(sim_width, sim_height, fbr_length),
            center=mp.Vector3(),
            material=mp.Medium(epsilon=n_clad**2)),

        mp.Cylinder(radius=radius_core, 
                    height=fbr_length, 
                    center=mp.Vector3(),
                    material=mp.Medium(epsilon=n_core**2))]

cross_section = mp.Volume(
        center=mp.Vector3(0, 0, 0),
        size=mp.Vector3(
            sim_width - 2*sim_bnd_thickness,
            sim_height - 2*sim_bnd_thickness,
            0)
    )

Fiber_mode = SL.find_mode_from_cross_section(
                geometry = geometry_fiber_mode, 
                cross_section = cross_section, 
                mode_order=1, 
                frequency=frequency, 
                sim_resolution=sim_resolution)

# =========================
# SIMULATION LOOP
# =========================
overlaps_TE = []
overlaps_TM = []
for wvg_width in wvg_widths:
    # MODE CALCULATION SiN WAVEGUIDE
    geometry_sin_mode = [
            mp.Block(
                size=mp.Vector3(sim_width, sim_height, wvg_length_mode),
                center=mp.Vector3(),
                material=mp.Medium(epsilon=n_air**2)
            ),

            # --- Cladding background substrate ---
            mp.Block(
                size=mp.Vector3(SiO2_width, SiO2_height, wvg_length_mode),
                center=mp.Vector3(0, -wvg_height),
                material=mp.Medium(epsilon=n_SiO2**2)
            ),

            # --- Silicon Nitride waveguide ---
            mp.Block(
                size=mp.Vector3(wvg_width, wvg_height, wvg_length_mode),
                center=mp.Vector3(),
                material=mp.Medium(epsilon=n_SiN**2)
            ),

            # --- Silicon bottom substrate --- Commented for avoid mode issues
            #mp.Block(
            #    size=mp.Vector3(sim_width, wvg_height, wvg_length_mode),
            #    center=mp.Vector3(0, - sim_height / 2 + wvg_height/2),
            #    material=mp.Medium(epsilon=n_Si**2)
            #)
            ]

    cross_section = mp.Volume(
        center=mp.Vector3(0, 0, 0),
        size=mp.Vector3(
            sim_width - 2*sim_bnd_thickness,
            sim_height - 2*sim_bnd_thickness,
            0)
    )
    
    mode = SL.find_mode_from_cross_section(
            geometry = geometry_sin_mode, 
            cross_section = cross_section, 
            mode_order=1, 
            frequency=frequency, 
            sim_resolution=sim_resolution,
            parity=mp.ODD_Y)

    # Calculate the overlap
    overlap_y = calculate_overlap(Fiber_mode['Ey'], 
                                    mode['Ey'])
    overlaps_TE.append(overlap_y)

    mode = SL.find_mode_from_cross_section(
            geometry = geometry_sin_mode, 
            cross_section = cross_section, 
            mode_order=1, 
            frequency=frequency, 
            sim_resolution=sim_resolution,
            parity=mp.EVEN_Y)

    # Calculate the overlap
    overlap_x = calculate_overlap(Fiber_mode['Ex'], 
                                    mode['Ex'])
    overlaps_TM.append(overlap_x)

# Plot the result
plt.plot(wvg_widths, overlaps_TE, marker='o', label='TE')
plt.plot(wvg_widths, overlaps_TM, marker='o', label='TM')
plt.grid()
plt.legend()
plt.show()



