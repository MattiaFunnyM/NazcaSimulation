import numpy as np
import meep as mp
import Library.Simulation3D as SL
import matplotlib.pyplot as plt

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
wvg_widths = np.linspace(0.1, 0.5, 21)
wvg_height = 0.4
wvg_length_mode = 1
SiO2_width = 6
SiO2_height = 7.6

# Fiber parameters
radius_core = 1.2
n_core = 1.4825
n_clad = 1.4447
fbr_length = 1

# Overall simulation parameters
sim_width  = 10
sim_height = 10
sim_resolution = 32

# Frequency parameters
wavelength = 1.55
frequency = 1/wavelength

# =========================
# CALCULATE FIBER MODE LOOP
# =========================
# Define the geometry of the fiber 
geometry_fiber_mode = [
        mp.Block(
            size=mp.Vector3(sim_width, sim_height, fbr_length),
            center=mp.Vector3(),
            material=mp.Medium(epsilon=n_clad**2)),

        mp.Cylinder(radius=radius_core, 
                    height=fbr_length, 
                    center=mp.Vector3(),
                    material=mp.Medium(epsilon=n_core**2))]

# Define the cross section where to perform the overlap
cross_section = mp.Volume(
        center=mp.Vector3(0, 0, 0),
        size=mp.Vector3(sim_width, sim_height))

# Find the mode in the given cross section at the wanted frequency
Fiber_mode = SL.find_mode_from_cross_section(
                geometry=geometry_fiber_mode, 
                cross_section=cross_section, 
                mode_order=1, 
                frequency=frequency, 
                sim_resolution=sim_resolution)

# =========================
# SIMULATION LOOP
# =========================
overlaps_TE = []
overlaps_TM = []
for wvg_width in wvg_widths:
    print(f"Working on width {wvg_width}")

    # Define the geometry of the silicon nitride waveguide
    geometry_sin_mode = [
            mp.Block(
                size=mp.Vector3(sim_width, sim_height, wvg_length_mode),
                center=mp.Vector3(),
                material=mp.Medium(epsilon=n_SiO2**2)),

            # --- Silicon Nitride waveguide ---
            mp.Block(
                size=mp.Vector3(wvg_width, wvg_height, wvg_length_mode),
                center=mp.Vector3(),
                material=mp.Medium(epsilon=n_SiN**2))]
    
    # Calculate the TE mode from the given geometry
    mode = SL.find_mode_from_cross_section(
            geometry = geometry_sin_mode, 
            cross_section = cross_section, 
            mode_order=1, 
            frequency=frequency, 
            sim_resolution=sim_resolution,
            parity=mp.EVEN_Y)

    # Calculate the overlap for TE light
    overlap_x = SL.calculate_overlap(Fiber_mode['Ex'], 
                                     mode['Ex'])
    overlaps_TE.append(overlap_x)

    # Calculate the TM mode from the given geometry
    mode = SL.find_mode_from_cross_section(
            geometry = geometry_sin_mode, 
            cross_section = cross_section, 
            mode_order=1, 
            frequency=frequency, 
            sim_resolution=sim_resolution,
            parity=mp.ODD_Y)

    # Calculate the overlap for TM ligth
    overlap_y = SL.calculate_overlap(Fiber_mode['Ey'], 
                                     mode['Ey'])
    overlaps_TM.append(overlap_y)

# Plot the result
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(wvg_widths, overlaps_TE, marker='o', linewidth=2.5, markersize=8, 
        label='TE', color='#1f77b4', markerfacecolor='white', markeredgewidth=2)
ax.plot(wvg_widths, overlaps_TM, marker='s', linewidth=2.5, markersize=8, 
        label='TM', color='#ff7f0e', markerfacecolor='white', markeredgewidth=2)
ax.set_xlabel("Width/µm", fontsize=18, fontweight='bold')
ax.set_ylabel("Modal Overlap Efficiency %", fontsize=18, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.legend(fontsize=11, framealpha=0.95, edgecolor='black', loc='best')
ax.set_facecolor('#f8f9fa')
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.show()



