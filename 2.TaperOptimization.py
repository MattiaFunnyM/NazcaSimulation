import numpy as np
import meep as mp
import SimLibrary as SL
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
wvg_widths = [0.28]
wvg_height = 0.4
wvg_length = 20
SiO2_width = 6
SiO2_height = 7.6
Si_wvg_distance = 4

# Fiber parameters
radius_core = 1.37
n_core = 1.49836
n_clad = 1.4447
fbr_length = 2

# Overall simulation parameters
sim_width  = 8.5
sim_height = 8.5
sim_length = fbr_length + wvg_length
sim_resolution = 16
sim_bnd_thickness = 0.3

# Frequency parameters
wavelength = 1.55
frequency = 1/wavelength

# =========================
# SIMULATION LOOP
# =========================

for wvg_width in wvg_widths:
    geometry = [
        ### Fiber Part ###
        mp.Block(
            size=mp.Vector3(sim_width, sim_height, fbr_length),
            center=mp.Vector3(0, 0, -fbr_length/2),
            material=mp.Medium(epsilon=n_clad**2)),

        mp.Cylinder(radius=radius_core, 
                    height=fbr_length, 
                    center=mp.Vector3(0, 0, -fbr_length/2),
                    material=mp.Medium(epsilon=n_core**2)),
        ### Waveguide Part ###
        # --- Air background substrate ---
        mp.Block(
            size=mp.Vector3(sim_width, sim_height, wvg_length),
            center=mp.Vector3(0, 0, wvg_length/2),
            material=mp.Medium(epsilon=n_air**2)
        ),

        # --- Cladding background substrate ---
        mp.Block(
            size=mp.Vector3(SiO2_width, SiO2_height, wvg_length),
            center=mp.Vector3(0, -wvg_height/2, wvg_length/2),
            material=mp.Medium(epsilon=n_SiO2**2)
        ),

        # --- Silicon Nitride waveguide ---
        mp.Block(
            size=mp.Vector3(wvg_width, wvg_height, wvg_length),
            center=mp.Vector3(0, 0, wvg_length/2),
            material=mp.Medium(epsilon=n_SiN**2)
        ),

        # --- Silicon bottom substrate ---
        mp.Block(
            size=mp.Vector3(sim_width, wvg_height, wvg_length),
            center=mp.Vector3(0, - sim_height / 2 + wvg_height/2, wvg_length/2),
            material=mp.Medium(epsilon=n_Si**2)
        )]

    # Calculate cell_size
    cell_size, cell_center = SL.compute_geometry_bounds(geometry)

    # Setting up the simulation
    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=[mp.PML(sim_bnd_thickness)],
        geometry=geometry,
        default_material=mp.air,
        resolution=sim_resolution,
        sources=[
            mp.EigenModeSource(
                src=mp.ContinuousSource(frequency=frequency, end_time=np.pi*6*frequency),
                center=mp.Vector3(0, 0, -fbr_length + sim_bnd_thickness + 1/sim_resolution),  
                size=mp.Vector3(sim_width-2*sim_bnd_thickness, sim_height-2*sim_bnd_thickness, 0),
                direction=mp.Z,
                eig_band=1)]
    )
   
    # Setting up the dft 
    dft = sim.add_dft_fields(
        [mp.Ex, mp.Ey, mp.Ez],
        frequency, 0, 1,
        center=cell_center,
        size=cell_size)

    sim.run(until=wvg_length * 2)  
    
    Ex = sim.get_dft_array(dft, mp.Ex, 0)
    Ey = sim.get_dft_array(dft, mp.Ey, 0)
    Ez = sim.get_dft_array(dft, mp.Ez, 0)
    E = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)
    SL.visualize_condensated_field(E, propagation_dir='z', wide_dir='x')

# TODO: Calculate Coupling Efficiency