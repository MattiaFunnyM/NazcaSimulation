import numpy as np
import meep as mp
import SimLibrary as SL
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


def generate_modal_source_2D(field_dict,
                             src_position,
                             src_size,
                             src_decay=3,
                             frequency=1):
    """
    Create 2D Meep sources from arbitrary field components.

    Parameters
    ----------
    field_dict : dict
        Dictionary of field components and their 2D distributions.
        Example:
        {
            "Ez": {"coords": (x, y), "field": Ez_xy},
            "Hy": {"coords": (x, y), "field": Hy_xy},
            ...
        }

        - coords must be 2 arrays: (x_grid, y_grid)
        - field must be a 2D array matching the grid

    src_position : mp.Vector3
        Center of the source region.

    src_size : mp.Vector3
        Size of the source region.

    src_decay : float
        Time after which the source amplitude is switched off.

    frequency : float
        Oscillation frequency.

    Returns
    -------
    list of mp.Source
        One Meep source per field component.
    """

    sources = []

    # --- Temporal profiles ---
    def temporal_E(t):
        if frequency * t > src_decay:
            return 0
        return np.cos(2 * np.pi * frequency * t)

    def temporal_H(t):
        if frequency * t > src_decay:
            return 0
        return np.sin(2 * np.pi * frequency * t)

    # --- Loop over all field components ---
    for comp_name, data in field_dict.items():

        # Extract grid and field
        x, y = data["coords"]
        field = np.real(data["field"])

        # Build 2D interpolator
        Field_function = RegularGridInterpolator((x, y),
                                                 field,
                                                 bounds_error=False,
                                                 fill_value=0)
       
        # Spatial profile for Meep
        def spatial_profile(r):
            return float(Field_function((r.x, r.y)))
        
        # Determine if component is E or H
        if comp_name.startswith("E"):
            temporal = temporal_E
        elif comp_name.startswith("H"):
            temporal = temporal_H
        else:
            raise ValueError(f"Unknown field component: {comp_name}")

        # Map string to Meep component
        try:
            meep_component = getattr(mp, comp_name)
        except AttributeError:
            raise ValueError(f"Invalid Meep component name: {comp_name}")

        # Create source
        src = mp.Source(
        src=mp.CustomSource(temporal),
        component=meep_component,
        center=src_position,
        size=src_size,
        amp_func=spatial_profile
        )

        sources.append(src)

    return sources

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
wvg_length_mode = 10
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

silicon_height = (sim_height/2) - wvg_height - SiO2_height/2
silicon_center_y = - (sim_height/4 + wvg_height/2 + SiO2_height/4)

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
        size=mp.Vector3(sim_width, sim_height))

Fiber_mode = SL.find_mode_from_cross_section(
                geometry = geometry_fiber_mode, 
                cross_section = cross_section, 
                mode_order=1, 
                frequency=frequency, 
                sim_resolution=sim_resolution)

x_grid = np.linspace(-sim_width/2, sim_width/2, int(sim_width*sim_resolution))
y_grid = np.linspace(-sim_height/2, sim_height/2, int(sim_height*sim_resolution))

field_dict = {
    "Ey": {
        "coords": (x_grid, y_grid),
        "field": Fiber_mode["Ey"]
    }
}

sources = generate_modal_source_2D(
    field_dict,
    src_position=mp.Vector3(0, 0, -wvg_length_mode/2+1/sim_resolution),
    src_size=mp.Vector3(sim_width, sim_height),
    src_decay=3,
    frequency=frequency
)


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
            mp.Block(
                size=mp.Vector3(sim_width, silicon_height, wvg_length_mode),
                center=mp.Vector3(0, silicon_center_y),
                material=mp.Medium(epsilon=n_Si**2)
            )
            ]
    
    sim_size, sim_center = SL.compute_geometry_bounds(geometry_sin_mode)
    
    # Initialize Simulation
    sim = mp.Simulation(
        cell_size=sim_size,
        geometry=geometry_sin_mode,
        resolution=sim_resolution,
        sources=sources,
        dimensions=3)

     # Prepare a Discrete Fourier Transform monitor to extract the complex field information
    dft = sim.add_dft_fields([mp.Ey], frequency, 0, 1, 
                            where=mp.Volume(center=mp.Vector3(0, 0, wvg_length_mode/2-1),
                                            size=mp.Vector3(sim_width, sim_height)))
    
    # Run the simulation
    sim.run(until=n_SiN*wvg_length_mode)

    Ey = np.transpose(sim.get_dft_array(dft, mp.Ey, 0))
    
    # Calculate the overlap
    overlap_y = calculate_overlap(Fiber_mode['Ey'], Ey)
    overlaps_TE.append(overlap_y)

# Plot the result
plt.plot(wvg_widths, overlaps_TE, marker='o', label='TE')
plt.grid()
plt.legend()
plt.show()



