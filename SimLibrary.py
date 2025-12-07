import meep as mp
import numpy as np

def compute_geometry_bounds(geometry_list):
    """
    Given the Meep geometry list, compute the bounding box that contains all objects.
    """
   
    # Init bounds with infinities
    xmin, ymin, zmin = np.inf, np.inf, np.inf
    xmax, ymax, zmax = -np.inf, -np.inf, -np.inf

    for obj in geometry_list:
        size = obj.size
        center = obj.center

        # Half-sizes
        hx = size.x / 2
        hy = size.y / 2
        hz = size.z / 2

        # Bounds for current object
        oxmin = center.x - hx
        oxmax = center.x + hx
        oymin = center.y - hy
        oymax = center.y + hy
        ozmin = center.z - hz
        ozmax = center.z + hz

        # Update global bounds
        xmin = min(xmin, oxmin)
        xmax = max(xmax, oxmax)
        ymin = min(ymin, oymin)
        ymax = max(ymax, oymax)
        zmin = min(zmin, ozmin)
        zmax = max(zmax, ozmax)

    # Create final mp.Vector3
    bounding_size = mp.Vector3(xmax - xmin, ymax - ymin, zmax - zmin)
    bounding_center = mp.Vector3((xmin + xmax)/2, (ymin + ymax)/2, (zmin + zmax)/2)

    return bounding_size, bounding_center

def finding_mode_from_geometry(geometry, mode=1, frequency=1, resolution=20, time=50):
    """
    This function generates the Ez and Hy field cross sections 
    for the TE modes of specified mode order of the geometry in input.
    The mode is calculates at the specified frequency.
    The precision of the simulation is determined by the resolution parameter.
    The simulations runs for the specified time.
    """
    
    # Simulation boundary from geometry
    sim_size, sim_center = compute_geometry_bounds(geometry)

    # Condition for edge of simulation
    pml_thickness = 1.0
    pml_layers = [mp.PML(pml_thickness)]

    # Define the source for the mode calculation
    src_position = -sim_size.x / 2 + pml_thickness + 1 / resolution
    source = [mp.EigenModeSource(
        src=mp.ContinuousSource(frequency=frequency, fwidth=frequency/2),
        center=mp.Vector3(src_position, 0),
        size=mp.Vector3(0, sim_size.y),
        direction=mp.X,
        eig_band=mode
    )]

    # Create the meep simulation
    sim = mp.Simulation(
        cell_size=sim_size,
        sources=source,
        geometry=geometry,
        resolution=resolution,
        boundary_layers=pml_layers)

    # Change output directory
    sim.use_output_directory("Output")

    # Prepare a Discrete Fourier Transform monitor to extract the complex field information
    frequency_center = 0 
    frequency_points = 1
    dft = sim.add_dft_fields([mp.Ez, mp.Hy], frequency, frequency_center, frequency_points, 
                             where=mp.Volume(center=mp.Vector3(),
                                             size=sim_size))

    # Define the variable to optimize# Physically run the simulation with current setup
    sim.run(mp.at_beginning(mp.output_epsilon),
            until=time)

    # Extract complex field component
    Ez1 = sim.get_dft_array(dft, mp.Ez, 0)
    Hy1 = sim.get_dft_array(dft, mp.Hy, 0)

    # Extract the cross section at that maxima (used for cross section and mode generation
    flat_index = np.argmax(np.real(Ez1))
    x_max_index, _ = np.unravel_index(flat_index, Ez1.shape)
    Ez1_cross = Ez1[x_max_index, :]
    Hy1_cross = Hy1[x_max_index, :]
    
    # Calculate the normalization from the fields optical power
    dy = sim_size.y / resolution  

    # Calculate the optical power as half of the product of electric field and magnetic field
    Power_vector = -0.5 * np.real(Ez1_cross * np.conj(Hy1_cross))

    # The power value of the cross section is the integral of the optical power vector
    Power_value = np.abs(np.sum(Power_vector) * dy)
    
    # To normalize the fields, we need to divide by the square root of the power.
    # In reality, np.abs(Ez1_cross) is normalized, not np.real(Ez1_cross) that oscillates
    # Byt later Ez1 and Hy1 are properly normalized
    Ez1_cross_norm = np.real(Ez1_cross) / np.sqrt(Power_value)
    Hy1_cross_norm = np.real(Hy1_cross) / np.sqrt(Power_value) 

    # Normalization by ratio to pass from np.real to np.abs
    Ez1_cross_norm *= np.max(np.abs(Ez1_cross)) / np.max(np.real(Ez1_cross))
    Hy1_cross_norm *= np.max(np.abs(Hy1_cross)) / np.max(np.real(Hy1_cross))
    
    # Get the material profile distribtuion
    eps_data = sim.get_array(
    component=mp.Dielectric,
    center=mp.Vector3(),           
    size=sim_size)

    # Get the cross section of the material profile
    eps_cross = eps_data[x_max_index, :]

    # Calculate refractive index
    eps_eff_cross = Power_vector * eps_cross / Power_value
    neff = np.sqrt(np.sum(eps_eff_cross) * dy)
                   
    # Return the field cross section 
    return Ez1_cross_norm, Hy1_cross_norm, neff
