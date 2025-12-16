import meep as mp
import numpy as np
from scipy.interpolate import interp1d

def compute_geometry_bounds(geometry_list):
    """
    Compute the bounding box that contains all objects in a Meep geometry list.

    This function calculates the smallest rectangular prism (bounding box) 
    that fully encloses all the objects in the provided Meep geometry list. 
    It returns both the size and the center of the bounding box.

    Parameters
    ----------
    geometry_list : list of mp.GeometricObject
        List of Meep geometric objects (e.g., blocks, prisms) to include 
        in the bounding box calculation.

    Returns
    -------
    bounding_size : mp.Vector3
        Size of the bounding box along each axis (x, y, z).
    bounding_center : mp.Vector3
        Center position of the bounding box.

    """
   
    # Init bounds with infinities
    xmin, ymin, zmin = np.inf, np.inf, np.inf
    xmax, ymax, zmax = -np.inf, -np.inf, -np.inf

    for obj in geometry_list:
        # Extract bounding points based on object type
        if isinstance(obj, mp.Prism):
            # For Prism: extract vertices and height
            vertices = obj.vertices
            height = obj.height if obj.height != mp.inf else 0
            axis = obj.axis if hasattr(obj, 'axis') else mp.Vector3(0, 0, 1)

            # Compute height vector along the axis
            V = mp.Vector3(axis.x * height / 2, axis.y * height / 2, axis.z * height / 2)
         
            # Create 3D points from 2D vertices +/- height vector
            points = []
            for vertex in vertices:
                # Add points at both ends along the axis
                points.extend([
                    (vertex.x - V.x, vertex.y - V.y, vertex.z - V.z),
                    (vertex.x + V.x, vertex.y + V.y, vertex.z + V.z),
                ])
        else:
            # For objects with .size: create corner points
            size = obj.size
            center = obj.center
        
            hx, hy, hz = size.x / 2, size.y / 2, size.z / 2
            
            # Generate all 8 corners of the bounding box
            points = [
                (center.x - hx, center.y - hy, center.z - hz),
                (center.x + hx, center.y - hy, center.z - hz),
                (center.x - hx, center.y + hy, center.z - hz),
                (center.x + hx, center.y + hy, center.z - hz),
                (center.x - hx, center.y - hy, center.z + hz),
                (center.x + hx, center.y - hy, center.z + hz),
                (center.x - hx, center.y + hy, center.z + hz),
                (center.x + hx, center.y + hy, center.z + hz),
            ]
        
        # Update global bounds from all points
        for x, y, z in points:
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)
            zmin = min(zmin, z)
            zmax = max(zmax, z)

    # Create final mp.Vector3
    bounding_size = mp.Vector3(xmax - xmin, ymax - ymin, zmax - zmin)
    bounding_center = mp.Vector3((xmin + xmax)/2, (ymin + ymax)/2, (zmin + zmax)/2)

    return bounding_size, bounding_center

def finding_mode_from_geometry(geometry, mode=1, frequency=1, resolution=20, time=50):
    """
    Compute the TE mode fields for a given geometry in a Meep simulation.

    This function calculates the cross-sectional profiles of the electric (Ez) 
    and magnetic (Hy) fields for a specified TE mode order. The fields are 
    computed at a given frequency, using a simulation with specified spatial 
    resolution and run time.

    Parameters
    ----------
    geometry : list of mp.GeometricObject
        The geometry of the simulation region (e.g., blocks, cylinders, etc.).
    mode : int, optional
        Mode order to compute (default is 1).
    frequency : float, optional
        Frequency at which to calculate the mode (default is 1).
    resolution : int, optional
        Spatial resolution of the simulation (default is 20).
    time : float, optional
        Total simulation time (in Meep time units) (default is 50).

    Returns
    -------
    Ez_cross_norm : np.ndarray
        Normalized electric field cross section along the y-axis.
    Hy_cross_norm : np.ndarray
        Normalized magnetic field cross section along the y-axis.
    neff : float
        Effective refractive index of the mode.

    Notes
    -----
    - The simulation uses a PML boundary with thickness 1.0.
    - The source is an eigenmode source centered at the left edge of the simulation cell.
    - Cross sections are taken at the x-coordinate where Ez reaches its maximum.
    - Fields are normalized to the total optical power of the mode.
    - The effective index is computed from the normalized field and material distribution.
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
    Hy1_cross_norm *= np.max(np.abs(Ez1_cross)) / np.max(np.real(Ez1_cross))
    
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

def generate_modal_source(Ez_cross, Hy_cross, cross_axis, src_position, src_size, src_decay=3, frequency=1):
    """
    Create a custom Meep source that generates an electromagnetic mode with specified 
    electric and magnetic field profiles along a given cross-sectional axis.

    Parameters
    ----------
    Ez_cross : array-like
        Electric field (Ez) profile along the cross-sectional axis.
    Hy_cross : array-like
        Magnetic field (Hy) profile along the cross-sectional axis.
    cross_axis : array-like
        Coordinates corresponding to the cross-sectional profiles (e.g., y-axis positions).
    src_position : mp.Vector3
        Position of the source in the simulation domain.
    src_size : mp.Vector3
        Spatial extent of the source.
    src_decay : float
        Amount of radiant after which the source amplitude is switched off.
    frequency : float, optional
        Oscillation frequency of the source (default is 1).

    Returns
    -------
    list of mp.Source
        A list containing the Meep sources for the Ez and Hy components, 
        with spatial profiles defined by interpolation of the provided cross sections 
        and temporal profiles as harmonic oscillations at the specified frequency.

    Notes
    -----
    - The Ez field is interpolated along the cross_axis to define its spatial dependence.
    - The Hy field is interpolated similarly.
    - Temporal profiles are sinusoidal: Ez ~ cos(2πft), Hy ~ sin(2πft).
    - Intended for use in simulations where both electric and magnetic field components
      of a mode are needed.
    """

    # Electric field cross section in z direction
    Ez_function = interp1d(
        cross_axis,
        Ez_cross,
        kind='cubic',
        bounds_error=False,
        fill_value=0.0
    )
    # Magnetic field cross section in y direction
    Hy_function = interp1d(
        cross_axis,
        Hy_cross,
        kind='cubic',
        bounds_error=False,
        fill_value=0.0
    )

    # The temporal dependence is an harmonic oscillation at the specified frequency
    def temporal_profile_Ez(t):
        if frequency * t > src_decay:
            return 0
        return np.cos(2 * np.pi * frequency * t) 
    
    def temporal_profile_Hy(t):
        if frequency * t > src_decay:
            return 0
        return np.sin(2 * np.pi * frequency * t)
    
    # The spatial dependence is an interpolation of the mode cross section
    def spatial_profile_Ez(r):
        return float(Ez_function(r.y))

    def spatial_profile_Hy(r):
        return float(Hy_function(r.y))
    
    # Generate a Meep source with electric and magnetic components
    source = [mp.Source(
    src=mp.CustomSource(temporal_profile_Ez),
    center=src_position,
    size=src_size,
    component = mp.Ez,
    amp_func = spatial_profile_Ez),
    mp.Source(
    src=mp.CustomSource(temporal_profile_Hy),
    center=src_position,
    size=src_size,
    component = mp.Hy,
    amp_func = spatial_profile_Hy)]

    return source


def calculate_modal_overlap(Ez_cross, Hy_cross, Ez_field, Hy_field, cross_axis=None, field_axis=None):
    """
    Compute the normalized modal overlap between a guided-mode cross section and 
    the electromagnetic fields obtained from a simulation.

    Parameters
    ----------
    Ez_cross : array-like
        Electric field (Ez) profile of the mode along the cross section.
    Hy_cross : array-like
        Magnetic field (Hy) profile of the mode along the cross section.
    Ez_field : array-like
        Electric field (Ez) distribution from the simulation domain. Must have a 
        compatible shape with Hy_field for element-wise multiplication.
    Hy_field : array-like
        Magnetic field (Hy) distribution from the simulation domain.
    cross_axis : array like
        coordinates where Ez_cross and Hy_cross are defined.
        If None, assumes same as field_axis.
    field_axis : array like
        coordinates where Ez_field and Hy_field are defined.
        If None, assumes same as cross_axis.

    Returns
    -------
    float or ndarray
        The normalized modal overlap value(s). If Ez_field and Hy_field contain 
        multiple slices along one axis (e.g., different z-planes), the function 
        returns an array of overlap values, one per slice.

    Notes
    -----
    - The overlap integral is computed as:
      
          0.5 * ∫ (Ez_cross * Hy_field* + Hy_cross * Ez_field*) dx

      evaluated element-wise along the specified axis.
    - The mode normalization is computed from:

          ∫ (Ez_cross * Hy_cross*) dx

    - The field normalization is evaluated at the position of the maximum real Ez 
      field component and uses:

          ∫ (Ez_field * Hy_field*) dx

    - The final output is the absolute value of the normalized overlap:

          |overlap / mode_norm / field_norm|

    - This metric is useful for estimating coupling efficiency between the analytical 
      mode profile and the simulated fields.

    """
    if cross_axis is not None and field_axis is not None:
        # Interpolate the mode cross sections to match the field axis
        Ez_cross_interp = interp1d(
            cross_axis,
            Ez_cross,
            kind='cubic',
            bounds_error=False,
            fill_value=0.0
        )
        Hy_cross_interp = interp1d(
            cross_axis,
            Hy_cross,
            kind='cubic',
            bounds_error=False,
            fill_value=0.0
        )

        # Use the field axis to get the mode in the same grid 
        Ez_cross = Ez_cross_interp(field_axis)
        Hy_cross = Hy_cross_interp(field_axis)

    # Calculate the overlap integral as the power between the modes and the fields.
    overlap = 0.5 * np.sum(np.abs(Ez_cross * np.conj(Hy_field) + Hy_cross * np.conj(Ez_field)), axis=1)

    # Calculate the normalization factor the the mode
    mode_norm = np.sqrt(np.abs(np.sum((Ez_cross * np.conj(Hy_cross)))))

    # Calculate the normalization factor for the fields, based on the maximum Ez field position
    x_imax = np.argmax(np.real(Ez_field))
    x_max_index, _ = np.unravel_index(x_imax, Ez_field.shape)
    field_norm = np.sqrt(np.abs(np.sum((Ez_field[x_max_index] * np.conj(Hy_field[x_max_index])))))

    # Normalize the overlap
    overlap_norm = abs(overlap / mode_norm / field_norm)

    return overlap_norm