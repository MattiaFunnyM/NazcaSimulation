import meep as mp
import numpy as np
from matplotlib.path import Path
from scipy.interpolate import interp1d

################################################################################
#################### FULL 3D SIMULATION FUNCTIONS ##############################
################################################################################

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

def find_mode_from_cross_section(geometry, cross_section, mode_order, frequency, sim_resolution=32):
    """
    Uses meep internal functions to find the mode of given mode_order from
    the provided cross_section. Cross section is taking by compiling the 
    geometry and taking the material at the cross_section volume.
    The mode is calculated at the given frequency.
    Field is expected to propagate along the z direction.
    Simulation resolution decide the amount of points per unit length.

    Return:
    dictionary with the field components and effective index of the mode.
    Parameters
    ----------
        frequency : float,
            Frequency at which to calculate the mode.
        mode_order : int,
            Mode order to compute (1 for fundamental mode).
        k_value : mp.Vector3,   
            k-point vector for mode calculation.
        Ex : ndarray
            2D complex array representing the x-component of the electric field.
        Ey : ndarray
            2D complex array representing the y-component of the electric field.
        Ez : ndarray
            2D complex array representing the z-component of the electric field.
        Hx : ndarray
            2D complex array representing the x-component of the magnetic field.
        Hy : ndarray
            2D complex array representing the y-component of the magnetic field.
        Hz : ndarray
            2D complex array representing the z-component of the magnetic field.
        neff : float
            Effective refractive index of the mode.
    """
    # Take the simulation size and center from its geometry
    sim_size, sim_center = compute_geometry_bounds(geometry)

    # Define the source needed only to initialize the fields
    sources = [mp.Source(
        mp.GaussianSource(frequency=1),
        component=mp.Ey,
        center=mp.Vector3(0, 0, 0))]
    
    # Initialize Simulation
    sim = mp.Simulation(
    cell_size=sim_size,
    geometry=geometry,
    resolution=sim_resolution,
    sources=sources,
    dimensions=3)

    # Run a short simulation to initialize fields
    sim.run(until=1/sim_resolution)
     
    # Perform the mode_calculation at the given cross section and frequency
    mode = sim.get_eigenmode(
        frequency=frequency,
        direction=mp.Z,
        band_num=mode_order,
        where=cross_section,
        kpoint=mp.Vector3(0, 0, 1),
        eigensolver_tol=1e-4
    )

    # Extract the field components over the entire cross section
    y_points = np.linspace(-sim_size.y/2, sim_size.y/2, int(sim_size.y*sim_resolution))
    x_points = np.linspace(-sim_size.x/2, sim_size.x/2, int(sim_size.x*sim_resolution))
    Ex = np.array([[mode.amplitude(component=mp.Ex, point=mp.Vector3(x, y, 0)) 
                    for x in x_points] for y in y_points])
    Ey = np.array([[mode.amplitude(component=mp.Ey, point=mp.Vector3(x, y, 0)) 
                    for x in x_points] for y in y_points])
    Ez = np.array([[mode.amplitude(component=mp.Ez, point=mp.Vector3(x, y, 0)) 
                    for x in x_points] for y in y_points])
    Hx = np.array([[mode.amplitude(component=mp.Hx, point=mp.Vector3(x, y, 0)) 
                    for x in x_points] for y in y_points])
    Hy = np.array([[mode.amplitude(component=mp.Hy, point=mp.Vector3(x, y, 0))
                    for x in x_points] for y in y_points])
    Hz = np.array([[mode.amplitude(component=mp.Hz, point=mp.Vector3(x, y, 0)) 
                    for x in x_points] for y in y_points])
    k_value = mode.k[2]
    neff = k_value / (2 * np.pi * frequency)

    # Create the output dictionary
    output = {
        'frequency': frequency,
        'mode_order': mode_order,
        'k_value': k_value,
        'Ex': Ex,
        'Ey': Ey,
        'Ez': Ez,
        'Hx': Hx,
        'Hy': Hy,
        'Hz': Hz,
        'neff': neff
    }

    return output

################################################################################
##################### RAW 2D SIMULATION FUNCTIONS ##############################
################################################################################

def polygon_to_materialgrid(vertices, medium_outside, medium_inside, sub_resolution=20, x_axis=None, y_axis=None):
    """
    Convert a polygon defined by vertices into a 2D MaterialGrid.

    This function generates a material grid representing a polygon by determining
    whether grid points lie inside or outside the polygon. It supports custom grid bounds 
    and allows for supersampling of boundary pixels to refine the material assignment.

    Parameters
    ----------
    vertices : list of objects
        List of vertex objects defining the polygon, each with attributes x and y.
    medium_outside : object
        Material property assigned to areas outside the polygon.
    medium_inside : object
        Material property assigned to areas inside the polygon.
    sub_resolution : int, optional
        Resolution factor for supersampling the grid boundary, default is 20.
    x_axis : tuple of float, optional
        Custom (xmin, xmax) boundaries for the grid. If None, calculated from vertices.
    y_axis : tuple of float, optional
        Custom (ymin, ymax) boundaries for the grid. If None, calculated from vertices.

    Returns
    -------
    matgrid : mp.MaterialGrid
        A MaterialGrid object defining varied materials inside and outside the polygon.
    """
     
    # Convert vertex objects to a NumPy array of (x, y) coordinates
    poly_xy = np.array([(v.x, v.y) for v in vertices], dtype=float)

    # Determine the bounds for the grid based on input axes or polygon vertices
    if x_axis is None:
        xmin, xmax = poly_xy[:, 0].min(), poly_xy[:, 0].max()
    else:
        xmin, xmax = x_axis

    if y_axis is None:
        ymin, ymax = poly_xy[:, 1].min(), poly_xy[:, 1].max()
    else:
        ymin, ymax = y_axis

    # Calculate the grid size
    Lx, Ly = xmax - xmin, ymax - ymin

    # Determine the number of subdivisions in each dimension
    Nx = max(1, int(np.ceil(Lx * sub_resolution)))
    Ny = max(1, int(np.ceil(Ly * sub_resolution)))
    Nz = 1

    # Calculate the resolution of each cell
    dx = Lx / Nx
    dy = Ly / Ny

    # Create a Matplotlib path object for the polygon
    poly_path = Path(poly_xy)

    # Calculate coordinates of pixel centers
    x_cent = xmin + (np.arange(Nx) + 0.5) * dx
    y_cent = ymin + (np.arange(Ny) + 0.5) * dy
    Xc, Yc = np.meshgrid(x_cent, y_cent, indexing="ij")
    pts_c = np.column_stack([Xc.ravel(), Yc.ravel()])

    # Determine which pixel centers are inside the polygon
    inside_c = poly_path.contains_points(pts_c).reshape(Nx, Ny)
    
    # Initialize weights based on inside status
    weights = inside_c.astype(np.float64)

    # Identify boundary pixels where inside/outside changes
    boundary = np.zeros((Nx, Ny), dtype=bool)
    boundary[1:, :]  |= (inside_c[1:, :] != inside_c[:-1, :])
    boundary[:-1, :] |= (inside_c[:-1, :] != inside_c[1:, :])
    boundary[:, 1:]  |= (inside_c[:, 1:] != inside_c[:, :-1])
    boundary[:, :-1] |= (inside_c[:, :-1] != inside_c[:, 1:])
    
    bi, bj = np.nonzero(boundary)
    nb = bi.size

    # Supersample boundary pixels for higher accuracy
    if nb > 0 and sub_resolution > 1:
        ox = (np.arange(sub_resolution) + 0.5) / sub_resolution
        oy = (np.arange(sub_resolution) + 0.5) / sub_resolution
        x0 = xmin + bi * dx
        y0 = ymin + bj * dy
        Xs = x0[:, None, None] + ox[None, :, None] * dx
        Ys = y0[:, None, None] + oy[None, None, :] * dy
        pts = np.column_stack([Xs.ravel(), Ys.ravel()])
        inside_sub = poly_path.contains_points(pts).reshape(nb, sub_resolution, 1)
        frac = inside_sub.mean(axis=(1, 2))
        weights[bi, bj] = frac

    # Create a 3D weight array for the MaterialGrid
    weights3d = weights[:, :, None]

    # Create and return the MaterialGrid object with the calculated weights
    matgrid = mp.MaterialGrid(
        mp.Vector3(Nx, Ny, Nz),
        medium1=medium_outside,
        medium2=medium_inside,
        weights=weights3d
    )
    return matgrid

def normalizing_mode_field(Ez, eps_cross, Hy=None, frequency=1, delta=1):
        """
        Normalize electromagnetic field components and calculate effective refractive index.

        This function extracts a cross-sectional slice of the electromagnetic field at the
        location of maximum electric field intensity, and
        normalizes the field components accordingly. It also calculates the effective 
        refractive index of the mode based on the permittivity distribution.

        Parameters
        ----------
        Ez : ndarray
            2D complex array representing the z-component of the electric field.
        Hy : ndarray
            2D complex array representing the y-component of the magnetic field.
            Not necessarly for the calculation. 
            If not None, it's normalization cross section is returned.
        eps_cross : ndarray
            1D array of permittivity values along the cross-section.
        frequency : float, optional
            Frequency at which to calculate the mode (default is 1).
        delta : float
            Spatial step size for numerical integration (default is 1).
        
        Returns
        -------
        Ez_cross_norm : ndarray
            Normalized real part of the electric field along the cross-section.
        neff : float
            Effective refractive index of the mode.
        """

        # Locate the position of maximum electric field intensity
        flat_index = np.argmax(np.real(Ez))
        x_max_index, _ = np.unravel_index(flat_index, Ez.shape)
    
        # Extract cross-sectional field profiles at the maximum intensity location
        Ez_cross = np.real(Ez[x_max_index, :])
        if Hy is not None:
            Hy_cross = np.real(Hy[x_max_index, :])

        # Calculate the intensity of the mode
        intensity = np.abs(Ez_cross)**2

        # Integral to normalize the result
        normalization_term = np.sum(intensity) * delta

        # Apply correction factor to account for complex field magnitude
        Ez_cross_norm = Ez_cross / np.sqrt(normalization_term)
        if Hy is not None:
             Hy_cross_norm = Hy_cross / np.sqrt(normalization_term)

        # Propagation constant
        k0 = 2 * np.pi * frequency

        # Integral to takes into account the field distribution
        permittivity_term = np.sum(k0**2 * eps_cross * intensity) * delta

        # Integral to takes into account the field confinment
        confirment_term = np.sum(np.abs(np.gradient(Ez_cross, delta))**2) * delta
        
        # neff is calculated from the beta value
        beta_squared = (permittivity_term - confirment_term) / normalization_term
        neff = np.sqrt(beta_squared) / k0
        
        if Hy is None:
             return Ez_cross_norm, neff
        else:
             return Ez_cross_norm, Hy_cross_norm, neff
             
def finding_mode_from_geometry(geometry, mode=1, frequency=1, resolution=20, time=50, eps_cross=None):
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
        eps_cross : ndarray
            1D array of permittivity values along the cross-section.
            If None, uses the one from simulation

        Returns
        -------
        Ez_cross_norm : np.ndarray
            Normalized electric field cross section along the y-axis.
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
            src=mp.ContinuousSource(frequency=frequency),
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
        Ez = sim.get_dft_array(dft, mp.Ez, 0)
        Hy = sim.get_dft_array(dft, mp.Hy, 0)
        
        # Calculate the positional variation needed for normalization
        delta = 1 / resolution  
        
        # Extract the information about the material
        if eps_cross is None:
            eps_data = sim.get_array(
                component=mp.Dielectric,
                center=mp.Vector3(),           
                size=sim_size)
            eps_cross = eps_data[int(len(eps_data)/2), :]
        
        # Return the normalize mode field cross section 
        return normalizing_mode_field(Ez=Ez, Hy=Hy, frequency=frequency, delta=delta, eps_cross=eps_cross)

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

def calculate_modal_overlap(Ez_cross, Hy_cross, Ez_field, Hy_field, cross_axis=None, field_axis=None, field_norm=None):
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
    field_norm : value
        numerical value used to normalized the Ez field.
        If None, the Ez field is used to determine the normalization

    Returns
    -------
    float or ndarray
        The normalized modal overlap value(s) between Ez_cross and Hy_cross.

    Notes
    -----
    - The overlap integral is computed as:
      
          0.5 * ∫ (Ez_cross * Hy_field* + Hy_cross * Ez_field*) dx

      evaluated element-wise along the specified axis.
    - The mode normalization is computed from:

          ∫ (Ez_cross * Hy_cross*) dx

    - The field normalization (if field_norm is not None) is evaluated as:

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
    overlap = 0.5 * np.sum(Ez_cross * np.conj(Hy_field) + Hy_cross * np.conj(Ez_field), axis=1)

    # Calculate the normalization factor the the mode
    mode_norm = np.sqrt(np.abs(0.5 * np.real(np.sum((Ez_cross * np.conj(Hy_cross) + Hy_cross * np.conj(Ez_cross))))))

    # Calculate the normalization factor for the fields, based on the maximum Ez field position
    if field_norm is None:
        x_max_index = np.argmax(np.abs(0.5 * np.real(np.sum(Ez_field * np.conj(Hy_field) + Hy_field * np.conj(Ez_field), axis=1))))
        field_norm = np.sqrt(np.abs(0.5 * np.real(np.sum(Ez_field[x_max_index] * np.conj(Hy_field[x_max_index]) + 
                                                        Hy_field[x_max_index] * np.conj(Ez_field[x_max_index])))))

    # Normalize the overlap
    overlap_norm = abs(overlap / mode_norm / field_norm)**2

    return overlap_norm, field_norm

def stop_when_field_exceeds(component, point, threshold):
    """
    Creates a termination condition for Meep to stop the simulation when 
    the field amplitude at a specific location exceeds a given threshold.

    This function returns a callable (closure) that Meep executes at every 
    time step. It is intended to be passed to the 'until' argument of 
    simulation.run().

    Parameters
    ----------
    component : mp.Component
        The field component to monitor (e.g., mp.Ez, mp.Hy).
    point : mp.Vector3
        The specific spatial coordinate to check.
    threshold : float
        The absolute magnitude value. The simulation stops when |field| > threshold.

    Returns
    -------
    Callable[[mp.Simulation], bool]
        A function that returns True (stop) or False (continue) based on the field value.
    """
    def check_condition(sim):
        # Extract the complex field value at the specific point
        field_value = sim.get_field_point(component, point)
        
        # Calculate the magnitude (absolute value)
        magnitude = np.abs(field_value)
        
        # Return True to stop simulation if threshold is breached
        return magnitude > threshold

    return check_condition

def simulate_without_reflections(simulation, number_windows, max_active_windows, window_volume, frequencies):
        """
        Propagate an electromagnetic pulse using a "sliding window" FDTD technique to 
        capture frequency-domain fields while minimizing memory usage and reflections.

        This method tracks the pulse as it moves through the geometry, activating and 
        deactivating Fourier Transform (DFT) monitors dynamically. This allows for the 
        simulation of long waveguides or large domains where capturing the global field 
        simultaneously would be computationally prohibitive or memory-intensive.

        Parameters
        ----------
        simulation : meep.Simulation
                The initialized Meep simulation object containing geometry, sources, and 
                boundary conditions.
        number_windows : int
                The total number of spatial windows (chunks) to record along the propagation direction.
        max_active_windows : int
                The size of the sliding buffer. Determines how many DFT monitors are active 
                simultaneously. Must be at least 3 to allow for robust overlap and triggering.
        window_volume : meep.Volume
                The volume of a single recording window. The code assumes these windows are 
                stacked sequentially along the x-axis.
        frequencies : list of float
                The list of frequencies to capture in the DFT monitors.

        Returns
        -------
        ez_dict : dict of ndarray
                Dictionary mapping each frequency to the stitched Electric field (Ez) matrix.
                The shape represents the full concatenated domain.
        hy_dict : dict of ndarray
                Dictionary mapping each frequency to the stitched Magnetic field (Hy) matrix.

        Notes
        -----
        - **Calibration:** The function performs a pre-run (resetting afterwards) to determine 
        the peak field amplitude. This is used to set a dynamic threshold for the 
        propagation trigger.
        - **Phase Stitching:** Since DFT windows are initialized at different simulation 
        times, their raw phases are not aligned. This function attempts to correct the 
        phase empirically by comparing the overlapping pixels of adjacent windows.
        - **Triggering:** The simulation advances based on a field-threshold trigger 
        (`stop_when_fields_above`) to detect when the pulse reaches the next window.

        """
        # Method works if there are at least 3 max_active windows
        if max_active_windows < 3:
                raise ValueError("Not enough active windows for the method (minimum 3 required).")
        
        # We run a short simulation to estimate the maximum source amplitude. 
        calibration_vol = mp.Volume(center=mp.Vector3(),
                                   size=mp.Vector3(window_volume.size.x * number_windows, window_volume.size.y))  
        simulation.run(until=30)         
        
        # Measure peak field to set a dynamic trigger threshold later
        temp_field = simulation.get_array(component=mp.Ez, vol=calibration_vol)
        max_field_amp = np.max(np.abs(temp_field))
        
        # Completely reset Meep state to start the real simulation
        simulation.reset_meep()

        # Initialization
        window_active_idx = 0
        extra_index_distance = int(max_active_windows / 2) + 1
        simulation_time = 0

        # Initialize storage dictionaries
        ez_dict = {f: [] for f in frequencies}
        hy_dict = {f: [] for f in frequencies}

        # Iterate enough times to cover all windows plus the buffer fade-out
        total_cycles = number_windows + max_active_windows - 1  
        for sim_idx in range(total_cycles):
                
                # Add New DFT Monitor (if we haven't reached the end of the geometry)
                if len(simulation.dft_objects) < max_active_windows and sim_idx < number_windows:
                        # Calculate the center position of the new window 
                        active_center = window_volume.center + mp.Vector3((window_volume.size.x) * (sim_idx + 0.5))
                        active_volume = mp.Volume(center=active_center, size=window_volume.size)
                        
                        # Create the monitor (fields are accumulated starting from NOW)
                        simulation.add_dft_fields([mp.Ez, mp.Hy], frequencies, where=active_volume) 
                
                # Propagate Pulse
                # Only start running when the buffer is full or we've created all necessary windows
                if sim_idx >= number_windows or len(simulation.dft_objects) >= max_active_windows:   
                
                        # Logic 1: Pulse is still traveling through the main domain
                        if window_active_idx < number_windows - extra_index_distance:
                                # Specific point ahead of the current window to check for signal arrival
                                trigger_pos = window_volume.center + mp.Vector3((window_volume.size.x) * (window_active_idx + extra_index_distance))
                                
                                # Run until the field at trigger_pos exceeds 5% of the calibrated max
                                simulation.run(until=stop_when_field_exceeds(mp.Ez, trigger_pos, max_field_amp * 0.05))
                                simulation_time = simulation.meep_time()
                        
                        # Logic 2: Pulse is exiting the domain (extrapolation)
                        else:
                                remaining_steps = number_windows - max_active_windows
                                simulation.run(until=simulation_time / remaining_steps)

                        # We always process the oldest active window (index 0)
                        dft_obj = simulation.dft_objects[0]

                        for idf, frequency in enumerate(frequencies):
                                Ez = simulation.get_dft_array(dft_obj, mp.Ez, idf)
                                Hy = simulation.get_dft_array(dft_obj, mp.Hy, idf)

                                # Because DFT monitors started at different times, their phases are offset.
                                # We align them by matching the phase of the overlapping pixels.
                                if window_active_idx > 0:  
                                        prev_Ez = ez_dict[frequency][-1]
                                        
                                        # We pick the center_idx of the Y-axis to avoid edge noise.
                                        center_idx = len(prev_Ez[-1]) // 2
                                        
                                        prev_phase = np.angle(prev_Ez[-1, center_idx]) 
                                        curr_phase = np.angle(Ez[0, center_idx])       
                                        phase_correction = curr_phase - prev_phase
                        
                                        # Apply correction to the whole window
                                        Ez = Ez * np.exp(-1j * phase_correction)
                                        Hy = Hy * np.exp(-1j * phase_correction)

                                # Remove the first row of pixels to avoid duplicating the stitching boundary
                                ez_dict[frequency].append(Ez[1:])
                                hy_dict[frequency].append(Hy[1:])

                        # Remove the oldest monitor to free up the slot
                        simulation.dft_objects.pop(0)
                        window_active_idx += 1

        # Stack the list of arrays into a single continuous matrix for each frequency
        for frequency in frequencies:
                ez_dict[frequency] = np.vstack(ez_dict[frequency])
                hy_dict[frequency] = np.vstack(hy_dict[frequency])
        
        return ez_dict, hy_dict
