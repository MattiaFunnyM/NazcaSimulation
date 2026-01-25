import meep as mp
import numpy as np
import matplotlib.pyplot as plt

################################################################################
#################### FULL 3D SIMULATION FUNCTIONS ##############################
################################################################################

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
                    (vertex.x + V.x, vertex.y + V.y, vertex.z + V.z)])
        if isinstance(obj, mp.Cylinder):
            # For cylinder we have the radius, the height and the center
            center = obj.center
            radius = obj.radius
            height = obj.height if obj.height != mp.inf else 0

            # Extract the axis to know in which direction the cylinder is oriented
            axis = obj.axis if hasattr(obj, "axis") else mp.Vector3(0, 0, 1)
            ax, ay, az = axis.x, axis.y, axis.z
  
            # Half-extents of the AABB
            dx = abs(ax)*(height/2) + radius*np.sqrt(max(0.0, 1-ax*ax))
            dy = abs(ay)*(height/2) + radius*np.sqrt(max(0.0, 1-ay*ay))
            dz = abs(az)*(height/2) + radius*np.sqrt(max(0.0, 1-az*az))

            # Calculate points for the cylinder
            points = [
                (center.x - dx, center.y - dy, center.z - dz),
                (center.x + dx, center.y - dy, center.z - dz),
                (center.x - dx, center.y + dy, center.z - dz),
                (center.x + dx, center.y + dy, center.z - dz),
                (center.x - dx, center.y - dy, center.z + dz),
                (center.x + dx, center.y - dy, center.z + dz),
                (center.x - dx, center.y + dy, center.z + dz),
                (center.x + dx, center.y + dy, center.z + dz)]
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


def find_mode_from_cross_section(geometry, cross_section, mode_order, frequency, sim_resolution=32, parity=None):
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
        eigensolver_tol=1e-5,
        parity=mp.NO_PARITY if parity is None else parity
    )

    # Extract the field components over the entire cross section
    x_min = cross_section.center.x - cross_section.size.x / 2
    x_max = cross_section.center.x + cross_section.size.x / 2
    y_min = cross_section.center.y - cross_section.size.y / 2
    y_max = cross_section.center.y + cross_section.size.y / 2
    y_points = np.linspace(y_min, y_max, int(sim_size.y*sim_resolution))
    x_points = np.linspace(x_min, x_max, int(sim_size.x*sim_resolution))
    Ex = np.array([[mode.amplitude(component=mp.Ex, point=mp.Vector3(x, -y, 0)) 
                    for x in x_points] for y in y_points])
    Ey = np.array([[mode.amplitude(component=mp.Ey, point=mp.Vector3(x, -y, 0)) 
                    for x in x_points] for y in y_points])
    Ez = np.array([[mode.amplitude(component=mp.Ez, point=mp.Vector3(x, -y, 0)) 
                    for x in x_points] for y in y_points])
    Hx = np.array([[mode.amplitude(component=mp.Hx, point=mp.Vector3(x, -y, 0)) 
                    for x in x_points] for y in y_points])
    Hy = np.array([[mode.amplitude(component=mp.Hy, point=mp.Vector3(x, -y, 0))
                    for x in x_points] for y in y_points])
    Hz = np.array([[mode.amplitude(component=mp.Hz, point=mp.Vector3(x, -y, 0)) 
                    for x in x_points] for y in y_points])
    k_value = mode.k[2]
    neff = k_value / frequency

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


def visualize_geometry(geometry, resolution=32):
    """
    Visualize the refractive-index distribution of a Meep geometry.

    This function initializes a minimal Meep simulation containing the provided
    geometry displays the result as a 2D color map. 

    Parameters
    ----------
    geometry : list of mp.GeometricObject
        List of Meep geometric objects (e.g., mp.Block) defining the material
        distribution in the simulation cell.
    resolution : int, optional
        Spatial resolution in pixels per micron used to discretize the geometry.
        This parameter affects only the visualization fidelity and not numerical
        accuracy of electromagnetic solutions. Default is 32.
    """
    
    cell_size, cell_center = compute_geometry_bounds(geometry)
    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        resolution=resolution)

    # Initialize simulation (no sources needed)
    sim.init_sim()

    # Get epsilon grid
    eps = sim.get_array(
        center=mp.Vector3(),
        size=cell_size,
        component=mp.Dielectric
    )

    # Get central cross section
    eps = eps[:, :, int(eps.shape[2]/2)]

    # Convert epsilon → refractive index
    n = np.sqrt(eps)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(
        n.T,
        interpolation="nearest",
        origin="lower",
        cmap="viridis",
        extent=[
            -cell_size.x/2, cell_size.x/2,
            -cell_size.y/2, cell_size.y/2
        ]
    )
    plt.colorbar(label="Refractive index n")
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.tight_layout()
    plt.show()


def visualize_condensated_field(Field3D, propagation_dir = 'z', wide_dir = 'x', x_range = None, y_range = None, colormap = None):
    """
    Condensate a 3D field by summing its magnitude over the remaining direction and visualize it.

    This function takes a 3D complex field, calculates the magnitude of each
    complex value, and then condenses the field into a 2D representation. This
    condensation is achieved by summing the magnitudes along the axis that is
    neither the specified `propagation_dir` nor the `wide_dir`.
    The resulting 2D field is then displayed as an image using matplotlib, with
    optional custom axis numerical ranges and a specified colormap.

    Parameters
    ----------
    Field3D : np.ndarray
              3D complex field with dimensions (Nx, Ny, Nz). The indices 0, 1, 2
              correspond internally to 'x', 'y', 'z' directions respectively.
    propagation_dir : str, optional
        'x', 'y', or 'z'. This specifies the primary direction of propagation.
    wide_dir : str, optional
        'x', 'y', or 'z'. This specifies the "wide" or transverse direction to
        be retained in the 2D field. 
    x_range : tuple or list of 2 floats, optional
        Custom numerical range `(min_val, max_val)` for the x-axis of the plot.
    y_range : tuple or list of 2 floats, optional
        Custom numerical range `(min_val, max_val)` for the y-axis of the plot.
    colormap : str, optional
        Name of a valid matplotlib colormap to use for the image.
    """

    dir_map = {'x': 0, 'y': 1, 'z': 2}

    if propagation_dir not in dir_map or wide_dir not in dir_map:
        raise ValueError("propagation_dir and wide_dir must be 'x', 'y', or 'z'")
    if propagation_dir == wide_dir:
        raise ValueError("Propagation and wide directions must be different.")

    prop_axis = dir_map[propagation_dir]
    wide_axis = dir_map[wide_dir]

    # Determine the axis to sum over
    cond_axis = list({0, 1, 2} - {prop_axis, wide_axis})[0]

    # Perform the condensation by summing the absolute values along the determined axis
    Field3D_cond = np.sum(np.abs(Field3D), axis=cond_axis)

    # --- Plotting the condensed field ---
    fig, ax = plt.subplots(figsize=(8, 6))

    # Use the specified colormap or default to None (matplotlib's default)
    current_cmap = colormap if colormap else None

    # Display the condensed field as an image
    im = ax.imshow(Field3D_cond, cmap=current_cmap)

    # Set x-axis range and bold tick labels if provided
    if x_range is not None:
        ax.set_xlim(x_range[0], x_range[1])
        for tick in ax.get_xticklabels():
            tick.set_fontweight('bold')
    else:
        # If no range is assigned, remove x-axis tick marks and labels
        ax.set_xticks([])
        ax.set_xticklabels([])

    # Set y-axis range and bold tick labels if provided
    if y_range is not None:
        ax.set_ylim(y_range[0], y_range[1])
        for tick in ax.get_yticklabels():
            tick.set_fontweight('bold')
    else:
        # If no range is assigned, remove x-axis tick marks and labels
        ax.set_yticks([])
        ax.set_yticklabels([])

    # Add a color bar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    for t in cbar.ax.get_yticklabels(): # For a vertical colorbar, these are y-tick labels
        t.set_fontweight('bold')

    plt.tight_layout() 
    plt.show()


