import meep as mp
import matplotlib.pyplot as plt

# Define Simulation Size - um
cell_width = 16   # X direction (-25, 25)
cell_height = 16  # Y direction (-5, 5)
cell_depth = 0    # Z direction - Absent
resolution = 20   # grid points per reference wavelength
cell_size = mp.Vector3(cell_width, cell_height, cell_depth) 

#(-25, 5)                    (25, 5)
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#(-25, -5)                  (25, -5)

# Define the Simulation Source
material_epsilon = 12     # Relative permittivity of the waveguide material
source_frequency = 0.1    # Frequency corresponding to the reference wavelength
source = mp.Source(
    mp.ContinuousSource(frequency=source_frequency),  # Gaussian pulse
    component=mp.Ez,            
    center=mp.Vector3(-7, -3.5, 0), # Leave a bit of space from the left boundary
    size=mp.Vector3(0, 1)           # Line source for better coupling
)

#(-25, 5)                    (25, 5)
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ···#o··#···#···#···#···#···#
#   ···#o··#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#(-25, -5)                  (25, -5)

# Define the waveguide geometry
horizontal = mp.Block(mp.Vector3(12,1,1), # Size
                     center=mp.Vector3(-2, -3.5, 0), # Center position
                     material=mp.Medium(epsilon=material_epsilon))
vertical = mp.Block(mp.Vector3(1,12,1), # Size
                    center=mp.Vector3(3.5, 2, 0), # Center
                    material=mp.Medium(epsilon=material_epsilon))
geometry = [horizontal, vertical]

#(-25, 5)                    (25, 5)
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ===#o==#===#===#===#===#===#
#   ===#o==#===#===#===#===#===#
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#(-25, -5)                  (25, -5)

# Define the simulation object
pml_layers = [mp.PML(1.0)]  # Perfectly matched layer boundary conditions to avoid reflections
sim = mp.Simulation(
    cell_size=cell_size,
    sources=[source],
    geometry=geometry,
    resolution=resolution,
    boundary_layers=pml_layers
)

# Run the simulation for 200 units
sim.run(mp.at_beginning(mp.output_epsilon), # We want to access structure only at the beginning,
        until=200)

# Visualize the geometry simulated
eps_data = sim.get_array(center=mp.Vector3(0, 0, 0), 
                         size=cell_size, 
                         component=mp.Dielectric)

# And the electric field
ez_data = sim.get_array(center=mp.Vector3(0, 0, 0), 
                        size=cell_size, 
                        component=mp.Ez)
plt.figure()
plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
plt.axis('off')
plt.show()