import meep as mp
import matplotlib.pyplot as plt

# Define Simulation Size - um
cell_width = 8   # X direction (-4, 4)
cell_height = 8  # Y direction (-4, 4)
cell_depth = 0   # Z direction - Absent
resolution = 10  # grid points per reference wavelength
cell_size = mp.Vector3(cell_width, cell_height, cell_depth) 

#(-4, 4)                    (4, 4)
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#(-4, -4)                  (4, -4)

# Define the Simulation Source
material_epsilon = 12     # Relative permittivity of the waveguide material
source_frequency = 0.1    # Frequency corresponding to the reference wavelength
source = mp.Source(
    mp.ContinuousSource(frequency=source_frequency),  # Gaussian pulse
    component=mp.Ez,            
    center=mp.Vector3(-3, 0, 0) # Leave a bit of space from the left boundary
)

#(-4, 4)                    (4, 4)
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ···#o··#···#···#···#···#···#
#   ···#o··#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#(-4, -4)                  (4, -4)

# Define the waveguide geometry
geometry = mp.Block(mp.Vector3(8,1,1), # Size
                     center=mp.Vector3(0, 0, 0), # Center position
                     material=mp.Medium(epsilon=material_epsilon))

#(-4, 4)                    (4, 4)
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ===#o==#===#===#===#===#===#
#   ===#o==#===#===#===#===#===#
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#   ···#···#···#···#···#···#···#
#(-4, -4)                  (4, -4)

# Define the simulation object
pml_layers = [mp.PML(1.0)]  # Perfectly matched layer boundary conditions to avoid reflections
sim = mp.Simulation(
    cell_size=cell_size,
    sources=[source],
    geometry=[geometry],
    resolution=resolution,
    boundary_layers=pml_layers
)

# Run the simulation for 200 units
sim.run(until=200)

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