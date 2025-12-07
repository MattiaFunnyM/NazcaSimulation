import h5py
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    mp.ContinuousSource(frequency=source_frequency),  # Continuous wave (CW) source
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

# Simulation naming
filename_prefix = "2D_bend"
filename_sim = "simulation"
sim.use_output_directory("Output")
sim.filename_prefix = filename_prefix

# Run the simulation for 200 units, saving the Ez field every 0.6 time units
sim.run(mp.at_beginning(mp.output_epsilon), # We want to access structure only at the beginning
        mp.to_appended(filename_sim, mp.at_every(0.6, mp.output_efield_z)),
        until=300)

# Open the generated file for electric field
with h5py.File("Output/" + filename_prefix + '-' + filename_sim + ".h5", "r") as f:
    Ez = f["ez"][:]        
    Nx, Ny, Nt = Ez.shape

# Load the generated file for material
with h5py.File("Output/" + filename_prefix + '-eps-000000.00.h5', "r") as f:
    eps_data = f["eps"][:]

# Create animation
fig, ax = plt.subplots()

# Draw the waveguide structure as background
ax.imshow(-eps_data.transpose(),
          cmap='gray',
          origin='upper',
          vmin=-material_epsilon, vmax=0,
          extent=[-8, 8, -8, 8])
ax.set_xlabel("x [µm]")
ax.set_ylabel("y [µm]")

# Draw the initial field on top of the waveguide
im = ax.imshow(Ez[:, :, 0].transpose(), interpolation='spline36', cmap='RdBu', alpha=0.8,
               vmin=np.min(Ez), vmax=np.max(Ez),
               extent=[-8, 8, -8, 8])
cbar = plt.colorbar(im, ax=ax)

# Update the frame with new information
def update(frame):
    im.set_data(Ez[:, :, frame].transpose())
    return [im]

anim = FuncAnimation(fig, update, frames=Nt, interval=50, blit=False)
plt.show()