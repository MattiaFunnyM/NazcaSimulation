import h5py
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec

# -----------------------------
# Simulation parameters
# -----------------------------
# Simulation in a short but wide volume for short time to have a good-mode-decomposition
sim1_length = 5
sim1_width = 6
sim1_height = 0
sim1_resolution = 20
sim1_time = 1500
sim1_size = mp.Vector3(sim1_length, sim1_width, sim1_height)

# Simulation in a long but narrow volume, for enough time to have the light propagating into it
sim2_length = 50
sim2_width = 5
sim2_height = 0
sim2_resolution = 20
sim2_time = 5000
sim2_size = mp.Vector3(sim2_length, sim2_width, sim2_height)

# Waveguide geometry in Simulation 1 
wvg1_length = sim1_length
wvg1_width = 1
wvg1_height = 0
cld1_neff = 1.72
wvg1_neff = 3.21

# Waveguide gometry + MMI in Simulation 2
wvg2_length = sim2_length
wvg2_width = 1
wvg2_height = 0
cld2_neff = 1.72
wvg2_neff = 3.21
mmi2_length = 10
mmi2_width = 2
mmi2_height = 0

# Source parameters in Simulation 1
src1_wvl = 1.550
src1_freq = 1 / src1_wvl

# Source parameters in Simulation 1
src2_wvl = 1.550
src2_freq = 1 / src2_wvl

# Output naming
filename_prefix1 = "Mode_Calculation"
filename_prefix2 = "Field_Propagation"
filename_sim = "simulation"
output_dir = "Output"

# Geometry for mode calculation: a straight waveguide surrounded by FIL material 
geometry1 = [
    mp.Block(size=mp.Vector3(sim1_length, sim1_width, sim1_height),
            center=mp.Vector3(),
            material=mp.Medium(epsilon=cld1_neff**2)),
    mp.Block(size=mp.Vector3(wvg1_length, wvg1_width, wvg1_height),
            center=mp.Vector3(),
            material=mp.Medium(epsilon=wvg1_neff**2)),
]

# Geometry for propagation simulation: an MMI with a waveguide in input and output
geometry2 = [
    mp.Block(size=mp.Vector3(sim2_length, sim2_width, sim2_height),
            center=mp.Vector3(),
            material=mp.Medium(epsilon=cld2_neff**2)),
    mp.Block(size=mp.Vector3(wvg2_length, wvg2_width, wvg2_height),
            center=mp.Vector3(),
            material=mp.Medium(epsilon=wvg2_neff**2)),
    mp.Block(size=mp.Vector3(mmi2_length, mmi2_width, mmi2_height),
            center=mp.Vector3(),
            material=mp.Medium(epsilon=wvg2_neff**2)),
]

# Define the source for the mode calculation
source1 = [mp.EigenModeSource(
        src=mp.ContinuousSource(frequency=src1_freq, fwidth=src1_freq/2),
        center=mp.Vector3(-sim1_length/2 + 1.0, 0),
        size=mp.Vector3(0, sim1_width),
        direction=mp.X,
        eig_band=1
)]

# Condition for edge of simulation
pml_layers = [mp.PML(1.0)]

# -----------------------------
# Simulation 1 setup - Mode calculation
# -----------------------------
sim = mp.Simulation(
    cell_size=sim1_size,
    sources=source1,
    geometry=geometry1,
    resolution=sim1_resolution,
    boundary_layers=pml_layers
)

# Properly set output directories
sim.use_output_directory(output_dir)
sim.filename_prefix = filename_prefix1

# -----------------------------
# Simulation - Mode calculation
# -----------------------------
# Prepare a Discrete Fourier Transform monitor to extract the complex field information
dft1 = sim.add_dft_fields([mp.Ez, mp.Hy], src1_freq, 0, 1, 
                          where=mp.Volume(center=mp.Vector3(),
                                          size=mp.Vector3(sim1_length, sim1_width, sim1_height)))

# Physically run the simulation with current setup
sim.run(mp.at_beginning(mp.output_epsilon),
        until=sim1_time)

# Extract complex field component
Ez1 = sim.get_dft_array(dft1, mp.Ez, 0)
Hy1 = sim.get_dft_array(dft1, mp.Hy, 0)

# Extract the cross section at that maxima (used for cross section and mode generation
x_max_index = int(1.0 * sim1_resolution) + 3
Ez1_cross = np.abs(Ez1[x_max_index, :])
Hy1_cross = np.abs(Hy1[x_max_index, :])

# Reset simulation before starting new one
sim.reset_meep()

# Interpolates the cross section so the meep simulator can access whatever value it wants in second simulation
y_cross = np.linspace(-sim1_width/2, sim1_width/2, len(Ez1_cross))

# Electric field cross section in z direction
Ez1_cross_interp = interp1d(
    y_cross,
    Ez1_cross,
    kind='cubic',
    bounds_error=False,
    fill_value=0.0
)
# Magnetic field cross section in y direction
Hy1_cross_interp = interp1d(
    y_cross,
    Hy1_cross,
    kind='cubic',
    bounds_error=False,
    fill_value=0.0
)

# Define temporal and spatial evolution of the mode source
def temporal_profile_Ez(t):
    return np.cos(2 * np.pi * src2_freq * t)

def spatial_profile_Ez(r):
    return float(Ez1_cross_interp(r.y))

def temporal_profile_Hy(t):
    return np.sin(2 * np.pi * src2_freq * t)

def spatial_profile_Hy(r):
    return float(Hy1_cross_interp(r.y))

# Define the source for the propagation simulation
source2 = [
    mp.Source(
    src=mp.CustomSource(temporal_profile_Ez),
    center=mp.Vector3(-sim2_length/2 + 1, 0),
    size=mp.Vector3(0, sim2_width),
    component = mp.Ez,
    amp_func = spatial_profile_Ez),
    mp.Source(
    src=mp.CustomSource(temporal_profile_Hy),
    center=mp.Vector3(-sim2_length/2 + 1, 0),
    size=mp.Vector3(0, sim2_width),
    component = mp.Hy,
    amp_func = spatial_profile_Hy)
]

# -----------------------------
# Simulation 2 setup - Propagation in MMI
# -----------------------------
sim2 = mp.Simulation(
    cell_size=sim2_size,
    geometry=geometry2,
    resolution=sim2_resolution,
    boundary_layers=pml_layers,
    sources=source2
)

# Properly set output directories
sim2.use_output_directory(output_dir)
sim2.filename_prefix = filename_prefix2

# -----------------------------
# Simulation - Propagation in MMI
# -----------------------------
# Prepare a Discrete Fourier Transform monitor to extract the complex field information
dft2 = sim2.add_dft_fields([mp.Ez], src2_freq, 0, 1, 
                           where=mp.Volume(center=mp.Vector3(),
                                           size=mp.Vector3(sim2_length, sim2_width, sim2_height)))

# Physically run the simulation
sim2.run(mp.at_beginning(mp.output_epsilon),
         until=sim2_time)

# Extract complex field component
Ez2 = sim2.get_dft_array(dft2, mp.Ez, 0)

# -----------------------------
# Plotting
# -----------------------------
# Load the generated file for material from 1st and 2nd simulation 
with h5py.File("Output/" + filename_prefix1 + '-eps-000000.00.h5', "r") as f: 
    eps_data1 = f["eps"][:]
with h5py.File("Output/" + filename_prefix2 + '-eps-000000.00.h5', "r") as f: 
    eps_data2 = f["eps"][:]

# Transpose the field
Ez1 = Ez1.T
Ez2 = Ez2.T

# Axes for simulation 1
x1 = np.linspace(-sim1_width/2, sim1_width/2, Ez1.shape[0])
y1 = np.linspace(-sim1_length/2, sim1_length/2, Ez1.shape[1])

# Axes for simulation 2
x2 = np.linspace(-sim2_width/2, sim2_width/2, Ez2.shape[0])
y2 = np.linspace(-sim2_length/2, sim2_length/2, Ez2.shape[1])

# Normalize the field of simulation 1
norm = 1 / np.sqrt(np.sum(np.abs(Ez1_cross)**2))
Ez1_cross *= norm
Ez1 *= norm

# Normalize the field of simulation 2
Ez1_cross_ = Ez1_cross_interp(x = x2)
norm = 1 / np.sqrt(np.sum(np.abs(Ez1_cross_)**2))
Ez1_cross_ *= norm
x_max_index = int(1.0 * sim2_resolution) + 3
norm = 1 / np.sqrt(np.sum(np.abs(Ez2[:, x_max_index])**2))
Ez2 *= norm

# Create the figure
plt_fontsize = 16
fig = plt.figure(figsize=(14, 8))

# Create a 2x2 grid with different column widths
gs = gridspec.GridSpec(2, 2, width_ratios=[1.8, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.5)

# Axes
ax1 = fig.add_subplot(gs[0, 0])  # Top-left (big)
ax2 = fig.add_subplot(gs[0, 1])  # Top-right (small)
ax3 = fig.add_subplot(gs[1, 0])  # Bottom-left (big)
ax4 = fig.add_subplot(gs[1, 1])  # Bottom-right (small)

# -----------------------------
# Plotting Simulation 1 field mode finding
# -----------------------------
ax1.imshow(-eps_data1.T, cmap='gray', origin='upper', extent=[y1[0], y1[-1], x1[0], x1[-1]])
im1 = ax1.imshow(np.abs(Ez1)**2, cmap='RdBu', alpha=0.8, origin='upper', 
                 interpolation='bilinear', extent=[y1[0], y1[-1], x1[0], x1[-1]])
ax1.set_title("Determine Mode", fontsize=plt_fontsize)
ax1.set_xlabel("Propagation Direction (µm)", fontsize=plt_fontsize)
ax1.set_ylabel("Section Direction (µm)", fontsize=plt_fontsize)
ax1.tick_params(axis='both', which='major', labelsize=plt_fontsize) 

# -----------------------------
# Plotting Simulation 2 propagation in MMI
# -----------------------------
ax3.imshow(-eps_data2.T, cmap='gray', origin='upper', extent=[y2[0], y2[-1], x2[0], x2[-1]])
im2 = ax3.imshow(np.abs(Ez2)**2, cmap='RdBu', alpha=0.8, origin='upper',
                 interpolation='bilinear', extent=[y2[0], y2[-1], x2[0], x2[-1]])
ax3.set_title("Mode Propagation", fontsize=plt_fontsize)
ax3.set_xlabel("Propagation Direction (µm)", fontsize=plt_fontsize)
ax3.set_ylabel("Section Direction (µm)", fontsize=plt_fontsize)
ax3.tick_params(axis='both', which='major', labelsize=plt_fontsize) 

# -----------------------------
# Plotting simulation 1 Ez1 overlap
# -----------------------------
overlap1 = np.abs(np.dot(Ez1_cross, Ez1))**2
ax2.plot(y1, overlap1, color='black', lw=2)
ax2.set_title("Ez1 Overlap", fontsize=plt_fontsize)
ax2.set_xlabel("y (µm)", fontsize=plt_fontsize)
ax2.set_ylabel("Overlap", fontsize=plt_fontsize)
ax2.tick_params(axis='both', which='major', labelsize=plt_fontsize) 
ax2.grid()
# -----------------------------
# Plotting simulation 2 - Ez2 overlap
# -----------------------------
overlap2 = np.abs(np.dot(Ez1_cross_, Ez2))**2
ax4.plot(y2, overlap2, color='black', lw=2)
ax4.set_title("Ez2 Overlap", fontsize=plt_fontsize)
ax4.set_xlabel("y (µm)", fontsize=plt_fontsize)
ax4.set_ylabel("Overlap", fontsize=plt_fontsize)
ax4.tick_params(axis='both', which='major', labelsize=plt_fontsize) 
ax4.grid()

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_aspect('auto')
plt.show()
