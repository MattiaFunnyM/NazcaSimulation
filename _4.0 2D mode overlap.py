import h5py
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec

# -----------------------------
# Simulation parameters
# -----------------------------
sim_length = 30
sim_width = 5
sim_height = 0
sim_resolution = 20
sim_time = 2000
sim_size = mp.Vector3(sim_length, sim_width, sim_height)

# Waveguide geometry
wvg_length = sim_length
wvg_width = 1
wvg_height = 0
cld_neff = 1.72
wvg_neff = 3.21

# MMI geometry
mmi_length = 10
mmi_width = 2
mmi_height = 0

# Source parameters
src_wvl = 1.550
src_freq = 1 / src_wvl

# Output naming
filename_prefix1 = "Mode_Calculation"
filename_prefix2 = "Field_Propagation"
filename_sim = "simulation"
output_dir = "Output"

# Geometry for the mode calculation
geometry = [
    mp.Block(size=mp.Vector3(sim_length, sim_width, sim_height),
            center=mp.Vector3(),
            material=mp.Medium(epsilon=cld_neff**2)),
    mp.Block(size=mp.Vector3(wvg_length, wvg_width, wvg_height),
            center=mp.Vector3(),
            material=mp.Medium(epsilon=wvg_neff**2)),
]

# Geometry modified for the propagation simulation
geometry2 = [
    mp.Block(size=mp.Vector3(sim_length, sim_width, sim_height),
            center=mp.Vector3(),
            material=mp.Medium(epsilon=cld_neff**2)),
    mp.Block(size=mp.Vector3(wvg_length, wvg_width, wvg_height),
            center=mp.Vector3(),
            material=mp.Medium(epsilon=wvg_neff**2)),
    mp.Block(size=mp.Vector3(mmi_length, mmi_width, mmi_height),
            center=mp.Vector3(),
            material=mp.Medium(epsilon=wvg_neff**2)),
]

# Define the source for the mode calculation
source = [mp.EigenModeSource(
        src=mp.ContinuousSource(frequency=src_freq, fwidth=src_freq/2),
        center=mp.Vector3(-sim_length/2 + 1, 0),
        size=mp.Vector3(0, sim_width),
        direction=mp.X,
        eig_band=1
)]

# -----------------------------
# Simulation setup - Mode calculation
# -----------------------------
pml_layers = [mp.PML(1.0)]
sim = mp.Simulation(
    cell_size=sim_size,
    sources=source,
    geometry=geometry,
    resolution=sim_resolution,
    boundary_layers=pml_layers
)

# Fix output directories
sim.use_output_directory(output_dir)
sim.filename_prefix = filename_prefix1

# -----------------------------
# Simulation - Mode calculation
# -----------------------------
# Prepare a dft to extract complex field component
dft1 = sim.add_dft_fields([mp.Ez, mp.Hy], src_freq, 0, 1, 
                          where=mp.Volume(center=mp.Vector3(),
                                          size=mp.Vector3(sim_length, sim_width, sim_height)))

# Physically run the simulation with current setup
sim.run(mp.at_beginning(mp.output_epsilon),
        until=sim_time)

# Extract complex field component
Ez1 = sim.get_dft_array(dft1, mp.Ez, 0)
Hy1 = sim.get_dft_array(dft1, mp.Hy, 0)

# Extract the cross section at that maxima (used for cross section and mode generation)
x_max_index = np.unravel_index(np.argmax(np.real(Ez1)), Ez1.shape)[0]   # row
Ez1_cross = np.real(Ez1[x_max_index, :])
Hy1_cross = np.real(Hy1[x_max_index, :])


# Reset simulation before starting new one
sim.reset_meep()

# Interpolates the cross section so the meep simulator can access whatever value it wants
y = np.linspace(-sim_width/2, sim_width/2, len(Ez1_cross))
Ez1_cross_interp = interp1d(
    y,
    Ez1_cross,
    kind='cubic',
    bounds_error=False,
    fill_value=0.0
)
Hy1_cross_interp = interp1d(
    y,
    Hy1_cross,
    kind='cubic',
    bounds_error=False,
    fill_value=0.0
)

# Define temporal and spatial evolution of the mode source
def temporal_profile_Ez(t):
    return np.cos(2 * np.pi * src_freq * t)

def spatial_profile_Ez(r):
    return float(Ez1_cross_interp(r.y))

def temporal_profile_Hy(t):
    return np.sin(2 * np.pi * src_freq * t)

def spatial_profile_Hy(r):
    return float(Hy1_cross_interp(r.y))


# Define the source for the propagation simulation
source2 = [
    mp.Source(
    src=mp.CustomSource(temporal_profile_Ez),
    center=mp.Vector3(-sim_length/2 + 1, 0),
    size=mp.Vector3(0, sim_width),
    component = mp.Ez,
    amp_func = spatial_profile_Ez),
    mp.Source(
    src=mp.CustomSource(temporal_profile_Hy),
    center=mp.Vector3(-sim_length/2 + 1, 0),
    size=mp.Vector3(0, sim_width),
    component = mp.Hy,
    amp_func = spatial_profile_Hy)
]

# -----------------------------
# Simulation setup - insert MMI
# -----------------------------
sim2 = mp.Simulation(
    cell_size=sim_size,
    geometry=geometry2,
    resolution=sim_resolution,
    boundary_layers=pml_layers,
    sources=source2
)

# Fix output directories
sim2.use_output_directory(output_dir)
sim2.filename_prefix = filename_prefix2

# -----------------------------
# Simulation - propagation MMI
# -----------------------------
# Prepare a dft to extract complex field component
dft2 = sim2.add_dft_fields([mp.Ez], src_freq, 0, 1, 
                           where=mp.Volume(center=mp.Vector3(),
                                           size=mp.Vector3(sim_length, sim_width, sim_height)))

# Physically run the simulation
sim2.run(mp.at_beginning(mp.output_epsilon),
         until=sim_time)

# Extract complex field component
Ez2 = sim2.get_dft_array(dft2, mp.Ez, 0)

# -----------------------------
# Plotting - 4 subplots
# -----------------------------
# Load the generated file for material from 1st simulation 
with h5py.File("Output/" + filename_prefix1 + '-eps-000000.00.h5', "r") as f: 
    eps_data1 = f["eps"][:]
with h5py.File("Output/" + filename_prefix2 + '-eps-000000.00.h5', "r") as f: 
    eps_data2 = f["eps"][:]

# Transpose the field
Ez1 = Ez1.T
Ez2 = Ez2.T

# Normalize the field
Ez1 /= np.max(abs(Ez1))
Ez2 /= np.max(abs(Ez2))

plt_fontsize = 16
fig = plt.figure(figsize=(14, 8))

# Create a 2x2 grid with different column widths
gs = gridspec.GridSpec(2, 2, width_ratios=[1.8, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.5)

# Axes
ax1 = fig.add_subplot(gs[0, 0])  # Top-left (big)
ax2 = fig.add_subplot(gs[0, 1])  # Top-right (small)
ax3 = fig.add_subplot(gs[1, 0])  # Bottom-left (big)
ax4 = fig.add_subplot(gs[1, 1])  # Bottom-right (small)

# Background material image (from sim1)
x = np.linspace(-sim_length/2, sim_length/2, Ez1.shape[1])
y = np.linspace(-sim_width/2, sim_width/2, Ez1.shape[0])

# -----------------------------
# Simulation 1 field (big)
# -----------------------------
ax1.imshow(-eps_data1.T, cmap='gray', origin='upper', extent=[x[0], x[-1], y[0], y[-1]])
im1 = ax1.imshow(np.abs(Ez1)**2, cmap='RdBu', alpha=0.8, origin='upper', 
                 interpolation='bilinear', vmin=-1, vmax=1, extent=[x[0], x[-1], y[0], y[-1]])
ax1.set_title("Determine Mode", fontsize=plt_fontsize)
ax1.set_xlabel("x (µm)", fontsize=plt_fontsize)
ax1.set_ylabel("y (µm)", fontsize=plt_fontsize)
ax1.tick_params(axis='both', which='major', labelsize=plt_fontsize) 

# -----------------------------
# Simulation 2 field (big)
# -----------------------------
ax3.imshow(-eps_data2.T, cmap='gray', origin='upper', extent=[x[0], x[-1], y[0], y[-1]])
im2 = ax3.imshow(np.abs(Ez2)**2, cmap='RdBu', alpha=0.8, origin='upper',
                 interpolation='bilinear', vmin=-1, vmax=1, extent=[x[0], x[-1], y[0], y[-1]])
ax3.set_title("Mode Propagation", fontsize=plt_fontsize)
ax3.set_xlabel("x (µm)", fontsize=plt_fontsize)
ax3.set_ylabel("y (µm)", fontsize=plt_fontsize)
ax3.tick_params(axis='both', which='major', labelsize=plt_fontsize) 
# -----------------------------
# Extra plot 1 - Ez1 cross-section 
# -----------------------------
overlap = np.abs(np.dot(Ez1_cross, Ez1))
overlap /= max(overlap)
ax2.plot(x, overlap, color='black', lw=2)
ax2.set_title("Ez1 Overlap", fontsize=plt_fontsize)
ax2.set_xlabel("y (µm)", fontsize=plt_fontsize)
ax2.set_ylabel("Overlap", fontsize=plt_fontsize)
ax2.tick_params(axis='both', which='major', labelsize=plt_fontsize) 
ax2.grid()
# -----------------------------
# Extra plot 2 - Ez2 cross-section
# -----------------------------
overlap = np.abs(np.dot(Ez1_cross, Ez2))
overlap /= max(overlap)
ax4.plot(x, overlap, color='black', lw=2)
ax4.set_title("Ez2 Overlap", fontsize=plt_fontsize)
ax4.set_xlabel("y (µm)", fontsize=plt_fontsize)
ax4.set_ylabel("Overlap", fontsize=plt_fontsize)
ax4.tick_params(axis='both', which='major', labelsize=plt_fontsize) 
ax4.grid()
plt.show()
