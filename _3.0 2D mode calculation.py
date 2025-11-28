import h5py
import meep as mp
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Simulation parameters
# -----------------------------
sim_length = 30
sim_width = 6
sim_height = 0
sim_resolution = 15
sim_time = 250
sim_size = mp.Vector3(sim_length, sim_width, sim_height)

# Waveguide geometry
wvg_length = sim_length
wvg_width = 1
wvg_height = 0
cld_neff = 1.72
wvg_neff = 3.21

geometry = [
    mp.Block(size=mp.Vector3(sim_length, sim_width, sim_height),
             center=mp.Vector3(),
             material=mp.Medium(epsilon=cld_neff**2)),
    mp.Block(size=mp.Vector3(wvg_length, wvg_width, wvg_height),
             center=mp.Vector3(),
             material=mp.Medium(epsilon=wvg_neff**2))
]

# Source parameters
src_wvl = 1.550
src_freq = 1 / src_wvl

# Output naming
filename_prefix = "Mode_Calculation"
filename_sim = "simulation"
output_dir = "Output"

# -----------------------------
# Simulation with EigenModeSource
# -----------------------------
source = [mp.EigenModeSource(
         src=mp.ContinuousSource(frequency=src_freq),
         center=mp.Vector3(-sim_length/2 + 1, 0),
         size=mp.Vector3(0, sim_width),
         direction=mp.X,
         eig_band=1
)]

pml_layers = [mp.PML(1.0)]

sim = mp.Simulation(
    cell_size=sim_size,
    sources=source,
    geometry=geometry,
    resolution=sim_resolution,
    boundary_layers=pml_layers
)

sim.use_output_directory(output_dir)
sim.filename_prefix = filename_prefix

sim.run(mp.at_beginning(mp.output_epsilon),
        mp.to_appended(filename_sim, mp.at_end(mp.output_efield_z)),
        until=sim_time)

# -----------------------------
# Load data from HDF5
# -----------------------------
with h5py.File(f"{output_dir}/{filename_prefix}-eps-000000.00.h5", "r") as f:
    eps_data = f["eps"][:]

with h5py.File(f"{output_dir}/{filename_prefix}-{filename_sim}.h5", "r") as f:
    Ez = f["ez"][:, :, 0]
    Ez_neff = Ez * np.sqrt(eps_data)
# -----------------------------
# Prepare axes
# -----------------------------
x = np.linspace(-sim_length/2, sim_length/2, Ez.shape[0])
y = np.linspace(-sim_width/2, sim_width/2, Ez.shape[1])
X, Y = np.meshgrid(x, y)

# -----------------------------
# Plot propagation and cross-section
# -----------------------------
fig_fontsize = 16
fig, axs = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

# --- 1) Field propagation along waveguide ---
axs[0].imshow(eps_data.T, extent=[x[0], x[-1], y[0], y[-1]],
              interpolation='spline36', cmap='binary')
im1 = axs[0].imshow(Ez.T, extent=[x[0], x[-1], y[0], y[-1]],
                    origin='lower', cmap='RdBu', aspect='auto', alpha=0.8)
axs[0].set_xlabel("x (µm)", fontsize=fig_fontsize)
axs[0].set_ylabel("y (µm)",  fontsize=fig_fontsize)
axs[0].set_title("Ez field propagation along waveguide",  fontsize=fig_fontsize)
axs[0].tick_params(axis='both', which='major', labelsize=fig_fontsize) 
cbar1 = fig.colorbar(im1, ax=axs[0])
cbar1.set_label("Ez amplitude")

# --- 2) Mode cross-section at center (x=0) ---
center_idx = Ez.shape[0] // 2
Ez_cross = Ez[center_idx, :]
Ez_cross_norm = Ez_cross/np.sqrt(np.sum(np.abs(Ez_cross)**2)) * np.sign(Ez_cross[int(len(Ez_cross)/2)])
Ez_neff_cross = Ez_neff[center_idx, :]
eps_cross = np.sqrt(eps_data[center_idx, :])
neff = np.sqrt(np.sum(np.abs(Ez_neff_cross)**2)/np.sum(np.abs(Ez_cross)**2))

eps_background = np.tile(eps_cross, (200, 1))
axs[1].imshow(eps_background, aspect='auto', extent=[y.min(), y.max(), Ez_cross_norm.min() - 0.1*Ez_cross_norm.max(), Ez_cross_norm.max() * 1.1],
              origin='lower', cmap='viridis', alpha=0.7)
axs[1].plot(y, Ez_cross_norm, label=f"Mode (n_eff={neff:.2f})", color='black', linewidth=2.5)
axs[1].set_xlabel("y (µm)", fontsize=fig_fontsize)
axs[1].set_ylabel("Ez amplitude", fontsize=fig_fontsize)
axs[1].set_title("Waveguide mode cross-section", fontsize=fig_fontsize)
axs[1].tick_params(axis='both', which='major', labelsize=fig_fontsize) 
axs[1].legend(fontsize=fig_fontsize)

plt.show()
