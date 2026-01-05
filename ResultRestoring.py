import numpy as np
import matplotlib.pyplot as plt
import meep as mp
import SimLibrary as SL
import time

time_start = time.time()
# -------------------------------
# USER PARAMETERS
# -------------------------------
n_f = 20
fs = np.linspace(0.1, 1.0, n_f)

sim_length = 2
sim_width = 3
sim_height = 2
sim_resolution = 32

wvg_neff = 3.45
cld_neff = 1.45
wvg_width = 0.5
wvg_height = 0.22

bnd_thickness = 0.5
max_bands = 4
k_tol = 2e-2

# -------------------------------
# GEOMETRY
# -------------------------------
geometry = [
    mp.Block(
        size=mp.Vector3(sim_length, sim_width, sim_height),
        center=mp.Vector3(),
        material=mp.Medium(epsilon=cld_neff**2)
    ),
    mp.Block(
        size=mp.Vector3(sim_length, wvg_width, wvg_height),
        center=mp.Vector3(),
        material=mp.Medium(epsilon=wvg_neff**2)
    )
]

sim_size, sim_center = SL.compute_geometry_bounds(geometry)

cross_section = mp.Volume(
    center=mp.Vector3(-0.5 * sim_length + bnd_thickness + 1, 0, 0),
    size=mp.Vector3(
        0,
        sim_width  - 2 * bnd_thickness,
        sim_height - 2 * bnd_thickness
    )
)

# -------------------------------
# STORAGE
# -------------------------------
dispersion = {}  

# -------------------------------
# FREQUENCY LOOP
# -------------------------------
for f in fs:

    kpoint = mp.Vector3(2 * np.pi * f / wvg_neff, 0, 0)

    sources = [
        mp.Source(
            mp.GaussianSource(frequency=f),
            component=mp.Ez,
            center=mp.Vector3(-sim_width/2 + bnd_thickness, 0, 0)
        )
    ]

    sim = mp.Simulation(
        cell_size=sim_size,
        geometry=geometry,
        resolution=sim_resolution,
        boundary_layers=[mp.PML(bnd_thickness)],
        sources=sources,
        dimensions=3,
        k_point=kpoint
    )

    sim.run(until=0.1)

    k_found = []

    for band in range(1, max_bands + 1):

        try:
            mode = sim.get_eigenmode(
                frequency=f,
                direction=mp.X,
                band_num=band,
                where=cross_section,
                kpoint=kpoint
            )
        except RuntimeError:
            break

        k_val = mode.k[0]
        f_val = mode.freq
        
        # Check uniqueness
        if any(abs(k_val - k_prev) < k_tol for k_prev in k_found):
            break

        if np.isnan(k_val) or k_val <= 0:
            break

        if k_val < f_val * cld_neff + k_tol:
            break

        if band not in dispersion:
            dispersion[band] = {'k': [], 'f': []}

        dispersion[band]['k'].append(abs(k_val))
        dispersion[band]['f'].append(f_val)

        k_val = abs(k_val)
        k_found.append(k_val)

# -------------------------------
# PLOT
# -------------------------------
plt.figure(figsize=(7, 5))

for band, data in dispersion.items():
    plt.plot(data['k'], data['f'], color='blue', lw=2)

f_max = max(max(data['f']) for data in dispersion.values())
f_light = np.linspace(0, f_max, 300)
k_light = f_light * cld_neff

plt.fill_between(
    k_light,
    f_light,
    y2 = f_max,
    color="#e6a249",
    alpha=0.9,
    label='Light line (oxide)'
)
plt.plot(k_light, f_light, color='black')

plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel("Wavevector K", fontsize=18, fontweight='bold')
plt.ylabel("Frequency f", fontsize=18, fontweight='bold')
plt.title("Waveguide dispersion", fontsize=18, fontweight='bold')
plt.tight_layout()
plt.xlim(0, 2)
plt.ylim(0, 1)
time_end = time.time()
print(time_end - time_start)
plt.show()
