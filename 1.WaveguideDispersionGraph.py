import numpy as np
import meep as mp
import SimLibrary as SL
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import RadioButtons

# =========================
# USER PARAMETERS
# =========================
document_url = "http://www.simpetus.com/projects_scheme.html"
n_f = 20
fs = np.linspace(0.4, 1.0, n_f)

sim_length = 2
sim_width = 2
sim_height = 2
sim_resolution = 32

wvg_neff = 3.45
cld_neff = 1.45
wvg_width = 0.5
wvg_height = 0.22

bnd_thickness = 0.25
max_bands = 4

# =========================
# GEOMETRY DEFINITION
# =========================
geometry = [
    mp.Block(
        size=mp.Vector3(sim_width, sim_height, sim_length),
        center=mp.Vector3(),
        material=mp.Medium(epsilon=1)
    ),
    mp.Block(
        size=mp.Vector3(wvg_width, wvg_height, sim_length),
        center=mp.Vector3(),
        material=mp.Medium(epsilon=wvg_neff**2)
    ),
    mp.Block(
        size=mp.Vector3(sim_width, sim_height/2-wvg_height/2, sim_length),
        center=mp.Vector3(0, -sim_height/4-wvg_height/4),
        material=mp.Medium(epsilon=cld_neff**2)
    )
]

cross_section = mp.Volume(
    center=mp.Vector3(0, 0, 0),
    size=mp.Vector3(
        sim_width - 2*bnd_thickness,
        sim_height - 2*bnd_thickness,
        0
    )
)

# =========================
# DISPERSION + MODE DATABASE
# =========================
modes_db = []
active_bands = list(range(1, max_bands + 1))

for f in reversed(fs):
    bands_to_remove = []
    for band in active_bands:
        output = SL.find_mode_from_cross_section(
            geometry = geometry, 
            cross_section = cross_section, 
            mode_order=band, 
            frequency=f, 
            sim_resolution=32)

        modes_db.append(output)

        # Remove unphysical modes
        if output['k_value'] <= 0 or output['k_value'] < output['frequency'] * cld_neff:
            bands_to_remove.append(band)
            continue

    for band in bands_to_remove:
        active_bands.remove(band)
    if not active_bands:
        break

# =========================
# INTERACTIVE VISUALIZATION
# =========================
fig = plt.figure(figsize=(14,6))
gs = GridSpec(1,2,width_ratios=[1.3,1])

ax_disp = fig.add_subplot(gs[0])
ax_mode = fig.add_subplot(gs[1])

# -------------------------
# Dispersion plot
# -------------------------
width = sim_width - 2*bnd_thickness
height = sim_height - 2*bnd_thickness
bands = list(set([m['mode_order'] for m in modes_db]))
scatter_dict = {}
lines_dict = {}

# Light line
f_max = max([m['frequency'] for m in modes_db])
f_light = np.linspace(0, f_max, 300)
k_light = f_light * cld_neff
ax_disp.fill_between(k_light, f_light, f_max, color="#e6a249")
ax_disp.plot(k_light, f_light, color="black", lw=1, zorder=1)

for band in bands:
    band_points = [m for m in modes_db if m['mode_order']==band]
    k_band = [m['k_value'] for m in band_points]
    f_band = [m['frequency'] for m in band_points]
    # Scatter points
    scatter_dict[band] = ax_disp.scatter(k_band, f_band, color='blue', s=20, zorder=0)
    # Join points with a line
    lines_dict[band], = ax_disp.plot(k_band, f_band, '-', color='blue', lw=2, zorder=0)

# Axis formatting
ax_disp.set_xlabel("Wavevector k", fontsize=16, fontweight='bold')
ax_disp.set_ylabel("Frequency f", fontsize=16, fontweight='bold')
ax_disp.tick_params(axis='both', which='major', labelsize=14)
ax_disp.set_xlim(0,2)
ax_disp.set_ylim(0,1)
ax_disp.set_title("Waveguide dispersion", fontsize=16, fontweight='bold')
for label in ax_disp.get_xticklabels() + ax_disp.get_yticklabels():
    label.set_fontweight('bold')
# -------------------------
# Initial mode plot
# -------------------------
component = 'Ex'  # default component
im = ax_mode.imshow(np.abs(modes_db[0][component]), cmap='YlGnBu',
                    origin='lower', extent=[-width/2, width/2, -height/2, height/2],
                    interpolation='bilinear')
ax_mode.set_title(f'Band {modes_db[0]["mode_order"]} | f={modes_db[0]["frequency"]:.3f}, k={modes_db[0]["k_value"]:.3f} | Ex',
                      fontsize=16, fontweight='bold')
ax_mode.tick_params(axis='both', which='major', labelsize=14)
plt.colorbar(im, ax=ax_mode, fraction=0.046)
for label in ax_mode.get_xticklabels() + ax_mode.get_yticklabels():
    label.set_fontweight('bold')
# -------------------------
# Component selector
# -------------------------
rax = plt.axes([0.88, 0.75, 0.05, 0.15])
radio = RadioButtons(rax, ('Ex','Ey','Ez'), active=0)
for label in radio.labels:
    label.set_fontsize(16)
    label.set_fontweight('bold')

def update_component(label):
    global component
    component = label
    # Update mode image with currently selected point
    idx = getattr(update_component, 'last_idx', 0)
    im.set_data(np.abs(modes_db[idx][component]))
    ax_mode.set_title(ax_mode.get_title()[:-4] + f'| {component}',
                      fontsize=16, fontweight='bold')
    fig.canvas.draw_idle()
radio.on_clicked(update_component)

# -------------------------
# Click interaction
# -------------------------
def on_click(event):
    if event.inaxes != ax_disp:
        return
    x_click = event.xdata
    y_click = event.ydata
    # Find nearest point
    distances = [( (m['k_value']-x_click)**2 + (m['frequency']-y_click)**2 , idx) for idx, m in enumerate(modes_db)]
    _, idx_closest = min(distances)
    update_component.last_idx = idx_closest
    mode_data = modes_db[idx_closest]
    im.set_data(np.abs(mode_data[component]))
    ax_mode.set_title(f'Band {mode_data["mode_order"]} | f={mode_data["frequency"]:.3f}, k={mode_data["k_value"]:.3f} | {component}',
                      fontsize=16, fontweight='bold')
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', on_click)
plt.tight_layout()
plt.show()
