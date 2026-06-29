"""
Regenerate fig1_motorway_3d_map.png.
Uses first 1800m of the motorway (northeast segment only) — this is the clean
open S-shaped portion that matches the original paper figure.
All contestants, teleport-filtered, 30m bins.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hires_plots')
os.makedirs(OUT, exist_ok=True)

capstone_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'capstone')
mtway_csv = os.path.join(capstone_dir, 'df_ntu_mtway.csv')

print('Loading...')
df = pd.read_csv(mtway_csv, usecols=[
    'pos_x', 'pos_y', 'cumulative_distance', 'speed_mps', 'contestant'
]).dropna()

# Use first 2500m — avoids the U-turn at the far east end
moving = df[(df['speed_mps'] > 1.0) & (df['cumulative_distance'] <= 2700)].copy()

# Remove teleport rows per contestant
clean_parts = []
for name, grp in moving.groupby('contestant'):
    grp = grp.reset_index(drop=True)
    dx = grp['pos_x'].diff().abs().fillna(0)
    dy = grp['pos_y'].diff().abs().fillna(0)
    mask = (dx <= 150) & (dy <= 150)
    clean_parts.append(grp[mask])

clean = pd.concat(clean_parts, ignore_index=True)
print(f'Clean rows: {len(clean)}, x {clean.pos_x.min():.0f}-{clean.pos_x.max():.0f}, '
      f'y {clean.pos_y.min():.0f}-{clean.pos_y.max():.0f}')

# Bin by cumulative_distance every 30 m
bin_w = 30
bins = np.arange(0, 2700 + bin_w, bin_w)
clean['d_bin'] = pd.cut(clean['cumulative_distance'], bins,
                        labels=bins[:-1], include_lowest=True).astype(float)

route = (clean.groupby('d_bin')
              .agg(pos_x=('pos_x', 'median'),
                   pos_y=('pos_y', 'median'))
              .dropna()
              .reset_index()
              .sort_values('d_bin'))

x = route['pos_x'].values
y = route['pos_y'].values
print(f'Route bins: {len(x)}, x {x.min():.0f}-{x.max():.0f}, y {y.min():.0f}-{y.max():.0f}')

# Gaussian smooth + cubic upsample
xs = gaussian_filter1d(x, sigma=2)
ys = gaussian_filter1d(y, sigma=2)

t = np.linspace(0, 1, len(xs))
t_fine = np.linspace(0, 1, 300)
xs = interp1d(t, xs, kind='cubic')(t_fine)
ys = interp1d(t, ys, kind='cubic')(t_fine)

dx = np.gradient(xs)
dy = np.gradient(ys)
heading = np.arctan2(dy, dx)
heading = np.unwrap(heading)
zs = gaussian_filter1d(heading, sigma=10)

trim = 10
xs = xs[trim:-trim]
ys = ys[trim:-trim]
zs = zs[trim:-trim]
zs = 2 * (zs - zs.min()) / (zs.max() - zs.min()) - 1

print(f'Final: {len(xs)} pts, x {xs.min():.0f}-{xs.max():.0f}, y {ys.min():.0f}-{ys.max():.0f}')

plt.style.use('dark_background')
fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot(xs, ys, zs, color='limegreen', linewidth=6)
ax.view_init(elev=25, azim=-60)
ax.set_xlabel('X', labelpad=8)
ax.set_ylabel('Y', labelpad=8)
ax.set_zlabel('Z', labelpad=8)
ax.set_title('Motorway 3D map', loc='left', pad=10, fontsize=11)
ax.grid(True, linestyle='--', alpha=0.4)
ax.tick_params(colors='white')
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#1f4e79')
ax.yaxis.pane.set_edgecolor('#1f4e79')
ax.zaxis.pane.set_edgecolor('#1f4e79')

plt.tight_layout()
out_path = os.path.join(OUT, 'fig1_motorway_3d_map.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='black')
plt.close(fig)
print(f'Saved: {out_path}')
