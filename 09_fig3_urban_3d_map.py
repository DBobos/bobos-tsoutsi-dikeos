"""
Regenerate fig3_urban_3d_map.png with thicker line (lw=6).
Reconstructs the L-shaped urban track spatially.
X=pos_x, Y=pos_y, Z=heading  (X and Z swapped per user request).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hires_plots')
os.makedirs(OUT, exist_ok=True)

capstone_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'capstone')
urblt_csv = os.path.join(capstone_dir, 'df_ntu_urblt.csv')

print('Loading urban CSV...')
df = pd.read_csv(urblt_csv, usecols=['pos_x', 'pos_y', 'speed_mps']).dropna()
moving = df[df['speed_mps'] > 1.0].copy()

# East arm: driving north on the east side (pos_x ≈ 690-815)
east = moving[(moving['pos_x'] >= 690) & (moving['pos_x'] <= 815)]
east_bins = np.arange(-560, 20, 15)
east['y_bin'] = pd.cut(east['pos_y'], east_bins,
                       labels=east_bins[:-1], include_lowest=True).astype(float)
east_route = (east.groupby('y_bin')
                  .agg(pos_x=('pos_x','median'), pos_y=('pos_y','median'))
                  .dropna().reset_index()
                  .sort_values('pos_y'))   # south → north

# North arm: driving west on the north side (pos_y ≈ -30 to 15)
north = moving[(moving['pos_y'] >= -30) & (moving['pos_y'] <= 15)]
north_bins = np.arange(-220, 820, 15)
north['x_bin'] = pd.cut(north['pos_x'], north_bins,
                        labels=north_bins[:-1], include_lowest=True).astype(float)
north_route = (north.groupby('x_bin')
                    .agg(pos_x=('pos_x','median'), pos_y=('pos_y','median'))
                    .dropna().reset_index()
                    .sort_values('pos_x', ascending=False))  # east → west

print(f'East arm: {len(east_route)} pts, x {east_route.pos_x.min():.0f}-{east_route.pos_x.max():.0f}, y {east_route.pos_y.min():.0f}-{east_route.pos_y.max():.0f}')
print(f'North arm: {len(north_route)} pts, x {north_route.pos_x.min():.0f}-{north_route.pos_x.max():.0f}, y {north_route.pos_y.min():.0f}-{north_route.pos_y.max():.0f}')

route = pd.concat([east_route[['pos_x','pos_y']],
                   north_route[['pos_x','pos_y']]], ignore_index=True)

x = route['pos_x'].values
y = route['pos_y'].values

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
zs = gaussian_filter1d(heading, sigma=8)

trim = 10
xs_t = xs[trim:-trim]
ys_t = ys[trim:-trim]
zs_t = zs[trim:-trim]
zs_t = 2 * (zs_t - zs_t.min()) / (zs_t.max() - zs_t.min()) - 1

print(f'Final: {len(xs_t)} pts, pos_x(X) {xs_t.min():.0f}-{xs_t.max():.0f}, '
      f'pos_y(Y) {ys_t.min():.0f}-{ys_t.max():.0f}, heading(Z) {zs_t.min():.2f}-{zs_t.max():.2f}')

plt.style.use('dark_background')
fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection='3d')

# X=pos_x, Y=pos_y, Z=heading
ax.plot(xs_t, ys_t, zs_t, color='limegreen', linewidth=6)
ax.set_zlim(-10, 10)
ax.set_xlabel('X', labelpad=8)
ax.set_ylabel('Y', labelpad=8)
ax.set_zlabel('Z', labelpad=8)
ax.set_title('Urban-low-traffic 3D map', loc='left', pad=10, fontsize=11)
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
out_path = os.path.join(OUT, 'fig3_urban_3d_map.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='black')
plt.close(fig)
print(f'Saved: {out_path}')
