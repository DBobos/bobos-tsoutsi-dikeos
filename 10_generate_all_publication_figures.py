"""
High-resolution plot regeneration for publication.
Generates all 6 figures from the depression/driving paper at 300 DPI.
Run from:  E:\TSOUTSI-DIKEOS\for the publication rerun\
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'hires_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DPI = 300


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS 1–4  (3D route maps + mean speed charts)
# ─────────────────────────────────────────────────────────────────────────────

def _get_route_and_speed(csv_path):
    """Return (route_df, speed_df) from a capstone timeseries CSV."""
    df = pd.read_csv(csv_path, usecols=[
        'pos_x', 'pos_y', 'orientation', 'route_x', 'route_y',
        'cumulative_distance', 'speed_mps',
        'contestant_class', 'contestant', 'bins'
    ])
    df = df.dropna()

    # Route: use the pre-computed mean-control route (route_x, route_y) with
    # mean orientation per distance bin → clean single smooth trace
    route = (df.groupby('bins')
               .agg(route_x=('route_x', 'first'),
                    route_y=('route_y', 'first'),
                    orientation=('orientation', 'mean'),
                    cumulative_distance=('cumulative_distance', 'mean'))
               .reset_index()
               .sort_values('cumulative_distance')
               .dropna())
    route = route.rename(columns={'route_x': 'pos_x', 'route_y': 'pos_y'})

    # Speed per class: mean speed binned by cumulative distance (every 10 m)
    bin_width = 10
    max_dist = df['cumulative_distance'].max()
    bins = np.arange(0, max_dist + bin_width, bin_width)
    df['dist_bin'] = pd.cut(df['cumulative_distance'], bins,
                            labels=bins[:-1], include_lowest=True)
    df['dist_bin'] = df['dist_bin'].astype(float)

    speed = (df.groupby(['contestant_class', 'dist_bin'])['speed_mps']
               .mean()
               .reset_index()
               .rename(columns={'dist_bin': 'cumulative_distance',
                                'speed_mps': 'mean_speed'}))
    return route, speed


def plot_3d_map(route, title, out_path):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')

    x = route['pos_x'].values
    y = route['pos_y'].values
    z = route['orientation'].values
    # Normalise orientation to [-1, 1] to match original
    z_range = z.max() - z.min()
    if z_range > 0:
        z = 2 * (z - z.min()) / z_range - 1

    ax.plot(x, y, z, color='limegreen', linewidth=2.5)
    ax.set_xlabel('X', labelpad=8)
    ax.set_ylabel('Y', labelpad=8)
    ax.set_zlabel('Z', labelpad=8)
    ax.set_title(title, loc='left', pad=10, fontsize=11)
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
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_mean_speed(speed_df, title, out_path):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(13, 5))

    for cls, color, label in [('Control', '#1f9bd6', 'Control'),
                               ('Patient', '#e03c3c', 'Patient')]:
        sub = speed_df[speed_df['contestant_class'] == cls].sort_values(
            'cumulative_distance')
        ax.plot(sub['cumulative_distance'], sub['mean_speed'],
                color=color, linewidth=0.9, label=label)

    ax.set_xlabel('Cumulative distance [m]', fontsize=11)
    ax.set_ylabel('Speed [m/s]', fontsize=11)
    ax.set_title(title, loc='left', pad=10, fontsize=11)
    ax.legend(title='Class', framealpha=0.3)
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # x-axis tick formatter: show "2k" style labels matching the original
    def km_fmt(val, pos):
        if val >= 1000:
            return f'{val/1000:.0f}k'
        return f'{val:.0f}'
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(km_fmt))

    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    print(f'Saved: {out_path}')


def generate_track_plots():
    plt.rcdefaults()
    capstone_dir = os.path.join(os.path.dirname(__file__), '..', 'capstone')
    mtway_csv = os.path.join(capstone_dir, 'df_ntu_mtway.csv')
    urblt_csv = os.path.join(capstone_dir, 'df_ntu_urblt.csv')

    print('Loading motorway data …')
    mtway_route, mtway_speed = _get_route_and_speed(mtway_csv)
    plot_3d_map(mtway_route, 'Motorway 3D map',
                os.path.join(OUTPUT_DIR, 'fig1_motorway_3d_map.png'))
    plot_mean_speed(mtway_speed, 'Mean speed during motorway track per class',
                    os.path.join(OUTPUT_DIR, 'fig2_motorway_mean_speed.png'))

    print('Loading urban-low-traffic data …')
    urblt_route, urblt_speed = _get_route_and_speed(urblt_csv)
    plot_3d_map(urblt_route, 'Urban-low-traffic 3D map',
                os.path.join(OUTPUT_DIR, 'fig3_urban_3d_map.png'))
    plot_mean_speed(urblt_speed, 'Mean speed during urban-low-traffic track per class',
                    os.path.join(OUTPUT_DIR, 'fig4_urban_mean_speed.png'))


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5  (Random Forest feature importance – horizontal bar, seaborn theme)
# ─────────────────────────────────────────────────────────────────────────────

def generate_rf_feature_importance():
    plt.rcdefaults()
    plt.style.use('default')

    agg_csv = os.path.join(os.path.dirname(__file__), 'aggregated_data.csv')
    df = pd.read_csv(agg_csv)

    df['contestant_class'].replace(('Control', 'Patient'), (0, 1), inplace=True)
    df['location'].replace(('NTU', 'HMU'), (0, 1), inplace=True)
    df = df[(df.contestant != 'CG2') & (df.contestant != 'CG5') & (df.contestant != 'CG7')]

    y = df['contestant_class']
    df = df.drop('contestant', axis=1)
    del df['contestant_class']

    X = df
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=1, shuffle=True)

    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                 max_depth=2)
    clf.fit(X_train, y_train)

    X_plot = X.drop('location', axis=1)
    sort = clf.feature_importances_[
        [list(X.columns).index(c) for c in X_plot.columns]
    ].argsort()

    fig, ax = plt.subplots(figsize=(16, 9))
    bars = ax.barh(X_plot.columns[sort],
                   clf.feature_importances_[
                       [list(X.columns).index(c) for c in X_plot.columns]
                   ][sort],
                   height=0.7, color='#e05c3a')
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_ylabel('Factors', fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.set_facecolor('#f5f5f5')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig5_rf_feature_importance.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6  (XGBoost feature importance – vertical bar)
# ─────────────────────────────────────────────────────────────────────────────

def generate_xgb_feature_importance():
    plt.rcdefaults()
    plt.style.use('default')
    import random
    from imblearn.under_sampling import RandomUnderSampler
    from xgboost.sklearn import XGBClassifier

    ts_csv = os.path.join(os.path.dirname(__file__), 'timeseries_data.csv')
    df = pd.read_csv(ts_csv)
    df = df.dropna(axis=0, how='any')
    df = df[df.location != 'HMU']

    columns = ['contestant_class', 'contestant', 'location', 'track',
               'event_class', 'time_s', 'cumulative_distance', 'speed_mps',
               'acceleration', 'gear', 'brake', 'steering', 'headway_s',
               'lane_offset_m']
    df = df[[c for c in columns]]
    df = df.loc[(df['speed_mps'] != -0.0) & (df['cumulative_distance'] != 0)]

    def encode_and_bind(original_dataframe, feature_to_encode):
        dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
        return pd.concat([original_dataframe, dummies], axis=1)

    df = encode_and_bind(df, 'event_class')

    df_dict = {g: d for g, d in df.groupby('contestant')}
    keys = list(df_dict.keys())
    random.seed(42)
    random.shuffle(keys)
    shuffled = {k: df_dict[k] for k in keys}

    for key in shuffled:
        shuffled[key] = shuffled[key].drop(['location', 'track', 'event_class'], axis=1)

    import re
    suffled_keys = list(shuffled.keys())
    keys_control = [k for k in suffled_keys if re.match('^CG.*', k)]
    keys_patient = [k for k in suffled_keys if re.match('^DP.*', k)]
    df_ctrl = {k: shuffled[k] for k in keys_control}
    df_pat = {k: shuffled[k] for k in keys_patient}

    train_size = int(len(shuffled) * 0.7)
    train_size_ctrl = int(train_size * 0.4)
    train_size_pat = train_size - train_size_ctrl

    train_set = list(df_ctrl.keys())[:train_size_ctrl] + list(df_pat.keys())[:train_size_pat]

    train_df = pd.DataFrame()
    for t in train_set:
        train_df = pd.concat([train_df, shuffled[t]])

    IDcol, target = 'contestant', 'contestant_class'
    predictors = [x for x in train_df.columns if x not in [target, IDcol]]

    train_df['contestant_class'] = train_df['contestant_class'].map(
        lambda x: 1 if x == 'Patient' else 0).astype(int)

    undersample = RandomUnderSampler(sampling_strategy='majority')
    x_train, y_train = undersample.fit_resample(
        train_df[predictors], train_df['contestant_class'])

    alg = XGBClassifier(
        learning_rate=1, n_estimators=250, max_depth=None, gamma=0,
        subsample=1, colsample_bytree=0.8, colsample_bynode=0.8,
        objective='binary:logistic', eval_metric='auc', scale_pos_weight=1,
        reg_alpha=10, nthread=-1, seed=120, use_label_encoder=False)
    alg.fit(x_train, y_train)

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    feat_imp.plot(kind='bar', ax=ax, color='#1f77b4')
    ax.set_title('Feature Importances')
    ax.set_ylabel('Feature Importance Score', fontsize=11)
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    fig.patch.set_facecolor('white')
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, 'fig6_xgb_feature_importance.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print('=== Generating track plots (figs 1–4) ===')
    generate_track_plots()
    print('=== Generating RF feature importance (fig 5) ===')
    generate_rf_feature_importance()
    print('=== Generating XGBoost feature importance (fig 6) ===')
    generate_xgb_feature_importance()
    print(f'\nAll plots saved to: {OUTPUT_DIR}')
