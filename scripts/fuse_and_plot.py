"""
Full IMU + distance sensor fusion: reconstruct trajectory and hit points in 3D.

Pipeline
--------
1. Integrate gyroscope (RK4) → estimated orientation at every IMU timestamp.
2. Rotate accelerometer measurements to world frame, subtract gravity, correct
   for the centripetal lever-arm term (omega × (omega × p_imu)).
3. Double-integrate world-frame acceleration → estimated velocity and position.
4. At each distance sensor sample, transform sensor rays to world frame using
   the integrated position and orientation, compute hit points.
5. Compare three tiers of reconstruction:
     GT     : ground-truth position  + ground-truth orientation   (reference)
     ori    : ground-truth position  + gyro-integrated orientation (isolates orientation error)
     full   : integrated position    + gyro-integrated orientation (fully sensor-based)

Lever-arm note
--------------
The accelerometer sits at p_imu (offset from COM).  The measurement includes
a centripetal term  omega × (omega × p_imu)  that is subtracted here using the
gyro reading.  The tangential term  alpha × p_imu  (needing d(omega)/dt) is
small and ignored.

Usage
-----
    python scripts/fuse_and_plot.py                      # default log
    python scripts/fuse_and_plot.py -i results/run.h5   # custom log
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simio.logger import load
from sim.physics  import quat_multiply, quat_normalize, quat_to_rotmat
from viz.playback import _wall_patch


# ---------------------------------------------------------------------------
# IMU integration
# ---------------------------------------------------------------------------

def _q_dot(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return 0.5 * quat_multiply(q, np.array([0.0, omega[0], omega[1], omega[2]]))


def integrate_imu(t_imu, accel_body, gyro, q0, p0, gravity, imu_pos_from_com):
    """
    Integrate gyroscope and accelerometer to estimate orientation, velocity,
    and position.

    Three passes:
      1. Precompute angular acceleration α = dω/dt via centered finite differences
         on the gyro signal.  Used for the tangential lever-arm correction.
      2. Integrate orientation with RK4 quaternion kinematics.
      3. Compute world-frame COM acceleration at every sample (full lever-arm
         correction: centripetal  ω×(ω×p)  and tangential  α×p), then
         integrate velocity and position with the trapezoid rule.

    Lever-arm corrections (body frame, subtracted after rotating to world):
        centripetal : ω × (ω × p_imu)   — from gyro directly
        tangential  : α × p_imu          — from finite-differenced gyro

    Trapezoid rule:
        v[i] = v[i-1] + dt/2 * (a[i-1] + a[i])
        p[i] = p[i-1] + dt/2 * (v[i-1] + v[i])

    Parameters
    ----------
    t_imu           : (N,)   sample timestamps (s)
    accel_body      : (N, 3) accelerometer readings, body frame (m/s²)
    gyro            : (N, 3) gyroscope readings, body frame (rad/s)
    q0              : (4,)   initial quaternion
    p0              : (3,)   initial COM position (world frame)
    gravity         : float  gravitational acceleration magnitude (m/s²)
    imu_pos_from_com: (3,)   IMU position relative to COM in body frame

    Returns
    -------
    quats       : (N, 4)
    vels        : (N, 3)  world frame
    pos         : (N, 3)  world frame
    accel_world : (N, 3)  estimated COM acceleration in world frame
    """
    N     = len(t_imu)
    p_imu = imu_pos_from_com
    g_vec = np.array([0.0, 0.0, -gravity])

    # ------------------------------------------------------------------
    # Pass 1: angular acceleration via centered finite differences
    # ------------------------------------------------------------------
    alpha = np.empty((N, 3))
    # Interior: centered (second-order accurate)
    dt2           = (t_imu[2:] - t_imu[:-2])[:, None]
    alpha[1:-1]   = (gyro[2:] - gyro[:-2]) / dt2
    # Endpoints: one-sided (first-order)
    alpha[0]      = (gyro[1]  - gyro[0])  / (t_imu[1]  - t_imu[0])
    alpha[-1]     = (gyro[-1] - gyro[-2]) / (t_imu[-1] - t_imu[-2])

    # ------------------------------------------------------------------
    # Pass 2: orientation — RK4 quaternion integration
    # ------------------------------------------------------------------
    quats    = np.empty((N, 4))
    quats[0] = quat_normalize(q0)
    for i in range(1, N):
        dt = t_imu[i] - t_imu[i - 1]
        q  = quats[i - 1]
        w  = gyro[i - 1]    # zero-order hold
        k1 = _q_dot(q,                              w)
        k2 = _q_dot(quat_normalize(q + 0.5*dt*k1),  w)
        k3 = _q_dot(quat_normalize(q + 0.5*dt*k2),  w)
        k4 = _q_dot(quat_normalize(q +     dt*k3),  w)
        quats[i] = quat_normalize(q + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4))

    # ------------------------------------------------------------------
    # Pass 3: world-frame COM acceleration, then trapezoid integration
    # ------------------------------------------------------------------
    accel_world = np.empty((N, 3))
    for i in range(N):
        R           = quat_to_rotmat(quats[i])
        a_imu_world = R @ accel_body[i] + g_vec
        lever_body  = (np.cross(gyro[i],    np.cross(gyro[i], p_imu))   # centripetal
                     + np.cross(alpha[i],   p_imu))                      # tangential
        accel_world[i] = a_imu_world - R @ lever_body

    vels    = np.zeros((N, 3))
    pos     = np.zeros((N, 3))
    pos[0]  = np.array(p0, dtype=float)
    for i in range(1, N):
        dt      = t_imu[i] - t_imu[i - 1]
        vels[i] = vels[i - 1] + 0.5 * dt * (accel_world[i - 1] + accel_world[i])
        pos[i]  = pos[i - 1]  + 0.5 * dt * (vels[i - 1]        + vels[i])

    return quats, vels, pos, accel_world


# ---------------------------------------------------------------------------
# Hit-point reconstruction
# ---------------------------------------------------------------------------

def nearest_index(t_query: float, t_ref: np.ndarray) -> int:
    return int(np.clip(np.searchsorted(t_ref, t_query, side='left'), 0, len(t_ref) - 1))


def compute_hit_points(t_dist, dist_readings,
                       t_pos, pos_arr,
                       t_ori, quat_arr,
                       sensor_pos_geom, com_offset, max_range):
    """
    Reconstruct 3D hit points from range measurements.

    Returns (P, S, 3) array; NaN where the reading equals max_range.
    """
    P, S     = dist_readings.shape
    hits     = np.full((P, S, 3), np.nan)
    dirs_body = sensor_pos_geom / np.linalg.norm(sensor_pos_geom, axis=1, keepdims=True)

    for i, t in enumerate(t_dist):
        pos_com  = pos_arr [nearest_index(t, t_pos)]
        R        = quat_to_rotmat(quat_arr[nearest_index(t, t_ori)])
        pos_geom = pos_com + R @ com_offset

        for j in range(S):
            d = float(dist_readings[i, j])
            if d >= max_range:
                continue
            hits[i, j] = (pos_geom + R @ sensor_pos_geom[j]) + d * (R @ dirs_body[j])

    return hits


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def quat_angle_error(q_est: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Orientation error in degrees; handles quaternion double-cover."""
    errors = np.zeros(len(q_est))
    for i in range(len(q_est)):
        qr   = q_ref[i]
        conj = np.array([qr[0], -qr[1], -qr[2], -qr[3]])
        q_e  = quat_normalize(quat_multiply(conj, q_est[i]))
        errors[i] = np.degrees(2.0 * np.arccos(np.clip(abs(q_e[0]), 0.0, 1.0)))
    return errors


def interp_gt(t_query_arr, t_gt, values_gt):
    """Nearest-neighbour lookup of GT array at query times."""
    return np.array([values_gt[nearest_index(t, t_gt)] for t in t_query_arr])


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def equal_3d_axes(ax, points: np.ndarray, margin: float = 0.5):
    pts = points[~np.isnan(points).any(axis=1)]
    if len(pts) == 0:
        return
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    lo[2]  = min(lo[2], 0.0)
    center = (lo + hi) / 2.0
    half   = (hi - lo).max() / 2.0 + margin
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(max(0.0, center[2] - half), center[2] + half)
    ax.set_box_aspect([1, 1, 1])


def _add_environment(ax, cfg, pos_gt):
    """Draw ground and wall on a 3D axes."""
    x0, x1 = pos_gt[:, 0].min() - 1, pos_gt[:, 0].max() + 1
    y0, y1 = pos_gt[:, 1].min() - 1, pos_gt[:, 1].max() + 1
    GX, GY = np.meshgrid([x0, x1], [y0, y1])
    ax.plot_surface(GX, GY, np.zeros_like(GX), alpha=0.08, color='tan')
    corners = _wall_patch(cfg['environment']['wall'])
    ax.add_collection3d(
        Poly3DCollection([corners], alpha=0.2, facecolor='steelblue', edgecolor='navy')
    )


# ---------------------------------------------------------------------------
# Figure 1: 3D hit points (three tiers)
# ---------------------------------------------------------------------------

def plot_3d_hits(cfg, gt, ds_t, hits_gt, hits_ori, hits_full):
    fig = plt.figure(figsize=(13, 10))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_title('Distance Sensor Hit Points in 3D World Space\n'
                 '▲ GT pos+ori (ref)   ● GT pos + gyro ori   ■ Integrated pos + gyro ori')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')

    pos = gt['position']
    _add_environment(ax, cfg, pos)

    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
            color='black', alpha=0.25, lw=1, label='GT trajectory')

    col = gt.get('collision_state')
    if col is not None and not np.isnan(col).any():
        ax.scatter(*col[:3], color='red', s=80, zorder=10, label='Impact')

    N_t    = len(ds_t)
    cmap   = cm.plasma
    colors = cmap(np.linspace(0, 1, N_t))
    N_sens = hits_gt.shape[1]

    # Set axis limits from all data upfront so they stay fixed during scrubbing
    all_pts = np.vstack([h.reshape(-1, 3) for h in [hits_gt, hits_ori, hits_full]] + [pos])
    equal_3d_axes(ax, all_pts)

    # Pre-compute valid masks and scatter artists per (tier, sensor)
    scatter_artists = []  # list of (artist, tier_hits, tier_colors_valid, valid_mask)
    tiers = [
        (hits_gt,   '^', 0.40, 'GT pos+ori'),
        (hits_ori,  'o', 0.70, 'GT pos, gyro ori'),
        (hits_full, 's', 0.70, 'Integrated pos+ori'),
    ]

    for j in range(N_sens):
        for ti, (hits, marker, alpha, base_lbl) in enumerate(tiers):
            lbl = base_lbl if j == 0 else '_'
            valid = ~np.isnan(hits[:, j, 0])
            if not valid.any():
                continue
            # Start empty — slider will populate
            artist = ax.scatter([], [], [], marker=marker, s=18, alpha=alpha, label=lbl)
            scatter_artists.append((artist, hits[:, j, :], colors, valid))

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=ds_t[0], vmax=ds_t[-1]))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, label='Time (s)')
    ax.legend(fontsize=8, loc='upper left')

    # --- Slider and play/pause controls ---
    fig.subplots_adjust(bottom=0.15)
    ax_slider = fig.add_axes([0.15, 0.04, 0.55, 0.03])
    ax_play   = fig.add_axes([0.75, 0.03, 0.08, 0.04])

    slider = Slider(ax_slider, 'Time (s)', ds_t[0], ds_t[-1],
                    valinit=ds_t[-1], valstep=(ds_t[-1] - ds_t[0]) / 200)
    btn_play = Button(ax_play, 'Play')

    def update(t_max):
        time_mask = ds_t <= t_max
        for artist, pts, cols, valid in scatter_artists:
            show = valid & time_mask
            if show.any():
                artist._offsets3d = (pts[show, 0], pts[show, 1], pts[show, 2])
                artist.set_facecolors(cols[show])
            else:
                artist._offsets3d = ([], [], [])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(ds_t[-1])  # show all initially

    # Animation state
    state = {'playing': False, 'anim': None}

    def animate(frame):
        val = slider.val + (ds_t[-1] - ds_t[0]) / 100
        if val > ds_t[-1]:
            val = ds_t[0]
        slider.set_val(val)

    def on_play(event):
        if state['playing']:
            if state['anim'] is not None:
                state['anim'].event_source.stop()
            state['playing'] = False
            btn_play.label.set_text('Play')
        else:
            state['anim'] = FuncAnimation(fig, animate, interval=50, cache_frame_data=False)
            state['playing'] = True
            btn_play.label.set_text('Pause')
        fig.canvas.draw_idle()

    btn_play.on_clicked(on_play)

    # Keep references alive
    fig._fuse_controls = (slider, btn_play, state)
    return fig


# ---------------------------------------------------------------------------
# Figure 2: Trajectory comparison
# ---------------------------------------------------------------------------

def plot_trajectory_comparison(t_imu, pos_integrated, pos_gt_at_imu, vel_integrated,
                               vel_gt_at_imu, throw_active_at_imu):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle('Integrated vs Ground-truth Trajectory')

    labels = ['X', 'Y', 'Z']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    # --- Position ---
    ax = axes[0]
    for k in range(3):
        ax.plot(t_imu, pos_integrated[:, k], color=colors[k],
                lw=1.5, label=f'integrated {labels[k]}')
        ax.plot(t_imu, pos_gt_at_imu[:, k],  color=colors[k],
                lw=1, ls='--', alpha=0.6, label=f'GT {labels[k]}')
    _shade_throw(ax, t_imu, throw_active_at_imu)
    ax.set_ylabel('Position (m)')
    ax.legend(fontsize=7, ncol=6)
    ax.grid(True, alpha=0.3)

    # --- Velocity ---
    ax = axes[1]
    for k in range(3):
        ax.plot(t_imu, vel_integrated[:, k], color=colors[k], lw=1.5,
                label=f'integrated {labels[k]}')
        ax.plot(t_imu, vel_gt_at_imu[:, k],  color=colors[k], lw=1,
                ls='--', alpha=0.6, label=f'GT {labels[k]}')
    _shade_throw(ax, t_imu, throw_active_at_imu)
    ax.set_ylabel('Velocity (m/s)')
    ax.legend(fontsize=7, ncol=6)
    ax.grid(True, alpha=0.3)

    # --- Position error magnitude ---
    ax = axes[2]
    pos_err = np.linalg.norm(pos_integrated - pos_gt_at_imu, axis=1)
    ax.plot(t_imu, pos_err, color='crimson', lw=1.5)
    _shade_throw(ax, t_imu, throw_active_at_imu)
    ax.set_ylabel('Position error (m)')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Orientation and sensor analysis
# ---------------------------------------------------------------------------

def plot_sensor_analysis(t_imu, angle_err, throw_active_at_imu, ds_t, dist_readings, cfg):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    fig.suptitle('Orientation Error & Distance Sensor Readings')

    ax = axes[0]
    ax.plot(t_imu, angle_err, color='crimson', lw=1)
    _shade_throw(ax, t_imu, throw_active_at_imu)
    ax.set_ylabel('Orientation error (°)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Gyro-integrated vs Ground-truth Orientation Error')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    max_range = float(cfg['sensors']['distance_sensors']['max_range'])
    for j in range(dist_readings.shape[1]):
        r = dist_readings[:, j].copy().astype(float)
        r[r >= max_range] = np.nan
        ax.plot(ds_t, r, label=f'DS{j}', alpha=0.8)
    ax.set_ylabel('Distance (m)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Distance Sensor Readings (max-range returns hidden)')
    ax.legend(fontsize=8, ncol=min(dist_readings.shape[1], 6))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def _shade_throw(ax, t, throw_mask):
    """Shade the throw phase on a 2D axes."""
    if throw_mask is None or not throw_mask.any():
        return
    t0 = t[throw_mask][0]
    t1 = t[throw_mask][-1]
    ax.axvspan(t0, t1, alpha=0.08, color='gold', label='throw phase')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Fuse IMU + distance sensors to reconstruct 3D trajectory and hit points.')
    p.add_argument('-i', '--input', default='sim_output_1.h5',
                   help='Path to HDF5 log file (default: sim_output_1.h5)')
    return p.parse_args()


def main():
    args = parse_args()
    data = load(args.input)
    cfg  = data['config']['parsed']
    gt   = data['ground_truth']
    imu  = data['imu']
    ds   = data['distance_sensors']

    com_offset      = np.array(cfg['ball']['center_of_mass'], dtype=float)
    sensor_pos_geom = np.array(cfg['sensors']['distance_sensors']['positions'], dtype=float)
    max_range       = float(cfg['sensors']['distance_sensors']['max_range'])
    gravity         = float(cfg['physics']['gravity'])
    p0              = np.array(cfg['initial_conditions']['position'], dtype=float)

    imu_pos_geom    = np.array(cfg['sensors']['imu']['position'], dtype=float)
    imu_pos_from_com = imu_pos_geom - com_offset

    if len(imu['t']) == 0:
        print("No IMU data found in log. Aborting.")
        return
    if len(ds['t']) == 0:
        print("No distance sensor data found in log. Aborting.")
        return

    print(f"Loaded {len(imu['t'])} IMU samples, {len(ds['t'])} distance samples.")

    # --- Full IMU integration ---
    print("Integrating IMU (gyro + accelerometer)...")
    q0 = gt['quaternion'][0]
    quats_est, vels_est, pos_est, _ = integrate_imu(
        imu['t'], imu['accelerometer'], imu['gyroscope'],
        q0, p0, gravity, imu_pos_from_com
    )

    # --- GT quantities interpolated to IMU times ---
    gt_quat_at_imu = interp_gt(imu['t'], gt['t'], gt['quaternion'])
    gt_pos_at_imu  = interp_gt(imu['t'], gt['t'], gt['position'])
    gt_vel_at_imu  = interp_gt(imu['t'], gt['t'], gt['vel'] if 'vel' in gt else gt['velocity'])

    throw_active_at_imu = interp_gt(imu['t'], gt['t'],
                                    gt['throw_active'].astype(float)).astype(bool)

    # --- Orientation error ---
    angle_err = quat_angle_error(quats_est, gt_quat_at_imu)
    pos_err   = np.linalg.norm(pos_est - gt_pos_at_imu, axis=1)
    print(f"Orientation error — mean: {angle_err.mean():.3f}°   max: {angle_err.max():.3f}°")
    print(f"Position error    — mean: {pos_err.mean():.4f} m   max: {pos_err.max():.4f} m")

    # --- Hit points: three tiers ---
    print("Reconstructing hit points...")

    # Tier 1: GT position + GT orientation (reference)
    hits_gt = compute_hit_points(
        ds['t'], ds['readings'],
        gt['t'], gt['position'],
        gt['t'], gt['quaternion'],
        sensor_pos_geom, com_offset, max_range,
    )

    # Tier 2: GT position + gyro-integrated orientation (isolates orientation error)
    hits_ori = compute_hit_points(
        ds['t'], ds['readings'],
        gt['t'], gt['position'],
        imu['t'], quats_est,
        sensor_pos_geom, com_offset, max_range,
    )

    # Tier 3: integrated position + gyro-integrated orientation (fully sensor-based)
    hits_full = compute_hit_points(
        ds['t'], ds['readings'],
        imu['t'], pos_est,
        imu['t'], quats_est,
        sensor_pos_geom, com_offset, max_range,
    )

    for label, hits in [('GT', hits_gt), ('ori', hits_ori), ('full', hits_full)]:
        n = int((~np.isnan(hits[:, :, 0]).all(axis=1)).sum())
        print(f"  [{label}] samples with ≥1 valid hit: {n} / {len(ds['t'])}")

    # Attach collision state for plot
    col = data.get('collision', {})
    if col.get('occurred', False):
        gt['collision_state'] = col['state']

    # --- Plots ---
    out_dir = Path(args.input).resolve().parent
    plot_3d_hits(cfg, gt, ds['t'], hits_gt, hits_ori, hits_full).savefig(out_dir / 'hits_plot.png')
    plot_trajectory_comparison(imu['t'], pos_est, gt_pos_at_imu,
                               vels_est, gt_vel_at_imu, throw_active_at_imu).savefig(out_dir / 'traj_compare.png')
    plot_sensor_analysis(imu['t'], angle_err, throw_active_at_imu,
                         ds['t'], ds['readings'], cfg).savefig(out_dir / 'sensors.png')

    plt.show()

if __name__ == '__main__':
    main()
