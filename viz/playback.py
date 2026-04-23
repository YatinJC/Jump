"""
Post-hoc visualization of a simulation run.

Produces two figures:
  1. 3D animated trajectory — ball position, orientation axes, sensor rays, wall, ground.
  2. Sensor time-series — IMU accelerometer/gyroscope, distance sensors, thruster forces.

Usage:
    from viz.playback import Visualizer
    v = Visualizer(result, cfg)
    v.show()
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sim.physics import quat_to_rotmat


# ---------------------------------------------------------------------------
# Helper: draw a sphere wireframe
# ---------------------------------------------------------------------------

def _sphere_lines(center, radius, n=10):
    """Return lists of (x,y,z) arrays for a wireframe sphere."""
    lines = []
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi,   n)
    # Latitude circles
    for vi in v[::2]:
        x = center[0] + radius * np.cos(u) * np.sin(vi)
        y = center[1] + radius * np.sin(u) * np.sin(vi)
        z = center[2] + radius * np.cos(vi) * np.ones_like(u)
        lines.append((x, y, z))
    # Longitude circles
    for ui in u[::2]:
        x = center[0] + radius * np.cos(ui) * np.sin(v)
        y = center[1] + radius * np.sin(ui) * np.sin(v)
        z = center[2] + radius * np.cos(v)
        lines.append((x, y, z))
    return lines


# ---------------------------------------------------------------------------
# Helper: draw a finite wall as a filled rectangle
# ---------------------------------------------------------------------------

def _wall_patch(wall_cfg):
    """Return the four corner vertices of the wall as a (4, 3) array."""
    center = np.array(wall_cfg['center'], dtype=float)
    normal = np.array(wall_cfg['normal'], dtype=float)
    normal /= np.linalg.norm(normal)
    w = float(wall_cfg['width'])
    h = float(wall_cfg['height'])

    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref); u /= np.linalg.norm(u)
    v = np.cross(normal, u);   v /= np.linalg.norm(v)
    if v[2] < 0:
        v = -v; u = -u

    corners = np.array([
        center - (w/2)*u - (h/2)*v,
        center + (w/2)*u - (h/2)*v,
        center + (w/2)*u + (h/2)*v,
        center - (w/2)*u + (h/2)*v,
    ])
    return corners


# ---------------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------------

class Visualizer:

    def __init__(self, result, cfg: dict):
        self.result = result
        self.cfg    = cfg
        self.viz_cfg = cfg.get('visualization', {})

        self.frame_skip      = int(self.viz_cfg.get('frame_skip', 5))
        self.show_sensor_rays = bool(self.viz_cfg.get('show_sensor_rays', True))
        self.show_body_axes   = bool(self.viz_cfg.get('show_body_axes', True))
        self.traj_alpha       = float(self.viz_cfg.get('trajectory_alpha', 0.6))

        self.radius = float(cfg['ball']['radius'])
        self.com_offset = np.array(cfg['ball']['center_of_mass'], dtype=float)

        # Thruster geometry in body frame (from geometric center)
        self.thruster_pos_geom = []    # positions on ball surface
        self.thruster_nominal_dirs = []  # nominal thrust direction unit vectors
        self.thruster_swing_dirs = []    # swing direction (None if no vectoring)
        for tcfg in cfg['thrusters']:
            pos = np.array(tcfg['position'], dtype=float)
            pos_hat = pos / np.linalg.norm(pos)
            self.thruster_pos_geom.append(pos)

            if 'direction' in tcfg:
                d = np.array(tcfg['direction'], dtype=float)
                nominal = d / np.linalg.norm(d)
            else:
                nominal = -pos_hat
            self.thruster_nominal_dirs.append(nominal)

            vcfg = tcfg.get('vectoring', {})
            if vcfg.get('enabled', False):
                ga = vcfg.get('gimbal_axis', 'tangential')
                if isinstance(ga, str):
                    if ga == 'tangential':
                        raw = np.cross(nominal, pos_hat)
                    elif ga == 'radial':
                        dot = np.dot(pos_hat, nominal)
                        raw = pos_hat - dot * nominal
                else:
                    raw = np.array(ga, dtype=float)
                    dot = np.dot(raw, nominal)
                    raw = raw - dot * nominal
                gimbal = raw / np.linalg.norm(raw)
                swing = np.cross(gimbal, nominal)
                swing /= np.linalg.norm(swing)
                self.thruster_swing_dirs.append(swing)
            else:
                self.thruster_swing_dirs.append(None)

        self.thruster_pos_geom = np.array(self.thruster_pos_geom)        # (T, 3)
        self.thruster_nominal_dirs = np.array(self.thruster_nominal_dirs)  # (T, 3)
        self.thruster_max_force = max(
            abs(float(t['max_force'])) for t in cfg['thrusters']
        )

        # Sensor directions in body frame (outward radial, unit vectors)
        self.sensor_dirs_body = []
        for pos in cfg['sensors']['distance_sensors']['positions']:
            p = np.array(pos, dtype=float)
            self.sensor_dirs_body.append(p / np.linalg.norm(p))
        self.sensor_dirs_body = np.array(self.sensor_dirs_body)  # (S, 3)

        # Sensor positions from COM in body frame
        self.sensor_pos_body = (
            np.array(cfg['sensors']['distance_sensors']['positions'], dtype=float)
            - self.com_offset
        )

        # Animation frames: indices into truth arrays
        self.frame_indices = np.arange(0, len(result.t_truth), self.frame_skip)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def show(self):
        """Display both the 3D animation and sensor time-series plots."""
        self.plot_sensors()
        self.animate_3d()
        plt.show()

    def animate_3d(self):
        """Create and display the 3D trajectory animation with scrub slider."""
        fig = plt.figure(figsize=(10, 9))
        # Reserve bottom strip for slider + button controls
        ax  = fig.add_axes([0.05, 0.15, 0.90, 0.80], projection='3d')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.set_title('Ball Trajectory & Orientation')

        pos  = self.result.pos
        quat = self.result.quat

        # Auto-set axis limits with equal scale on all axes
        margin = self.radius * 3
        all_pts = np.vstack([
            pos,
            _wall_patch(self.cfg['environment']['wall']),
        ])
        lo = all_pts.min(axis=0) - margin
        hi = all_pts.max(axis=0) + margin
        lo[2] = min(lo[2], 0.0)   # always include ground
        hi[2] = max(hi[2], 0.0)
        center = (lo + hi) / 2
        half   = ((hi - lo).max()) / 2   # same half-range for all axes
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(max(0.0, center[2] - half), center[2] + half)
        ax.set_box_aspect([1, 1, 1])

        # --- Static elements ---
        # Ground grid
        gx = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 5)
        gy = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 5)
        GX, GY = np.meshgrid(gx, gy)
        ax.plot_surface(GX, GY, np.zeros_like(GX), alpha=0.1, color='tan', zorder=0)

        # Wall
        corners = _wall_patch(self.cfg['environment']['wall'])
        poly = Poly3DCollection([corners], alpha=0.25, facecolor='steelblue', edgecolor='navy')
        ax.add_collection3d(poly)

        # Full trajectory (static ghost)
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                'gray', alpha=self.traj_alpha * 0.5, lw=1, zorder=1)

        # Phase transition markers
        phase_names = {
            0: 'INACTIVE', 1: 'YAW_DESPIN',
            3: 'TILT_DESPIN', 4: 'ORI_BURN', 5: 'BURN',
            6: 'FINAL_ORI',
            10: 'C_ORI_B', 11: 'C_BOOST', 12: 'C_ORI_LAND',
            13: 'C_APPR',
        }
        phase_colors = {
            0: 'gray', 1: 'orange',
            3: 'purple', 4: 'cyan', 5: 'red',
            6: 'gold',
            10: 'yellowgreen', 11: 'lime', 12: 'violet',
            13: 'deepskyblue',
        }

        if hasattr(self.result, 'controller_phase') and len(self.result.controller_phase) > 0:
            cp_arr = self.result.controller_phase
            for i in range(1, len(cp_arr)):
                ph = int(cp_arr[i])
                ph_prev = int(cp_arr[i-1])

                if ph != ph_prev:
                    p_trans = pos[i]
                    color = phase_colors.get(ph, 'black')
                    name = phase_names.get(ph, f'?{ph}')
                    ax.scatter(*p_trans, color=color, s=60, zorder=9,
                               edgecolors='black', linewidths=0.5)
                    ax.text(p_trans[0], p_trans[1], p_trans[2] + 0.08,
                            name, fontsize=6, ha='center', color=color, zorder=9)

        # Collision marker
        if self.result.collision_occurred:
            cp = self.result.collision_state[:3]
            ax.scatter(*cp, color='red', s=80, zorder=10, label='Impact')

        # --- Dynamic elements (will be updated each frame) ---
        traj_line,   = ax.plot([], [], [], 'b-', lw=1.5, alpha=self.traj_alpha)
        sphere_lines = [ax.plot([], [], [], 'k-', lw=0.5, alpha=0.6)[0] for _ in range(20)]
        axis_x,      = ax.plot([], [], [], 'r-', lw=2)
        axis_y,      = ax.plot([], [], [], 'g-', lw=2)
        axis_z,      = ax.plot([], [], [], 'b-', lw=2)
        ray_lines    = [ax.plot([], [], [], 'c-', lw=0.8, alpha=0.5)[0]
                        for _ in self.sensor_dirs_body]
        # Thruster markers (dots on ball surface) and force arrows
        n_thrusters = len(self.thruster_pos_geom)
        thruster_dots = ax.plot([], [], [], 'o', color='orange', ms=5, zorder=8)[0]
        thrust_arrows = [ax.plot([], [], [], '-', color='orangered', lw=2.5, alpha=0.85)[0]
                         for _ in range(n_thrusters)]
        time_text    = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)

        # Pre-compute geometric center positions
        geom_pos = np.array([
            pos[i] + quat_to_rotmat(quat[i]) @ self.com_offset
            for i in range(len(pos))
        ])

        # Distance readings interpolated to truth timesteps
        # (use last known reading)
        dist_readings = self.result.dist_readings  # (P, S)
        t_dist        = self.result.t_dist
        t_truth       = self.result.t_truth

        def _get_dist_at(frame_t):
            if len(t_dist) == 0:
                return None
            idx = np.searchsorted(t_dist, frame_t, side='right') - 1
            idx = max(0, min(idx, len(dist_readings) - 1))
            return dist_readings[idx]

        # Thrust actual readings interpolated to truth timesteps
        thrust_actual = self.result.thrust_actual  # (P, T)
        vectoring_actual = self.result.vectoring_actual  # (P, T)
        t_thrust      = self.result.t_thrust

        def _get_thrust_at(frame_t):
            if len(t_thrust) == 0:
                return None, None
            idx = np.searchsorted(t_thrust, frame_t, side='right') - 1
            idx = max(0, min(idx, len(thrust_actual) - 1))
            return thrust_actual[idx], vectoring_actual[idx]

        # Scale factor: map max thruster force to this length in world units
        thrust_arrow_scale = self.radius * 3.0 / max(self.thruster_max_force, 1e-6)

        axis_len = self.radius * 2.0

        def update(fi):
            i = self.frame_indices[fi]
            p = geom_pos[i]
            q = quat[i]
            R = quat_to_rotmat(q)
            t = t_truth[i]

            # Trajectory up to now
            traj_line.set_data(geom_pos[:i+1, 0], geom_pos[:i+1, 1])
            traj_line.set_3d_properties(geom_pos[:i+1, 2])

            # Sphere wireframe
            slines = _sphere_lines(p, self.radius, n=10)
            for sl, (sx, sy, sz) in zip(sphere_lines, slines + [([], [], [])] * (20 - len(slines))):
                sl.set_data(sx, sy)
                sl.set_3d_properties(sz)

            # Body-frame axes
            if self.show_body_axes:
                for line, col_idx in [(axis_x, 0), (axis_y, 1), (axis_z, 2)]:
                    tip = p + axis_len * R[:, col_idx]
                    line.set_data([p[0], tip[0]], [p[1], tip[1]])
                    line.set_3d_properties([p[2], tip[2]])

            # Sensor rays
            if self.show_sensor_rays:
                dists = _get_dist_at(t)
                for ri, (sens_dir, sens_pos) in enumerate(zip(self.sensor_dirs_body, self.sensor_pos_body)):
                    ray_origin = p + R @ (sens_pos)
                    ray_dir    = R @ sens_dir
                    d = float(dists[ri]) if dists is not None else 1.0
                    tip = ray_origin + d * ray_dir
                    ray_lines[ri].set_data([ray_origin[0], tip[0]], [ray_origin[1], tip[1]])
                    ray_lines[ri].set_3d_properties([ray_origin[2], tip[2]])

            # Thruster positions and force arrows
            thruster_world_pts = np.array([
                p + R @ (tp - self.com_offset) for tp in self.thruster_pos_geom
            ])  # (T, 3) — world-frame positions on ball surface
            thruster_dots.set_data(thruster_world_pts[:, 0], thruster_world_pts[:, 1])
            thruster_dots.set_3d_properties(thruster_world_pts[:, 2])

            forces, deflections = _get_thrust_at(t)
            for ti in range(n_thrusters):
                if forces is not None and abs(forces[ti]) > 1e-4:
                    origin = thruster_world_pts[ti]
                    # Compute deflected thrust direction in body frame
                    nom = self.thruster_nominal_dirs[ti]
                    swing = self.thruster_swing_dirs[ti]
                    if swing is not None and deflections is not None:
                        delta = deflections[ti]
                        dir_body = np.cos(delta) * nom + np.sin(delta) * swing
                    else:
                        dir_body = nom
                    direction = R @ dir_body
                    tip = origin + direction * forces[ti] * thrust_arrow_scale
                    thrust_arrows[ti].set_data([origin[0], tip[0]], [origin[1], tip[1]])
                    thrust_arrows[ti].set_3d_properties([origin[2], tip[2]])
                else:
                    thrust_arrows[ti].set_data([], [])
                    thrust_arrows[ti].set_3d_properties([])

            time_text.set_text(f't = {t:.3f} s')
            return (traj_line, *sphere_lines, axis_x, axis_y, axis_z,
                    *ray_lines, thruster_dots, *thrust_arrows, time_text)

        n_frames    = len(self.frame_indices)
        dt_sim      = float(self.cfg['simulation']['dt'])
        interval_ms = max(1, int(self.frame_skip * dt_sim * 1000))

        # --- Scrub slider ---
        ax_slider = fig.add_axes([0.10, 0.07, 0.65, 0.03])
        t_min = t_truth[self.frame_indices[0]]
        t_max = t_truth[self.frame_indices[-1]]
        slider = Slider(ax_slider, 'Time (s)', t_min, t_max,
                        valinit=t_min, valstep=(t_max - t_min) / max(n_frames - 1, 1))

        # --- Play / Pause button ---
        ax_button = fig.add_axes([0.80, 0.05, 0.10, 0.05])
        btn = Button(ax_button, 'Pause')

        # Mutable state shared between callbacks
        state = {'playing': True, 'syncing': False}

        def _update_and_draw(fi):
            """Call the frame-update function and redraw."""
            update(fi)
            fig.canvas.draw_idle()

        def on_slider_changed(val):
            # Ignore programmatic slider moves that come from within update()
            if state['syncing']:
                return
            # Pause auto-play when the user grabs the slider
            if state['playing']:
                ani.pause()
                state['playing'] = False
                btn.label.set_text('Play')
            # Map slider time value → nearest frame index
            t_val = slider.val
            fi = int(np.argmin(np.abs(t_truth[self.frame_indices] - t_val)))
            _update_and_draw(fi)

        slider.on_changed(on_slider_changed)

        def on_button_clicked(event):
            if state['playing']:
                ani.pause()
                state['playing'] = False
                btn.label.set_text('Play')
            else:
                ani.resume()
                state['playing'] = True
                btn.label.set_text('Pause')

        btn.on_clicked(on_button_clicked)

        # Wrap update so the slider tracks the current playback position
        original_update = update

        def update_with_slider_sync(fi):
            result = original_update(fi)
            # Move slider to current time without triggering on_slider_changed
            state['syncing'] = True
            slider.set_val(t_truth[self.frame_indices[fi]])
            state['syncing'] = False
            return result

        ani = animation.FuncAnimation(
            fig, update_with_slider_sync, frames=n_frames,
            interval=interval_ms, blit=False
        )
        self._ani_3d = ani   # keep reference to prevent GC
        return fig, ani

    def plot_sensors(self):
        """Plot IMU data, distance sensor readings, and thruster forces."""
        r = self.result
        n_dist     = r.dist_readings.shape[1] if len(r.dist_readings) > 0 else 0
        n_thrusters = r.thrust_actual.shape[1] if len(r.thrust_actual) > 0 else 0

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
        fig.suptitle('Sensor & Actuator Data', fontsize=13)

        # --- IMU Accelerometer ---
        ax = axes[0]
        if len(r.t_imu) > 0:
            labels = ['ax', 'ay', 'az']
            for i, lbl in enumerate(labels):
                ax.plot(r.t_imu, r.imu_accel[:, i], label=lbl)
        ax.set_ylabel('Accel (m/s²)')
        ax.set_title('IMU Accelerometer (body frame)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        if self.result.collision_occurred:
            ax.axvline(self.result.collision_time, color='r', ls='--', alpha=0.6, label='impact')

        # --- IMU Gyroscope ---
        ax = axes[1]
        if len(r.t_imu) > 0:
            labels = ['ωx', 'ωy', 'ωz']
            for i, lbl in enumerate(labels):
                ax.plot(r.t_imu, r.imu_gyro[:, i], label=lbl)
        ax.set_ylabel('Gyro (rad/s)')
        ax.set_title('IMU Gyroscope (body frame)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        if self.result.collision_occurred:
            ax.axvline(self.result.collision_time, color='r', ls='--', alpha=0.6)

        # --- Distance sensors ---
        ax = axes[2]
        if len(r.t_dist) > 0 and n_dist > 0:
            for i in range(n_dist):
                ax.plot(r.t_dist, r.dist_readings[:, i], label=f'DS{i}', alpha=0.8)
        ax.set_ylabel('Distance (m)')
        ax.set_title('Distance Sensor Readings')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        if self.result.collision_occurred:
            ax.axvline(self.result.collision_time, color='r', ls='--', alpha=0.6)

        # --- Thruster forces ---
        ax = axes[3]
        if len(r.t_thrust) > 0 and n_thrusters > 0:
            for i in range(n_thrusters):
                ax.plot(r.t_thrust, r.thrust_actual[:, i], label=f'T{i}', alpha=0.8)
        ax.set_ylabel('Force (N)')
        ax.set_xlabel('Time (s)')
        ax.set_title('Thruster Forces (actual)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        if self.result.collision_occurred:
            ax.axvline(self.result.collision_time, color='r', ls='--', alpha=0.6)

        plt.tight_layout()
        return fig
