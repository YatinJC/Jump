"""
JumpController — flight controller for the ball after throw.

Phases
------
INACTIVE  : During throw phase, thrusters silent.
BALLISTIC   : Post-throw.  Yaw despin via reaction torques + thrust
              vectoring.  Transitions to TILT_DESPIN once yaw settled.
TILT_DESPIN : Bang-bang despin of roll/pitch angular velocity.  Maximum
              opposing torque via the torque-only pseudoinverse for fastest
              possible despin (minimises time under lateral-producing thrust).
              If yaw grows above threshold, transitions back to BALLISTIC.
              Transitions to REORIENT when tilt rates settle.
REORIENT    : Zero-force PD attitude control to point the thrust axis
              against the lateral velocity error accumulated during despin.
              Torque-only (via wrench_to_commands with zero force request)
              so no additional lateral drift is introduced.  Transitions
              to LATERAL_CORRECT when attitude is aligned.
LATERAL_CORRECT : Brief axial burn to cancel the lateral velocity error.
              Fires along the thrust axis (already aligned by REORIENT)
              with proportional velocity control, while PD attitude torque
              holds alignment.  Transitions to TERMINAL when velocity error
              is small, and refreshes the ballistic trajectory snapshot.
TERMINAL    : Placeholder — terminal guidance algorithm to be implemented.

Estimation
----------
IncrementalEstimator integrates gyro (RK4) + accelerometer (trapezoid)
with full lever-arm correction at 8 kHz.

A lightweight position-correction step uses distance-sensor hit points
near known surfaces (floor z=0, wall from config) to reduce double-
integration drift.  For each hit point whose surface is recognised,
a gain-weighted correction is applied to the estimated position along
the constrained axis only.

Surface detection
-----------------
The ballistic parabola  r(τ) = p + v·τ + ½·g·τ²  is sampled at
N_traj points (truncated when z < 0).

For each sample point, all accumulated hitpoints within cluster_radius
are gathered.  If enough points, SVD on the vectors from centroid to
points assesses coplanarity (smallest/second-smallest singular value).
Valid coplanar clusters yield a surface normal and centroid.  The closest
valid surface to the ball is selected as the landing target.  Detection
runs continuously so newly visible surfaces can update the target.

Default target: ground plane at z = ball_radius.

Control
-------
At construction the force rows of the control allocation matrix B are
decomposed via SVD to identify P_force (projection onto achievable
body-frame forces) and thrust_axis_body (primary force direction).

Yaw despin uses propeller reaction torques + thrust vectoring: one
diagonal pair at +max_force, the other at −min_force (reverse), so
reaction torques add while z-forces partially cancel.  All four
thrusters tilt to add moment-arm yaw torque via vectoring.  Tilt despin
uses bang-bang control: the perpendicular spin axis is identified and
maximum opposing torque is applied via the torque-only pseudoinverse.
After despin, a zero-force reorientation phase steers the thrust axis
to oppose the accumulated lateral velocity error, followed by a brief
axial correction burn.

wrench_to_commands() (SVD pseudoinverse) projects the 6-vector onto the
rank-4 achievable subspace and clips to per-thruster limits.
"""

import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.estimator import IncrementalEstimator


class JumpController:

    INACTIVE         = 0
    BALLISTIC        = 1   # yaw despin
    TILT_DESPIN      = 2   # bang-bang tilt despin
    REORIENT         = 3   # zero-force attitude steering toward correction direction
    LATERAL_CORRECT  = 4   # axial burn to cancel lateral velocity
    TERMINAL         = 5   # landing control

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, cfg: dict, thruster_array):
        """
        Parameters
        ----------
        cfg            : full parsed config dict
        thruster_array : ThrusterArray instance (for wrench_to_commands)
        """
        self.cfg            = cfg
        self.thruster_array = thruster_array
        self.n_thrusters    = thruster_array.n

        # Physics
        self.mass    = float(cfg['ball']['mass'])
        self.gravity = float(cfg['physics']['gravity'])
        self.g_vec   = np.array([0.0, 0.0, -self.gravity])

        # Throw timing: stay INACTIVE until throw completes
        self.throw_duration = float(cfg['throw_phase']['duration'])

        # Geometry
        self.ball_radius     = float(cfg['ball']['radius'])
        self.com_offset      = np.array(cfg['ball']['center_of_mass'], dtype=float)
        self.max_range       = float(cfg['sensors']['distance_sensors']['max_range'])
        sensor_pos_geom      = np.array(cfg['sensors']['distance_sensors']['positions'],
                                        dtype=float)
        self.sensor_pos_geom = sensor_pos_geom
        self.sensor_dirs_body = (sensor_pos_geom /
                                 np.linalg.norm(sensor_pos_geom, axis=1, keepdims=True))

        # Distance sensor period (s)
        self._ds_period = 1.0 / float(cfg['sensors']['distance_sensors']['sample_rate'])

        # IMU estimator
        q0               = np.array(cfg['initial_conditions']['quaternion'], dtype=float)
        p0               = np.array(cfg['initial_conditions']['position'],   dtype=float)
        v0               = np.zeros(3)
        imu_pos_geom     = np.array(cfg['sensors']['imu']['position'], dtype=float)
        imu_pos_from_com = imu_pos_geom - self.com_offset
        self.estimator   = IncrementalEstimator(q0, p0, v0, self.gravity, imu_pos_from_com)

        # EKF position correction
        wall_cfg             = cfg['environment']['wall']
        self.wall_x          = float(wall_cfg['center'][0])
        self.floor_z         = 0.0
        self.ekf_gain_floor  = 0.25   # correction gain on p.z from floor hits
        self.ekf_gain_wall   = 0.25   # correction gain on p.x from wall hits
        self.ekf_surf_tol    = 0.25   # m — how close hit must be to surface

        # Hit-point accumulation (unbounded to keep data from entire flight)
        self.hit_points: deque = deque()

        # Surface detection parameters
        self.N_traj             = 20    # trajectory samples
        self._cluster_radius_initial = 1.0   # m — start wide to detect early
        self._cluster_radius_final   = 0.5   # m — shrink for precision
        self._cluster_radius_tau     = 0.5   # s — time constant for shrinking
        self.cluster_min        = 8     # min points in a cluster to attempt SVD
        self.quality_th         = 0.25  # max S[-1]/S[-2] for valid coplanar fit

        # Detection rate and timing
        self._detect_period = 1.0 / 50.0  # 50 Hz
        self._last_detect_t = -1.0

        # Detection results — defaults overwritten after thrust_axis_body is known (below)
        self.target_normal = np.array([0.0, 0.0, 1.0])
        self.target_pos    = np.array([0.0, 0.0, 0.0])

        # Attitude control gains
        self.Kp_att = 0.8    # N·m/rad
        self.Kd_att = 0.15   # N·m·s/rad

        # Terminal guidance sub-phases:
        #   ORIENT_DIVERT → DIVERT → ORIENT_BRAKE → BRAKE → FINAL_ORIENT
        self._landing_speed = 1.0         # m/s — desired approach speed at impact
        self._divert_done_deg = 10.0      # degrees — incidence within this → orient_brake
        self._terminal_sub = 'orient_divert'
        self._divert_dir: np.ndarray | None = None   # fixed divert thrust direction
        self._divert_normal: np.ndarray | None = None  # surface normal used to compute _divert_dir
        self._orient_is_yawing = False                 # True during yaw align within orient

        # Yaw optimization: rotate about thrust axis to align tilt rotation
        # axis with a principal torque axis (x or y body) for max authority.
        self._yaw_opt_Kp = 2.0    # yaw P gain for axis alignment
        self._yaw_opt_Kd = 2.0    # yaw rate damping (computed once)

        # Lateral correction
        self._lateral_v_threshold = 0.05   # m/s — correction complete
        self._att_align_cos = 0.95         # cos(~18°) — aligned enough to start burn
        self._lateral_Kp_v = 10.0          # 1/s — proportional velocity gain for burn

        # Inertia tensor (body frame) — needed for time-optimal reorientation
        self._J = np.array(cfg['ball']['moment_of_inertia']['matrix'], dtype=float)

        # Generalised tilt-to-brake: analyse B matrix once at construction.
        # Use nominal (undeflected) directions for stable precomputation.
        B = thruster_array.nominal_allocation_matrix()
        U_f, S_f, _ = np.linalg.svd(B[:3, :])
        rank_f = int(np.sum(S_f > 1e-6))
        self.P_force = U_f[:, :rank_f] @ U_f[:, :rank_f].T  # project onto achievable forces
        self.thrust_axis_body = U_f[:, 0]                    # primary force axis in body frame
        # SVD sign is arbitrary; ensure axis aligns with mean thruster direction.
        mean_dir = np.mean([t.nominal_direction for t in thruster_array.thrusters], axis=0)
        if np.dot(self.thrust_axis_body, mean_dir) < 0:
            self.thrust_axis_body = -self.thrust_axis_body

        # COM-to-surface offset: when the ball rests on a surface, the COM
        # is (ball_radius + com_offset·thrust_axis) above the contact point
        # because thrust_axis ≈ surface normal at landing.
        self._com_surface_offset = (
            self.ball_radius + np.dot(self.com_offset, self.thrust_axis_body)
        )
        self.target_pos = np.array([0.0, 0.0, self._com_surface_offset])

        # Torque-only pseudoinverse: solves for thruster commands from a
        # 3-vector torque request using only the torque rows of B.
        # No force rows → no zero-force constraint → full torque authority.
        self._B_tau = B[3:, :]                       # (3, n)
        self._B_tau_pinv = np.linalg.pinv(self._B_tau)  # (n, 3)

        # Vectoring yaw-torque coupling: for each thruster, the yaw torque
        # from the lateral force of a deflected thrust vector is
        #   τ_z_vec_i = f_i · sin(δ_i) · c_i
        # where c_i = r_x·swing_y − r_y·swing_x  (z-component of r × swing).
        # The optimal deflection sign is chosen so that f_i·sin(δ_i)·c_i
        # opposes the yaw spin.
        self._vec_coupling = np.zeros(self.n_thrusters)  # c_i per thruster
        for i, thr in enumerate(thruster_array.thrusters):
            if thr.vectoring_enabled:
                r = thr.pos_from_com
                s = thr.swing_direction
                self._vec_coupling[i] = r[0] * s[1] - r[1] * s[0]

        # Thrust budget
        self._F_max_total = sum(t.max_force for t in thruster_array.thrusters)
        self._F_balanced = min(min(t.max_force for t in thruster_array.thrusters),
                               min(abs(t.min_force) for t in thruster_array.thrusters))
        self._thruster_tau = max(t.tau for t in thruster_array.thrusters)
        self._yaw_despin_threshold = 0.2    # rad/s: yaw settled → tilt despin
        self._yaw_despin_active = True     # start in yaw despin
        self._tilt_despin_threshold = 0.2  # rad/s: tilt settled → near-zero residual spin
        self._tilt_despin_active = True    # start in tilt despin

        # Internal bookkeeping
        self._phase       = self.INACTIVE
        self._last_dist_t = -1.0
        self._last_gyro   = np.zeros(3)   # latest gyro ≈ body angular velocity

        # Reorientation gains for uniform-scaling PD.  Only the ratio
        # matters (direction is normalized then scaled to F_balanced).
        # Ratio ≈ sqrt(2·α_max) gives near-optimal braking: the PD
        # switching surface approximates the braking curve in phase space.
        self._reorient_Kp = 22.0   # N·m/rad   (ratio Kp/Kd ≈ 22)
        self._reorient_Kd = 1.0    # N·m·s/rad

        # Precompute average tilt angular acceleration for adaptive gains.
        # α_max ≈ τ_max / I_eff along a typical tilt axis.
        I_tilt = float(np.diag(self._J)[:2].mean())  # average of Ixx, Iyy
        unit_x = np.array([1.0, 0.0, 0.0])
        tau_x = self._B_tau @ (self._F_balanced * np.sign(self._B_tau_pinv @ unit_x))
        self._alpha_tilt = abs(np.dot(tau_x, unit_x)) / I_tilt

        # Snapshot of state at throw end — used for frozen ballistic trajectory
        self._throw_end_p: np.ndarray | None = None
        self._throw_end_v: np.ndarray | None = None

        # Per-call diagnostic log (drained by the simulation loop each step)
        self._log_hit_points: list = []   # [(t, pos_3d), ...]
        self._log_detections: list = []   # [(t, normal_3d, pos_3d), ...]  pos=NaN if none

    # ------------------------------------------------------------------
    # Main callable
    # ------------------------------------------------------------------

    def __call__(self, t: float, _state, imu_data, dist_data) -> np.ndarray:
        """
        Called at simulation rate (8 kHz).

        Parameters
        ----------
        t         : current simulation time (s)
        state     : (13,) ground-truth state — NOT used (sensor-only)
        imu_data  : (accel_body, gyro_body) tuple or None
        dist_data : (n_sensors,) range array or None

        Returns
        -------
        commands : (n_thrusters,) thruster force commands
        """
        zero = np.zeros(self.n_thrusters)

        # Phase transition: throw complete → begin ballistic coast + despin
        if self._phase == self.INACTIVE and t >= self.throw_duration:
            self._phase = self.BALLISTIC
            self._throw_end_p = self.estimator.p.copy()
            self._throw_end_v = self.estimator.v.copy()

        # Always update IMU estimator (estimator guards dt ≤ 0 internally)
        if imu_data is not None:
            accel, gyro = imu_data
            self.estimator.update(t, accel, gyro)
            self._last_gyro = gyro

        if self._phase == self.INACTIVE:
            return zero

        # Current state estimate
        p     = self.estimator.p
        v     = self.estimator.v
        R     = self.estimator.R
        omega = self._last_gyro

        # Distance sensor update (runs at sensor rate)
        if dist_data is not None and t - self._last_dist_t >= self._ds_period - 1e-9:
            self._process_distance(t, p, R, dist_data)
            self._last_dist_t = t

        # Surface detection (runs at detection rate, decoupled from sensors)
        if (self._throw_end_p is not None
                and self._throw_end_v is not None
                and t - self._last_detect_t >= self._detect_period - 1e-9):
            tau_elapsed = t - self.throw_duration
            traj = self._predict_trajectory(
                self._throw_end_p, self._throw_end_v, t_start=tau_elapsed)
            detection = self._detect_surface(traj, p, tau_elapsed)
            if detection is not None:
                normal, centroid = detection
                # Only commit to this surface if the trajectory actually reaches it
                p0 = self._throw_end_p
                v0 = self._throw_end_v
                a_coeff = 0.5 * np.dot(self.g_vec, normal)
                b_coeff = np.dot(v0, normal)
                c_coeff = np.dot(p0 - centroid, normal)
                intersection = self._solve_plane_intersection(
                    a_coeff, b_coeff, c_coeff, p0, v0, self.g_vec)
                if intersection is not None:
                    self.target_normal = normal
                    self.target_pos = intersection + self._com_surface_offset * normal
                    self._log_detections.append((t, normal.copy(), self.target_pos.copy()))
                else:
                    nan3 = np.full(3, np.nan)
                    self._log_detections.append((t, nan3, nan3))
            else:
                nan3 = np.full(3, np.nan)
                self._log_detections.append((t, nan3, nan3))
            self._last_detect_t = t

        # Phase transitions
        omega_z_mag = abs(np.dot(omega, self.thrust_axis_body))

        if self._phase == self.BALLISTIC and not self._yaw_despin_active:
            self._phase = self.TILT_DESPIN
            self._zero_vectoring()

        if self._phase == self.TILT_DESPIN:
            if omega_z_mag > self._yaw_despin_threshold:
                self._yaw_despin_active = True
                self._phase = self.BALLISTIC
            else:
                omega_perp_mag = np.linalg.norm(
                    omega - np.dot(omega, self.thrust_axis_body) * self.thrust_axis_body)
                if omega_perp_mag < self._tilt_despin_threshold * 0.5:
                    self._throw_end_p = p.copy()
                    self._throw_end_v = v.copy()
                    self._phase = self.TERMINAL

        if self._phase == self.REORIENT:
            v_error = self._ballistic_velocity_error(v, t)
            v_err_mag = np.linalg.norm(v_error)
            omega_perp_mag = np.linalg.norm(
                omega - np.dot(omega, self.thrust_axis_body) * self.thrust_axis_body)
            if v_err_mag < self._lateral_v_threshold:
                self._throw_end_p = p.copy()
                self._throw_end_v = v.copy()
                self._phase = self.TERMINAL
            elif omega_perp_mag < self._tilt_despin_threshold:
                thrust_world = R @ self.thrust_axis_body
                desired_dir = -v_error / v_err_mag
                if np.dot(thrust_world, desired_dir) > self._att_align_cos:
                    self._phase = self.LATERAL_CORRECT

        if self._phase == self.LATERAL_CORRECT:
            v_error = self._ballistic_velocity_error(v, t)
            if np.linalg.norm(v_error) < self._lateral_v_threshold:
                self._throw_end_p = p.copy()
                self._throw_end_v = v.copy()
                self._phase = self.TERMINAL

        # Dispatch
        if self._phase == self.BALLISTIC:
            return self._compute_yaw_despin(omega)

        # All phases after yaw despin: compute commands then zero vectoring
        if self._phase == self.TILT_DESPIN:
            cmds = self._compute_tilt_despin(omega)
        elif self._phase == self.REORIENT:
            cmds = self._compute_reorient(R, omega, v, t)
        elif self._phase == self.LATERAL_CORRECT:
            cmds = self._compute_lateral_correct(R, omega, v, t)
        else:
            cmds = self._compute_terminal(p, v, R, omega)

        # Zero vectoring unless yaw optimization is active in an orient phase
        if self._terminal_sub not in ('orient_divert', 'orient_brake'):
            self._zero_vectoring()
        return cmds

    # ------------------------------------------------------------------
    # Vectoring reset
    # ------------------------------------------------------------------

    def _zero_vectoring(self) -> None:
        """Command all vectoring angles to zero."""
        for thr in self.thruster_array.thrusters:
            if thr.vectoring_enabled:
                thr.set_vector_command(0.0)

    def _compute_yaw_align(self, att_err_body: np.ndarray,
                           omega: np.ndarray) -> np.ndarray | None:
        """
        Proportional yaw maneuver to align the tilt rotation axis with
        a principal torque axis (x or y body frame).

        Uses differential thrust (reaction torques) plus vectoring,
        both scaled proportionally to the desired yaw torque.  Thrust
        magnitudes are balanced (clamped to ±F_balanced) for zero net
        axial force.  Returns thruster commands, or None if aligned.
        """
        ax = self.thrust_axis_body
        # Tilt error projected to body xy-plane
        err_xy = att_err_body - np.dot(att_err_body, ax) * ax
        err_xy_mag = np.linalg.norm(err_xy)
        if err_xy_mag < 0.01:
            return None

        # Angle of tilt axis in body xy-plane
        phi = np.arctan2(err_xy[1], err_xy[0])

        # Yaw error to nearest principal axis (period pi/2, wrap to [-pi/4, pi/4])
        yaw_error = phi % (np.pi / 2)
        if yaw_error > np.pi / 4:
            yaw_error -= np.pi / 2

        # Check if aligned and settled
        omega_z = np.dot(omega, ax)
        if abs(yaw_error) < np.radians(5.0) and abs(omega_z) < 0.5:
            return None

        # Desired yaw torque (PD)
        tau_z_des = -self._yaw_opt_Kp * yaw_error - self._yaw_opt_Kd * omega_z

        # --- Proportional differential thrust (reaction torques) ---
        # f_i = -τ_z_des / (n_thrusters × k_q × spin_i), clamped to ±F_balanced
        commands = np.zeros(self.n_thrusters)
        for i, thr in enumerate(self.thruster_array.thrusters):
            f_i = -tau_z_des / (self.n_thrusters * thr.k_q * thr.spin_direction)
            commands[i] = np.clip(f_i, -self._F_balanced, self._F_balanced)

        # --- Proportional vectoring ---
        desired_tau_sign = np.sign(tau_z_des)
        scale = min(abs(tau_z_des) / 0.05, 1.0)  # ramp deflection with torque demand
        for i, thr in enumerate(self.thruster_array.thrusters):
            if not thr.vectoring_enabled:
                continue
            c_i = self._vec_coupling[i]
            if abs(c_i) < 1e-10 or abs(commands[i]) < 1e-10:
                thr.set_vector_command(0.0)
                continue
            delta_sign = desired_tau_sign * np.sign(commands[i]) * np.sign(c_i)
            thr.set_vector_command(delta_sign * thr.max_deflection * scale)

        return commands

    # ------------------------------------------------------------------
    # Saturation scaling
    # ------------------------------------------------------------------

    def _saturate_scale(self, wrench: np.ndarray) -> np.ndarray:
        """
        Scale a wrench so the pseudoinverse solution fits within actuator
        limits *before* clipping.  This preserves the zero-net-force
        property of torque-only wrenches that would otherwise be destroyed
        by per-thruster clipping.

        Delegates to ThrusterArray.saturate_scale_wrench which handles
        both fixed-direction and vectoring-enabled layouts.
        """
        return self.thruster_array.saturate_scale_wrench(wrench)

    # ------------------------------------------------------------------
    # Despin
    # ------------------------------------------------------------------

    def _compute_yaw_despin(self, omega: np.ndarray) -> np.ndarray:
        """
        Yaw despin via propeller reaction torques AND thrust vectoring.

        Two torque mechanisms act together:

        1. Reaction torque: τ_z_i = −k_q · spin_i · f_i
           One diagonal pair at +max_force, the other at −min_force.
           All four reaction torques oppose the spin; z-forces partially cancel.

        2. Vectoring moment-arm torque: τ_z_i = f_i · sin(δ_i) · c_i
           Each thruster tilts so the lateral force component creates a
           moment about z that also opposes the spin.  The deflection sign
           is chosen per-thruster based on the precomputed coupling c_i.

        A proportional law (Kd_att · omega_z) smoothly reduces force
        commands as omega_z → 0.  Deflections are set to ±max when active.
        """
        omega_z = np.dot(omega, self.thrust_axis_body)

        # Yaw hysteresis
        if abs(omega_z) > self._yaw_despin_threshold:
            self._yaw_despin_active = True
        elif abs(omega_z) < self._yaw_despin_threshold * 0.5:
            self._yaw_despin_active = False

        if not self._yaw_despin_active:
            return np.zeros(self.n_thrusters)

        # --- Force commands (reaction torque) ---
        # Desired yaw torque: τ_z = −Kd · omega_z
        # Per-thruster: τ_z_i = −k_q · spin_i · f_i
        # Solve for f_i: f_i = −τ_z_des / (n · k_q · spin_i)
        tau_z_des = -self.Kd_att * omega_z

        commands = np.zeros(self.n_thrusters)
        for i, thr in enumerate(self.thruster_array.thrusters):
            f_i = -tau_z_des / (self.n_thrusters * thr.k_q * thr.spin_direction)
            commands[i] = np.clip(f_i, thr.min_force, thr.max_force)

        # --- Vectoring commands (moment-arm torque) ---
        # δ_i chosen so f_i · sin(δ_i) · c_i opposes omega_z.
        # sign(δ_i) = −sign(omega_z) · sign(f_i) · sign(c_i)
        desired_tau_sign = -np.sign(omega_z)
        for i, thr in enumerate(self.thruster_array.thrusters):
            if not thr.vectoring_enabled:
                continue
            c_i = self._vec_coupling[i]
            if abs(c_i) < 1e-10 or abs(commands[i]) < 1e-10:
                thr.set_vector_command(0.0)
                continue
            delta_sign = desired_tau_sign * np.sign(commands[i]) * np.sign(c_i)
            thr.set_vector_command(delta_sign * thr.max_deflection)

        return commands

    # ------------------------------------------------------------------
    # Tilt despin (two-stage: bang-bang → PD attitude steering)
    # ------------------------------------------------------------------

    def _compute_tilt_despin(self, omega: np.ndarray) -> np.ndarray:
        """
        Bang-bang tilt despin: maximum opposing torque via the torque-only
        pseudoinverse.  Fast despin to minimise time under lateral-producing
        thrust.
        """
        ax = self.thrust_axis_body
        omega_perp = omega - np.dot(omega, ax) * ax
        omega_perp_mag = np.linalg.norm(omega_perp)

        if omega_perp_mag < 1e-6:
            return np.zeros(self.n_thrusters)

        spin_axis = omega_perp / omega_perp_mag
        unit_commands = self._B_tau_pinv @ (-spin_axis)

        commands = np.zeros(self.n_thrusters)
        for i, thr in enumerate(self.thruster_array.thrusters):
            if unit_commands[i] > 1e-10:
                commands[i] = thr.max_force
            elif unit_commands[i] < -1e-10:
                commands[i] = thr.min_force
        return commands

    # ------------------------------------------------------------------
    # Reorientation (zero-force, torque only)
    # ------------------------------------------------------------------

    def _compute_reorient(self, R: np.ndarray, omega: np.ndarray,
                          v: np.ndarray, t: float,
                          override_dir: np.ndarray | None = None,
                          yaw_align: bool = False) -> np.ndarray:
        """
        Reactive reorientation via uniformly-scaled PD (zero net force).

        A PD attitude torque (Kp on orientation error, Kd on angular
        velocity) is converted to thruster commands via B_tau_pinv,
        then uniformly scaled so the largest command reaches F_balanced.
        This smoothly tracks the rotating error axis — unlike bang-bang
        sign quantization which locks onto a few discrete torque
        directions.  The Kp/Kd ratio ≈ sqrt(2·α_max) approximates the
        optimal braking curve in phase space.

        If override_dir is provided, it is used as the desired direction
        instead of the lateral velocity error direction.
        """
        if override_dir is not None:
            desired_dir = override_dir
        else:
            v_error = self._ballistic_velocity_error(v, t)
            v_err_mag = np.linalg.norm(v_error)
            if v_err_mag > self._lateral_v_threshold:
                desired_dir = -v_error / v_err_mag
            else:
                desired_dir = np.array([0.0, 0.0, 1.0])

        # Attitude error
        ax = self.thrust_axis_body
        current_axis_world = R @ ax
        cross = np.cross(current_axis_world, desired_dir)
        sin_th = np.linalg.norm(cross)
        cos_th = float(np.clip(np.dot(current_axis_world, desired_dir), -1.0, 1.0))
        angle = np.arctan2(sin_th, cos_th)
        if sin_th > 1e-6:
            att_err_body = R.T @ (angle * (cross / sin_th))
        else:
            att_err_body = np.zeros(3)

        omega_perp = omega - np.dot(omega, ax) * ax
        tau = self._reorient_Kp * att_err_body - self._reorient_Kd * omega_perp
        tau -= np.dot(tau, ax) * ax  # no thrust-axis component

        # Sequential yaw alignment: if the tilt axis isn't on a principal
        # torque axis AND the tilt angle is large enough to benefit,
        # yaw first using balanced differential thrust + vectoring.
        if yaw_align:
            yaw_cmds = self._compute_yaw_align(att_err_body, omega)
            if yaw_cmds is not None:
                self._orient_is_yawing = True
                return yaw_cmds
            # Yaw done — zero vectoring before tilting
            self._zero_vectoring()
        self._orient_is_yawing = False

        # Uniformly scale to max balanced thrust
        raw_commands = self._B_tau_pinv @ tau
        max_abs = np.abs(raw_commands).max()
        if max_abs < 1e-10:
            return np.zeros(self.n_thrusters)

        return raw_commands * (self._F_balanced / max_abs)

    # ------------------------------------------------------------------
    # Lateral velocity correction
    # ------------------------------------------------------------------

    def _compute_lateral_correct(self, R: np.ndarray, omega: np.ndarray,
                                 v: np.ndarray, t: float) -> np.ndarray:
        """
        Axial burn to cancel despin-induced lateral velocity error.

        The thrust axis was pre-aligned by tilt despin stage 2.  A
        proportional controller drives the velocity error to zero while
        PD attitude torque holds alignment.
        """
        v_error = self._ballistic_velocity_error(v, t)
        v_err_mag = np.linalg.norm(v_error)

        if v_err_mag < self._lateral_v_threshold:
            return np.zeros(self.n_thrusters)

        # Force: proportional to velocity error, along thrust axis
        F_mag = min(self._lateral_Kp_v * v_err_mag * self.mass,
                    self._F_max_total)
        F_body = F_mag * self.thrust_axis_body

        # Attitude hold: keep thrust axis opposing velocity error
        desired_dir = -v_error / v_err_mag
        tau = self._attitude_torque(R, omega, desired_dir)

        wrench = np.concatenate([F_body, tau])
        return self.thruster_array.wrench_to_commands(wrench)

    # ------------------------------------------------------------------
    # Terminal guidance
    # ------------------------------------------------------------------

    def _compute_terminal(self, p: np.ndarray, v: np.ndarray,
                          R: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """
        Four-phase terminal guidance for perpendicular surface landing.

        Phase 1 — Divert: cancel tangential velocity at the predicted
        ballistic impact.  No gravity compensation — just zero tangential.
        Uses actual current velocity for incidence check.

        Phase 2 — Orient to brake: rotate to the brake direction, which
        simultaneously decelerates the approach and fights gravity:
            d_brake = normalize(a_brake · n - g_vec)

        Phase 3 — Brake: max thrust along the brake direction.  The
        gravity component keeps the ball from drifting tangentially.
        Fires until approach speed reaches landing speed.

        Phase 4 — Final orient: rotate the last ~18° to align thrust
        axis with surface normal for contact.  Coast at landing speed.
        """
        n = self.target_normal
        v_approach_now = -np.dot(v, n)
        dist_to_surface = np.dot(p - self.target_pos, n)

        # --- Brake direction: away from surface + cancel gravity + cancel tangential ---
        # During braking: a_desired = a_brake·n - v_tan/t_brake
        # F_thrust = m · (a_brake·n - v_tan/t_brake - g_vec)
        a_max_total = self._F_max_total / self.mass
        v_tan_now = v - np.dot(v, n) * n

        # Step 1: gravity-only brake estimate for t_brake seed
        g_perp_mag = np.linalg.norm(self.g_vec - np.dot(self.g_vec, n) * n)
        a_brake_base = np.sqrt(max(a_max_total ** 2 - g_perp_mag ** 2, 0.0))
        a_brake_eff_base = a_brake_base + np.dot(self.g_vec, n)
        t_brake_est = max(v_approach_now - self._landing_speed, 0.01) / max(a_brake_eff_base, 0.1)

        # Step 2: tangential correction — only the component that gravity won't fix.
        # Remove the gravity-tangential component: gravity changes v_tan during
        # coast, so the current v_tan_z is transient. Only correct the part
        # that persists (perpendicular to both n and g_tangent).
        g_tangent = self.g_vec - np.dot(self.g_vec, n) * n
        g_tan_mag = np.linalg.norm(g_tangent)
        if g_tan_mag > 1e-6:
            g_tan_dir = g_tangent / g_tan_mag
            # Project out the gravity-tangent direction from v_tan
            v_tan_persistent = v_tan_now - np.dot(v_tan_now, g_tan_dir) * g_tan_dir
        else:
            v_tan_persistent = v_tan_now

        v_tan_pers_mag = np.linalg.norm(v_tan_persistent)
        if t_brake_est > 0.01 and v_tan_pers_mag > 0.01:
            a_tan_corr = -v_tan_persistent / t_brake_est
        else:
            a_tan_corr = np.zeros(3)

        # Step 3: recompute brake accel with full perpendicular load
        perp_load = -self.g_vec + a_tan_corr
        perp_perp = perp_load - np.dot(perp_load, n) * n
        perp_perp_mag = np.linalg.norm(perp_perp)

        if perp_perp_mag < a_max_total:
            a_brake_max = np.sqrt(max(a_max_total ** 2 - perp_perp_mag ** 2, 0.0))
        else:
            a_brake_max = a_brake_base
            a_tan_corr = np.zeros(3)

        a_brake_eff = a_brake_max + np.dot(self.g_vec, n)

        # Brake thrust direction (world frame)
        brake_accel = a_brake_max * n + a_tan_corr - self.g_vec
        brake_dir = brake_accel / np.linalg.norm(brake_accel)

        # Ground-like surface: no final orient needed (brake dir ≈ surface normal)
        is_ground = np.dot(n, np.array([0.0, 0.0, 1.0])) > 0.7

        # Braking distance at effective deceleration
        braking_dist = max(v_approach_now ** 2 - self._landing_speed ** 2, 0.0) / (
            2.0 * a_brake_eff) if a_brake_eff > 0.1 else 0.0

        # Log alignment between thrust axis and brake direction
        thrust_axis_world = R @ self.thrust_axis_body
        cos_align = np.dot(thrust_axis_world, brake_dir)
        self._brake_align_deg = float(np.degrees(np.arccos(np.clip(cos_align, -1.0, 1.0))))

        # --- Sub-phase transitions ---
        if self._terminal_sub in ('orient_divert', 'divert'):
            # Recompute divert direction if surface changed or not yet computed
            surface_changed = (self._divert_normal is not None
                               and np.dot(self._divert_normal, n) < 0.99)
            if self._divert_dir is None or surface_changed:
                t_impact = self._predict_surface_impact(p, v, n)
                if t_impact is not None:
                    v_impact = v + self.g_vec * t_impact
                    v_normal_mag = np.dot(v_impact, n)
                    v_tangent = v_impact - v_normal_mag * n
                    v_tan_mag = np.linalg.norm(v_tangent)
                    if v_tan_mag > 1e-6:
                        self._divert_dir = -v_tangent / v_tan_mag
                        self._divert_normal = n.copy()
                        # If surface changed mid-divert, go back to orient
                        if surface_changed and self._terminal_sub == 'divert':
                            self._terminal_sub = 'orient_divert'
                    else:
                        self._terminal_sub = 'orient_brake'

        if self._terminal_sub == 'orient_divert':
            # Check if aligned AND settled
            if self._divert_dir is not None:
                thrust_w = R @ self.thrust_axis_body
                ax = self.thrust_axis_body
                omega_perp_mag = np.linalg.norm(
                    omega - np.dot(omega, ax) * ax)
                if (np.dot(thrust_w, self._divert_dir) > 0.95
                        and omega_perp_mag < 2.0):
                    self._terminal_sub = 'divert'

        if self._terminal_sub == 'divert':
            # Recalculate incidence each step to check if correction is done
            t_impact = self._predict_surface_impact(p, v, n)
            if t_impact is not None:
                v_impact = v + self.g_vec * t_impact
                v_normal_mag = np.dot(v_impact, n)
                v_tangent = v_impact - v_normal_mag * n
                v_tan_mag = np.linalg.norm(v_tangent)

                v_for_inc = v_tangent + self._landing_speed * (-n)
                v_fi_mag = np.linalg.norm(v_for_inc)
                if v_fi_mag > 1e-6:
                    cos_inc = self._landing_speed / v_fi_mag
                    incidence = np.degrees(np.arccos(np.clip(cos_inc, 0.0, 1.0)))
                else:
                    incidence = 0.0

                if incidence < self._divert_done_deg:
                    self._terminal_sub = 'orient_brake'
                # Stop if tangential has flipped (overcorrected)
                elif (v_tan_mag > 1e-6
                      and np.dot(-v_tangent / v_tan_mag, self._divert_dir) < 0):
                    self._terminal_sub = 'orient_brake'

        if self._terminal_sub == 'orient_brake':
            if v_approach_now > self._landing_speed:
                if is_ground:
                    offset = 0.0
                else:
                    final_orient_time = 0.05
                    offset = self._landing_speed * final_orient_time
                if braking_dist >= (dist_to_surface - offset) * 0.8 or dist_to_surface < offset + 0.05:
                    self._terminal_sub = 'brake'

        elif self._terminal_sub == 'brake':
            if v_approach_now <= self._landing_speed and v_approach_now > 0:
                if is_ground:
                    return np.zeros(self.n_thrusters)
                else:
                    self._terminal_sub = 'final_orient'

        # --- Phase 4: Final orient (align thrust axis with surface normal) ---
        if self._terminal_sub == 'final_orient':
            return self._compute_reorient(R, omega, v, 0.0,
                                          override_dir=n)

        # --- Phase 3: Brake (all thrusters at max + attitude hold via B_tau_pinv) ---
        if self._terminal_sub == 'brake':
            # Full thrust: all thrusters at max_force
            commands = np.array([thr.max_force for thr in self.thruster_array.thrusters])

            # Small attitude correction toward gravity-only brake direction
            tau = self._attitude_torque_adaptive(R, omega, brake_dir)
            raw_tau_cmds = self._B_tau_pinv @ tau
            max_abs = np.abs(raw_tau_cmds).max()
            if max_abs > 1e-10:
                tau_cmds = raw_tau_cmds * (self._F_balanced * 0.3 / max_abs)
                commands = commands + tau_cmds

            # Clip to per-thruster limits
            for i, thr in enumerate(self.thruster_array.thrusters):
                commands[i] = np.clip(commands[i], thr.min_force, thr.max_force)
            return commands

        # --- Phase 2: Orient to brake direction (yaw align then tilt) ---
        if self._terminal_sub == 'orient_brake':
            return self._compute_reorient(R, omega, v, 0.0,
                                          override_dir=brake_dir,
                                          yaw_align=True)

        # --- Phase 0: Orient to divert direction (yaw align then tilt) ---
        if self._terminal_sub == 'orient_divert':
            if self._divert_dir is not None:
                return self._compute_reorient(R, omega, v, 0.0,
                                              override_dir=self._divert_dir,
                                              yaw_align=True)
            return np.zeros(self.n_thrusters)

        # --- Phase 1: Divert (all thrusters at max + attitude hold via B_tau_pinv) ---
        if self._terminal_sub == 'divert':
            assert self._divert_dir is not None
            # Full thrust: all thrusters at max_force
            commands = np.array([thr.max_force for thr in self.thruster_array.thrusters])

            # Small attitude correction via zero-force torque (doesn't affect thrust)
            tau = self._attitude_torque_adaptive(R, omega, self._divert_dir)
            raw_tau_cmds = self._B_tau_pinv @ tau
            max_abs = np.abs(raw_tau_cmds).max()
            if max_abs > 1e-10:
                tau_cmds = raw_tau_cmds * (self._F_balanced * 0.3 / max_abs)
                commands = commands + tau_cmds

            for i, thr in enumerate(self.thruster_array.thrusters):
                commands[i] = np.clip(commands[i], thr.min_force, thr.max_force)
            return commands

        return np.zeros(self.n_thrusters)

    def _predict_surface_impact(self, p: np.ndarray, v: np.ndarray,
                                 n: np.ndarray) -> float | None:
        """Solve for when the ballistic trajectory hits the target surface."""
        a_c = 0.5 * np.dot(self.g_vec, n)
        b_c = np.dot(v, n)
        c_c = np.dot(p - self.target_pos, n)
        disc = b_c ** 2 - 4.0 * a_c * c_c
        if disc >= 0 and abs(a_c) > 1e-10:
            sq = np.sqrt(disc)
            roots = [(-b_c + sq) / (2 * a_c), (-b_c - sq) / (2 * a_c)]
            pos_roots = [r for r in roots if r > 0.01]
            if pos_roots:
                return min(pos_roots)
        elif abs(b_c) > 1e-10:
            t_cand = -c_c / b_c
            if t_cand > 0.01:
                return t_cand
        return None

    # ------------------------------------------------------------------
    # Attitude torque (PD)
    # ------------------------------------------------------------------

    def _attitude_torque(self, R: np.ndarray, omega: np.ndarray,
                         desired_axis_world: np.ndarray) -> np.ndarray:
        """
        PD torque to align thrust axis with a desired world-frame direction.

        Kp drives the axis-angle orientation error, Kd damps perpendicular
        angular velocity.  Both are projected out of the thrust axis to
        avoid competing with yaw control.
        """
        current_axis_world = R @ self.thrust_axis_body
        cross = np.cross(current_axis_world, desired_axis_world)
        sin_th = np.linalg.norm(cross)
        cos_th = float(np.clip(np.dot(current_axis_world, desired_axis_world),
                                -1.0, 1.0))
        angle = np.arctan2(sin_th, cos_th)
        if sin_th > 1e-6:
            att_err_body = R.T @ (angle * (cross / sin_th))
        else:
            att_err_body = np.zeros(3)

        ax = self.thrust_axis_body
        omega_perp = omega - np.dot(omega, ax) * ax

        tau = self.Kp_att * att_err_body - self.Kd_att * omega_perp
        tau -= np.dot(tau, ax) * ax  # no thrust-axis component
        return tau

    def _attitude_torque_adaptive(self, R: np.ndarray, omega: np.ndarray,
                                   desired_axis_world: np.ndarray) -> np.ndarray:
        """
        Adaptive-gain PD attitude torque for terminal guidance.

        Kp is computed from the current angle error so the Kp/Kd ratio
        matches the optimal braking curve:  Kp/Kd = 2·sqrt(α/θ).
        This gives aggressive response at small angles and damped
        response at large angles.
        """
        current_axis_world = R @ self.thrust_axis_body
        cross = np.cross(current_axis_world, desired_axis_world)
        sin_th = np.linalg.norm(cross)
        cos_th = float(np.clip(np.dot(current_axis_world, desired_axis_world),
                                -1.0, 1.0))
        angle = np.arctan2(sin_th, cos_th)
        if sin_th > 1e-6:
            att_err_body = R.T @ (angle * (cross / sin_th))
        else:
            att_err_body = np.zeros(3)

        ax = self.thrust_axis_body
        omega_perp = omega - np.dot(omega, ax) * ax

        # Adaptive Kp with extra damping for cross-axis + thruster lag
        Kp = 2.0 * np.sqrt(self._alpha_tilt / max(angle, 0.01))
        Kd = 2.0

        tau = Kp * att_err_body - Kd * omega_perp
        tau -= np.dot(tau, ax) * ax
        return tau

    # ------------------------------------------------------------------
    # Reorientation torque for divert (fills actuator headroom)
    # ------------------------------------------------------------------

    def _reorient_torque_for_divert(self, R: np.ndarray, omega: np.ndarray,
                                     desired_axis_world: np.ndarray,
                                     F_cmd_body: np.ndarray) -> np.ndarray:
        """
        Aggressive PD reorientation torque scaled to fill the actuator
        headroom left by the force command.

        Uses the same high-authority PD gains as _compute_reorient.  The
        torque is uniformly scaled so the combined wrench (force + torque)
        stays within actuator limits: when force is small (large
        misalignment), nearly the full budget goes to torque for fast
        rotation; as alignment improves and force grows, torque scales
        down gracefully.
        """
        ax = self.thrust_axis_body
        current_axis_world = R @ ax
        cross = np.cross(current_axis_world, desired_axis_world)
        sin_th = np.linalg.norm(cross)
        cos_th = float(np.clip(np.dot(current_axis_world, desired_axis_world),
                                -1.0, 1.0))
        angle = np.arctan2(sin_th, cos_th)
        if sin_th > 1e-6:
            att_err_body = R.T @ (angle * (cross / sin_th))
        else:
            att_err_body = np.zeros(3)

        omega_perp = omega - np.dot(omega, ax) * ax
        Kp = 2.0 * np.sqrt(self._alpha_tilt / max(angle, 0.01))
        Kd = 2.0  # extra damping to absorb cross-axis momentum + thruster lag
        tau = Kp * att_err_body - Kd * omega_perp
        tau -= np.dot(tau, ax) * ax  # no thrust-axis component

        # Estimate actuator headroom: find what fraction of per-thruster
        # limits the force command alone consumes, then scale torque to
        # fill the remainder.
        force_wrench = np.concatenate([F_cmd_body, np.zeros(3)])
        force_cmds = self.thruster_array.wrench_to_commands(force_wrench)
        force_usage = 0.0
        for i, thr in enumerate(self.thruster_array.thrusters):
            limit = thr.max_force if force_cmds[i] >= 0 else abs(thr.min_force)
            if limit > 1e-10:
                force_usage = max(force_usage, abs(force_cmds[i]) / limit)

        headroom = max(1.0 - force_usage, 0.0)
        if headroom < 1e-6:
            return np.zeros(3)

        # Scale torque to fill headroom: compute what the torque alone
        # would need, then cap at headroom fraction of balanced thrust.
        raw_commands = self._B_tau_pinv @ tau
        max_abs = np.abs(raw_commands).max()
        if max_abs < 1e-10:
            return np.zeros(3)

        # Target command magnitude: headroom * F_balanced
        desired_scale = headroom * self._F_balanced / max_abs
        return tau * desired_scale

    # ------------------------------------------------------------------
    # Ballistic velocity error
    # ------------------------------------------------------------------

    def _ballistic_velocity_error(self, v: np.ndarray, t: float) -> np.ndarray:
        """
        Velocity deviation from the expected ballistic trajectory.

        Returns v_current − v_ballistic, where v_ballistic is the
        gravity-only prediction from the throw-end snapshot.  Any nonzero
        result is entirely from thruster forces (despin side-effects).
        """
        assert self._throw_end_v is not None
        dt = t - self.throw_duration
        v_ballistic = self._throw_end_v + self.g_vec * dt
        return v - v_ballistic

    # ------------------------------------------------------------------
    # Distance processing & EKF correction
    # ------------------------------------------------------------------

    def _process_distance(self, t: float, p: np.ndarray, R: np.ndarray,
                          dist_readings: np.ndarray) -> None:
        """
        Compute world-frame hit points and apply lightweight EKF corrections.

        For each sensor whose ray is broadly aligned with a known surface
        and whose hit point lands near that surface, the position estimate
        is nudged along the constrained axis by a small gain-weighted step.
        """
        pos_geom = p - R @ self.com_offset

        for j, d in enumerate(dist_readings):
            if d >= self.max_range:
                continue

            ray_world    = R @ self.sensor_dirs_body[j]
            sensor_world = pos_geom + R @ self.sensor_pos_geom[j]
            hit          = sensor_world + d * ray_world
            self.hit_points.append(hit)
            self._log_hit_points.append((t, hit.copy()))

            # Floor correction (z=0): ray must point predominantly downward
            if (ray_world[2] < -0.5
                    and abs(hit[2] - self.floor_z) < self.ekf_surf_tol):
                residual = self.floor_z - hit[2]
                self.estimator._p[2] += self.ekf_gain_floor * residual   # noqa: SLF001

            # Wall correction (x=wall_x): ray must point toward wall (+x)
            if (ray_world[0] > 0.5
                    and abs(hit[0] - self.wall_x) < self.ekf_surf_tol):
                residual = self.wall_x - hit[0]
                self.estimator._p[0] += self.ekf_gain_wall * residual    # noqa: SLF001

    # ------------------------------------------------------------------
    # Trajectory–plane intersection
    # ------------------------------------------------------------------

    @staticmethod
    def _solve_plane_intersection(a: float, b: float, c: float,
                                  p0: np.ndarray, v0: np.ndarray,
                                  g_vec: np.ndarray) -> np.ndarray | None:
        """
        Solve  a·τ² + b·τ + c = 0  for the smallest positive τ and
        return the trajectory point  p0 + v0·τ + ½·g·τ².

        Returns None if no positive root exists.
        """
        if abs(a) > 1e-10:
            disc = b * b - 4.0 * a * c
            if disc < 0:
                return None
            sq = np.sqrt(disc)
            roots = [(-b + sq) / (2.0 * a), (-b - sq) / (2.0 * a)]
        elif abs(b) > 1e-10:
            roots = [-c / b]
        else:
            return None

        pos_roots = [r for r in roots if r > 0.01]
        if not pos_roots:
            return None
        tau = min(pos_roots)
        return p0 + v0 * tau + 0.5 * g_vec * tau ** 2

    # ------------------------------------------------------------------
    # Ballistic trajectory prediction
    # ------------------------------------------------------------------

    def _predict_trajectory(self, p: np.ndarray, v: np.ndarray,
                             t_start: float = 0.0) -> np.ndarray:
        """
        Sample  r(τ) = p + v·τ + ½·g·τ²  from τ=t_start to ground impact.

        Parameters
        ----------
        t_start : time already elapsed since p, v snapshot — samples
                  begin here so they cover only the remaining flight.

        Returns (M, 3) array of world-frame positions.
        """
        # Time to ground: solve  p.z + v.z·τ - ½·g·τ² = 0
        a = -0.5 * self.gravity
        b = v[2]
        c = p[2]
        disc = b * b - 4.0 * a * c
        t_ground = 10.0
        if disc >= 0:
            roots = [(-b + np.sqrt(disc)) / (2*a),
                     (-b - np.sqrt(disc)) / (2*a)]
            pos_roots = [r for r in roots if r > 0.05]
            if pos_roots:
                t_ground = min(pos_roots)

        ts   = np.linspace(max(t_start, 0.0), min(t_ground, 10.0), self.N_traj)
        traj = (p[None, :]
                + v[None, :] * ts[:, None]
                + 0.5 * self.g_vec[None, :] * (ts * ts)[:, None])
        return traj

    # ------------------------------------------------------------------
    # Surface detection
    # ------------------------------------------------------------------

    def _detect_surface(self, trajectory: np.ndarray,
                        p: np.ndarray,
                        tau_elapsed: float = 0.0) -> tuple | None:
        """
        Cluster-based surface detection along the frozen trajectory.

        For each sample point on the trajectory, grab all hitpoints within
        the adaptive cluster radius.  The radius starts large (1.0 m) for
        early detection and shrinks toward 0.5 m as time progresses.

        If enough points cluster, their centroid and coplanarity are
        assessed via SVD.  Among all valid coplanar clusters, the one
        whose surface is closest to the ball is selected.

        Returns (normal, centroid) or None if no valid surface found.
        """
        if len(self.hit_points) < self.cluster_min:
            return None

        pts = np.array(self.hit_points)          # (K, 3)
        traj = trajectory                         # (M, 3)

        # Vectorised squared distances: (M, K)
        # ||traj[j] - pts[i]||² = ||traj||² + ||pts||² - 2·traj·ptsᵀ
        traj_sq = (traj * traj).sum(axis=1)       # (M,)
        pts_sq  = (pts * pts).sum(axis=1)          # (K,)
        dist_sq = traj_sq[:, None] + pts_sq[None, :] - 2.0 * (traj @ pts.T)
        np.maximum(dist_sq, 0.0, out=dist_sq)

        # Adaptive radius: exponential decay from initial to final
        blend = 1.0 - np.exp(-tau_elapsed / self._cluster_radius_tau)
        radius = (self._cluster_radius_initial
                  + blend * (self._cluster_radius_final - self._cluster_radius_initial))
        r_sq = radius ** 2

        best = None
        best_dist = float('inf')

        for j in range(len(traj)):
            mask = dist_sq[j] < r_sq
            count = mask.sum()
            if count < self.cluster_min:
                continue

            nearby = pts[mask]
            centroid = nearby.mean(axis=0)
            # Find the plane of the hitpoints themselves
            vecs = nearby - centroid
            _, S, Vt = np.linalg.svd(vecs, full_matrices=False)

            # Coplanarity: smallest / second-smallest singular value
            if S[-1] / (S[-2] + 1e-10) > self.quality_th:
                continue

            normal = Vt[-1]

            # Normal must point toward the ball
            if np.dot(normal, p - centroid) < 0:
                normal = -normal

            # Pick the closest valid surface
            dist_to_ball = abs(np.dot(p - centroid, normal))
            if dist_to_ball < best_dist:
                best_dist = dist_to_ball
                best = (normal, centroid)

        return best

