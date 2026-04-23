"""
FlightController — phase-plane switching controller for the Jump ball.

Flight sequence:
  YAW_DESPIN → TILT_DESPIN → ORIENT_BURN → BURN → FINAL_ORIENT

Orientation maneuvers use phase-plane switching (near time-optimal,
zero overshoot) on single axes:
  - Far from target: ±tau_max (bang-bang via switching curve)
  - Near target: PD with reference prefilter (smooth settling)

Despin phases use max-torque configs (residual Z force OK).
Orient phases use zero-Z configs (no stray forces).
"""

import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.estimator import IncrementalEstimator


# ===================================================================
#  Phase-plane switching controller (single axis)
# ===================================================================

def phase_plane_torque(error, omega, alpha_max, tau_max, tau_f,
                       Kp, Kd, lag_margin=2.15):
    """
    Phase-plane switching + PD settling for a single rotation axis.

    Far from target: command ±tau_max based on switching curve.
    Near target: PD with reference prefilter (handled externally
    by the caller via the returned mode).

    Parameters
    ----------
    error     : angle error (rad), positive = need positive torque
    omega     : angular rate (rad/s)
    alpha_max : max angular acceleration (tau_max / I)
    tau_max   : max torque magnitude (N·m)
    tau_f     : force time constant (s)
    Kp, Kd    : PD gains (triple-pole critical damping)
    lag_margin: switching curve margin multiplier

    Returns
    -------
    tau_cmd : commanded torque (N·m), clipped to ±tau_max
    mode    : 'bang' or 'pd'
    """
    # PD threshold: use a fraction of tau_max/Kp to minimize PD settle time
    pd_threshold = 0.3 * tau_max / Kp

    # Check if we're in the PD region
    if abs(error) < pd_threshold and abs(omega) < alpha_max * 3 * tau_f:
        # PD mode — caller handles prefilter
        return np.clip(-Kp * error - Kd * omega, -tau_max, tau_max), 'pd'

    # Phase-plane switching
    if abs(omega) > 1e-10:
        stop_dist = omega**2 / (2 * alpha_max)
        margin = abs(omega) * tau_f * lag_margin
    else:
        stop_dist = 0.0
        margin = 0.0

    # error > 0 means we need to move in positive direction
    # omega > 0 means moving in positive direction
    if error > 0:
        # Need positive rotation
        if omega >= 0:
            # Moving toward target — check if we'd overshoot
            if stop_dist + margin >= error:
                return -tau_max, 'bang'  # brake
            else:
                return tau_max, 'bang'   # accelerate
        else:
            # Moving away from target — accelerate toward it
            return tau_max, 'bang'
    else:
        # Need negative rotation (error < 0)
        if omega <= 0:
            # Moving toward target — check if we'd overshoot
            if stop_dist + margin >= abs(error):
                return tau_max, 'bang'   # brake
            else:
                return -tau_max, 'bang'  # accelerate
        else:
            # Moving away — accelerate toward target
            return -tau_max, 'bang'


def is_settled(error, omega, tol_angle=0.1, tol_rate=1.0):
    """Check if a single axis is settled (degrees)."""
    return abs(np.degrees(error)) < tol_angle and abs(np.degrees(omega)) < tol_rate


# ===================================================================
#  Main controller
# ===================================================================

class FlightController:

    INACTIVE       = 0
    YAW_DESPIN     = 1
    YAW_ALIGN      = 2
    TILT_DESPIN    = 3
    ORIENT_BURN    = 4
    BURN           = 5
    FINAL_ORIENT   = 6
    CEIL_ORI_BOOST = 10   # orient upward (zero-Z)
    CEIL_BOOST     = 11   # brake/boost + kill tangential
    CEIL_ORI_LAND  = 12   # orient inverted for landing (zero-Z)
    CEIL_APPROACH  = 13   # final approach at landing speed

    PHASE_NAMES = {
        0: 'INACTIVE', 1: 'YAW_DESPIN', 2: 'YAW_ALIGN',
        3: 'TILT_DESPIN', 4: 'ORIENT_BURN', 5: 'BURN',
        6: 'FINAL_ORIENT',
        10: 'C_ORI_B', 11: 'C_BOOST', 12: 'C_ORI_LAND',
        13: 'C_APPR',
    }

    def __init__(self, cfg: dict, thruster_array):
        self.cfg = cfg
        self.thruster_array = thruster_array
        self.n_thrusters = thruster_array.n

        # ---- Physics ----
        self.mass = float(cfg['ball']['mass'])
        self.gravity = float(cfg['physics']['gravity'])
        self.g_vec = np.array([0.0, 0.0, -self.gravity])
        self.ball_radius = float(cfg['ball']['radius'])
        self.com_offset = np.array(cfg['ball']['center_of_mass'], dtype=float)
        I_diag = np.array(cfg['ball']['moment_of_inertia']['diagonal'])
        self.Ixx, self.Iyy, self.Izz = I_diag

        # ---- Timing ----
        self.throw_duration = float(cfg['throw_phase']['duration'])
        tau_f = float(cfg['thrusters'][0]['time_constant'])
        tau_s = float(cfg['thrusters'][0]['vectoring']['time_constant'])
        self.tau_f = tau_f
        self.tau_s = tau_s

        # ---- Thrust budget ----
        f_max = float(cfg['thrusters'][0]['max_force'])
        f_min = float(cfg['thrusters'][0]['min_force'])
        f_cap = min(f_max, abs(f_min))
        self.F_total = sum(t.max_force for t in thruster_array.thrusters)
        self.f_max = f_max
        self.f_min = f_min
        self.f_cap = f_cap

        # ---- Thrust axis ----
        B = thruster_array.nominal_allocation_matrix()
        U_f, S_f, _ = np.linalg.svd(B[:3, :])
        self.thrust_axis_body = U_f[:, 0]
        mean_dir = np.mean([t.nominal_direction for t in thruster_array.thrusters], axis=0)
        if np.dot(self.thrust_axis_body, mean_dir) < 0:
            self.thrust_axis_body = -self.thrust_axis_body

        # ---- Torque-only pseudoinverse ----
        self._B_tau = B[3:, :]
        self._B_tau_pinv = np.linalg.pinv(self._B_tau)
        # Pitch/roll only (no yaw coupling) for tilt despin
        self._B_pr_pinv = np.linalg.pinv(self._B_tau[:2, :])

        # ---- Virtual channel decomposition (for simultaneous orient) ----
        B_v, M = thruster_array.virtual_input_basis()
        self._B_v = B_v  # (6, 4) wrench per unit virtual input
        self._M = M       # (4, 4) pattern matrix
        # Channel mapping: ch1=pitch(ty), ch2=roll(tx), ch3=yaw(tz)
        self._vc_pitch_gain = B_v[4, 1]   # ty per unit v1 (negative)
        self._vc_roll_gain = B_v[3, 2]    # tx per unit v2 (positive)
        self._vc_yaw_gain = B_v[5, 3]     # tz per unit v3 (positive)

        # ---- Landing ----
        self.landing_speed = 1.0  # m/s
        self._com_surface_offset = (
            self.ball_radius + np.dot(self.com_offset, self.thrust_axis_body))

        # ---- Precompute thruster configs ----
        self._precompute_configs(cfg, thruster_array)

        # ---- PD gains (triple pole at p = 1/(3·tau_f)) ----
        p = 1.0 / (3.0 * tau_f)
        self.Kp_yaw = self.Izz * p**3 * tau_f
        self.Kd_yaw = 3 * self.Izz * p**2 * tau_f
        self.Kp_pr = self.Ixx * p**3 * tau_f
        self.Kd_pr = 3 * self.Ixx * p**2 * tau_f
        self.lag_margin = 2.15

        # ---- Attitude hold gains (for burns) ----
        self.Kp_att = 0.8
        self.Kd_att = 0.15

        # ---- IMU estimator ----
        q0 = np.array(cfg['initial_conditions']['quaternion'], dtype=float)
        p0 = np.array(cfg['initial_conditions']['position'], dtype=float)
        imu_pos_geom = np.array(cfg['sensors']['imu']['position'], dtype=float)
        imu_pos_from_com = imu_pos_geom - self.com_offset
        self.estimator = IncrementalEstimator(q0, p0, np.zeros(3),
                                              self.gravity, imu_pos_from_com)

        # ---- Distance sensors ----
        self.max_range = float(cfg['sensors']['distance_sensors']['max_range'])
        sensor_pos_geom = np.array(cfg['sensors']['distance_sensors']['positions'],
                                   dtype=float)
        self.sensor_pos_geom = sensor_pos_geom
        self.sensor_dirs_body = (sensor_pos_geom /
                                 np.linalg.norm(sensor_pos_geom, axis=1, keepdims=True))
        self._ds_period = 1.0 / float(cfg['sensors']['distance_sensors']['sample_rate'])

        # EKF correction
        wall_cfg = cfg['environment']['wall']
        self.wall_x = float(wall_cfg['center'][0])
        self.floor_z = 0.0
        self.ekf_gain = 0.25
        self.ekf_tol = 0.25

        # Hit-point accumulation
        self.hit_points: deque = deque()

        # Per-sensor surface detection
        n_sensors = len(cfg['sensors']['distance_sensors']['positions'])
        self._sensor_buffers = [[] for _ in range(n_sensors)]
        self._normal_lookback_near = 20   # T-20: samples back for near vector
        self._normal_lookback_far = 40    # T-40: samples back for far vector
        self._surface_library = []        # list of [normal, d_offset, count]
        self._surface_match_cos = 0.9     # normal similarity threshold for voting
        self._surface_count_threshold = 3  # min votes to consider surface "real"
        self.N_traj = 20

        # ---- State ----
        self._phase = self.INACTIVE
        self._last_dist_t = -1.0
        self._last_gyro = np.zeros(3)
        self._throw_end_p = None
        self._throw_end_v = None

        # Surface detection results
        self.target_normal = np.array([0.0, 0.0, 1.0])
        self.target_pos = np.array([0.0, 0.0, self._com_surface_offset])

        # Orient sub-state
        self._orient_sub = 'yaw'  # 'yaw' or 'tilt'
        self._orient_target_dir = None
        self._orient_tilt_axis = None  # 'x' or 'y'
        self._orient_tilt_sign = 1.0

        # Prefilter states for PD settling
        self._yaw_ref = 0.0
        self._tilt_ref = 0.0
        self._yaw_in_pd = False
        self._tilt_in_pd = False

        # Latch-and-integrate tracking for yaw/tilt maneuvers
        self._yaw_align_target = None    # total yaw angle to rotate (rad)
        self._yaw_align_progress = 0.0   # accumulated yaw rotation (rad)
        self._orient_yaw_target = None   # latched yaw target for orient phases
        self._orient_yaw_progress = 0.0
        self._orient_tilt_target = None  # latched tilt target (computed after yaw done)
        self._orient_tilt_progress = 0.0
        self._orient_tilt_axis = None    # 'x' or 'y'

        # Divert
        self._divert_dir = None
        self._divert_v_tan = None  # tangential velocity vector to kill
        self._divert_surface_normal = None  # surface used to compute divert

        # Final orient latched axis
        self._fo_latched_axis = None

        # Ceiling mode
        self._throw_is_vertical = False
        self._ceiling_mode = False

        # Per-step debug state (read by simulation logger)
        self._yaw_error = 0.0        # rad
        self._tilt_error = 0.0       # rad
        self._control_mode = -1      # -1=none, 0=bang, 1=pd
        self._orient_target_dir_log = np.full(3, np.nan)
        self._brake_align_deg = float('nan')
        self._first_surface_t = None       # timestamp of first surface detection
        self._first_qualified_t = None     # timestamp when a surface first reaches vote threshold

        # Logging
        self._log_hit_points = []
        self._log_detections = []

    # ---------------------------------------------------------------
    #  Thruster config precomputation
    # ---------------------------------------------------------------

    def _precompute_configs(self, cfg, thruster_array):
        """Precompute force/deflection configs for all maneuver types."""
        from scripts.bang_bang_timing import (
            parse_thrusters, compute_wrench,
            find_max_yaw_config, find_max_yaw_zero_z,
        )
        thrusters = parse_thrusters(cfg)

        # ---- Yaw max torque ----
        f1, d1, w1 = find_max_yaw_config(thrusters)
        f2 = [f1[1], f1[0], f1[3], f1[2]]
        w2 = compute_wrench(thrusters, f2, d1)
        if w1[5] < 0:
            f1, f2 = f2, f1
            w1, w2 = w2, w1
        self.yaw_max_fwd = np.array(f1)
        self.yaw_max_rev = np.array(f2)
        self.yaw_max_defls = np.array(d1)
        self.tau_yaw_max = abs(w1[5])

        # ---- Yaw zero-Z ----
        f1, d1, w1 = find_max_yaw_zero_z(thrusters)
        f2 = [f1[1], f1[0], f1[3], f1[2]]
        w2 = compute_wrench(thrusters, f2, d1)
        if w1[5] < 0:
            f1, f2 = f2, f1
            w1, w2 = w2, w1
        self.yaw_zz_fwd = np.array(f1)
        self.yaw_zz_rev = np.array(f2)
        self.yaw_zz_defls = np.array(d1)
        self.tau_yaw_zz = abs(w1[5])

        # ---- Roll (tau_x) max torque ----
        fm, fn = self.f_max, self.f_min
        fc = self.f_cap
        self.roll_max_fwd = np.array([fm, fn, fm, fn])
        self.roll_max_rev = np.array([fn, fm, fn, fm])
        w = compute_wrench(thrusters, list(self.roll_max_fwd), [0,0,0,0])
        self.tau_roll_max = abs(w[3])

        # ---- Roll zero-Z ----
        self.roll_zz_fwd = np.array([fc, -fc, fc, -fc])
        self.roll_zz_rev = np.array([-fc, fc, -fc, fc])
        w = compute_wrench(thrusters, list(self.roll_zz_fwd), [0,0,0,0])
        self.tau_roll_zz = abs(w[3])

        # ---- Pitch (tau_y) max torque ----
        self.pitch_max_fwd = np.array([fn, fn, fm, fm])
        self.pitch_max_rev = np.array([fm, fm, fn, fn])
        w = compute_wrench(thrusters, list(self.pitch_max_fwd), [0,0,0,0])
        self.tau_pitch_max = abs(w[4])

        # ---- Pitch zero-Z ----
        self.pitch_zz_fwd = np.array([-fc, -fc, fc, fc])
        self.pitch_zz_rev = np.array([fc, fc, -fc, -fc])
        w = compute_wrench(thrusters, list(self.pitch_zz_fwd), [0,0,0,0])
        self.tau_pitch_zz = abs(w[4])

    # ---------------------------------------------------------------
    #  Main callable
    # ---------------------------------------------------------------

    def __call__(self, t, _state, imu_data, dist_data):
        zero = np.zeros(self.n_thrusters)

        # Throw → active transition
        if self._phase == self.INACTIVE and t >= self.throw_duration:
            self._phase = self.YAW_DESPIN
            self._throw_end_p = self.estimator.p.copy()
            self._throw_end_v = self.estimator.v.copy()
            v_dir = self._throw_end_v / (np.linalg.norm(self._throw_end_v) + 1e-10)
            self._throw_is_vertical = v_dir[2] > np.cos(np.radians(20))

        # Update estimator
        if imu_data is not None:
            accel, gyro = imu_data
            self.estimator.update(t, accel, gyro)
            self._last_gyro = gyro

        if self._phase == self.INACTIVE:
            return zero

        # Current state
        p = self.estimator.p
        v = self.estimator.v
        R = self.estimator.R
        omega = self._last_gyro

        # Distance processing
        if dist_data is not None and t - self._last_dist_t >= self._ds_period - 1e-9:
            self._process_distance(t, p, R, dist_data)
            self._last_dist_t = t

        # ---- Dispatch ----
        if self._phase == self.YAW_DESPIN:
            return self._do_yaw_despin(omega)

        if self._phase == self.TILT_DESPIN:
            return self._do_tilt_despin(R, omega)

        if self._phase == self.ORIENT_BURN:
            return self._do_orient(R, omega, p, v, zero_z=True,
                                   next_phase=self.BURN)

        if self._phase == self.BURN:
            return self._do_burn(R, omega, p, v)

        if self._phase == self.FINAL_ORIENT:
            return self._do_final_orient(R, omega)

        if self._phase == self.CEIL_ORI_BOOST:
            return self._do_ceil_ori_boost(R, omega, p, v)

        if self._phase == self.CEIL_BOOST:
            return self._do_ceil_boost(R, omega, p, v)

        if self._phase == self.CEIL_ORI_LAND:
            return self._do_ceil_ori_land(R, omega, p, v)

        if self._phase == self.CEIL_APPROACH:
            return self._do_ceil_approach(R, omega, p, v)

        return zero

    # ---------------------------------------------------------------
    #  YAW DESPIN
    # ---------------------------------------------------------------

    def _do_yaw_despin(self, omega):
        """Pure yaw rate kill — just drive omega_z to zero."""
        ax = self.thrust_axis_body
        omega_z = np.dot(omega, ax)

        # Bang-bang: oppose the yaw rate
        tau_max = self.tau_yaw_max
        if abs(omega_z) > np.radians(1.0):
            tau_cmd = -tau_max if omega_z > 0 else tau_max
            mode = 'bang'
        else:
            # PD settling near zero rate
            tau_cmd = np.clip(-self.Kd_yaw * omega_z, -tau_max, tau_max)
            mode = 'pd'

        # Debug logging
        self._yaw_error = 0.0  # no angle target in this phase
        self._tilt_error = 0.0
        self._control_mode = 0 if mode == 'bang' else 1

        # Check settled: yaw rate near zero
        if abs(omega_z) < np.radians(0.5):
            self._phase = self.TILT_DESPIN
            self._yaw_in_pd = False
            self._zero_vectoring()
            return np.zeros(self.n_thrusters)

        return self._alloc_yaw(tau_cmd, self.yaw_max_fwd, self.yaw_max_rev,
                               self.yaw_max_defls, tau_max)

    # ---------------------------------------------------------------
    #  YAW ALIGN
    # ---------------------------------------------------------------

    def _do_yaw_align(self, R, omega):
        """Yaw to align the tilt rate axis with a principal body axis.

        The target yaw angle is latched at entry and progress is tracked
        by integrating omega_z.  This is necessary because omega_perp
        components are constant in the body frame (Ixx = Iyy → no
        gyroscopic coupling), so recomputing the error from atan2(wy,wx)
        would always give the same value regardless of how much the
        ball has actually yawed.
        """
        ax = self.thrust_axis_body
        omega_z = np.dot(omega, ax)
        omega_perp = omega - omega_z * ax
        omega_perp_mag = np.linalg.norm(omega_perp)

        # If tilt rate is negligible, skip alignment
        if omega_perp_mag < np.radians(1.0):
            self._phase = self.TILT_DESPIN
            self._yaw_in_pd = False
            self._yaw_align_target = None
            self._zero_vectoring()
            return np.zeros(self.n_thrusters)

        # Latch target on first entry
        if self._yaw_align_target is None:
            phi = np.arctan2(omega_perp[1], omega_perp[0])
            phi_target = np.round(phi / (np.pi / 2)) * (np.pi / 2)
            self._yaw_align_target = phi - phi_target
            # Wrap to [-pi, pi]
            self._yaw_align_target = (
                (self._yaw_align_target + np.pi) % (2 * np.pi) - np.pi)
            self._yaw_align_progress = 0.0

        # Integrate progress (omega_z is the yaw rate)
        dt = 1.0 / 8000  # simulation rate
        self._yaw_align_progress += omega_z * dt

        # Error = how much yaw remains
        # Target is to yaw by _yaw_align_target total.
        # We've yawed _yaw_align_progress so far.
        # The ball needs to yaw in the NEGATIVE direction by _yaw_align_target
        # (yawing the body moves phi in the opposite direction).
        remaining = -self._yaw_align_target - self._yaw_align_progress

        # Phase-plane switching
        alpha_max = self.tau_yaw_max / self.Izz
        tau_cmd, mode = phase_plane_torque(
            remaining, omega_z, alpha_max, self.tau_yaw_max,
            self.tau_f, self.Kp_yaw, self.Kd_yaw, self.lag_margin)

        if mode == 'pd':
            if not self._yaw_in_pd:
                self._yaw_in_pd = True
                self._yaw_ref = 0.0
            ref_rate = self.Kp_yaw / self.Kd_yaw
            self._yaw_ref += (remaining - self._yaw_ref) * ref_rate * dt
            tau_cmd = np.clip(
                -self.Kp_yaw * (0 - self._yaw_ref) - self.Kd_yaw * omega_z,
                -self.tau_yaw_max, self.tau_yaw_max)

        # Debug logging
        self._yaw_error = remaining
        self._tilt_error = 0.0
        self._control_mode = 0 if mode == 'bang' else 1

        # Check settled (loose — tilt despin handles any residual)
        if is_settled(remaining, omega_z, tol_angle=2.0, tol_rate=5.0):
            self._phase = self.TILT_DESPIN
            self._yaw_in_pd = False
            self._yaw_align_target = None
            self._zero_vectoring()
            return np.zeros(self.n_thrusters)

        return self._alloc_yaw(tau_cmd, self.yaw_max_fwd, self.yaw_max_rev,
                               self.yaw_max_defls, self.tau_yaw_max)

    # ---------------------------------------------------------------
    #  TILT DESPIN
    # ---------------------------------------------------------------

    def _do_tilt_despin(self, R, omega):
        """Kill remaining roll/pitch angular rate.

        Handles arbitrary omega_perp directions by blending roll/pitch configs.
        """
        ax = self.thrust_axis_body
        omega_perp = omega - np.dot(omega, ax) * ax
        omega_perp_mag = np.linalg.norm(omega_perp)

        if omega_perp_mag < np.radians(0.5):
            # Settled — transition depends on ceiling detection
            if self._throw_is_vertical and self._has_ceiling_surface():
                self._phase = self.CEIL_ORI_BOOST
                self._ceiling_mode = True
            else:
                self._phase = self.ORIENT_BURN
            self._tilt_in_pd = False
            self._control_mode = -1
            return np.zeros(self.n_thrusters)

        # Wait for vectoring servos to settle near zero before applying forces
        # (YAW_DESPIN leaves servos at ~20° deflection)
        max_defl = max(abs(thr.actual_deflection) for thr in self.thruster_array.thrusters
                       if thr.vectoring_enabled)
        if max_defl > np.radians(2.0):
            self._zero_vectoring()
            self.thruster_array.set_commands(np.zeros(self.n_thrusters))
            return np.zeros(self.n_thrusters)

        # Despin both roll and pitch simultaneously using precomputed configs.
        # Each config is yaw-neutral by construction.
        ox, oy = omega_perp[0], omega_perp[1]

        # Roll (tau_x) component: oppose omega_x
        if abs(ox) > 1e-6:
            roll_sign = -1.0 if ox > 0 else 1.0
            roll_cmds = roll_sign * self.roll_max_fwd if roll_sign > 0 \
                else (-roll_sign) * self.roll_max_rev
            roll_frac = abs(ox) / omega_perp_mag
        else:
            roll_cmds = np.zeros(self.n_thrusters)
            roll_frac = 0.0

        # Pitch (tau_y) component: oppose omega_y
        if abs(oy) > 1e-6:
            pitch_sign = -1.0 if oy > 0 else 1.0
            pitch_cmds = pitch_sign * self.pitch_max_fwd if pitch_sign > 0 \
                else (-pitch_sign) * self.pitch_max_rev
            pitch_frac = abs(oy) / omega_perp_mag
        else:
            pitch_cmds = np.zeros(self.n_thrusters)
            pitch_frac = 0.0

        # Blend by fraction and clamp
        raw_commands = roll_frac * roll_cmds + pitch_frac * pitch_cmds
        raw_commands = np.clip(raw_commands, self.f_min, self.f_max)

        self._tilt_error = 0.0
        self._yaw_error = 0.0
        self._control_mode = 0  # bang

        self._zero_vectoring()
        self.thruster_array.set_commands(raw_commands)
        return raw_commands

    # ---------------------------------------------------------------
    #  ORIENT (shared by orient_divert, orient_brake, final_orient)
    # ---------------------------------------------------------------

    def _do_orient(self, R, omega, p, v, zero_z, next_phase):
        """Direct axis-angle tilt toward target direction using pseudoinverse."""

        # Compute or recompute target direction
        need_recompute = self._orient_target_dir is None
        if (self._phase == self.ORIENT_BURN
                and self._orient_target_dir is not None
                and self._divert_surface_normal is not None
                and np.dot(self._divert_surface_normal, self.target_normal) < 0.99):
            need_recompute = True

        if need_recompute:
            if self._phase == self.ORIENT_BURN:
                self._setup_orient_burn(p, v)

        desired_dir = self._orient_target_dir
        if desired_dir is None:
            if next_phase is not None:
                self._phase = next_phase
            return np.zeros(self.n_thrusters)

        self._orient_target_dir_log = desired_dir.copy()

        ax = self.thrust_axis_body
        omega_z = np.dot(omega, ax)
        omega_perp = omega - omega_z * ax

        # Axis-angle from thrust axis to desired direction
        thrust_w = R @ ax
        cross = np.cross(thrust_w, desired_dir)
        sin_th = np.linalg.norm(cross)
        cos_th = float(np.clip(np.dot(thrust_w, desired_dir), -1.0, 1.0))
        angle = np.arctan2(sin_th, cos_th)

        # If already well-aligned, skip orient
        if angle < np.radians(5.0) and np.linalg.norm(omega_perp) < np.radians(10.0):
            self._orient_target_dir = None
            if next_phase is not None:
                self._phase = next_phase
            return np.zeros(self.n_thrusters)

        # Completion check
        if angle < np.radians(2.0):
            self._orient_target_dir = None
            if next_phase is not None:
                self._phase = next_phase
            return np.zeros(self.n_thrusters)

        # Rotation axis in body frame
        if sin_th > 1e-6:
            rot_axis_body = R.T @ (cross / sin_th)
        else:
            rot_axis_body = np.array([1.0, 0.0, 0.0])
        # Project out thrust axis (tilt only)
        rot_axis_body -= np.dot(rot_axis_body, ax) * ax
        rot_norm = np.linalg.norm(rot_axis_body)
        if rot_norm < 1e-6:
            return np.zeros(self.n_thrusters)
        rot_axis_body /= rot_norm

        omega_tilt = np.dot(omega, rot_axis_body)

        tau_max = self.tau_roll_zz if zero_z else self.tau_roll_max
        alpha_max = tau_max / self.Ixx

        # Body-frame PD below 20° to avoid axis-angle singularity
        if angle < np.radians(20.0):
            err_body = R.T @ (angle * cross / sin_th) if sin_th > 1e-6 else np.zeros(3)
            err_body -= np.dot(err_body, ax) * ax
            tau_body = 30.0 * err_body - 2.5 * omega_perp
            tau_mag = np.linalg.norm(tau_body)
            if tau_mag > tau_max:
                tau_body *= tau_max / tau_mag
            self._control_mode = 1
        else:
            tau_cmd, mode = phase_plane_torque(
                angle, omega_tilt, alpha_max, tau_max,
                self.tau_f, self.Kp_pr, self.Kd_pr, self.lag_margin)
            tau_body = tau_cmd * rot_axis_body
            self._control_mode = 0 if mode == 'bang' else 1

        self._yaw_error = 0.0
        self._tilt_error = angle

        # Allocate via torque-only pseudoinverse
        raw_commands = self._B_tau_pinv @ tau_body
        if zero_z:
            raw_commands -= np.mean(raw_commands)
        fc = self.f_cap if zero_z else self.f_max
        f_lo = -self.f_cap if zero_z else self.f_min
        max_abs = np.abs(raw_commands).max()
        if max_abs > fc and max_abs > 1e-10:
            raw_commands *= fc / max_abs
        raw_commands = np.clip(raw_commands, f_lo, fc)
        self._zero_vectoring()
        self.thruster_array.set_commands(raw_commands)
        return raw_commands

    def _decompose_rotation(self, R, desired_dir):
        """Decompose rotation from current to desired into yaw + tilt."""
        d_body = R.T @ desired_dir
        tilt_angle = np.arccos(np.clip(d_body[2], -1.0, 1.0))

        if tilt_angle < np.radians(2.0):
            return 0.0, 0.0, 'x'

        # Angle in body xy-plane
        phi = np.arctan2(d_body[1], d_body[0])

        # Yaw to nearest principal axis (multiples of 90°)
        phi_nearest = np.round(phi / (np.pi / 2)) * (np.pi / 2)
        yaw_error = phi - phi_nearest
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi

        # After yaw, determine tilt axis from which quadrant
        # Simulate the yaw rotation
        c, s = np.cos(-yaw_error), np.sin(-yaw_error)
        d_after = np.array([c * d_body[0] + s * d_body[1],
                            -s * d_body[0] + c * d_body[1],
                            d_body[2]])

        # Tilt axis = cross(z_hat, d_after)
        tilt_ax = np.cross([0, 0, 1], d_after)
        tilt_ax_mag = np.linalg.norm(tilt_ax)
        if tilt_ax_mag < 1e-6:
            return yaw_error, 0.0, 'x'
        tilt_ax /= tilt_ax_mag

        # Determine roll vs pitch and sign
        if abs(tilt_ax[0]) > abs(tilt_ax[1]):
            axis = 'x'
            tilt_sign = np.sign(tilt_ax[0])
        else:
            axis = 'y'
            tilt_sign = np.sign(tilt_ax[1])

        return yaw_error, tilt_angle * tilt_sign, axis

    # ---------------------------------------------------------------
    #  DIVERT
    # ---------------------------------------------------------------

    def _compute_final_orient_time(self, n):
        """Compute the rotation time and angle for final orient to surface normal.

        The rotation angle is between the vertical (thrust fighting gravity)
        and the surface normal.  The time comes from the bang-bang formula
        for zero-Z pitch/roll.
        """
        # Rotation angle: vertical [0,0,1] to surface normal n
        cos_angle = np.clip(n[2], -1.0, 1.0)  # dot([0,0,1], n)
        angle = np.arccos(cos_angle)

        if angle < np.radians(2.0):
            return 0.0, angle  # nearly aligned, no rotation needed

        # Bang-bang time: T ≈ 2 * sqrt(theta * I / tau_max)
        # Use zero-Z pitch/roll torque (final orient uses zero-Z)
        tau_max = self.tau_roll_zz
        T = 2.0 * np.sqrt(angle * self.Ixx / tau_max)
        # Add actuator lag overhead (~4ms)
        T += self.tau_f * (2 + np.log(2))

        return T, angle

    def _setup_orient_burn(self, p, v):
        """Compute the delta-v direction for the combined burn.

        The burn target is NOT the surface itself, but a standoff point
        where the ball begins its final 90-degree rotation.  During the
        rotation (free-fall), gravity cancels the upward velocity so the
        ball arrives at the surface with zero tangential velocity.

        v_ideal at standoff = -v_land * n - g_vec * t_rotate
        d_standoff = v_land * t_rotate + 0.5 * dot(g, n) * t_rotate^2
        """
        n = self.target_normal
        self._divert_surface_normal = n.copy()

        # Compute final orient rotation time from ball capabilities
        t_rotate, rot_angle = self._compute_final_orient_time(n)
        self._final_orient_time = t_rotate

        # Standoff distance and ideal velocity
        g_n = np.dot(self.g_vec, n)
        d_standoff = self.landing_speed * t_rotate + 0.5 * g_n * t_rotate**2

        v_ideal = -self.landing_speed * n - self.g_vec * t_rotate

        # Predict ballistic velocity at the standoff point
        # Shift target_pos outward by d_standoff along normal
        target_pos_standoff = self.target_pos + d_standoff * n

        # Predict time to reach standoff point
        a_c = 0.5 * np.dot(self.g_vec, n)
        b_c = np.dot(v, n)
        c_c = np.dot(p - target_pos_standoff, n)

        disc = b_c**2 - 4 * a_c * c_c
        if disc < 0:
            self._orient_target_dir = None
            return

        sq = np.sqrt(disc)
        if abs(a_c) > 1e-10:
            roots = [(-b_c + sq) / (2 * a_c), (-b_c - sq) / (2 * a_c)]
        elif abs(b_c) > 1e-10:
            roots = [-c_c / b_c]
        else:
            self._orient_target_dir = None
            return

        # Only future intersections
        tau_elapsed = 0.0  # p, v are current state
        pos_roots = [r for r in roots if r > 0.01]
        if not pos_roots:
            self._orient_target_dir = None
            return

        t_to_standoff = min(pos_roots)
        v_at_standoff = v + self.g_vec * t_to_standoff
        dv = v_ideal - v_at_standoff
        dv_mag = np.linalg.norm(dv)
        if dv_mag < 0.01:
            self._orient_target_dir = None
            return

        self._orient_target_dir = dv / dv_mag
        self._orient_sub = 'yaw'

    def _do_burn(self, R, omega, p, v):
        """Combined burn: recompute dv each step toward the standoff state.

        The target is the standoff point (d_standoff from the surface)
        with velocity v_ideal = -v_land * n - g * t_rotate.  From there,
        the ball rotates 90 degrees during free-fall and gravity cancels
        the upward velocity.
        """
        n = self.target_normal

        # Recompute final orient time (surface might have changed)
        t_rotate, _ = self._compute_final_orient_time(n)

        g_n = np.dot(self.g_vec, n)
        d_standoff = self.landing_speed * t_rotate + 0.5 * g_n * t_rotate**2

        v_ideal = -self.landing_speed * n - self.g_vec * t_rotate
        target_pos_standoff = self.target_pos + d_standoff * n

        # Check distance to standoff
        dist_to_standoff = np.dot(p - target_pos_standoff, n)
        if dist_to_standoff <= 0:
            # At or past the standoff point — begin final orient
            self._phase = self.FINAL_ORIENT
            self._orient_target_dir = n
            self._orient_sub = 'yaw'
            self._fo_latched_axis = None
            return np.zeros(self.n_thrusters)

        # If moving away from standoff, coast — thrusting is counterproductive
        v_approach = -np.dot(v, n)
        if v_approach < 0:
            self._yaw_error = 0.0
            self._tilt_error = 0.0
            self._control_mode = -1
            return np.zeros(self.n_thrusters)

        # Predict time to standoff from current state
        a_c = 0.5 * np.dot(self.g_vec, n)
        b_c = np.dot(v, n)
        c_c = np.dot(p - target_pos_standoff, n)

        disc = b_c**2 - 4 * a_c * c_c
        t_remaining = None
        if disc >= 0:
            sq = np.sqrt(disc)
            if abs(a_c) > 1e-10:
                roots = [(-b_c + sq) / (2 * a_c), (-b_c - sq) / (2 * a_c)]
            elif abs(b_c) > 1e-10:
                roots = [-c_c / b_c]
            else:
                roots = []
            pos_roots = [r for r in roots if 0.001 < r < 30.0]
            if pos_roots:
                t_remaining = min(pos_roots)

        if t_remaining is None:
            # Can't reach standoff — transition to final orient now
            self._phase = self.FINAL_ORIENT
            self._orient_target_dir = n
            self._orient_sub = 'yaw'
            self._fo_latched_axis = None
            return np.zeros(self.n_thrusters)

        v_at_standoff = v + self.g_vec * t_remaining
        dv_remaining = v_ideal - v_at_standoff
        dv_mag = np.linalg.norm(dv_remaining)

        if dv_mag < 0.1:
            self._phase = self.FINAL_ORIENT
            self._orient_target_dir = n
            self._orient_sub = 'yaw'
            self._fo_latched_axis = None
            return np.zeros(self.n_thrusters)

        burn_dir = dv_remaining / dv_mag

        # Coast-then-full-thrust: compute how long the burn takes at max
        # thrust, and coast until t_to_standoff equals that time.
        t_burn = self.mass * dv_mag / self.F_total
        if t_remaining > t_burn + 0.02:
            # Coast: hold attitude toward burn direction with zero-Z torque
            ax = self.thrust_axis_body
            current = R @ ax
            cross = np.cross(current, burn_dir)
            sin_th = np.linalg.norm(cross)
            cos_th = float(np.clip(np.dot(current, burn_dir), -1, 1))
            angle_att = np.arctan2(sin_th, cos_th)
            if sin_th > 1e-6:
                err_body = R.T @ (angle_att * cross / sin_th)
            else:
                err_body = np.zeros(3)
            err_body -= np.dot(err_body, ax) * ax
            omega_perp = omega - np.dot(omega, ax) * ax
            tau_body = 8.0 * err_body - 0.4 * omega_perp
            raw = self._B_tau_pinv @ tau_body
            raw -= np.mean(raw)
            fc = self.f_cap
            max_abs = np.abs(raw).max()
            if max_abs > fc and max_abs > 1e-10:
                raw *= fc / max_abs
            raw = np.clip(raw, -fc, fc)
            self._zero_vectoring()
            self.thruster_array.set_commands(raw)
            self._yaw_error = dv_mag
            self._tilt_error = 0.0
            self._control_mode = -1
            return raw

        # Full thrust burn
        F_cmd = self.F_total

        # Debug logging
        thrust_w = R @ self.thrust_axis_body
        cos_align = np.dot(thrust_w, burn_dir)
        self._brake_align_deg = float(np.degrees(np.arccos(np.clip(cos_align, -1, 1))))
        self._control_mode = -1
        self._orient_target_dir_log = burn_dir.copy()
        # Repurpose yaw/tilt error for burn diagnostics:
        # yaw_error = |dv_remaining| (m/s), tilt_error = F_cmd / F_total (thrust fraction)
        self._yaw_error = dv_mag
        self._tilt_error = F_cmd / self.F_total

        # Thrust along thrust axis + aggressive PD attitude torque
        # Higher Kp reduces tracking lag as the burn direction rotates
        F_body = F_cmd * self.thrust_axis_body
        current = R @ self.thrust_axis_body
        cross = np.cross(current, burn_dir)
        sin_th = np.linalg.norm(cross)
        cos_th = float(np.clip(np.dot(current, burn_dir), -1, 1))
        angle = np.arctan2(sin_th, cos_th)
        if sin_th > 1e-6:
            err_body = R.T @ (angle * cross / sin_th)
        else:
            err_body = np.zeros(3)
        ax = self.thrust_axis_body
        omega_perp = omega - np.dot(omega, ax) * ax
        tau = 40.0 * err_body - 2.0 * omega_perp
        tau -= np.dot(tau, ax) * ax
        wrench = np.concatenate([F_body, tau])
        return self.thruster_array.wrench_to_commands(wrench)

    # ---------------------------------------------------------------
    #  FINAL ORIENT (yaw despin then direct tilt)
    # ---------------------------------------------------------------

    def _do_final_orient(self, R, omega):
        """Final orient: kill yaw rate, then tilt directly to surface normal.

        Unlike _do_orient (sequential yaw align + tilt), this skips the
        yaw alignment and tilts directly along the shortest rotation path.
        This avoids the yaw phase that was counterproductive (spending time
        while unchecked tilt rate degraded alignment).

        Phase 1: Zero-Z yaw despin (kill omega_z)
        Phase 2: Direct tilt via torque-only pseudoinverse toward surface normal
        """
        ax = self.thrust_axis_body
        omega_z = np.dot(omega, ax)
        n = self.target_normal

        # Phase 1: Kill yaw rate first (fast — typically <10ms)
        if abs(omega_z) > np.radians(5.0):
            tau_max = self.tau_yaw_zz
            tau_cmd = -tau_max if omega_z > 0 else tau_max
            self._yaw_error = 0.0
            self._tilt_error = 0.0
            self._control_mode = 0
            return self._alloc_yaw(tau_cmd, self.yaw_zz_fwd, self.yaw_zz_rev,
                                   self.yaw_zz_defls, tau_max)

        # Phase 2: Direct tilt toward surface normal
        # Compute axis-angle rotation from thrust axis to surface normal
        thrust_w = R @ ax
        cross = np.cross(thrust_w, n)
        sin_th = np.linalg.norm(cross)
        cos_th = float(np.clip(np.dot(thrust_w, n), -1.0, 1.0))
        angle = np.arctan2(sin_th, cos_th)

        # Debug
        self._brake_align_deg = np.degrees(angle)
        self._orient_target_dir_log = n.copy()

        if angle < np.radians(2.0):
            # Aligned — done
            self._yaw_error = 0.0
            self._tilt_error = 0.0
            return np.zeros(self.n_thrusters)

        # Rotation axis in body frame
        if sin_th > 1e-6:
            rot_axis_world = cross / sin_th
            rot_axis_body = R.T @ rot_axis_world
        else:
            rot_axis_body = np.array([1.0, 0.0, 0.0])
        # Project out thrust axis (tilt only)
        rot_axis_body -= np.dot(rot_axis_body, ax) * ax
        rot_norm = np.linalg.norm(rot_axis_body)
        if rot_norm < 1e-6:
            return np.zeros(self.n_thrusters)
        rot_axis_body /= rot_norm

        omega_tilt = np.dot(omega, rot_axis_body)
        omega_perp = omega - np.dot(omega, ax) * ax

        # Estimate time to wall vs time to complete rotation
        v_est = self.estimator.v
        p_est = self.estimator.p
        v_approach = -np.dot(v_est, self.target_normal)
        dist_to_wall = np.dot(p_est - self.target_pos, self.target_normal)
        t_to_wall = dist_to_wall / max(v_approach, 0.1)

        stop_time = abs(omega_tilt) / (self.tau_roll_zz / self.Ixx)
        remaining_after_stop = angle + omega_tilt**2 / (2 * self.tau_roll_zz / self.Ixx) \
            if omega_tilt < 0 else max(angle - omega_tilt**2 / (2 * self.tau_roll_zz / self.Ixx), 0)
        rotate_time = 2.0 * np.sqrt(max(remaining_after_stop, 0) * self.Ixx / self.tau_roll_zz)
        t_orient_est = stop_time + rotate_time

        # Emergency: use max torque if zero-Z won't finish in time
        if t_orient_est > t_to_wall * 0.9:
            tau_max = self.tau_roll_max
        else:
            tau_max = self.tau_roll_zz

        alpha_max = tau_max / self.Ixx

        # Below 20°: body-frame PD (axis-angle singularity avoidance).
        # Pure damping dominates at high rate; position term takes over
        # as rate dies. This avoids the overshoot caused by the rotation
        # axis flipping sign near zero angle.
        if angle < np.radians(20.0):
            err_body = R.T @ (angle * cross / sin_th) if sin_th > 1e-6 else np.zeros(3)
            err_body -= np.dot(err_body, ax) * ax
            Kp_fo = 30.0
            Kd_fo = 2.5
            tau_body = Kp_fo * err_body - Kd_fo * omega_perp
            tau_mag = np.linalg.norm(tau_body)
            if tau_mag > tau_max:
                tau_body *= tau_max / tau_mag
            self._control_mode = 1
        else:
            tau_cmd, mode = phase_plane_torque(
                angle, omega_tilt, alpha_max, tau_max,
                self.tau_f, self.Kp_pr, self.Kd_pr, self.lag_margin)
            tau_body = tau_cmd * rot_axis_body
            self._control_mode = 0 if mode == 'bang' else 1

        self._yaw_error = 0.0
        self._tilt_error = angle

        # Allocate via torque-only pseudoinverse
        emergency = (tau_max == self.tau_roll_max)
        raw_commands = self._B_tau_pinv @ tau_body
        if not emergency:
            raw_commands -= np.mean(raw_commands)  # zero-Z

        # Scale to fit within actuator limits
        if emergency:
            f_limit = self.f_max
            f_lo = self.f_min
        else:
            f_limit = self.f_cap
            f_lo = -self.f_cap
        max_abs = np.abs(raw_commands).max()
        if max_abs > f_limit and max_abs > 1e-10:
            raw_commands *= f_limit / max_abs
        raw_commands = np.clip(raw_commands, f_lo, f_limit)
        self._zero_vectoring()
        self.thruster_array.set_commands(raw_commands)
        return raw_commands

    # ---------------------------------------------------------------
    #  Ceiling landing
    # ---------------------------------------------------------------

    def _has_ceiling_surface(self):
        """Check if the surface library contains a ceiling with enough votes."""
        min_count = self._surface_count_threshold
        if self._surface_library:
            max_count = max(c for _, _, c in self._surface_library)
            min_count = max(min_count, int(max_count * 0.1))
        for normal, d_plane, count in self._surface_library:
            if count >= min_count and normal[2] < -0.7:
                return True
        return False

    def _get_ceiling_surface(self):
        """Return the best ceiling surface (normal, position) from the library."""
        best = None
        best_count = 0
        for normal, d_plane, count in self._surface_library:
            if normal[2] < -0.7 and count > best_count:
                best = (normal, d_plane, count)
                best_count = count
        if best is None:
            return None, None
        normal, d_plane, _ = best
        ceiling_z = d_plane / normal[2]
        pos = np.array([0.0, 0.0, ceiling_z])
        return normal, pos + self._com_surface_offset * normal

    def _do_ceil_ori_boost(self, R, omega, p, v):
        """Zero-Z orient upward (toward ceiling).

        Uses _do_orient to align thrust axis with ceiling_dir.
        Transitions to CEIL_BOOST once within 10°.
        """
        if self.target_normal is None:
            n, tpos = self._get_ceiling_surface()
            if n is not None:
                self.target_normal = n
                self.target_pos = tpos

        ceiling_dir = -self.target_normal
        self._orient_target_dir = ceiling_dir

        # Early transition once within 10°
        thrust_w = R @ self.thrust_axis_body
        angle = np.arccos(np.clip(np.dot(thrust_w, ceiling_dir), -1, 1))
        if angle < np.radians(10.0):
            self._orient_target_dir = None
            self._phase = self.CEIL_BOOST
            return np.zeros(self.n_thrusters)

        return self._do_orient(R, omega, p, v, zero_z=True,
                               next_phase=self.CEIL_BOOST)

    def _do_ceil_boost(self, R, omega, p, v):
        """Brake or boost so ballistic arrival at ceiling = 0.1 m/s.

        Ball is oriented upward. Uses positive thrust (upward = toward
        ceiling) to boost, or reverse thrust to brake. The suicide-burn
        approach: coast when possible, full thrust when needed.

        Transitions to CEIL_ORI_TAN when predicted arrival speed is
        within tolerance of 0.1 m/s.
        """
        ceiling_dir = -self.target_normal
        v_toward = np.dot(v, ceiling_dir)  # positive = toward ceiling
        dist = np.dot(self.target_pos - p, ceiling_dir)

        # Target: arrive at ceiling with v_arrival = 0.1 m/s
        v_target = 0.1
        g_toward = np.dot(self.g_vec, ceiling_dir)  # negative (gravity opposes)

        # Predict ballistic arrival speed at ceiling
        disc = v_toward**2 + 2 * g_toward * dist
        if disc >= 0 and v_toward > 0:
            v_arrival = np.sqrt(disc)
            reaches = True
        else:
            v_arrival = 0.0
            reaches = False

        # Transition when arrival speed is close to target
        if reaches and abs(v_arrival - v_target) < 0.15 and dist > 0:
            self._orient_target_dir = None
            self._phase = self.CEIL_ORI_LAND
            return np.zeros(self.n_thrusters)

        # Tangential velocity
        v_tan = v - v_toward * ceiling_dir
        v_tan_mag = np.linalg.norm(v_tan)

        # Thrust direction: 90% ceiling, 10% anti-tangential
        desired_dir = ceiling_dir.copy()
        if v_tan_mag > 0.01:
            # tan_weight such that cos(angle) = 0.9 → sin = 0.436
            desired_dir = 0.9 * ceiling_dir - 0.436 * (v_tan / v_tan_mag)
            desired_dir /= np.linalg.norm(desired_dir)

        # Determine thrust magnitude
        if not reaches or v_arrival < v_target:
            F_body_z = self.F_total
        elif v_arrival > v_target:
            F_body_z = self.f_min * self.n_thrusters
        else:
            F_body_z = 0.0

        tau = self._attitude_torque(R, omega, desired_dir)
        F_body = np.array([0.0, 0.0, F_body_z])
        wrench = np.concatenate([F_body, tau])
        commands = self.thruster_array.wrench_to_commands(wrench)

        self._yaw_error = v_arrival
        self._tilt_error = v_toward
        self._control_mode = -1
        self._orient_target_dir_log = ceiling_dir.copy()
        self._brake_align_deg = float(dist)

        return commands

    def _do_ceil_ori_land(self, R, omega, p, v):
        """Zero-Z orient to inverted (thrust axis → -ceiling_dir).

        Uses _do_orient (phase-plane controller) for proper deceleration.
        Transitions to CEIL_APPROACH on completion.
        """
        ceiling_dir = -self.target_normal
        desired_dir = -ceiling_dir  # thrust axis away from ceiling

        self._orient_target_dir = desired_dir

        return self._do_orient(R, omega, p, v, zero_z=True,
                               next_phase=self.CEIL_APPROACH)

    def _do_ceil_approach(self, R, omega, p, v):
        """Inverted velocity controller: approach ceiling at landing_speed.

        The ball is upside down: body +Z ≈ -ceiling_dir.
        Positive thrust → away from ceiling (brake).
        Reverse thrust → toward ceiling (boost).
        """
        n = self.target_normal
        ceiling_dir = -n

        v_toward = np.dot(v, ceiling_dir)
        v_tan = v - v_toward * ceiling_dir
        v_tan_mag = np.linalg.norm(v_tan)

        # Desired velocity: approach at landing speed, zero tangential
        v_desired = self.landing_speed * ceiling_dir
        v_err = v - v_desired

        # PD velocity control + gravity compensation
        Kp_v = 10.0
        F_desired = -Kp_v * self.mass * v_err - self.mass * self.g_vec

        # Decompose into ceiling-normal force
        F_toward = np.dot(F_desired, ceiling_dir)

        # Thrust axis stays pointed away from ceiling, tilted for tangential
        desired_dir = -ceiling_dir.copy()
        if v_tan_mag > 0.05:
            if F_toward > 0:
                desired_dir += v_tan / max(v_tan_mag, 0.3) * min(v_tan_mag, 0.5)
            else:
                desired_dir -= v_tan / max(v_tan_mag, 0.3) * min(v_tan_mag, 0.5)
            desired_dir /= np.linalg.norm(desired_dir)

        # F_toward > 0 → need force toward ceiling → reverse thrust (F_body_z < 0)
        F_body_z = -F_toward
        F_max_pos = self.f_max * self.n_thrusters
        F_max_rev = abs(self.f_min) * self.n_thrusters
        F_body_z = np.clip(F_body_z, -F_max_rev, F_max_pos)

        tau = self._attitude_torque(R, omega, desired_dir)
        F_body = np.array([0.0, 0.0, F_body_z])
        wrench = np.concatenate([F_body, tau])
        commands = self.thruster_array.wrench_to_commands(wrench)

        self._yaw_error = v_toward
        self._tilt_error = v_tan_mag
        self._control_mode = -1
        self._orient_target_dir_log = desired_dir.copy()

        return commands

    # ---------------------------------------------------------------
    #  Force allocation
    # ---------------------------------------------------------------

    def _alloc_yaw(self, tau_cmd, f_fwd, f_rev, defls, tau_max):
        """Map yaw torque command to thruster forces + vectoring."""
        ratio = np.clip(tau_cmd / tau_max, -1.0, 1.0)
        if ratio >= 0:
            forces = ratio * f_fwd
        else:
            forces = (-ratio) * f_rev

        # Set vectoring
        for i, thr in enumerate(self.thruster_array.thrusters):
            if thr.vectoring_enabled:
                thr.set_vector_command(defls[i])

        self.thruster_array.set_commands(forces)
        return forces

    def _alloc_tilt(self, tau_cmd, f_fwd, f_rev, tau_max):
        """Map pitch/roll torque command to thruster forces (no vectoring)."""
        ratio = np.clip(tau_cmd / tau_max, -1.0, 1.0)
        if ratio >= 0:
            forces = ratio * f_fwd
        else:
            forces = (-ratio) * f_rev

        self._zero_vectoring()
        self.thruster_array.set_commands(forces)
        return forces

    def _zero_vectoring(self):
        for thr in self.thruster_array.thrusters:
            if thr.vectoring_enabled:
                thr.set_vector_command(0.0)

    # ---------------------------------------------------------------
    #  Attitude torque (PD for burn attitude hold)
    # ---------------------------------------------------------------

    def _attitude_torque(self, R, omega, desired_dir):
        """PD torque to align thrust axis with desired world direction."""
        current = R @ self.thrust_axis_body
        cross = np.cross(current, desired_dir)
        sin_th = np.linalg.norm(cross)
        cos_th = float(np.clip(np.dot(current, desired_dir), -1, 1))
        angle = np.arctan2(sin_th, cos_th)

        if sin_th > 1e-6:
            err_body = R.T @ (angle * cross / sin_th)
        else:
            err_body = np.zeros(3)

        ax = self.thrust_axis_body
        omega_perp = omega - np.dot(omega, ax) * ax
        tau = self.Kp_att * err_body - self.Kd_att * omega_perp
        tau -= np.dot(tau, ax) * ax
        return tau

    # ---------------------------------------------------------------
    #  Surface detection & estimation (from existing controller)
    # ---------------------------------------------------------------

    def _predict_surface_impact(self, p, v, n):
        a_c = 0.5 * np.dot(self.g_vec, n)
        b_c = np.dot(v, n)
        c_c = np.dot(p - self.target_pos, n)
        disc = b_c**2 - 4.0 * a_c * c_c
        if disc >= 0 and abs(a_c) > 1e-10:
            sq = np.sqrt(disc)
            roots = [(-b_c + sq) / (2 * a_c), (-b_c - sq) / (2 * a_c)]
            pos = [r for r in roots if r > 0.01]
            if pos:
                return min(pos)
        elif abs(b_c) > 1e-10:
            t = -c_c / b_c
            if t > 0.01:
                return t
        return None

    def _process_distance(self, t, p, R, dist_readings):
        """Process distance sensor readings: EKF correction + surface detection."""
        pos_geom = p - R @ self.com_offset
        lb_near = self._normal_lookback_near
        lb_far = self._normal_lookback_far

        for j, d in enumerate(dist_readings):
            if d >= self.max_range:
                continue
            ray_w = R @ self.sensor_dirs_body[j]
            sensor_w = pos_geom + R @ self.sensor_pos_geom[j]
            hit = sensor_w + d * ray_w
            self.hit_points.append(hit)
            self._log_hit_points.append((t, hit.copy()))

            # EKF position corrections
            if ray_w[2] < -0.5 and abs(hit[2] - self.floor_z) < self.ekf_tol:
                self.estimator._p[2] += self.ekf_gain * (self.floor_z - hit[2])
            if ray_w[0] > 0.5 and abs(hit[0] - self.wall_x) < self.ekf_tol:
                self.estimator._p[0] += self.ekf_gain * (self.wall_x - hit[0])

            # Add to per-sensor buffer
            buf = self._sensor_buffers[j]
            buf.append(hit.copy())

            # Need enough hits for the arc cross-product (T, T-near, T-far)
            if len(buf) <= lb_far:
                continue

            T = buf[-1]
            T_near = buf[-1 - lb_near]
            T_far = buf[-1 - lb_far]

            # Vectors from T to the earlier points (along the arc)
            v1 = T_near - T
            v2 = T_far - T

            # Cross product gives surface normal from arc curvature
            n = np.cross(v1, v2)
            n_len = np.linalg.norm(n)
            if n_len < 1e-10:
                continue  # degenerate (collinear arc)
            n = n / n_len

            # Orient normal toward the ball
            if np.dot(n, p - T) < 0:
                n = -n

            # Plane offset
            d_plane = np.dot(n, T)

            # Vote in surface library
            matched = False
            for k in range(len(self._surface_library)):
                lib_n, lib_d, lib_count = self._surface_library[k]
                if np.dot(n, lib_n) > self._surface_match_cos:
                    # Update with count-weighted running average
                    new_count = lib_count + 1
                    new_n = (lib_n * lib_count + n) / new_count
                    new_n = new_n / np.linalg.norm(new_n)
                    new_d = (lib_d * lib_count + d_plane) / new_count
                    self._surface_library[k] = [new_n, new_d, new_count]
                    # Log when a surface first reaches the vote threshold
                    if (lib_count < self._surface_count_threshold
                            and new_count >= self._surface_count_threshold
                            and self._first_qualified_t is None):
                        self._first_qualified_t = t
                    matched = True
                    break

            if not matched:
                self._surface_library.append([n.copy(), d_plane, 1])
                if self._first_surface_t is None:
                    self._first_surface_t = t
                self._log_detections.append((t, n.copy(), T.copy()))

        # Select target surface from library (locked during BURN/FINAL_ORIENT)
        if self._phase not in (self.BURN, self.FINAL_ORIENT,
                               self.CEIL_ORI_BOOST, self.CEIL_BOOST,
                               self.CEIL_ORI_LAND, self.CEIL_APPROACH):
            self._select_target_surface(t, p)

    def _select_target_surface(self, t, p):
        """Pick the target surface from the library using the ballistic trajectory.

        Only considers surfaces with enough votes (above count threshold).
        Picks the surface with the earliest trajectory intersection.
        """
        if not self._surface_library or self._throw_end_p is None:
            return

        p0 = self._throw_end_p
        v0 = self._throw_end_v

        # Relative threshold: need at least 10% of the best surface's votes
        max_count = max(c for _, _, c in self._surface_library)
        min_count = max(self._surface_count_threshold,
                        int(max_count * 0.1))

        best_t = float('inf')
        best_surface = None

        for normal, d_plane, count in self._surface_library:
            if count < min_count:
                continue
            # Solve: n · (p0 + v0*τ + 0.5*g*τ²) = d_plane
            a_c = 0.5 * np.dot(self.g_vec, normal)
            b_c = np.dot(v0, normal)
            c_c = np.dot(p0, normal) - d_plane

            # Find earliest positive intersection time
            disc = b_c**2 - 4 * a_c * c_c
            if disc < 0:
                continue
            sq = np.sqrt(disc)
            if abs(a_c) > 1e-10:
                roots = [(-b_c + sq) / (2 * a_c), (-b_c - sq) / (2 * a_c)]
            elif abs(b_c) > 1e-10:
                roots = [-c_c / b_c]
            else:
                continue
            # Only consider FUTURE intersections (τ > elapsed time since throw)
            tau_elapsed = t - self.throw_duration
            pos_roots = [r for r in roots if r > tau_elapsed + 0.01]
            if pos_roots and min(pos_roots) < best_t:
                best_t = min(pos_roots)
                pt = p0 + v0 * best_t + 0.5 * self.g_vec * best_t**2
                best_surface = (normal, pt)

        # Vertical throw ceiling override: if the throw is nearly vertical
        # and a ceiling surface has enough votes, select it even if the
        # ballistic intersection fails (the ball will use thrust to reach it).
        if best_surface is None or best_surface[0][2] > -0.5:
            v0_dir = v0 / (np.linalg.norm(v0) + 1e-10)
            if v0_dir[2] > np.cos(np.radians(20)):  # within 20° of vertical
                for normal, d_plane, count in self._surface_library:
                    if count < min_count:
                        continue
                    if normal[2] < -0.7:  # ceiling-like
                        ceiling_z = d_plane / normal[2] if abs(normal[2]) > 0.01 else 0
                        impact_pt = np.array([p[0], p[1], ceiling_z])
                        best_surface = (normal, impact_pt)
                        break

        if best_surface is not None:
            normal, impact_pt = best_surface
            prev_normal = self.target_normal
            self.target_normal = normal
            self.target_pos = impact_pt + self._com_surface_offset * normal
            # Log target changes (new target or normal changed significantly)
            if np.dot(normal, prev_normal) < 0.99:
                self._log_detections.append((t, normal.copy(), self.target_pos.copy()))

    @staticmethod
    def _solve_plane_intersection(a, b, c, p0, v0, g_vec):
        if abs(a) > 1e-10:
            disc = b * b - 4.0 * a * c
            if disc < 0:
                return None
            sq = np.sqrt(disc)
            roots = [(-b + sq) / (2 * a), (-b - sq) / (2 * a)]
        elif abs(b) > 1e-10:
            roots = [-c / b]
        else:
            return None
        pos = [r for r in roots if r > 0.01]
        if not pos:
            return None
        tau = min(pos)
        return p0 + v0 * tau + 0.5 * g_vec * tau**2
