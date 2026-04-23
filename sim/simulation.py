"""
Main simulation loop.

Runs a fixed-step RK4 simulation.  Sensor sampling occurs at integer multiples
of the sim timestep.  The loop ends on the first collision or max_time.

The optional `controller` callable has signature:
    commands = controller(t, state, imu_data, distance_data) -> array (n_thrusters,)
where imu_data = (accel, gyro) from the last IMU sample (or None before first sample)
and distance_data = array of distances (or None).
"""

import numpy as np

from sim.physics      import build_initial_state, rk4_step, quat_to_rotmat
from sim.thrusters    import ThrusterArray
from sim.sensors      import SensorSuite
from sim.environment  import Environment
from sim.throw        import ThrowPhase


class SimResult:
    """Container for all simulation output data."""

    def __init__(self):
        # Ground truth (logged at every simulation step)
        self.t_truth       = []   # float
        self.pos           = []   # (3,)
        self.vel           = []   # (3,)
        self.quat          = []   # (4,)
        self.omega         = []   # (3,)
        self.accel_world   = []   # (3,) linear acceleration of COM
        self.alpha_body    = []   # (3,) angular acceleration in body frame
        self.throw_active  = []   # bool: True during throw phase

        # Thruster states (logged at every simulation step)
        self.t_thrust         = []
        self.thrust_commanded = []  # (n_thrusters,)
        self.thrust_actual    = []  # (n_thrusters,)
        self.vectoring_commanded = []  # (n_thrusters,) radians
        self.vectoring_actual    = []  # (n_thrusters,) radians

        # IMU (logged at IMU sample rate)
        self.t_imu    = []
        self.imu_accel = []   # (3,)
        self.imu_gyro  = []   # (3,)

        # Distance sensors (logged at distance sample rate)
        self.t_dist   = []
        self.dist_readings = []  # (n_sensors,)

        # Controller phase (logged at every simulation step)
        self.controller_phase = []     # int: phase ID
        self.terminal_sub_phase = []   # int: orient sub-state (-1=N/A, unused)
        self.brake_align_deg = []      # float: thrust axis vs target direction
        self.estimator_pos = []        # (3,) estimator position estimate
        self.estimator_vel = []        # (3,) estimator velocity estimate
        self.estimator_quat = []       # (4,) estimator quaternion estimate
        self.ctrl_yaw_error = []       # float: yaw error (rad)
        self.ctrl_tilt_error = []      # float: tilt error (rad)
        self.ctrl_mode = []            # int: -1=none, 0=bang, 1=pd
        self.ctrl_target_dir = []      # (3,) orient target direction (world)

        # Controller diagnostics (logged by the controller at its own rates)
        self.hit_points_t   = []   # float — timestamp per hit point
        self.hit_points_pos = []   # (3,)  — world-frame position
        self.detect_t       = []   # float — timestamp per detection attempt
        self.detect_normal  = []   # (3,)  — detected surface normal (NaN if none)
        self.detect_pos     = []   # (3,)  — detected target position (NaN if none)

        # Collision info
        self.collision_occurred = False
        self.collision_time     = None
        self.collision_surface  = None   # 'wall' or 'ground'
        self.collision_state    = None   # state vector at impact
        self.collision_relative = None   # relative state dict

    def finalize(self):
        """Convert lists to numpy arrays."""
        self.t_truth       = np.array(self.t_truth)
        self.pos           = np.array(self.pos)
        self.vel           = np.array(self.vel)
        self.quat          = np.array(self.quat)
        self.omega         = np.array(self.omega)
        self.accel_world   = np.array(self.accel_world)
        self.alpha_body    = np.array(self.alpha_body)
        self.t_thrust         = np.array(self.t_thrust)
        self.thrust_commanded = np.array(self.thrust_commanded)
        self.thrust_actual    = np.array(self.thrust_actual)
        self.vectoring_commanded = np.array(self.vectoring_commanded)
        self.vectoring_actual    = np.array(self.vectoring_actual)
        self.t_imu    = np.array(self.t_imu)
        self.imu_accel = np.array(self.imu_accel)
        self.imu_gyro  = np.array(self.imu_gyro)
        self.t_dist   = np.array(self.t_dist)
        self.dist_readings = np.array(self.dist_readings)
        self.throw_active = np.array(self.throw_active, dtype=bool)
        self.controller_phase = np.array(self.controller_phase, dtype=np.int8)
        self.terminal_sub_phase = np.array(self.terminal_sub_phase, dtype=np.int8)
        self.brake_align_deg = np.array(self.brake_align_deg, dtype=np.float32)
        self.estimator_pos = np.array(self.estimator_pos) if self.estimator_pos else np.empty((0, 3))
        self.estimator_vel = np.array(self.estimator_vel) if self.estimator_vel else np.empty((0, 3))
        self.estimator_quat = np.array(self.estimator_quat) if self.estimator_quat else np.empty((0, 4))
        self.ctrl_yaw_error = np.array(self.ctrl_yaw_error, dtype=np.float32)
        self.ctrl_tilt_error = np.array(self.ctrl_tilt_error, dtype=np.float32)
        self.ctrl_mode = np.array(self.ctrl_mode, dtype=np.int8)
        self.ctrl_target_dir = np.array(self.ctrl_target_dir) if self.ctrl_target_dir else np.empty((0, 3))
        self.hit_points_t   = np.array(self.hit_points_t)
        self.hit_points_pos = np.array(self.hit_points_pos) if self.hit_points_pos else np.empty((0, 3))
        self.detect_t       = np.array(self.detect_t)
        self.detect_normal  = np.array(self.detect_normal) if self.detect_normal else np.empty((0, 3))
        self.detect_pos     = np.array(self.detect_pos) if self.detect_pos else np.empty((0, 3))


class Simulation:

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dt  = float(cfg['simulation']['dt'])
        self.max_time = float(cfg['simulation']['max_time'])

        sim_rate = 1.0 / self.dt
        imu_rate  = float(cfg['sensors']['imu']['sample_rate'])
        dist_rate = float(cfg['sensors']['distance_sensors']['sample_rate'])

        imu_steps  = sim_rate / imu_rate
        dist_steps = sim_rate / dist_rate

        assert abs(imu_steps  - round(imu_steps))  < 1e-9, \
            f"IMU sample rate {imu_rate} Hz must divide evenly into sim rate {sim_rate} Hz"
        assert abs(dist_steps - round(dist_steps)) < 1e-9, \
            f"Distance sensor rate {dist_rate} Hz must divide evenly into sim rate {sim_rate} Hz"

        self.imu_every  = int(round(imu_steps))
        self.dist_every = int(round(dist_steps))

        com_offset = np.array(cfg['ball']['center_of_mass'], dtype=float)
        self.thruster_array = ThrusterArray(cfg['thrusters'], com_offset)
        self.sensors        = SensorSuite(cfg['sensors'], com_offset)
        self.environment    = Environment(cfg['environment'])
        self.throw_phase    = ThrowPhase(cfg['throw_phase'], cfg.get('ball'))

        # Print control allocation matrix at startup
        B = self.thruster_array.control_allocation_matrix()
        print("=== Control Allocation Matrix B (6 x n_thrusters) ===")
        print("  Rows 0-2: force (body frame x/y/z)")
        print("  Rows 3-5: torque (body frame x/y/z)")
        print(np.round(B, 6))
        print()

    def run(self, controller=None, seed: int = 42) -> SimResult:
        """
        Run the simulation.

        Parameters
        ----------
        controller : callable or None
            controller(t, state, imu_data, dist_data) -> ndarray (n_thrusters,)
        seed       : int  random seed for reproducible sensor noise

        Returns
        -------
        SimResult
        """
        rng    = np.random.default_rng(seed)
        result = SimResult()
        state  = build_initial_state(self.cfg)
        cfg    = self.cfg
        dt     = self.dt
        r_ball = float(cfg['ball']['radius'])
        com_offset = np.array(cfg['ball']['center_of_mass'], dtype=float)
        gravity = float(cfg['physics']['gravity'])

        last_imu_data  = None
        last_dist_data = None

        step = 0
        t    = 0.0
        total_steps = int(round(self.max_time / dt))
        log_every   = max(1, total_steps // 20)  # ~5% increments

        while t <= self.max_time:
            if step % log_every == 0:
                pct = 100.0 * step / total_steps
                print(f"\r  sim: {pct:5.1f}%  t={t:.3f}s  z={state[2]:.3f}m  vz={state[5]:.3f}m/s", end="", flush=True)

            # --- Geometric center for collision (COM - R @ com_offset) ---
            R = quat_to_rotmat(state[6:10])
            pos_geom = state[:3] - R @ com_offset

            # --- Collision check ---
            surface_name, surface_obj = self.environment.check_collision(pos_geom, r_ball)
            if surface_name is not None:
                result.collision_occurred = True
                result.collision_time     = t
                result.collision_surface  = surface_name
                result.collision_state    = state.copy()
                result.collision_relative = surface_obj.relative_state(
                    state[:3], state[3:6], state[6:10], state[10:13], r_ball
                )
                print(f"\nCollision with {surface_name} at t={t:.4f} s")
                break

            # --- Throw phase: compute external force/torque at current time ---
            if self.throw_phase.is_active(t):
                ext_force  = self.throw_phase.force_world(t)
                ext_torque = self.throw_phase.torque_world(t)
            else:
                ext_force  = None
                ext_torque = None

            # --- Sample sensors ---
            # Compute accel/alpha at current state (before integration) for sensor sampling.
            from sim.physics import compute_forces_and_accel, compute_torques_and_alpha
            _, accel_world = compute_forces_and_accel(state, self.thruster_array, cfg, ext_force)
            _, alpha_body  = compute_torques_and_alpha(state, self.thruster_array, cfg, ext_torque)

            if step % self.imu_every == 0:
                accel_meas, gyro_meas = self.sensors.sample_imu(
                    state, accel_world, alpha_body, gravity, rng
                )
                last_imu_data = (accel_meas.copy(), gyro_meas.copy())
                result.t_imu.append(t)
                result.imu_accel.append(accel_meas)
                result.imu_gyro.append(gyro_meas)

            if step % self.dist_every == 0:
                dist_meas = self.sensors.sample_distance(state, self.environment, rng)
                last_dist_data = dist_meas.copy()
                result.t_dist.append(t)
                result.dist_readings.append(dist_meas)

            # --- Controller ---
            if controller is not None:
                commands = controller(t, state, last_imu_data, last_dist_data)
                self.thruster_array.set_commands(commands)
                result.controller_phase.append(getattr(controller, '_phase', -1))
                # Orient sub-state: no longer used (direct axis-angle tilt)
                result.terminal_sub_phase.append(-1)
                result.brake_align_deg.append(getattr(controller, '_brake_align_deg', float('nan')))
                result.ctrl_yaw_error.append(getattr(controller, '_yaw_error', 0.0))
                result.ctrl_tilt_error.append(getattr(controller, '_tilt_error', 0.0))
                result.ctrl_mode.append(getattr(controller, '_control_mode', -1))
                td = getattr(controller, '_orient_target_dir_log', None)
                result.ctrl_target_dir.append(td.copy() if td is not None else np.full(3, np.nan))
                est = getattr(controller, 'estimator', None)
                if est is not None:
                    result.estimator_pos.append(est.p.copy())
                    result.estimator_vel.append(est.v.copy())
                    result.estimator_quat.append(est.q.copy())
                else:
                    result.estimator_pos.append(np.full(3, np.nan))
                    result.estimator_vel.append(np.full(3, np.nan))
                    result.estimator_quat.append(np.full(4, np.nan))

                # Drain controller diagnostic logs
                for hp_t, hp_pos in getattr(controller, '_log_hit_points', []):
                    result.hit_points_t.append(hp_t)
                    result.hit_points_pos.append(hp_pos)
                if hasattr(controller, '_log_hit_points'):
                    controller._log_hit_points.clear()

                for det_t, det_n, det_p in getattr(controller, '_log_detections', []):
                    result.detect_t.append(det_t)
                    result.detect_normal.append(det_n)
                    result.detect_pos.append(det_p)
                if hasattr(controller, '_log_detections'):
                    controller._log_detections.clear()
            else:
                result.controller_phase.append(-1)
                result.terminal_sub_phase.append(-1)
                result.brake_align_deg.append(float('nan'))
                result.ctrl_yaw_error.append(0.0)
                result.ctrl_tilt_error.append(0.0)
                result.ctrl_mode.append(-1)
                result.ctrl_target_dir.append(np.full(3, np.nan))
                result.estimator_pos.append(np.full(3, np.nan))
                result.estimator_vel.append(np.full(3, np.nan))
                result.estimator_quat.append(np.full(4, np.nan))

            # --- Log ground truth ---
            result.t_truth.append(t)
            result.pos.append(state[:3].copy())
            result.vel.append(state[3:6].copy())
            result.quat.append(state[6:10].copy())
            result.omega.append(state[10:13].copy())
            result.accel_world.append(accel_world.copy())
            result.alpha_body.append(alpha_body.copy())
            result.throw_active.append(self.throw_phase.is_active(t))

            result.t_thrust.append(t)
            result.thrust_commanded.append(self.thruster_array.get_commanded_forces().copy())
            result.thrust_actual.append(self.thruster_array.get_actual_forces().copy())
            result.vectoring_commanded.append(self.thruster_array.get_commanded_deflections().copy())
            result.vectoring_actual.append(self.thruster_array.get_actual_deflections().copy())

            # --- Integrate thrusters (Euler, exact for linear first-order lag) ---
            self.thruster_array.update(dt)

            # --- RK4 physics step ---
            state, accel_world, alpha_body = rk4_step(
                state, dt, self.thruster_array, cfg, ext_force, ext_torque
            )

            step += 1
            t = step * dt   # avoid float accumulation error

        else:
            print(f"\nSimulation reached max_time={self.max_time} s without collision.")

        # Snapshot the surface library at end of simulation
        result.surface_library = getattr(controller, '_surface_library', [])

        result.finalize()
        return result
