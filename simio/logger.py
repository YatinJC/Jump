"""
HDF5 logger for simulation results.

File structure
--------------
/ground_truth/
    t            (N,)    simulation time
    position     (N, 3)  COM in world frame
    velocity     (N, 3)  world frame
    quaternion   (N, 4)  body-to-world [w,x,y,z]
    omega        (N, 3)  body frame rad/s
    accel_world  (N, 3)  linear acceleration of COM, world frame
    alpha_body   (N, 3)  angular acceleration, body frame
    controller_phase (N,) int8  controller phase ID
    terminal_sub_phase (N,) int8  orient sub-state (0=yaw, 1=tilt, -1=N/A)
    brake_align_deg (N,) float32  thrust axis vs target direction (degrees)
    estimator_quat (N, 4) float  estimator quaternion [w,x,y,z]
    ctrl_yaw_error (N,) float32  yaw error (rad)
    ctrl_tilt_error (N,) float32  tilt error (rad)
    ctrl_mode (N,) int8  control mode (-1=none, 0=bang, 1=pd)
    ctrl_target_dir (N, 3) float  orient target direction (world frame)

/thrusters/
    t            (N,)
    commanded    (N, K)
    actual       (N, K)
    vectoring_commanded  (N, K)  gimbal deflection in radians
    vectoring_actual     (N, K)  gimbal deflection in radians

/imu/
    t            (M,)
    accelerometer (M, 3)  body frame m/s^2
    gyroscope     (M, 3)  body frame rad/s

/distance_sensors/
    t            (P,)
    readings     (P, S)   one column per sensor

/controller/
    hit_points_t   (H,)    timestamp per hit point
    hit_points_pos (H, 3)  world-frame hit point position
    detect_t       (D,)    timestamp per detection attempt
    detect_normal  (D, 3)  detected surface normal (NaN if none)
    detect_pos     (D, 3)  detected target position (NaN if none)

/collision/
    occurred     scalar bool
    time         scalar float (NaN if no collision)
    surface      scalar str ('wall', 'ground', or '')
    state        (13,)  state vector at impact (zeros if no collision)
    relative_state  variable-length JSON string

/config/
    json         scalar str  (entire config as JSON)
"""

import json
import h5py
import numpy as np
from pathlib import Path

from sim.simulation import SimResult


def save(result: SimResult, cfg: dict, path: str):
    """Write simulation result and config to an HDF5 file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, 'w') as f:

        # --- Ground truth ---
        gt = f.create_group('ground_truth')
        gt.create_dataset('t',            data=result.t_truth,      compression='gzip')
        gt.create_dataset('position',    data=result.pos,          compression='gzip')
        gt.create_dataset('velocity',    data=result.vel,          compression='gzip')
        gt.create_dataset('throw_active', data=result.throw_active, compression='gzip')
        gt.create_dataset('quaternion',  data=result.quat,        compression='gzip')
        gt.create_dataset('omega',       data=result.omega,       compression='gzip')
        gt.create_dataset('accel_world', data=result.accel_world, compression='gzip')
        gt.create_dataset('alpha_body',  data=result.alpha_body,  compression='gzip')
        gt.create_dataset('controller_phase', data=result.controller_phase, compression='gzip')
        gt.create_dataset('terminal_sub_phase', data=result.terminal_sub_phase, compression='gzip')
        gt.create_dataset('brake_align_deg', data=result.brake_align_deg, compression='gzip')
        gt.create_dataset('estimator_pos', data=result.estimator_pos, compression='gzip')
        gt.create_dataset('estimator_vel', data=result.estimator_vel, compression='gzip')
        gt.create_dataset('estimator_quat', data=result.estimator_quat, compression='gzip')
        gt.create_dataset('ctrl_yaw_error', data=result.ctrl_yaw_error, compression='gzip')
        gt.create_dataset('ctrl_tilt_error', data=result.ctrl_tilt_error, compression='gzip')
        gt.create_dataset('ctrl_mode', data=result.ctrl_mode, compression='gzip')
        gt.create_dataset('ctrl_target_dir', data=result.ctrl_target_dir, compression='gzip')

        # --- Thrusters ---
        th = f.create_group('thrusters')
        th.create_dataset('t',         data=result.t_thrust,         compression='gzip')
        th.create_dataset('commanded', data=result.thrust_commanded, compression='gzip')
        th.create_dataset('actual',    data=result.thrust_actual,    compression='gzip')
        th.create_dataset('vectoring_commanded', data=result.vectoring_commanded, compression='gzip')
        th.create_dataset('vectoring_actual',    data=result.vectoring_actual,    compression='gzip')

        # --- IMU ---
        imu = f.create_group('imu')
        imu.create_dataset('t',             data=result.t_imu,    compression='gzip')
        imu.create_dataset('accelerometer', data=result.imu_accel, compression='gzip')
        imu.create_dataset('gyroscope',     data=result.imu_gyro,  compression='gzip')

        # --- Distance sensors ---
        ds = f.create_group('distance_sensors')
        ds.create_dataset('t',        data=result.t_dist,        compression='gzip')
        ds.create_dataset('readings', data=result.dist_readings, compression='gzip')

        # --- Controller diagnostics ---
        diag = f.create_group('controller')
        diag.create_dataset('hit_points_t',   data=result.hit_points_t,   compression='gzip')
        diag.create_dataset('hit_points_pos', data=result.hit_points_pos, compression='gzip')
        diag.create_dataset('detect_t',       data=result.detect_t,       compression='gzip')
        diag.create_dataset('detect_normal',  data=result.detect_normal,  compression='gzip')
        diag.create_dataset('detect_pos',     data=result.detect_pos,     compression='gzip')
        # Surface library snapshot: JSON array of [normal, d_offset, count]
        lib = getattr(result, 'surface_library', [])
        lib_json = json.dumps([[n.tolist(), float(d), int(c)] for n, d, c in lib])
        diag.create_dataset('surface_library', data=lib_json)

        # --- Collision ---
        col = f.create_group('collision')
        col.create_dataset('occurred', data=bool(result.collision_occurred))
        col.create_dataset('time',     data=result.collision_time if result.collision_occurred else float('nan'))
        col.create_dataset('surface',  data=result.collision_surface or '')
        col.create_dataset('state',    data=result.collision_state if result.collision_occurred else np.zeros(13))
        rel_json = json.dumps(
            {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in result.collision_relative.items()}
        ) if result.collision_relative else '{}'
        col.create_dataset('relative_state', data=rel_json)

        # --- Config ---
        cfg_group = f.create_group('config')
        cfg_group.create_dataset('json', data=json.dumps(cfg, default=str))

    print(f"Saved simulation output to {path}")


def load(path: str) -> dict:
    """
    Load a saved HDF5 file.

    Returns a plain dict mirroring the HDF5 structure, with numpy arrays
    for all numerical datasets.
    """
    out = {}
    with h5py.File(path, 'r') as f:
        for group_name in f.keys():
            group = f[group_name]
            if isinstance(group, h5py.Dataset):
                out[group_name] = group[()]
            else:
                out[group_name] = {}
                for key in group.keys():
                    val = group[key][()]
                    if isinstance(val, bytes):
                        val = val.decode()
                    out[group_name][key] = val

    # Parse config JSON
    if 'config' in out and 'json' in out['config']:
        out['config']['parsed'] = json.loads(out['config']['json'])

    # Parse collision relative_state JSON
    if 'collision' in out:
        rs = out['collision'].get('relative_state', '{}')
        if isinstance(rs, bytes):
            rs = rs.decode()
        out['collision']['relative_state_parsed'] = json.loads(rs)

    # Parse surface library JSON
    if 'controller' in out:
        sl = out['controller'].get('surface_library', '[]')
        if isinstance(sl, bytes):
            sl = sl.decode()
        out['controller']['surface_library_parsed'] = json.loads(sl)

    return out
