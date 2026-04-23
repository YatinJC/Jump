"""
Entry point for the ball simulator.

Usage:
    python main.py                          # use default config, save to sim_output.h5
    python main.py -c config/my_scene.yaml  # custom config
    python main.py -o results/run1.h5       # custom output path
    python main.py --no-viz                 # skip visualization
    python main.py --seed 123              # set random seed for reproducible noise
    python main.py --replay results/run1.h5 # replay a saved run (no re-simulation)
"""

import argparse
import sys
from pathlib import Path

from simio.config_loader import load_config
from simio.logger        import save, load
from sim.simulation      import Simulation
from viz.playback        import Visualizer
from scripts.flight_controller import FlightController


def parse_args():
    p = argparse.ArgumentParser(description='Ball simulator with spin, sensors, and thrusters.')
    p.add_argument('-c', '--config',  default='config/default.yaml',
                   help='Path to YAML config file (default: config/default.yaml)')
    p.add_argument('-o', '--output',  default=None,
                   help='Path for HDF5 output (overrides config logging.output_file)')
    p.add_argument('--no-viz', action='store_true',
                   help='Skip visualization after simulation')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for sensor noise (default: 42)')
    p.add_argument('--replay', default=None,
                   help='Path to an existing HDF5 log file to replay (skips simulation)')
    return p.parse_args()


def run_simulation(args):
    cfg = load_config(args.config)

    output_path = args.output or cfg['logging']['output_file']

    sim        = Simulation(cfg)
    controller = FlightController(cfg, sim.thruster_array)
    result     = sim.run(controller=controller, seed=args.seed)

    save(result, cfg, output_path)

    if result.collision_occurred:
        print(f"\n--- Impact summary ---")
        print(f"  Surface : {result.collision_surface}")
        print(f"  Time    : {result.collision_time:.4f} s")
        cv = result.collision_state[3:6]
        print(f"  Velocity: vx={cv[0]:+.3f}  vy={cv[1]:+.3f}  vz={cv[2]:+.3f} m/s")
        print(f"  Speed   : {(cv[0]**2 + cv[1]**2 + cv[2]**2)**0.5:.3f} m/s")
        rel = result.collision_relative
        if rel:
            for k, v in rel.items():
                print(f"  {k:<20}: {v}")

    return result, cfg


def replay(args):
    data = load(args.replay)
    cfg  = data['config']['parsed']

    # Reconstruct a minimal SimResult-like object from the loaded data
    from sim.simulation import SimResult
    r = SimResult.__new__(SimResult)

    import numpy as np
    gt = data['ground_truth']
    r.t_truth     = gt['t']
    r.pos         = gt['position']
    r.vel         = gt['velocity']
    r.quat        = gt['quaternion']
    r.omega       = gt['omega']
    r.accel_world = gt['accel_world']
    r.alpha_body  = gt['alpha_body']
    r.throw_active = gt['throw_active']
    r.controller_phase = gt.get('controller_phase', np.array([], dtype=np.int8))
    r.terminal_sub_phase = gt.get('terminal_sub_phase', np.array([], dtype=np.int8))
    r.brake_align_deg = gt.get('brake_align_deg', np.array([], dtype=np.float32))
    r.estimator_quat = gt.get('estimator_quat', np.empty((0, 4)))
    r.ctrl_yaw_error = gt.get('ctrl_yaw_error', np.array([], dtype=np.float32))
    r.ctrl_tilt_error = gt.get('ctrl_tilt_error', np.array([], dtype=np.float32))
    r.ctrl_mode = gt.get('ctrl_mode', np.array([], dtype=np.int8))
    r.ctrl_target_dir = gt.get('ctrl_target_dir', np.empty((0, 3)))

    th = data['thrusters']
    r.t_thrust            = th['t']
    r.thrust_commanded    = th['commanded']
    r.thrust_actual       = th['actual']
    r.vectoring_commanded = th['vectoring_commanded']
    r.vectoring_actual    = th['vectoring_actual']

    imu = data['imu']
    r.t_imu     = imu['t']
    r.imu_accel = imu['accelerometer']
    r.imu_gyro  = imu['gyroscope']

    ds = data['distance_sensors']
    r.t_dist       = ds['t']
    r.dist_readings = ds['readings']

    col = data['collision']
    r.collision_occurred = bool(col['occurred'])
    r.collision_time     = float(col['time']) if r.collision_occurred else None
    surface = col['surface']
    r.collision_surface  = surface.decode() if isinstance(surface, bytes) else surface
    r.collision_state    = col['state'] if r.collision_occurred else None
    r.collision_relative = data['collision']['relative_state_parsed']

    print(f"Loaded replay from {args.replay}")
    return r, cfg


def main():
    args = parse_args()

    if args.replay:
        result, cfg = replay(args)
    else:
        result, cfg = run_simulation(args)

    if not args.no_viz:
        viz = Visualizer(result, cfg)
        viz.show()


if __name__ == '__main__':
    main()
