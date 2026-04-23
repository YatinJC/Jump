# Jump

A throwable, ball-shaped single-lens camera that self-orients and sticks to arbitrary surfaces. You throw it; during the ~500 ms terminal phase, onboard thrust-vectoring motors despin, orient, and brake the ball so a sticky pad on the side opposite the lens contacts the wall at a safe, near-normal impact — camera aimed into the room.

This repo is the Georgia Tech MS Robotics capstone work-in-progress: a physics-accurate simulator, a working model-based terminal-guidance controller, and analysis tooling. The hardware prototype and factor-graph state estimator are the next stages. See [Research/Proposal.pdf](Research/Proposal.pdf) for the full proposal.

## What works today

In simulation the ball lands on a vertical wall at ~0.94 m/s with ~11° orientation error after a ~1.1 s flight. The terminal maneuver decomposes into:

```
YAW_DESPIN (20ms) → YAW_ALIGN (147ms) → TILT_DESPIN (38ms)
  → ORI_BURN (327ms) → BURN (320ms) → FINAL_ORI (290ms)
```

A single constant-direction burn (`dv = v_ideal − v_impact`) handles deceleration, tangential matching, and gravity during ballistic coast.

## Repo layout

| Path | Contents |
| --- | --- |
| [main.py](main.py) | Simulator entry point (run, replay, visualize) |
| [config/](config/) | YAML scene/actuator configs ([default.yaml](config/default.yaml)) |
| [sim/](sim/) | 8 kHz rigid-body dynamics, thruster + IMU + ToF models, environment |
| [simio/](simio/) | Config loader and HDF5 logger |
| [scripts/](scripts/) | Flight controller, bang-bang timing, PD gain tuning, sensor fusion |
| [scripts/analysis/](scripts/analysis/) | Ad-hoc inspection / plotting scripts for HDF5 runs |
| [viz/](viz/) | 3D playback / animation |
| [data/](data/) | Saved simulation runs (`.h5`) |
| [plots/](plots/) | Generated figures |
| [Research/](Research/) | Capstone proposal (LaTeX source + PDF) |

Key module: [scripts/flight_controller.py](scripts/flight_controller.py). Timing analysis: [scripts/bang_bang_timing.py](scripts/bang_bang_timing.py). Gain tuning: [scripts/pd_gain_tuning.py](scripts/pd_gain_tuning.py).

## Running

All commands run from the repo root.

```bash
# Set up the environment
conda env create -f environment.yml
conda activate jump

# Simulate with the default config, save to data/sim_output_test_quadrotor_vertical.h5, then visualize
python main.py

# Custom config / output
python main.py -c config/default.yaml -o data/my_run.h5

# Replay a saved run without re-simulating
python main.py --replay data/sim_output_test_quadrotor_vertical.h5

# Skip visualization
python main.py --no-viz

# Poke at a saved run
python scripts/analysis/analyze_log.py
python scripts/analysis/analyze_despin.py
```

The analyze scripts read from `data/sim_output_test_quadrotor_vertical.h5` by default; point them elsewhere by editing the filename or re-running `main.py -o` first.

## Roadmap

1. **Controller** — convex re-planner respecting actuator saturation and vectoring limits; quantitative success envelope over throw speed, angle, spin, and surface orientation.
2. **Estimator** — GTSAM factor graph fusing IMU preintegration, sparse ToF range factors, thrust/torque priors, and optional monocular features; incremental iSAM2 solve for the ball's 15-DoF state + the target plane pose.
3. **Hardware** — ~10–12 cm shell, 4× 1106-class brushless motors with ±20° thrust vectoring, STM32H7 flight controller, ICM-42688-P IMU, VL53L5CX ToF, single-lens camera, 1S–3S LiPo.
4. **Closed-loop throws** — netted test space with motion-capture ground truth; vertical walls → oblique walls → tumbling throws.

See [Research/Proposal.pdf](Research/Proposal.pdf) for full methods, schedule, and references.
