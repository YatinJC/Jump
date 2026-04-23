# Jump Project Notes

## Current State

Flight controller for a ball with 4 thrust-vectoring thrusters. Successfully lands
on walls with near-target velocity and orientation.

## Flight Sequence

YAW_DESPIN (20ms) -> YAW_ALIGN (147ms) -> TILT_DESPIN (38ms) ->
ORI_BURN (327ms) -> BURN (320ms) -> FINAL_ORI (290ms)

## Controller Architecture

- `scripts/flight_controller.py` -- main controller
- `scripts/bang_bang_timing.py` -- timing analysis
- `scripts/pd_gain_tuning.py` -- gain tuning (phase-plane switching + PD settling)

### Key Design Decisions

- Orient phases use recompute-each-step decomposition (not latch-and-integrate, which drifts)
- Combined burn: single dv = v_ideal - v_impact handles deceleration + tangential + gravity
- dv is constant during ballistic coast (proven: gravity terms cancel)
- Surface detection: per-sensor arc cross-product (T, T-20, T-40) with voting library
- Surface selection: relative threshold (10% of max count), future-only intersections
- Surface selection locked during BURN and FINAL_ORIENT

### Burn Configuration

- ORI_BURN: zero-Z (preserves ballistic trajectory so dv direction stays constant)
- BURN: proportional thrust F = m x |dv| / t_remaining with aggressive PD attitude hold (Kp=8, Kd=0.4)
- BURN uses alignment compensation: F_cmd = F_needed / cos(alignment)
- FINAL_ORI: max-torque emergency mode (always triggers for wall landings)
- FINAL_ORI uses direct tilt (axis-angle, no sequential yaw+tilt decomposition)

## Wall Landing Physics

For a wall landing, the ball can't simultaneously fight gravity and be oriented perpendicular
to the wall (single thrust axis). The solution:

1. During BURN, approach horizontally at landing speed while oriented vertically (fighting gravity),
   with upward velocity v_z = g x t_rotate
2. At standoff distance d = v_land x t_rotate from the wall, cut thrust and begin rotation
3. During FINAL_ORI (free-fall + max-torque rotation), gravity cancels the upward velocity
4. Ball arrives at wall perpendicular, at landing speed, with reduced vertical velocity

Key parameters (computed at runtime):

- t_rotate = bang-bang time for arccos(n_z) rotation at zero-Z pitch/roll
- v_ideal at standoff = -v_land x n - g x t_rotate
- d_standoff = v_land x t_rotate + 0.5 x dot(g, n) x t_rotate squared

## Current Landing Performance (wall at x=3.0)

- Speed: 0.94 m/s (target: 1.0)
- Normal: 0.82 m/s
- Tangential: 0.46 m/s
- Orientation: 11 deg from perpendicular

### Known Limitations

- Burn alignment lag: the dv direction rotates about 75 deg during the burn (decelerate to
  gravity fight). The PD tracks with about 12 deg lag at Kp=8. Higher gains don't help
  (actuator saturated).
- FINAL_ORI enters with about 340 deg/s opposing tilt rate from the burn's PD tracking.
  Must reverse this rate before rotating, consuming about 64ms of the 290ms coast time.
- Max-torque FINAL_ORI adds about 0.46 m/s tangential from residual force during rotation.
- These are physical limits of this ball's actuator authority, not controller bugs.
