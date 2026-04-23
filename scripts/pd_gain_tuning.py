#!/usr/bin/env python3
"""
Optimal single-axis orientation controller for the Jump ball.

Phase-plane switching + linear PD settling:

  1. FAR FROM TARGET: command ±tau_max based on a switching curve.
     The switching curve accounts for actuator lag — it triggers
     deceleration early enough that the ball stops at the target
     even with the first-order force lag.

     Switching condition (decelerate when):
       theta + omega^2 / (2*alpha_max) + omega*tau_margin >= theta_target
     where tau_margin compensates for the actuator lag.

  2. NEAR TARGET (|error| < threshold): linear PD with reference
     prefilter for smooth, zero-overshoot settling.

     The prefilter cancels the closed-loop zero from the derivative
     term, giving a pure triple-pole response.

This gives near-time-optimal response for large maneuvers (essentially
bang-bang as feedback) and smooth settling for small errors.

For yaw, the servo ramp from 0° is included.
For pitch/roll, torque is linear in force (no vectoring).
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from bang_bang_timing import (
    load_config, parse_thrusters, compute_wrench,
    find_max_yaw_config, find_max_yaw_zero_z,
    solve_analytical, solve_yaw_numerical,
)


def compute_pd_gains(I, tau_f):
    """Critical damping gains (triple pole at p = 1/(3·τ_f))."""
    p = 1.0 / (3.0 * tau_f)
    Kp = I * p**3 * tau_f
    Kd = 3 * I * p**2 * tau_f
    return Kp, Kd


# ---------------------------------------------------------------------------
#  Pitch/Roll simulation
# ---------------------------------------------------------------------------

def simulate_pr(I, tau_f, tau_max, Kp, Kd, theta_target,
                lag_mult=3.0, dt=0.0001, t_max=2.0):
    """
    Simulate phase-plane switching + PD controller for pitch/roll.

    Plant: I*alpha = tau_actual, dtau/dt = (tau_cmd - tau)/tau_f
    lag_mult: switching curve margin = lag_mult * omega * tau_f
    """
    alpha_max = tau_max / I
    ref_rate = Kp / Kd if Kd > 0 else 1e6

    # Transition threshold: PD can handle without saturating
    pd_threshold = tau_max / Kp

    theta = 0.0
    omega = 0.0
    tau_act = 0.0
    theta_ref = 0.0  # prefilter state (used in PD mode)

    N = int(t_max / dt)
    overshoot = 0.0
    settled_at = None
    settle_start = None
    settle_tol_th = np.radians(0.1)
    settle_tol_w = np.radians(1.0)
    settle_hold = 0.005
    in_pd_mode = False

    for i in range(N):
        t = i * dt

        # Overshoot check
        if theta > theta_target:
            os_deg = np.degrees(theta - theta_target)
            if os_deg > overshoot:
                overshoot = os_deg

        # Settling check
        if settled_at is None:
            if (abs(theta - theta_target) < settle_tol_th and
                    abs(omega) < settle_tol_w):
                if settle_start is None:
                    settle_start = t
                elif t - settle_start >= settle_hold:
                    settled_at = settle_start
            else:
                settle_start = None

        error = theta - theta_target

        if not in_pd_mode and abs(error) < pd_threshold and abs(omega) < alpha_max * 3 * tau_f:
            in_pd_mode = True
            theta_ref = theta  # initialize prefilter at current position

        if in_pd_mode:
            # Linear PD with prefilter
            theta_ref += (theta_target - theta_ref) * ref_rate * dt
            tau_cmd = -Kp * (theta - theta_ref) - Kd * omega
            tau_cmd = np.clip(tau_cmd, -tau_max, tau_max)
        else:
            # Phase-plane switching
            # Stopping distance from current omega (accounting for actuator lag)
            if abs(omega) > 1e-10:
                stop_dist = omega**2 / (2 * alpha_max)
                # Margin for actuator lag: extra distance during torque reversal
                lag_margin = abs(omega) * tau_f * lag_mult
            else:
                stop_dist = 0.0
                lag_margin = 0.0

            if omega >= 0:
                # Moving toward target (positive direction)
                if theta + stop_dist + lag_margin >= theta_target:
                    tau_cmd = -tau_max  # decelerate
                else:
                    tau_cmd = tau_max   # accelerate
            else:
                # Moving away from target (or negative direction)
                if theta - stop_dist - lag_margin <= theta_target:
                    tau_cmd = tau_max   # decelerate (positive torque)
                else:
                    tau_cmd = -tau_max  # accelerate away?? — shouldn't happen for normal maneuver

        # RK4 plant step
        def derivs(th, w, ta):
            return w, ta / I, (tau_cmd - ta) / tau_f

        k1 = derivs(theta, omega, tau_act)
        k2 = derivs(theta + .5*dt*k1[0], omega + .5*dt*k1[1], tau_act + .5*dt*k1[2])
        k3 = derivs(theta + .5*dt*k2[0], omega + .5*dt*k2[1], tau_act + .5*dt*k2[2])
        k4 = derivs(theta + dt*k3[0], omega + dt*k3[1], tau_act + dt*k3[2])

        theta  += dt/6 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        omega  += dt/6 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        tau_act += dt/6 * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])

    return {
        'settling_time': settled_at,
        'overshoot_deg': overshoot,
    }


# ---------------------------------------------------------------------------
#  Yaw simulation
# ---------------------------------------------------------------------------

def simulate_yaw(thrusters, I, tau_f, tau_s, tau_max_ss,
                 forces_fwd, forces_rev, defls_target,
                 Kp, Kd, theta_target, lag_mult=3.0, dt=0.0001, t_max=2.0):
    """
    Simulate phase-plane switching + PD controller for yaw.
    Servos start from 0.  Force-only switching for torque reversal.
    lag_mult: switching curve margin = lag_mult * omega * tau_f
    """
    n = len(thrusters)
    alpha_max = tau_max_ss / I
    ref_rate = Kp / Kd if Kd > 0 else 1e6
    pd_threshold = tau_max_ss / Kp

    # Pre-extract geometry
    px = np.array([t['pos_from_com'][0] for t in thrusters])
    py = np.array([t['pos_from_com'][1] for t in thrusters])
    nx_ = np.array([t['nom'][0] for t in thrusters])
    ny_ = np.array([t['nom'][1] for t in thrusters])
    nz_ = np.array([t['nom'][2] for t in thrusters])
    sx_ = np.array([t['swing'][0] for t in thrusters])
    sy_ = np.array([t['swing'][1] for t in thrusters])
    sz_ = np.array([t['swing'][2] for t in thrusters])
    kq = np.array([t['k_q'] for t in thrusters])
    sp = np.array([t['spin'] for t in thrusters])
    f_fwd = np.array(forces_fwd, dtype=float)
    f_rev = np.array(forces_rev, dtype=float)
    d_tgt = np.array(defls_target, dtype=float)

    def tz_from_state(F, D):
        cd = np.cos(D); sd = np.sin(D)
        dx = cd * nx_ + sd * sx_
        dy = cd * ny_ + sd * sy_
        dz = cd * nz_ + sd * sz_
        return (np.sum(px * (F * dy) - py * (F * dx))
                + np.sum(kq * F * sp * (-dz)))

    theta = 0.0
    omega = 0.0
    theta_ref = 0.0
    F_act = np.zeros(n)
    D_act = np.zeros(n)

    N = int(t_max / dt)
    overshoot = 0.0
    settled_at = None
    settle_start = None
    settle_tol_th = np.radians(0.1)
    settle_tol_w = np.radians(1.0)
    settle_hold = 0.005
    in_pd_mode = False
    dt_tf = dt / tau_f
    dt_ts = dt / tau_s

    for i in range(N):
        t = i * dt
        tz = tz_from_state(F_act, D_act)

        if theta > theta_target:
            os_deg = np.degrees(theta - theta_target)
            if os_deg > overshoot:
                overshoot = os_deg

        if settled_at is None:
            if (abs(theta - theta_target) < settle_tol_th and
                    abs(omega) < settle_tol_w):
                if settle_start is None:
                    settle_start = t
                elif t - settle_start >= settle_hold:
                    settled_at = settle_start
            else:
                settle_start = None

        error = theta - theta_target

        if not in_pd_mode and abs(error) < pd_threshold and abs(omega) < alpha_max * 3 * tau_f:
            in_pd_mode = True
            theta_ref = theta

        if in_pd_mode:
            theta_ref += (theta_target - theta_ref) * ref_rate * dt
            tau_des = -Kp * (theta - theta_ref) - Kd * omega
            ratio = np.clip(tau_des / tau_max_ss, -1.0, 1.0)
        else:
            # Phase-plane switching
            if abs(omega) > 1e-10:
                stop_dist = omega**2 / (2 * alpha_max)
                lag_margin = abs(omega) * tau_f * lag_mult
            else:
                stop_dist = 0.0
                lag_margin = 0.0

            if omega >= 0:
                if theta + stop_dist + lag_margin >= theta_target:
                    ratio = -1.0  # decelerate
                else:
                    ratio = 1.0   # accelerate
            else:
                if theta - stop_dist - lag_margin <= theta_target:
                    ratio = 1.0
                else:
                    ratio = -1.0

        # Force allocation
        if ratio >= 0:
            F_cmd = ratio * f_fwd
        else:
            F_cmd = (-ratio) * f_rev

        # Actuator dynamics
        F_act += (F_cmd - F_act) * dt_tf
        D_act += (d_tgt - D_act) * dt_ts

        # Angular dynamics
        alpha = tz / I
        omega_new = omega + alpha * dt
        theta += 0.5 * (omega + omega_new) * dt
        omega = omega_new

    return {
        'settling_time': settled_at,
        'overshoot_deg': overshoot,
    }


# ---------------------------------------------------------------------------
#  Lag margin tuning
# ---------------------------------------------------------------------------

def find_optimal_lag_margin(sim_factory, angles_deg, n_bisect=30):
    """
    Bisect on the lag margin multiplier to find the smallest margin
    that gives zero overshoot across all angles.

    sim_factory(lag_mult, angle_deg) -> dict with 'overshoot_deg'
    """
    overshoot_tol = 0.01  # degrees

    def has_overshoot(lag_mult):
        for deg in angles_deg:
            result = sim_factory(lag_mult, deg)
            if result['overshoot_deg'] > overshoot_tol:
                return True
        return False

    # Generous margin always works; zero margin may not
    lo, hi = 0.0, 10.0
    # Verify hi works
    if has_overshoot(hi):
        hi = 20.0
    # Bisect: find smallest margin without overshoot
    for _ in range(n_bisect):
        mid = (lo + hi) / 2
        if has_overshoot(mid):
            lo = mid
        else:
            hi = mid

    return hi  # conservative side


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    I_diag = np.array(cfg['ball']['moment_of_inertia']['diagonal'])
    Ixx, Iyy, Izz = I_diag
    thrusters = parse_thrusters(cfg)

    tau_f = float(cfg['thrusters'][0]['time_constant'])
    tau_s = float(cfg['thrusters'][0]['vectoring']['time_constant'])
    f_max = thrusters[0]['f_max']
    f_min = thrusters[0]['f_min']
    f_cap = min(f_max, abs(f_min))

    Kp_pr, Kd_pr = compute_pd_gains(Ixx, tau_f)
    Kp_yaw, Kd_yaw = compute_pd_gains(Izz, tau_f)

    print("=" * 78)
    print("  PHASE-PLANE SWITCHING + PD SETTLING CONTROLLER")
    print("=" * 78)
    print()
    print(f"  MOI: Ixx={Ixx:.6f}  Iyy={Iyy:.6f}  Izz={Izz:.6f}  kg*m^2")
    print(f"  Force time constant: tau_f = {tau_f*1000:.1f} ms")
    print(f"  Servo time constant: tau_s = {tau_s*1000:.1f} ms")
    print()
    print("  Strategy:")
    print("    FAR:  ±tau_max (bang-bang via switching curve)")
    print("    NEAR: PD with prefilter (zero-overshoot settling)")
    print()
    print(f"  PD gains (triple pole at p = {1/(3*tau_f):.1f} rad/s):")
    print(f"    Pitch/Roll: Kp = {Kp_pr:.4f}  Kd = {Kd_pr:.6f}")
    print(f"    Yaw:        Kp = {Kp_yaw:.4f}  Kd = {Kd_yaw:.6f}")
    pd_thresh_pr = np.degrees(Ixx * (1/(3*tau_f))**3 * tau_f)  # tau_max/Kp in degrees... no
    print(f"    PD takeover threshold: ~{np.degrees(f_max / Kp_pr):.1f} deg (pitch/roll), "
          f"~{np.degrees(f_max / Kp_yaw):.1f} deg (yaw)")
    print()

    angles_deg = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180]

    # ==================================================================
    #  Build cases
    # ==================================================================
    cases = []

    # Pitch/Roll max torque
    f1_roll = [f_max, f_min, f_max, f_min]
    f2_roll = [f_min, f_max, f_min, f_max]
    d_zero = [0, 0, 0, 0]
    w1_roll = compute_wrench(thrusters, f1_roll, d_zero)
    tau_roll = abs(w1_roll[3])
    cases.append({
        'name': 'Pitch/Roll — max torque (residual Fz)',
        'type': 'pr', 'I': Ixx, 'tau_max': tau_roll,
        'Kp': Kp_pr, 'Kd': Kd_pr,
        'f_fwd': f1_roll, 'f_rev': f2_roll,
    })

    # Pitch/Roll zero-Z
    f1_rz = [f_cap, -f_cap, f_cap, -f_cap]
    f2_rz = [-f_cap, f_cap, -f_cap, f_cap]
    w1_rz = compute_wrench(thrusters, f1_rz, d_zero)
    tau_rz = abs(w1_rz[3])
    cases.append({
        'name': 'Pitch/Roll — zero Z force',
        'type': 'pr', 'I': Ixx, 'tau_max': tau_rz,
        'Kp': Kp_pr, 'Kd': Kd_pr,
        'f_fwd': f1_rz, 'f_rev': f2_rz,
    })

    # Yaw max torque
    f1_yaw, d_yaw, w1_yaw = find_max_yaw_config(thrusters)
    f2_yaw = [f1_yaw[1], f1_yaw[0], f1_yaw[3], f1_yaw[2]]
    w2_yaw = compute_wrench(thrusters, f2_yaw, d_yaw)
    if w1_yaw[5] < 0:
        f1_yaw, f2_yaw = f2_yaw, f1_yaw
        w1_yaw, w2_yaw = w2_yaw, w1_yaw
    tau_yaw = abs(w1_yaw[5])
    cases.append({
        'name': 'Yaw — max torque (residual Fz)',
        'type': 'yaw', 'I': Izz, 'tau_max': tau_yaw,
        'Kp': Kp_yaw, 'Kd': Kd_yaw,
        'f_fwd': f1_yaw, 'f_rev': f2_yaw, 'defls': d_yaw,
    })

    # Yaw zero-Z
    f1_yz, d_yz, w1_yz = find_max_yaw_zero_z(thrusters)
    f2_yz = [f1_yz[1], f1_yz[0], f1_yz[3], f1_yz[2]]
    w2_yz = compute_wrench(thrusters, f2_yz, d_yz)
    if w1_yz[5] < 0:
        f1_yz, f2_yz = f2_yz, f1_yz
        w1_yz, w2_yz = w2_yz, w1_yz
    tau_yz = abs(w1_yz[5])
    cases.append({
        'name': 'Yaw — zero Z force',
        'type': 'yaw', 'I': Izz, 'tau_max': tau_yz,
        'Kp': Kp_yaw, 'Kd': Kd_yaw,
        'f_fwd': f1_yz, 'f_rev': f2_yz, 'defls': d_yz,
    })

    # ==================================================================
    #  Process each case
    # ==================================================================
    for case in cases:
        I = case['I']
        tau_max = case['tau_max']
        Kp = case['Kp']
        Kd = case['Kd']
        alpha_max = tau_max / I

        print("-" * 78)
        print(f"  {case['name']}")
        print("-" * 78)
        print()
        print(f"  I = {I:.6f}  tau_max = {tau_max:.6f} N*m  "
              f"alpha_max = {alpha_max:.1f} rad/s^2")
        print(f"  PD threshold: {np.degrees(tau_max/Kp):.2f} deg")
        print()

        # Find optimal lag margin
        print("  Tuning switching curve lag margin ...", flush=True)

        if case['type'] == 'pr':
            def sim_factory(lag_mult, deg,
                            _I=I, _tf=tau_f, _tm=tau_max, _Kp=Kp, _Kd=Kd):
                return simulate_pr(_I, _tf, _tm, _Kp, _Kd,
                                   np.radians(deg), lag_mult=lag_mult)
        else:
            _ff = case['f_fwd']; _fr = case['f_rev']; _dd = case['defls']
            def sim_factory(lag_mult, deg,
                            _thr=thrusters, _I=I, _tf=tau_f, _ts=tau_s,
                            _tm=tau_max, _Kp=Kp, _Kd=Kd,
                            _ff=_ff, _fr=_fr, _dd=_dd):
                return simulate_yaw(_thr, _I, _tf, _ts, _tm,
                                    _ff, _fr, _dd, _Kp, _Kd,
                                    np.radians(deg), lag_mult=lag_mult)

        opt_margin = find_optimal_lag_margin(sim_factory, angles_deg)
        print(f"  Optimal lag margin: {opt_margin:.2f} × tau_f")
        print()

        # Compute bang-bang times
        bb_times = {}
        for deg in angles_deg:
            theta = np.radians(deg)
            if case['type'] == 'pr':
                _, _, _, T_bb, _ = solve_analytical(tau_f, I, tau_max, tau_max, theta)
            else:
                _, _, _, T_bb, _ = solve_yaw_numerical(
                    thrusters, I, tau_f, tau_s,
                    case['f_fwd'], case['f_rev'], case['defls'],
                    tau_max, theta)
            bb_times[deg] = T_bb

        # Results table
        hdr = (f"  {'Angle':>7s}  {'T_settle':>10s}  {'T_bang':>10s}  "
               f"{'Ratio':>7s}  {'Overshoot':>10s}")
        units = (f"  {'':>7s}  {'(ms)':>10s}  {'(ms)':>10s}  "
                 f"{'':>7s}  {'(deg)':>10s}")
        print(hdr)
        print(units)
        print("  " + "-" * 72)

        for deg in angles_deg:
            if case['type'] == 'pr':
                result = simulate_pr(I, tau_f, tau_max, Kp, Kd,
                                     np.radians(deg), lag_mult=opt_margin)
            else:
                result = simulate_yaw(thrusters, I, tau_f, tau_s, tau_max,
                                      case['f_fwd'], case['f_rev'], case['defls'],
                                      Kp, Kd, np.radians(deg), lag_mult=opt_margin)

            T_s = result['settling_time']
            T_bb = bb_times[deg]
            os_d = result['overshoot_deg']

            if T_s is not None:
                ratio = T_s / T_bb
                print(f"  {deg:>5d} deg  {T_s*1000:>10.1f}  {T_bb*1000:>10.1f}  "
                      f"{ratio:>7.2f}  {os_d:>10.4f}")
            else:
                print(f"  {deg:>5d} deg  {'> 2s':>10s}  {T_bb*1000:>10.1f}  "
                      f"{'—':>7s}  {os_d:>10.4f}")

        print()

    # Notes
    print("=" * 78)
    print("  NOTES")
    print("=" * 78)
    print()
    print("  SWITCHING CURVE")
    print("    The deceleration trigger accounts for actuator lag via a margin")
    print("    proportional to omega*tau_f. The margin is tuned by bisection to")
    print("    be the smallest value that prevents overshoot at all angles.")
    print()
    print("  PD SETTLING")
    print("    When error falls below tau_max/Kp, the controller transitions to")
    print("    linear PD with a reference prefilter. The prefilter cancels the")
    print("    closed-loop zero, giving a pure triple-pole step response (no")
    print("    overshoot in the linear regime).")
    print()
    print("  RATIO COLUMN")
    print("    T_settle / T_bang_bang. Ratio > 1.0 means slower than time-optimal.")
    print("    Values near 1.0 for large maneuvers show the switching controller")
    print("    approaches bang-bang performance.")
    print()


if __name__ == '__main__':
    main()
