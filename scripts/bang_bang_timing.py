#!/usr/bin/env python3
"""
Bang-bang rest-to-rest orientation control timing for the Jump ball,
with thruster and vectoring servo dynamics.

Three-phase bang-bang for exact rest-to-rest without oscillation:
  Phase 1 [0, d1]:          command +tau  (accelerate)
  Phase 2 [d1, d1+d2]:      command -tau  (decelerate)
  Phase 3 [d1+d2, d1+d2+d3]: command +tau  (zero the actuator state)

Yaw: thrust vectoring with servos starting from 0 degrees.  Torque reversal
uses force-only switching (swap forces between diagonal pairs, deflection
commands stay constant).  The servo ramp reduces torque during early phase 1
because moment-arm torque ~ F*sin(delta).  This is handled by a numerical
simulation of phases 1 & 2 (coupled force/servo dynamics), with phase 3
solved analytically (servos are settled by then).

Pitch/roll: differential thrust, no vectoring.  Fully analytical solver.
"""

import numpy as np
import yaml
import os


def load_config():
    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default.yaml')
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def parse_thrusters(cfg):
    """Parse thruster geometry from config."""
    com = np.array(cfg['ball']['center_of_mass'], dtype=float)
    thrusters = []
    for tcfg in cfg['thrusters']:
        pos_geom = np.array(tcfg['position'], dtype=float)
        d_nom = np.array(tcfg['direction'], dtype=float)
        d_nom /= np.linalg.norm(d_nom)
        pos_hat = pos_geom / np.linalg.norm(pos_geom)

        vcfg = tcfg.get('vectoring', {})
        ga = vcfg.get('gimbal_axis', 'radial')
        if ga == 'radial':
            dot = np.dot(pos_hat, d_nom)
            raw = pos_hat - dot * d_nom
        elif ga == 'tangential':
            raw = np.cross(d_nom, pos_hat)
        else:
            raise ValueError(f"Unknown gimbal_axis: {ga}")
        gimbal = raw / np.linalg.norm(raw)
        swing = np.cross(gimbal, d_nom)
        swing /= np.linalg.norm(swing)

        thrusters.append({
            'pos_from_com': pos_geom - com,
            'nom': d_nom,
            'spin': float(tcfg['spin_direction']),
            'k_q': float(tcfg['torque_to_thrust_ratio']),
            'swing': swing,
            'f_min': float(tcfg['min_force']),
            'f_max': float(tcfg['max_force']),
            'max_defl': np.radians(float(vcfg.get('max_deflection', 20))),
        })
    return thrusters


def compute_wrench(thrusters, forces, deflections):
    """Compute body-frame wrench [Fx, Fy, Fz, taux, tauy, tauz]."""
    F_total = np.zeros(3)
    tau_total = np.zeros(3)
    for i, t in enumerate(thrusters):
        d = np.cos(deflections[i]) * t['nom'] + np.sin(deflections[i]) * t['swing']
        F_vec = forces[i] * d
        F_total += F_vec
        tau_total += np.cross(t['pos_from_com'], F_vec)
        tau_total += t['k_q'] * forces[i] * t['spin'] * (-d)
    return np.concatenate([F_total, tau_total])


# ---------------------------------------------------------------------------
#  Analytical phase evaluation (shared by both solvers)
# ---------------------------------------------------------------------------

def phase_end(d, tf, I, U, tau0, w0, th0):
    """
    Analytical end state after a constant-command phase.
    System: dtau/dt = (U - tau)/tf,  dw/dt = tau/I,  dth/dt = w
    """
    if d < 1e-15:
        return tau0, w0, th0
    e = np.exp(-d / tf)
    a = tau0 - U
    tau = a * e + U
    J1 = a * tf * (1 - e) + U * d
    J2 = a * tf * (d - tf * (1 - e)) + U * d**2 / 2
    return tau, w0 + J1 / I, th0 + w0 * d + J2 / I


# ---------------------------------------------------------------------------
#  Analytical three-phase solver (pitch/roll — no servo dynamics)
# ---------------------------------------------------------------------------

def solve_analytical(tf, I, U_pos, U_neg, theta_des):
    """
    Three-phase bang-bang with first-order torque lag only.
    Returns: (d1, d2, d3, T, omega_peak)
    """
    def eval_d1(d1):
        tau1, w1, th1 = phase_end(d1, tf, I, U_pos, 0, 0, 0)

        def omega_final(d2):
            tau2, _, _ = phase_end(d2, tf, I, -U_neg, tau1, w1, th1)
            if tau2 >= 0:
                return 1e6
            d3 = tf * np.log((U_pos - tau2) / U_pos)
            _, w3, _ = phase_end(d3, tf, I, U_pos, tau2,
                                 *phase_end(d2, tf, I, -U_neg, tau1, w1, th1)[1:])
            return w3

        lo, hi = 1e-9, 10.0
        for _ in range(70):
            mid = (lo + hi) / 2
            tau2, w2, th2 = phase_end(mid, tf, I, -U_neg, tau1, w1, th1)
            if tau2 >= 0:
                lo = mid
                continue
            d3 = tf * np.log((U_pos - tau2) / U_pos)
            _, w3, _ = phase_end(d3, tf, I, U_pos, tau2, w2, th2)
            if w3 > 0:
                lo = mid
            else:
                hi = mid
        d2 = (lo + hi) / 2

        tau2, w2, th2 = phase_end(d2, tf, I, -U_neg, tau1, w1, th1)
        d3 = tf * np.log((U_pos - tau2) / U_pos) if tau2 < 0 else 0
        _, _, th3 = phase_end(d3, tf, I, U_pos, tau2, w2, th2)
        return d2, d3, th3

    lo, hi = 1e-9, 10.0
    for _ in range(70):
        mid = (lo + hi) / 2
        _, _, th = eval_d1(mid)
        if th < theta_des:
            lo = mid
        else:
            hi = mid
    d1 = (lo + hi) / 2
    d2, d3, _ = eval_d1(d1)
    T = d1 + d2 + d3

    # Peak omega: in phase 2 when tau crosses zero
    tau1, w1, _ = phase_end(d1, tf, I, U_pos, 0, 0, 0)
    if tau1 > 0 and U_neg > 0:
        t_zc = tf * np.log((tau1 + U_neg) / U_neg)
        if t_zc < d2:
            _, w_peak, _ = phase_end(t_zc, tf, I, -U_neg, tau1, w1, 0)
        else:
            w_peak = w1
    else:
        w_peak = w1

    return d1, d2, d3, T, w_peak


# ---------------------------------------------------------------------------
#  Numerical three-phase solver (yaw — coupled force + servo dynamics)
# ---------------------------------------------------------------------------

def solve_yaw_numerical(thrusters, I, tau_f, tau_s,
                        forces_fwd, forces_rev, defls_target,
                        tau_fwd_ss, theta_des,
                        dt=0.0005, n_bisect=30):
    """
    Three-phase bang-bang for yaw with servo ramp from 0.

    F(t) and delta(t) are computed analytically (exponential responses to
    piecewise-constant commands), then torque, omega, and theta are computed
    via vectorized numpy operations.  Phase 3 is analytical (servos settled).

    tau_fwd_ss: steady-state forward torque magnitude (for analytical d3).
    """
    n = len(thrusters)

    # Pre-extract geometry as column vectors for broadcasting
    _px = np.array([t['pos_from_com'][0] for t in thrusters])
    _py = np.array([t['pos_from_com'][1] for t in thrusters])
    _nx = np.array([t['nom'][0] for t in thrusters])
    _ny = np.array([t['nom'][1] for t in thrusters])
    _nz = np.array([t['nom'][2] for t in thrusters])
    _sx = np.array([t['swing'][0] for t in thrusters])
    _sy = np.array([t['swing'][1] for t in thrusters])
    _sz = np.array([t['swing'][2] for t in thrusters])
    _kq = np.array([t['k_q'] for t in thrusters])
    _sp = np.array([t['spin'] for t in thrusters])

    f_fwd = np.array(forces_fwd, dtype=float)
    f_rev = np.array(forces_rev, dtype=float)
    d_tgt = np.array(defls_target, dtype=float)

    U = tau_fwd_ss

    def compute_tz(F, D):
        """Yaw torque from (n,N) force and deflection arrays."""
        cd = np.cos(D); sd = np.sin(D)
        dx = cd * _nx[:, None] + sd * _sx[:, None]
        dy = cd * _ny[:, None] + sd * _sy[:, None]
        dz = cd * _nz[:, None] + sd * _sz[:, None]
        tz = np.sum(_px[:, None] * (F * dy) - _py[:, None] * (F * dx), axis=0)
        tz += np.sum(_kq[:, None] * F * _sp[:, None] * (-dz), axis=0)
        return tz

    def sim_phase1(d1):
        """Phase 1: F ramps to f_fwd, servos ramp from 0 to d_tgt.
        Returns (F1, D1, omega1, theta1, omega_peak)."""
        N = max(2, int(round(d1 / dt)))
        h = d1 / N
        t = np.linspace(0, d1, N + 1)

        F = f_fwd[:, None] * (1 - np.exp(-t[None, :] / tau_f))
        D = d_tgt[:, None] * (1 - np.exp(-t[None, :] / tau_s))
        tz = compute_tz(F, D)

        alpha = tz / I
        # Euler integration: omega[k] = h * sum(alpha[0:k])
        omega = np.empty(N + 1)
        omega[0] = 0.0
        omega[1:] = np.cumsum(alpha[:N]) * h

        theta1 = np.sum(omega[:N]) * h
        return F[:, -1].copy(), D[:, -1].copy(), omega[-1], theta1, np.max(np.abs(omega))

    def sim_phase2(F1, D1, w1, th1, d2):
        """Phase 2: F transitions to f_rev, servos continue toward d_tgt.
        Returns (tz_end, omega_end, theta_end, omega_peak)."""
        N = max(2, int(round(d2 / dt)))
        h = d2 / N
        t = np.linspace(0, d2, N + 1)

        etf = np.exp(-t[None, :] / tau_f)
        ets = np.exp(-t[None, :] / tau_s)
        F = (F1 - f_rev)[:, None] * etf + f_rev[:, None]
        D = (D1 - d_tgt)[:, None] * ets + d_tgt[:, None]
        tz = compute_tz(F, D)

        alpha = tz / I
        omega = np.empty(N + 1)
        omega[0] = w1
        omega[1:] = w1 + np.cumsum(alpha[:N]) * h

        theta_end = th1 + np.sum(omega[:N]) * h
        return tz[-1], omega[-1], theta_end, np.max(np.abs(omega))

    def eval_d1(d1):
        """Simulate phase 1, bisect d2 for omega=0. Return (d2, d3, theta, wpk)."""
        F1, D1, w1, th1, wp1 = sim_phase1(d1)

        lo, hi = 1e-6, 5.0
        for _ in range(n_bisect):
            mid = (lo + hi) / 2
            tz2, w2, th2, _ = sim_phase2(F1, D1, w1, th1, mid)
            if tz2 >= 0:
                lo = mid
                continue
            d3 = tau_f * np.log((U - tz2) / U)
            _, w3, _ = phase_end(d3, tau_f, I, U, tz2, w2, th2)
            if w3 > 0:
                lo = mid
            else:
                hi = mid
        d2 = (lo + hi) / 2

        tz2, w2, th2, wp2 = sim_phase2(F1, D1, w1, th1, d2)
        wpk = max(wp1, wp2)
        if tz2 < 0:
            d3 = tau_f * np.log((U - tz2) / U)
            _, _, th3 = phase_end(d3, tau_f, I, U, tz2, w2, th2)
        else:
            d3 = 0.0
            th3 = th2
        return d2, d3, th3, wpk

    # Outer bisection on d1 for theta
    lo, hi = 1e-6, 5.0
    for _ in range(n_bisect):
        mid = (lo + hi) / 2
        _, _, th, _ = eval_d1(mid)
        if th < theta_des:
            lo = mid
        else:
            hi = mid
    d1 = (lo + hi) / 2
    d2, d3, _, wpk = eval_d1(d1)
    T = d1 + d2 + d3

    return d1, d2, d3, T, wpk


# ---------------------------------------------------------------------------
#  Configuration search
# ---------------------------------------------------------------------------

def find_max_yaw_config(thrusters):
    delta_max = thrusters[0]['max_defl']
    f_max = thrusters[0]['f_max']
    f_min = thrusters[0]['f_min']
    best = (0.0, None)
    for fa in [f_max, f_min]:
        for fb in [f_max, f_min]:
            if fa == fb:
                continue
            for sa in [+1, -1]:
                for sb in [+1, -1]:
                    forces = [fa, fb, fb, fa]
                    defls = [sa * delta_max, sb * delta_max,
                             sb * delta_max, sa * delta_max]
                    w = compute_wrench(thrusters, forces, defls)
                    if abs(w[5]) > abs(best[0]):
                        best = (w[5], (forces, defls, w))
    return best[1]


def find_max_yaw_zero_z(thrusters):
    delta_max = thrusters[0]['max_defl']
    f_cap = min(thrusters[0]['f_max'], abs(thrusters[0]['f_min']))
    best = (0.0, None)
    for sf_a in [+1, -1]:
        for sf_b in [+1, -1]:
            if sf_a == sf_b:
                continue
            for sd_a in [+1, -1]:
                for sd_b in [+1, -1]:
                    forces = [sf_a * f_cap, sf_b * f_cap,
                              sf_b * f_cap, sf_a * f_cap]
                    defls = [sd_a * delta_max, sd_b * delta_max,
                             sd_b * delta_max, sd_a * delta_max]
                    w = compute_wrench(thrusters, forces, defls)
                    if abs(w[5]) > abs(best[0]) and abs(w[2]) < 1e-6:
                        best = (w[5], (forces, defls, w))
    return best[1]


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    I_diag = np.array(cfg['ball']['moment_of_inertia']['diagonal'])
    Ixx, Iyy, Izz = I_diag
    com = np.array(cfg['ball']['center_of_mass'])
    thrusters = parse_thrusters(cfg)

    tau_f = float(cfg['thrusters'][0]['time_constant'])
    tau_s = float(cfg['thrusters'][0]['vectoring']['time_constant'])
    f_max = thrusters[0]['f_max']
    f_min = thrusters[0]['f_min']
    f_cap = min(f_max, abs(f_min))
    delta_max = thrusters[0]['max_defl']

    print("=" * 78)
    print("  BANG-BANG REST-TO-REST TIMING WITH ACTUATOR DYNAMICS")
    print("=" * 78)
    print()
    print(f"  MOI: Ixx={Ixx:.6f}  Iyy={Iyy:.6f}  Izz={Izz:.6f}  kg*m^2")
    print(f"  Force range: [{f_min:.3f}, {f_max:.3f}] N")
    print(f"  Vectoring: {np.degrees(delta_max):.0f} deg max")
    print(f"  Force time constant:  tau_f = {tau_f*1000:.1f} ms")
    print(f"  Servo time constant:  tau_s = {tau_s*1000:.1f} ms")
    print()

    print("  Thruster geometry (body frame, from COM):")
    for i, t in enumerate(thrusters):
        p = t['pos_from_com']
        print(f"    T{i}: pos=[{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}]  "
              f"spin={t['spin']:+.0f}  "
              f"swing=[{t['swing'][0]:+.3f}, {t['swing'][1]:+.3f}, {t['swing'][2]:+.3f}]")
    print()

    angles_deg = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180]

    def print_table_header():
        hdr = (f"  {'Angle':>7s}  {'d1':>8s}  {'d2':>8s}  {'d3':>8s}  "
               f"{'T':>8s}  {'T_ideal':>8s}  {'dT':>7s}  {'w_pk':>10s}")
        units = (f"  {'':>7s}  {'(ms)':>8s}  {'(ms)':>8s}  {'(ms)':>8s}  "
                 f"{'(ms)':>8s}  {'(ms)':>8s}  {'(ms)':>7s}  {'(deg/s)':>10s}")
        print(hdr)
        print(units)
        print("  " + "-" * 74)

    def print_row(deg, d1, d2, d3, T, T_ideal, w_peak):
        dT = (T - T_ideal) * 1000
        print(f"  {deg:>5d} deg  {d1*1000:>8.2f}  {d2*1000:>8.2f}  {d3*1000:>8.2f}  "
              f"{T*1000:>8.2f}  {T_ideal*1000:>8.2f}  {dT:>+7.2f}  "
              f"{np.degrees(w_peak):>10.1f}")

    def print_example(d1, d2, d3, T, f_fwd, f_rev, defls=None):
        t1 = d1 * 1000
        t2 = (d1 + d2) * 1000
        t3 = T * 1000
        print(f"    Phase 1 [0, {t1:.1f} ms]:       "
              f"forces = [{', '.join(f'{f:+.3f}' for f in f_fwd)}]")
        print(f"    Phase 2 [{t1:.1f}, {t2:.1f} ms]: "
              f"forces = [{', '.join(f'{f:+.3f}' for f in f_rev)}]")
        print(f"    Phase 3 [{t2:.1f}, {t3:.1f} ms]: "
              f"forces = [{', '.join(f'{f:+.3f}' for f in f_fwd)}]")
        print(f"    After {t3:.1f} ms:               "
              f"forces = [0, 0, 0, 0]")
        if defls is not None and any(d != 0 for d in defls):
            print(f"    Deflections: command [{', '.join(f'{np.degrees(d):+.1f}' for d in defls)}] deg")
            print(f"    from t=0 (servos start at 0 deg, ramp with tau_s={tau_s*1000:.0f} ms)")

    # ==================================================================
    #  YAW CASES (numerical — servo ramp from 0)
    # ==================================================================
    yaw_configs = []

    f1_yaw, d_yaw, w1_yaw = find_max_yaw_config(thrusters)
    f2_yaw = [f1_yaw[1], f1_yaw[0], f1_yaw[3], f1_yaw[2]]
    w2_yaw = compute_wrench(thrusters, f2_yaw, d_yaw)
    # Ensure forward config has positive tau_z
    if w1_yaw[5] < 0:
        f1_yaw, f2_yaw = f2_yaw, f1_yaw
        w1_yaw, w2_yaw = w2_yaw, w1_yaw
    yaw_configs.append(('Yaw — max torque (residual Fz)',
                        f1_yaw, f2_yaw, d_yaw, w1_yaw, w2_yaw))

    f1_yz, d_yz, w1_yz = find_max_yaw_zero_z(thrusters)
    f2_yz = [f1_yz[1], f1_yz[0], f1_yz[3], f1_yz[2]]
    w2_yz = compute_wrench(thrusters, f2_yz, d_yz)
    # Ensure forward config has positive tau_z
    if w1_yz[5] < 0:
        f1_yz, f2_yz = f2_yz, f1_yz
        w1_yz, w2_yz = w2_yz, w1_yz
    yaw_configs.append(('Yaw — zero Z force',
                        f1_yz, f2_yz, d_yz, w1_yz, w2_yz))

    for (name, f_fwd, f_rev, defls, w_fwd, w_rev) in yaw_configs:
        tau_fwd = abs(w_fwd[5])
        tau_rev = abs(w_rev[5])

        print("-" * 78)
        print(f"  {name}")
        print("-" * 78)
        print()
        print(f"  Forward:  forces=[{', '.join(f'{f:+.3f}' for f in f_fwd)}]  "
              f"defl=[{', '.join(f'{np.degrees(d):+.1f}' for d in defls)}] deg")
        print(f"            tau_z={w_fwd[5]:+.6f} N*m   Fz={w_fwd[2]:+.4f} N")
        print(f"  Reverse:  forces=[{', '.join(f'{f:+.3f}' for f in f_rev)}]  (same deflections)")
        print(f"            tau_z={w_rev[5]:+.6f} N*m   Fz={w_rev[2]:+.4f} N")
        print(f"  Torque symmetry: {abs(tau_fwd - tau_rev)/max(tau_fwd,tau_rev)*100:.4f}%")
        print()
        print(f"  I = {Izz:.6f} kg*m^2   tau_max = {tau_fwd:.6f} N*m   "
              f"alpha = {tau_fwd/Izz:.1f} rad/s^2")
        print(f"  Servos start from 0 deg (ramp included in timing)")
        print(f"  Force-only switching at phase transitions")
        print()
        print(f"  Three-phase: {{+tau, -tau, +tau}}  (numerical, servo ramp from 0)")
        print()
        print_table_header()

        for deg in angles_deg:
            theta = np.radians(deg)
            T_ideal = 2.0 * np.sqrt(theta * Izz / tau_fwd)

            d1, d2, d3, T, wpk = solve_yaw_numerical(
                thrusters, Izz, tau_f, tau_s,
                f_fwd, f_rev, defls, tau_fwd, theta)

            print_row(deg, d1, d2, d3, T, T_ideal, wpk)

        print()

        # Example for 90 deg
        d1, d2, d3, T, _ = solve_yaw_numerical(
            thrusters, Izz, tau_f, tau_s,
            f_fwd, f_rev, defls, tau_fwd, np.radians(90))
        print(f"  Example command sequence for 90 deg maneuver:")
        print_example(d1, d2, d3, T, f_fwd, f_rev, defls)
        print()

    # ==================================================================
    #  PITCH/ROLL CASES (analytical — no servo dynamics)
    # ==================================================================
    d_zero = [0, 0, 0, 0]

    f1_roll = [f_max, f_min, f_max, f_min]
    f2_roll = [f_min, f_max, f_min, f_max]
    w1_roll = compute_wrench(thrusters, f1_roll, d_zero)
    w2_roll = compute_wrench(thrusters, f2_roll, d_zero)

    f1_rz = [f_cap, -f_cap, f_cap, -f_cap]
    f2_rz = [-f_cap, f_cap, -f_cap, f_cap]
    w1_rz = compute_wrench(thrusters, f1_rz, d_zero)
    w2_rz = compute_wrench(thrusters, f2_rz, d_zero)

    # Verify pitch == roll
    f_p = [f_max, f_max, f_min, f_min]
    w_p = compute_wrench(thrusters, f_p, d_zero)
    print(f"  Pitch/Roll symmetry: |tau_x| = {abs(w1_roll[3]):.6f}, "
          f"|tau_y| = {abs(w_p[4]):.6f}  =>  "
          f"{'MATCH' if abs(abs(w_p[4]) - abs(w1_roll[3])) < 1e-9 else 'MISMATCH'}")
    print()

    pr_cases = [
        ('Pitch/Roll — max torque (residual Fz)', 0,
         f1_roll, f2_roll, w1_roll, w2_roll),
        ('Pitch/Roll — zero Z force', 0,
         f1_rz, f2_rz, w1_rz, w2_rz),
    ]

    for (name, ax, f_fwd, f_rev, w_fwd, w_rev) in pr_cases:
        tau_fwd = abs(w_fwd[3 + ax])
        tau_rev = abs(w_rev[3 + ax])

        print("-" * 78)
        print(f"  {name}")
        print("-" * 78)
        print()
        print(f"  Forward:  forces=[{', '.join(f'{f:+.3f}' for f in f_fwd)}]")
        print(f"            tau_x={w_fwd[3+ax]:+.6f} N*m   Fz={w_fwd[2]:+.4f} N")
        print(f"  Reverse:  forces=[{', '.join(f'{f:+.3f}' for f in f_rev)}]")
        print(f"            tau_x={w_rev[3+ax]:+.6f} N*m   Fz={w_rev[2]:+.4f} N")
        print()
        print(f"  I = {Ixx:.6f} kg*m^2   tau_max = {tau_fwd:.6f} N*m   "
              f"alpha = {tau_fwd/Ixx:.1f} rad/s^2")
        print(f"  No vectoring — analytical solver (force lag only)")
        print()
        print(f"  Three-phase: {{+tau, -tau, +tau}}  (analytical)")
        print()
        print_table_header()

        for deg in angles_deg:
            theta = np.radians(deg)
            T_ideal = 2.0 * np.sqrt(theta * Ixx / tau_fwd)
            d1, d2, d3, T, wpk = solve_analytical(
                tau_f, Ixx, tau_fwd, tau_rev, theta)
            print_row(deg, d1, d2, d3, T, T_ideal, wpk)

        print()

        # Verify
        d1, d2, d3, T, _ = solve_analytical(
            tau_f, Ixx, tau_fwd, tau_rev, np.radians(90))
        t1, w1, th1 = phase_end(d1, tau_f, Ixx, tau_fwd, 0, 0, 0)
        t2, w2, th2 = phase_end(d2, tau_f, Ixx, -tau_rev, t1, w1, th1)
        t3, w3, th3 = phase_end(d3, tau_f, Ixx, tau_fwd, t2, w2, th2)
        print(f"  Verification (90 deg): tau_end={t3:.2e} N*m, "
              f"omega_end={np.degrees(w3):.4e} deg/s, "
              f"theta_end={np.degrees(th3):.6f} deg")
        print()

        print(f"  Example command sequence for 90 deg maneuver:")
        print_example(d1, d2, d3, T, f_fwd, f_rev)
        print()

    # ==================================================================
    #  Notes
    # ==================================================================
    print("=" * 78)
    print("  NOTES")
    print("=" * 78)
    print()
    print("  THREE-PHASE BANG-BANG")
    print("    Phase 3 brings the actuator torque to zero at the same instant")
    print("    omega reaches zero. Without it, residual torque causes drift:")
    for label, I_val, tau_val in [("Pitch/Roll", Ixx, abs(w1_roll[3])),
                                   ("Yaw", Izz, abs(w1_yaw[5]))]:
        drift = tau_val * tau_f / I_val
        print(f"      {label}: {np.degrees(drift):.1f} deg/s drift — NOT acceptable")
    print()
    print("  YAW SERVO RAMP")
    print(f"    Servos start from 0 and ramp with tau_s = {tau_s*1000:.0f} ms.")
    print("    Moment-arm torque ~ F*sin(delta) builds up slower than force alone.")
    print("    The numerical solver captures this nonlinear coupling.")
    print("    For small angles (d1 < ~3*tau_s), the servo ramp significantly")
    print("    extends the maneuver vs. the ideal case.")
    print()
    print("  FORCE-ONLY SWITCHING")
    print("    Yaw torque reversal swaps force commands between diagonal pairs,")
    print("    keeping deflection commands constant. Only force lag governs")
    print("    the phase transitions.")
    print()


if __name__ == '__main__':
    main()
