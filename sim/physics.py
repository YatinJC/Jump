"""
Core physics: quaternion math, force/torque computation, RK4 integration.

Conventions:
  World frame : right-handed, z-up, origin arbitrary.
  Body frame  : origin at center of mass (COM).
  Quaternion  : Hamilton convention, scalar-first [w, x, y, z], represents
                rotation FROM body frame TO world frame.
  Angular vel : expressed in body frame.
  Position    : COM in world frame.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------

def quat_multiply(q1, q2):
    """Hamilton product q1 ⊗ q2.  Both [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_to_rotmat(q):
    """Rotation matrix R such that v_world = R @ v_body.  q = [w, x, y, z]."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def quat_derivative(q, omega_body):
    """Time derivative of q given angular velocity in body frame.
    q_dot = 0.5 * q ⊗ [0, omega]
    """
    omega_q = np.array([0.0, omega_body[0], omega_body[1], omega_body[2]])
    return 0.5 * quat_multiply(q, omega_q)


def quat_normalize(q):
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# Force and torque computation
# ---------------------------------------------------------------------------

def _load_inertia(ball_cfg):
    """Return 3×3 inertia matrix and its inverse from config."""
    moi = ball_cfg['moment_of_inertia']
    if 'matrix' in moi:
        I = np.array(moi['matrix'], dtype=float)
    else:
        I = np.diag(moi['diagonal'])
    return I, np.linalg.inv(I)


def compute_forces_and_accel(state, thruster_array, cfg, external_force_world=None):
    """
    Compute total force on the ball (world frame) and linear acceleration of COM.

    Parameters
    ----------
    state         : np.ndarray shape (13,) — [pos(3), vel(3), quat(4), omega(3)]
    thruster_array: ThrusterArray or None
    cfg           : full config dict

    Returns
    -------
    F_world : (3,) total force in world frame
    accel   : (3,) linear acceleration of COM in world frame
    """
    vel   = state[3:6]
    quat  = state[6:10]
    omega = state[10:13]

    R    = quat_to_rotmat(quat)
    mass = cfg['ball']['mass']
    r    = cfg['ball']['radius']
    pcfg = cfg['physics']

    # --- Gravity ---
    F = np.array([0.0, 0.0, -mass * pcfg['gravity']])

    # --- Translational drag: -0.5 * Cd * rho * A * |v| * v ---
    if pcfg['drag']['enabled']:
        rho  = pcfg['air_density']
        Cd   = pcfg['drag']['coefficient']
        A    = np.pi * r**2
        v_sq = np.dot(vel, vel)
        if v_sq > 1e-20:
            F += -0.5 * Cd * rho * A * np.sqrt(v_sq) * vel

    # --- Magnus: Cl * rho * pi * r^3 * (omega_world × v) ---
    if pcfg['magnus']['enabled']:
        rho       = pcfg['air_density']
        Cl        = pcfg['magnus']['lift_coefficient']
        k_magnus  = Cl * rho * np.pi * r**3
        omega_world = R @ omega
        F += k_magnus * np.cross(omega_world, vel)

    # --- Thruster forces ---
    if thruster_array is not None:
        F_body, _ = thruster_array.get_forces_and_torques()
        F += R @ F_body

    # --- External force (e.g. throw) in world frame ---
    if external_force_world is not None:
        F += external_force_world

    return F, F / mass


def compute_torques_and_alpha(state, thruster_array, cfg, external_torque_world=None):
    """
    Compute total torque on the ball (body frame) and angular acceleration.

    Returns
    -------
    tau   : (3,) total torque in body frame
    alpha : (3,) angular acceleration in body frame
    """
    omega = state[10:13]
    I, I_inv = _load_inertia(cfg['ball'])
    pcfg = cfg['physics']

    tau = np.zeros(3)

    # --- Thruster torques ---
    if thruster_array is not None:
        _, tau_body = thruster_array.get_forces_and_torques()
        tau += tau_body

    # --- Rotational drag: -k * omega ---
    if pcfg['rotational_drag']['enabled']:
        tau += -pcfg['rotational_drag']['coefficient'] * omega

    # --- External torque (e.g. throw) converted from world to body frame ---
    if external_torque_world is not None:
        R = quat_to_rotmat(state[6:10])
        tau += R.T @ external_torque_world

    # --- Euler's equation: I*alpha = tau - omega × (I*omega) ---
    alpha = I_inv @ (tau - np.cross(omega, I @ omega))

    return tau, alpha


# ---------------------------------------------------------------------------
# State derivative and RK4
# ---------------------------------------------------------------------------

def state_derivative(state, thruster_array, cfg,
                     external_force_world=None, external_torque_world=None):
    """
    Compute d(state)/dt.

    State layout: [pos(3), vel(3), quat(4), omega(3)]

    Returns
    -------
    dstate : (13,)
    accel  : (3,)  linear acceleration of COM in world frame
    alpha  : (3,)  angular acceleration in body frame
    """
    vel   = state[3:6]
    quat  = state[6:10]
    omega = state[10:13]

    _, accel = compute_forces_and_accel(state, thruster_array, cfg, external_force_world)
    _, alpha = compute_torques_and_alpha(state, thruster_array, cfg, external_torque_world)
    q_dot   = quat_derivative(quat, omega)

    dstate = np.concatenate([vel, accel, q_dot, alpha])
    return dstate, accel, alpha


def rk4_step(state, dt, thruster_array, cfg,
             external_force_world=None, external_torque_world=None):
    """
    Advance state by dt using RK4.
    Thruster and throw forces are held constant across the step (piecewise constant).

    Returns
    -------
    new_state : (13,)
    accel     : (3,)  linear acceleration at the start of the step
    alpha     : (3,)  angular acceleration at the start of the step
    """
    ef = external_force_world
    et = external_torque_world

    k1, accel, alpha = state_derivative(state,             thruster_array, cfg, ef, et)
    k2, _,     _     = state_derivative(state + 0.5*dt*k1, thruster_array, cfg, ef, et)
    k3, _,     _     = state_derivative(state + 0.5*dt*k2, thruster_array, cfg, ef, et)
    k4, _,     _     = state_derivative(state +     dt*k3,  thruster_array, cfg, ef, et)

    new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # Renormalize quaternion to prevent drift
    new_state[6:10] = quat_normalize(new_state[6:10])

    return new_state, accel, alpha


# ---------------------------------------------------------------------------
# Initial state builder
# ---------------------------------------------------------------------------

def build_initial_state(cfg):
    """Construct the 13-element state vector.  Ball starts from rest."""
    ic  = cfg['initial_conditions']
    pos  = np.array(ic['position'],   dtype=float)
    quat = np.array(ic['quaternion'], dtype=float)
    quat = quat_normalize(quat)
    return np.concatenate([pos, np.zeros(3), quat, np.zeros(3)])
