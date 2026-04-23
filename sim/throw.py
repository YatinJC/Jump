"""
Throw phase model.

Applies an external force (world frame) and torque (world frame) to the ball
over a configurable duration starting from t=0.  Both magnitude and direction
are fixed; only the magnitude envelope varies over time.

Force and torque directions are specified in the world frame and normalized
internally.  For a short throw duration the ball rotates little, so world-frame
direction ≈ body-frame direction at release — but the conversion to body frame
is done correctly at every timestep regardless.

Profile shapes
--------------
constant   : full magnitude for the entire duration
trapezoid  : linear ramp-up → hold → linear ramp-down
             ramp_fraction sets the fraction of duration spent on each ramp
gaussian   : bell curve, peak at the midpoint, ±3σ fits within [0, duration]
"""

import numpy as np


def _envelope(t: float, duration: float, profile: str, ramp_fraction: float) -> float:
    """Return scalar multiplier in [0, 1] for time t within [0, duration]."""
    if t <= 0 or t >= duration:
        return 0.0
    tau = t / duration
    if profile == 'constant':
        return 1.0
    elif profile == 'trapezoid':
        r = float(ramp_fraction)
        if tau < r:
            return tau / r
        elif tau < 1.0 - r:
            return 1.0
        else:
            return (1.0 - tau) / r
    elif profile == 'gaussian':
        mu    = 0.5
        sigma = 1.0 / 6.0   # 3σ at each end → nearly zero at boundaries
        return float(np.exp(-0.5 * ((tau - mu) / sigma) ** 2))
    else:
        raise ValueError(f"Unknown throw profile '{profile}'. "
                         "Choose from: constant, trapezoid, gaussian")


def _avg_envelope(duration: float, profile: str, ramp_fraction: float,
                   n_samples: int = 200) -> float:
    """Average value of the envelope over [0, duration]."""
    ts = np.linspace(0, duration, n_samples)
    vals = [_envelope(t, duration, profile, ramp_fraction) for t in ts]
    return float(np.mean(vals))


class ThrowPhase:
    """Computes throw force and torque (both world frame) as a function of time.

    Supports two config modes:

    1. Direct (original): specify force direction, peak_magnitude, profile.
    2. End-state: specify final_velocity [vx,vy,vz] and final_omega [wx,wy,wz].
       The peak force and torque are computed automatically from the desired
       end-state, ball mass/inertia, throw duration, and profile shape.
       Gravity is subtracted from the required force.

    End-state mode is activated when 'final_velocity' is present in the config.
    """

    def __init__(self, throw_cfg: dict, ball_cfg: dict | None = None):
        self.duration = float(throw_cfg['duration'])

        if 'final_velocity' in throw_cfg or 'final_speed' in throw_cfg:
            # --- End-state mode ---
            if ball_cfg is None:
                raise ValueError("ball_cfg required for end-state throw mode")

            mass = float(ball_cfg['mass'])
            inertia = np.array(ball_cfg['moment_of_inertia']['matrix'], dtype=float)
            gravity = np.array([0.0, 0.0, -9.81])

            # Force profile
            fcfg = throw_cfg.get('force', {})
            self.force_profile = str(fcfg.get('profile', 'gaussian'))
            self.force_ramp    = float(fcfg.get('ramp_fraction', 0.2))
            avg_f = _avg_envelope(self.duration, self.force_profile, self.force_ramp)

            if 'final_velocity' in throw_cfg:
                v_final = np.array(throw_cfg['final_velocity'], dtype=float)
            else:
                speed = float(throw_cfg['final_speed'])
                direction = np.array(throw_cfg['direction'], dtype=float)
                direction = direction / np.linalg.norm(direction)
                v_final = speed * direction
            # impulse = m * v_final, but gravity contributes m*g*duration
            # so F_avg * duration = m * v_final - m * g * duration
            # F_avg = m * (v_final - g * duration) / duration
            # F_peak = F_avg / avg_envelope
            required_impulse = mass * v_final - mass * gravity * self.duration
            F_avg = required_impulse / self.duration
            F_avg_mag = np.linalg.norm(F_avg)
            if F_avg_mag > 1e-6:
                self.force_dir  = F_avg / F_avg_mag
                self.force_peak = float(F_avg_mag / avg_f)
            else:
                self.force_dir  = np.array([0.0, 0.0, 1.0])
                self.force_peak = 0.0

            # Torque profile
            tcfg = throw_cfg.get('torque', {})
            self.torque_profile = str(tcfg.get('profile', 'gaussian'))
            self.torque_ramp    = float(tcfg.get('ramp_fraction', 0.2))
            avg_t = _avg_envelope(self.duration, self.torque_profile, self.torque_ramp)

            if 'final_omega' in throw_cfg:
                omega_final = np.array(throw_cfg['final_omega'], dtype=float)
            elif 'spin_rate' in throw_cfg:
                spin_rate = float(throw_cfg['spin_rate'])
                spin_axis = np.array(throw_cfg.get('spin_axis', [0, 0, 1]), dtype=float)
                spin_axis = spin_axis / np.linalg.norm(spin_axis)
                omega_final = spin_rate * spin_axis
            else:
                omega_final = np.zeros(3)
            required_angular_impulse = inertia @ omega_final
            tau_avg = required_angular_impulse / self.duration
            tau_avg_mag = np.linalg.norm(tau_avg)
            if tau_avg_mag > 1e-6:
                self.torque_axis = tau_avg / tau_avg_mag
                self.torque_peak = float(tau_avg_mag / avg_t)
            else:
                self.torque_axis = np.array([0.0, 0.0, 1.0])
                self.torque_peak = 0.0
        else:
            # --- Direct mode (original) ---
            fcfg = throw_cfg['force']
            fdir = np.array(fcfg['direction'], dtype=float)
            self.force_dir      = fdir / np.linalg.norm(fdir)
            self.force_peak     = float(fcfg['peak_magnitude'])
            self.force_profile  = str(fcfg.get('profile', 'trapezoid'))
            self.force_ramp     = float(fcfg.get('ramp_fraction', 0.2))

            tcfg = throw_cfg['torque']
            taxis = np.array(tcfg['axis'], dtype=float)
            self.torque_axis    = taxis / np.linalg.norm(taxis)
            self.torque_peak    = float(tcfg['peak_magnitude'])
            self.torque_profile = str(tcfg.get('profile', 'trapezoid'))
            self.torque_ramp    = float(tcfg.get('ramp_fraction', 0.2))

    def is_active(self, t: float) -> bool:
        return 0.0 <= t < self.duration

    def force_world(self, t: float) -> np.ndarray:
        """External throw force in world frame at time t (N)."""
        scale = _envelope(t, self.duration, self.force_profile, self.force_ramp)
        return self.force_peak * scale * self.force_dir

    def torque_world(self, t: float) -> np.ndarray:
        """External throw torque in world frame at time t (N·m)."""
        scale = _envelope(t, self.duration, self.torque_profile, self.torque_ramp)
        return self.torque_peak * scale * self.torque_axis
