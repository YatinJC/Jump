"""
IncrementalEstimator — online IMU integration, one sample at a time.

Mirrors the batch integrate_imu() pipeline in fuse_and_plot.py but processes
each (accel, gyro) pair as it arrives, suitable for a real-time control loop.

State
-----
  q  : (4,)   quaternion [w,x,y,z], body-to-world
  v  : (3,)   velocity of COM in world frame (m/s)
  p  : (3,)   position of COM in world frame (m)
  t  : float  timestamp of the last update (s)

Integration scheme
------------------
  Orientation : RK4 quaternion kinematics (zero-order hold on current gyro)
  Velocity    : trapezoid rule
  Position    : trapezoid rule
  α (angular accel) : backward finite difference on gyro  (α = Δω/Δt)
                      — first sample uses α = 0 (no prior available)

Lever-arm correction (full, matching batch script)
------------------------------------------------------
  centripetal : ω × (ω × p_imu)
  tangential  : α × p_imu
Both are in body frame and subtracted after rotating to world frame.

Usage
-----
    est = IncrementalEstimator(q0, p0, v0, gravity=9.81, imu_pos_from_com=p_imu)

    for t, accel_b, gyro_b in imu_stream:
        est.update(t, accel_b, gyro_b)
        R, v, p = est.R, est.v, est.p
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.physics import quat_multiply, quat_normalize, quat_to_rotmat


def _q_dot(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Quaternion kinematic derivative: dq/dt = 0.5 * q ⊗ [0, ω]."""
    return 0.5 * quat_multiply(q, np.array([0.0, omega[0], omega[1], omega[2]]))


class IncrementalEstimator:
    """
    Online IMU dead-reckoning estimator.

    Parameters
    ----------
    q0              : (4,)  initial quaternion [w,x,y,z], body-to-world
    p0              : (3,)  initial COM position in world frame (m)
    v0              : (3,)  initial velocity in world frame (m/s)
    gravity         : float gravitational acceleration magnitude (m/s²)
    imu_pos_from_com: (3,)  IMU offset from COM in body frame (m)
    """

    def __init__(
        self,
        q0: np.ndarray,
        p0: np.ndarray,
        v0: np.ndarray,
        gravity: float,
        imu_pos_from_com: np.ndarray,
    ):
        self._q = quat_normalize(np.asarray(q0, dtype=float))
        self._v = np.asarray(v0, dtype=float).copy()
        self._p = np.asarray(p0, dtype=float).copy()
        self._t: float | None = None

        self._gravity = float(gravity)
        self._g_vec   = np.array([0.0, 0.0, -self._gravity])
        self._p_imu   = np.asarray(imu_pos_from_com, dtype=float).copy()

        # Previous-step cache (needed for trapezoid rule and backward diff)
        self._prev_gyro:        np.ndarray | None = None
        self._prev_accel_world: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public state accessors
    # ------------------------------------------------------------------

    @property
    def q(self) -> np.ndarray:
        """Current quaternion [w,x,y,z], body-to-world."""
        return self._q.copy()

    @property
    def R(self) -> np.ndarray:
        """Current 3×3 rotation matrix, body-to-world."""
        return quat_to_rotmat(self._q)

    @property
    def v(self) -> np.ndarray:
        """Current velocity in world frame (m/s)."""
        return self._v.copy()

    @property
    def p(self) -> np.ndarray:
        """Current COM position in world frame (m)."""
        return self._p.copy()

    @property
    def t(self) -> float | None:
        """Timestamp of the last processed sample (s), or None before first update."""
        return self._t

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self,
        t: float,
        accel_body: np.ndarray,
        gyro_body: np.ndarray,
    ) -> None:
        """
        Ingest one IMU sample and advance the state estimate.

        Parameters
        ----------
        t          : timestamp of this sample (s)
        accel_body : (3,) accelerometer reading in body frame (m/s²)
        gyro_body  : (3,) gyroscope reading in body frame (rad/s)
        """
        accel_body = np.asarray(accel_body, dtype=float)
        gyro_body  = np.asarray(gyro_body,  dtype=float)

        # First sample: initialise cache, nothing to integrate yet
        if self._t is None:
            self._t = float(t)
            self._prev_gyro        = gyro_body.copy()
            self._prev_accel_world = self._compute_accel_world(
                accel_body, gyro_body, alpha=np.zeros(3)
            )
            return

        dt = float(t) - self._t
        if dt <= 0.0:
            return  # guard against non-monotone timestamps

        # --- Angular acceleration (backward finite difference) ---
        alpha = (gyro_body - self._prev_gyro) / dt

        # --- RK4 quaternion integration (zero-order hold: use current gyro) ---
        q  = self._q
        w  = gyro_body
        k1 = _q_dot(q,                              w)
        k2 = _q_dot(quat_normalize(q + 0.5*dt*k1),  w)
        k3 = _q_dot(quat_normalize(q + 0.5*dt*k2),  w)
        k4 = _q_dot(quat_normalize(q +     dt*k3),  w)
        self._q = quat_normalize(q + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4))

        # --- World-frame COM acceleration at current sample ---
        accel_world_cur = self._compute_accel_world(accel_body, gyro_body, alpha)

        # --- Trapezoid integration ---
        a_prev = self._prev_accel_world
        a_cur  = accel_world_cur

        v_new = self._v + 0.5 * dt * (a_prev + a_cur)
        p_new = self._p + 0.5 * dt * (self._v + v_new)

        self._v = v_new
        self._p = p_new
        self._t = float(t)
        self._prev_gyro        = gyro_body.copy()
        self._prev_accel_world = accel_world_cur.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_accel_world(
        self,
        accel_body: np.ndarray,
        gyro_body: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        """
        Convert an accelerometer reading to world-frame COM acceleration.

        The accelerometer measures specific force at p_imu (not at the COM).
        Steps:
          1. Rotate specific force to world frame and add gravity.
          2. Subtract lever-arm accelerations rotated to world frame:
               centripetal : ω × (ω × p_imu)
               tangential  : α × p_imu
        """
        R           = quat_to_rotmat(self._q)
        a_imu_world = R @ accel_body + self._g_vec
        lever_body  = (np.cross(gyro_body, np.cross(gyro_body, self._p_imu))
                     + np.cross(alpha,     self._p_imu))
        return a_imu_world - R @ lever_body
