"""
Sensor models: IMU and distance sensors.

IMU model
---------
The IMU is at a fixed position in the body frame (not necessarily at the COM).
  Accelerometer: measures specific force in body frame,
      a_meas = R^T @ (a_imu_world - g_world)
  where
      a_imu_world = a_com_world + R @ (alpha × p_imu + omega × (omega × p_imu))
  p_imu is the IMU position relative to the COM in the body frame.
  g_world = [0, 0, -gravity]  (gravitational acceleration, not force)

  Note: a real accelerometer reads +g (upward) when stationary because specific
  force = a - g_field, and g_field = [0,0,-g].  Stationary: a=0, so specific
  force = -g_field = [0,0,g].

  Gyroscope: measures angular velocity in body frame directly.

Noise model (each enabled independently):
  - White Gaussian noise added each sample.
  - Constant bias added always (set to zero to disable).

Distance sensor model
---------------------
Each sensor fires a single ray from its surface position outward (radially).
Returns the distance to the nearest environment surface (ground or wall).
"""

import numpy as np
from sim.physics import quat_to_rotmat


# ---------------------------------------------------------------------------
# IMU
# ---------------------------------------------------------------------------

class IMUSensor:
    """6-DOF inertial measurement unit."""

    def __init__(self, imu_cfg: dict, com_offset: np.ndarray):
        """
        Parameters
        ----------
        imu_cfg    : dict from config sensors.imu
        com_offset : (3,) COM offset from geometric center, body frame
        """
        pos_geom = np.array(imu_cfg['position'], dtype=float)
        # IMU position relative to COM
        self.pos_from_com = pos_geom - com_offset

        self.accel_noise_std = float(imu_cfg['accelerometer_noise_std'])
        self.gyro_noise_std  = float(imu_cfg['gyroscope_noise_std'])
        self.accel_bias = np.array(imu_cfg['accelerometer_bias'], dtype=float)
        self.gyro_bias  = np.array(imu_cfg['gyroscope_bias'],  dtype=float)
        self.noise_enabled = bool(imu_cfg.get('noise_enabled', True))

    def sample(self, state: np.ndarray, accel_com_world: np.ndarray,
               alpha_body: np.ndarray, gravity: float, rng: np.random.Generator):
        """
        Produce one IMU measurement.

        Parameters
        ----------
        state          : (13,) simulation state
        accel_com_world: (3,)  linear acceleration of COM in world frame
        alpha_body     : (3,)  angular acceleration in body frame
        gravity        : float gravitational acceleration magnitude (m/s^2)
        rng            : numpy random Generator (for reproducibility)

        Returns
        -------
        accel_meas : (3,)  accelerometer reading (body frame, m/s^2)
        gyro_meas  : (3,)  gyroscope reading (body frame, rad/s)
        """
        quat  = state[6:10]
        omega = state[10:13]
        R     = quat_to_rotmat(quat)
        p     = self.pos_from_com

        # Acceleration of the IMU point in world frame (lever arm effect)
        # a_imu = a_com + R @ (alpha × p + omega × (omega × p))
        a_imu_world = (accel_com_world
                       + R @ (np.cross(alpha_body, p)
                              + np.cross(omega, np.cross(omega, p))))

        # Specific force in body frame: R^T @ (a_imu - g_world)
        g_world = np.array([0.0, 0.0, -gravity])
        accel_meas = R.T @ (a_imu_world - g_world)

        # Gyroscope: angular velocity in body frame
        gyro_meas = omega.copy()

        # Add bias
        accel_meas = accel_meas + self.accel_bias
        gyro_meas  = gyro_meas  + self.gyro_bias

        # Add white noise
        if self.noise_enabled:
            accel_meas = accel_meas + rng.standard_normal(3) * self.accel_noise_std
            gyro_meas  = gyro_meas  + rng.standard_normal(3) * self.gyro_noise_std

        return accel_meas, gyro_meas


# ---------------------------------------------------------------------------
# Distance sensor
# ---------------------------------------------------------------------------

class DistanceSensor:
    """Single-ray distance sensor, fires radially outward from ball surface."""

    def __init__(self, position_geom: np.ndarray, com_offset: np.ndarray,
                 noise_std: float, max_range: float, noise_enabled: bool):
        """
        Parameters
        ----------
        position_geom : (3,) sensor position from geometric center (body frame)
        com_offset    : (3,) COM offset from geometric center (body frame)
        """
        self.pos_from_com = np.array(position_geom, dtype=float) - com_offset
        # Direction is outward from the geometric center (sensor is on surface)
        self.direction_body = np.array(position_geom, dtype=float)
        self.direction_body /= np.linalg.norm(self.direction_body)

        self.noise_std     = noise_std
        self.max_range     = max_range
        self.noise_enabled = noise_enabled

    def sample(self, state: np.ndarray, environment, rng: np.random.Generator) -> float:
        """
        Fire ray and return measured distance.

        Returns max_range if no surface is hit within range.
        """
        pos_com = state[:3]
        quat    = state[6:10]
        R       = quat_to_rotmat(quat)

        # Ray origin: sensor position in world frame
        origin    = pos_com + R @ self.pos_from_com
        direction = R @ self.direction_body

        dist = environment.raycast(origin, direction)
        dist = min(dist if dist is not None else self.max_range, self.max_range)

        if self.noise_enabled and dist < self.max_range:
            dist = max(0.0, dist + rng.standard_normal() * self.noise_std)

        return dist


# ---------------------------------------------------------------------------
# Sensor suite
# ---------------------------------------------------------------------------

class SensorSuite:
    """Manages the full set of sensors on the ball."""

    def __init__(self, sensors_cfg: dict, com_offset: np.ndarray):
        com_offset = np.array(com_offset, dtype=float)

        self.imu = IMUSensor(sensors_cfg['imu'], com_offset)

        ds_cfg = sensors_cfg['distance_sensors']
        self.distance_sensors = [
            DistanceSensor(
                position_geom=np.array(pos, dtype=float),
                com_offset=com_offset,
                noise_std=float(ds_cfg['noise_std']),
                max_range=float(ds_cfg['max_range']),
                noise_enabled=bool(ds_cfg.get('noise_enabled', True)),
            )
            for pos in ds_cfg['positions']
        ]
        self.n_distance = len(self.distance_sensors)

    def sample_imu(self, state, accel_com_world, alpha_body, gravity, rng):
        return self.imu.sample(state, accel_com_world, alpha_body, gravity, rng)

    def sample_distance(self, state, environment, rng):
        return np.array([
            ds.sample(state, environment, rng)
            for ds in self.distance_sensors
        ])
