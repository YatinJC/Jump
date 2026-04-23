"""
Environment: ground plane and finite wall.

Ground : z = 0 plane (infinite).
Wall   : finite rectangle defined by center, outward normal, width, height.
         The ball can only collide from the front (normal side).

Coordinate system: world frame, z-up.
"""

import numpy as np


class Ground:
    """Infinite ground plane at z = 0."""

    def check_collision(self, pos_geom: np.ndarray, radius: float) -> bool:
        """True if the ball sphere (centered at pos_geom) touches the ground."""
        return pos_geom[2] <= radius

    def raycast(self, origin: np.ndarray, direction: np.ndarray):
        """
        Intersect ray with z=0 plane.

        Returns distance t > 0 if hit, else None.
        """
        if abs(direction[2]) < 1e-12:
            return None
        t = -origin[2] / direction[2]
        return t if t > 1e-9 else None

    def surface_normal(self) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])

    def relative_state(self, pos_com: np.ndarray, vel: np.ndarray,
                       quat: np.ndarray, omega: np.ndarray, ball_radius: float):
        """
        Express position, velocity, and orientation relative to the ground surface.

        Returns dict with:
          pos_rel   : COM position in surface frame (z = height above ground)
          vel_rel   : velocity in surface frame (same as world for flat ground)
          normal_vel: velocity component along surface normal
          tangent_vel: velocity component along surface
        """
        n = self.surface_normal()
        normal_vel  = np.dot(vel, n)
        tangent_vel = vel - normal_vel * n
        return {
            'pos_rel':    pos_com.copy(),
            'vel_rel':    vel.copy(),
            'normal_vel': normal_vel,
            'tangent_vel_mag': np.linalg.norm(tangent_vel),
        }


class Wall:
    """Finite rectangular wall."""

    def __init__(self, wall_cfg: dict):
        self.center = np.array(wall_cfg['center'], dtype=float)
        n = np.array(wall_cfg['normal'], dtype=float)
        self.normal = n / np.linalg.norm(n)
        self.width  = float(wall_cfg['width'])
        self.height = float(wall_cfg['height'])

        # Build orthonormal tangent frame (u horizontal, v vertical-ish)
        ref = np.array([1.0, 0.0, 0.0]) if abs(self.normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(self.normal, ref)
        u /= np.linalg.norm(u)
        v = np.cross(self.normal, u)   # = normal × u; will be roughly vertical
        v /= np.linalg.norm(v)

        # Ensure v has positive z component (points "up" if possible)
        if v[2] < 0:
            v = -v
            u = -u

        self.tangent_u = u  # horizontal across wall
        self.tangent_v = v  # vertical across wall

    def _local_coords(self, point: np.ndarray):
        """Return (signed distance from plane, u-coord, v-coord) for a world point."""
        diff = point - self.center
        d = np.dot(diff, self.normal)
        local_u = np.dot(diff, self.tangent_u)
        local_v = np.dot(diff, self.tangent_v)
        return d, local_u, local_v

    def _within_bounds(self, local_u: float, local_v: float, margin: float = 0.0) -> bool:
        return (abs(local_u) <= self.width  / 2 + margin and
                abs(local_v) <= self.height / 2 + margin)

    def check_collision(self, pos_geom: np.ndarray, radius: float) -> bool:
        """
        True if the ball sphere (centered at pos_geom, radius r) collides with
        this wall from the front side.
        """
        d, local_u, local_v = self._local_coords(pos_geom)

        # Must be approaching from the front (normal side)
        if d < 0:
            return False
        # Too far from plane
        if d > radius:
            return False

        # Find the closest point on the finite rectangle to pos_geom projection
        clamped_u = np.clip(local_u, -self.width  / 2, self.width  / 2)
        clamped_v = np.clip(local_v, -self.height / 2, self.height / 2)
        closest = self.center + clamped_u * self.tangent_u + clamped_v * self.tangent_v

        return bool(np.linalg.norm(pos_geom - closest) <= radius)

    def raycast(self, origin: np.ndarray, direction: np.ndarray):
        """
        Intersect ray with the wall plane, checking finite bounds.

        Returns distance t > 0 if hit, else None.
        """
        denom = np.dot(direction, self.normal)
        if abs(denom) < 1e-12:
            return None   # parallel to wall

        t = np.dot(self.center - origin, self.normal) / denom
        if t <= 1e-9:
            return None   # behind or at origin

        hit = origin + t * direction
        _, local_u, local_v = self._local_coords(hit)

        if self._within_bounds(local_u, local_v):
            return t
        return None

    def surface_normal(self) -> np.ndarray:
        return self.normal.copy()

    def relative_state(self, pos_com: np.ndarray, vel: np.ndarray,
                       quat: np.ndarray, omega: np.ndarray, ball_radius: float):
        """Express state relative to the wall surface frame."""
        n = self.normal
        normal_vel  = np.dot(vel, n)
        tangent_vel = vel - normal_vel * n

        d, local_u, local_v = self._local_coords(pos_com)

        return {
            'dist_to_wall':    d,
            'local_u':         local_u,
            'local_v':         local_v,
            'normal_vel':      normal_vel,       # negative = approaching
            'tangent_vel_mag': np.linalg.norm(tangent_vel),
        }


class Environment:
    """Aggregates all surfaces and provides unified collision/raycast API."""

    def __init__(self, env_cfg: dict):
        self.ground = Ground()
        self.wall   = Wall(env_cfg['wall'])

    def check_collision(self, pos_geom: np.ndarray, radius: float):
        """
        Returns ('wall', Wall) or ('ground', Ground) or None.
        Wall collision is checked first.
        """
        if self.wall.check_collision(pos_geom, radius):
            return 'wall', self.wall
        if self.ground.check_collision(pos_geom, radius):
            return 'ground', self.ground
        return None, None

    def raycast(self, origin: np.ndarray, direction: np.ndarray):
        """Return the smallest positive hit distance across all surfaces, or None."""
        distances = []
        d = self.ground.raycast(origin, direction)
        if d is not None:
            distances.append(d)
        d = self.wall.raycast(origin, direction)
        if d is not None:
            distances.append(d)
        return min(distances) if distances else None
