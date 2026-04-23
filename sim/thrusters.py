"""
Thruster model.

Each thruster sits on the ball surface.  It behaves like a propeller:
  - Produces a force along its direction_body unit vector.
  - Produces a reaction torque along -direction_body (opposite to thrust axis),
    with sign determined by spin_direction (+1 or -1) and the sign of the force.
  - First-order lag dynamics between commanded and actual force.

The thrust direction is specified per-thruster via the 'direction' config key
(a unit vector in body frame).  If omitted it defaults to -pos_hat (radially
inward toward the geometric center), preserving the original behaviour.

Thrust vectoring (optional):
  Each thruster can optionally gimbal about a single axis, deflecting the
  thrust direction within ±max_deflection of the nominal direction.  The
  gimbal axis can be specified as an explicit body-frame vector or as a
  keyword ('tangential' or 'radial') resolved from the thruster's geometry.
  Paired thrusters can be constrained to deflect in opposite directions.

If the ball has a COM offset (center_of_mass ≠ [0,0,0]), the force no longer
passes through the COM, so an additional moment-arm torque arises automatically
from the cross-product term.

Coordinate convention: all positions in body frame (origin = COM).
"""

import numpy as np


class Thruster:
    """Single propeller-style thruster with optional single-axis thrust vectoring."""

    def __init__(self, tcfg: dict, com_offset: np.ndarray):
        """
        Parameters
        ----------
        tcfg       : dict with keys position, spin_direction, min_force,
                     max_force, time_constant, torque_to_thrust_ratio, and
                     optionally direction (unit vector in body frame) and
                     vectoring (dict with enabled, max_deflection, time_constant,
                     gimbal_axis, and optionally pair_index / pair_mode).
                     If 'direction' is absent the thrust defaults to radially
                     inward (-pos_hat).
        com_offset : (3,) COM offset from geometric center in body frame (m)
        """
        pos_geom = np.array(tcfg['position'], dtype=float)  # from geom center
        self.pos_geom_hat = pos_geom / np.linalg.norm(pos_geom)

        # Position of thruster from COM (used for moment-arm torque)
        self.pos_from_com = pos_geom - com_offset

        # Nominal thrust direction: explicit unit vector, or radially inward
        if 'direction' in tcfg:
            d = np.array(tcfg['direction'], dtype=float)
            self.nominal_direction = d / np.linalg.norm(d)
        else:
            self.nominal_direction = -self.pos_geom_hat

        self.spin_direction = float(tcfg['spin_direction'])   # +1 or -1
        self.min_force = float(tcfg['min_force'])
        self.max_force = float(tcfg['max_force'])
        self.tau = float(tcfg['time_constant'])               # seconds
        self.k_q = float(tcfg['torque_to_thrust_ratio'])      # m

        self.actual_force = 0.0     # current force after lag dynamics
        self.commanded_force = 0.0

        # --- Thrust vectoring ---
        vcfg = tcfg.get('vectoring', {})
        self.vectoring_enabled = vcfg.get('enabled', False)
        self.actual_deflection = 0.0
        self.commanded_deflection = 0.0

        if self.vectoring_enabled:
            self.max_deflection = np.radians(float(vcfg['max_deflection']))
            self.tau_vectoring = float(vcfg['time_constant'])

            # Resolve gimbal axis (must be perpendicular to nominal direction)
            ga = vcfg.get('gimbal_axis', 'tangential')
            if isinstance(ga, str):
                if ga == 'tangential':
                    raw = np.cross(self.nominal_direction, self.pos_geom_hat)
                elif ga == 'radial':
                    # Radial component perpendicular to nominal direction
                    dot = np.dot(self.pos_geom_hat, self.nominal_direction)
                    raw = self.pos_geom_hat - dot * self.nominal_direction
                else:
                    raise ValueError(f"Unknown gimbal_axis keyword: {ga}")
            else:
                raw = np.array(ga, dtype=float)
                # Project out component along nominal direction
                dot = np.dot(raw, self.nominal_direction)
                raw = raw - dot * self.nominal_direction

            norm = np.linalg.norm(raw)
            assert norm > 1e-6, (
                "gimbal_axis must not be parallel to thrust direction"
            )
            self.gimbal_axis = raw / norm

            # Swing direction: direction thrust deflects for positive delta
            self.swing_direction = np.cross(self.gimbal_axis, self.nominal_direction)
            sw_norm = np.linalg.norm(self.swing_direction)
            assert sw_norm > 1e-6
            self.swing_direction /= sw_norm

            # Pair config (validated by ThrusterArray)
            self.pair_index = vcfg.get('pair_index', None)
            self.pair_mode = vcfg.get('pair_mode', None)
        else:
            self.gimbal_axis = None
            self.swing_direction = None
            self.pair_index = None
            self.pair_mode = None

    @property
    def direction_body(self) -> np.ndarray:
        """Current thrust direction in body frame, accounting for deflection."""
        if self.vectoring_enabled and abs(self.actual_deflection) > 1e-10:
            return (np.cos(self.actual_deflection) * self.nominal_direction
                    + np.sin(self.actual_deflection) * self.swing_direction)
        return self.nominal_direction

    def set_command(self, command: float):
        """Set desired force (clamped to [min_force, max_force])."""
        self.commanded_force = float(np.clip(command, self.min_force, self.max_force))

    def set_vector_command(self, delta: float):
        """Set desired gimbal deflection in radians (clamped to ±max_deflection)."""
        if not self.vectoring_enabled:
            return
        self.commanded_deflection = float(
            np.clip(delta, -self.max_deflection, self.max_deflection)
        )

    def update(self, dt: float):
        """Advance thruster dynamics by dt (first-order lag for both force and vectoring)."""
        self.actual_force += (self.commanded_force - self.actual_force) * dt / self.tau
        if self.vectoring_enabled:
            self.actual_deflection += (
                (self.commanded_deflection - self.actual_deflection)
                * dt / self.tau_vectoring
            )

    def force_vector_body(self) -> np.ndarray:
        """Force contributed by this thruster in body frame (3,)."""
        return self.actual_force * self.direction_body

    def torque_vector_body(self) -> np.ndarray:
        """
        Torque about COM in body frame from this thruster (3,).

        Two contributions:
          1. Moment-arm torque: r_from_com × F_body
             (nonzero only when COM ≠ geometric center)
          2. Propeller reaction torque: k_q * force * spin_direction * (-direction_body)
             The reaction torque axis is opposite to the thrust direction (propeller axis).
        """
        F_body = self.force_vector_body()
        tau_moment   = np.cross(self.pos_from_com, F_body)
        tau_reaction = self.k_q * self.actual_force * self.spin_direction * (-self.direction_body)
        return tau_moment + tau_reaction


class ThrusterArray:
    """Collection of thrusters with unified command/update interface.

    When any thruster has vectoring enabled, wrench_to_commands uses a unified
    allocation that solves for both force magnitudes and gimbal deflections
    simultaneously.  The extended control allocation matrix decomposes each
    vectoring thruster's force into a nominal component (along the undeflected
    direction) and a swing component (perpendicular, in the gimbal plane).
    """

    def __init__(self, thrusters_cfg: list, com_offset: np.ndarray):
        self.thrusters = [Thruster(t, com_offset) for t in thrusters_cfg]
        self.n = len(self.thrusters)

        # Identify vectoring DOFs — one per vectoring-enabled thruster
        self._vec_indices = [i for i, t in enumerate(self.thrusters)
                             if t.vectoring_enabled]
        self._has_vectoring = len(self._vec_indices) > 0

        # Map from thruster index → column index in the swing block of B_ext
        self._vec_col = {idx: j for j, idx in enumerate(self._vec_indices)}
        self.n_vec = len(self._vec_indices)

    def set_commands(self, commands):
        """Set commanded force for each thruster.  commands: array-like length n."""
        for t, c in zip(self.thrusters, commands):
            t.set_command(c)

    def update(self, dt: float):
        """Advance all thruster dynamics (force lag + vectoring servo lag) by dt."""
        for t in self.thrusters:
            t.update(dt)

    def get_forces_and_torques(self):
        """
        Return total force (body frame) and total torque (body frame) from all thrusters.

        Returns
        -------
        F_body   : (3,)
        tau_body : (3,)
        """
        F   = np.zeros(3)
        tau = np.zeros(3)
        for t in self.thrusters:
            F   += t.force_vector_body()
            tau += t.torque_vector_body()
        return F, tau

    def get_actual_forces(self) -> np.ndarray:
        """Return array of actual force magnitudes, shape (n,)."""
        return np.array([t.actual_force for t in self.thrusters])

    def get_commanded_forces(self) -> np.ndarray:
        """Return array of commanded force magnitudes, shape (n,)."""
        return np.array([t.commanded_force for t in self.thrusters])

    def get_actual_deflections(self) -> np.ndarray:
        """Return array of actual gimbal deflections in radians, shape (n,)."""
        return np.array([t.actual_deflection for t in self.thrusters])

    def get_commanded_deflections(self) -> np.ndarray:
        """Return array of commanded gimbal deflections in radians, shape (n,)."""
        return np.array([t.commanded_deflection for t in self.thrusters])

    def control_allocation_matrix(self) -> np.ndarray:
        """
        Build the 6×n control allocation matrix B where [F_body; tau_body] = B @ u.
        Rows 0-2: force components (body frame).
        Rows 3-5: torque components (body frame).
        u is a vector of unit force commands for each thruster.

        Uses the current (deflected) direction for each thruster.
        """
        B = np.zeros((6, self.n))
        for i, t in enumerate(self.thrusters):
            d = t.direction_body
            B[:3, i] = d
            B[3:, i] = np.cross(t.pos_from_com, d) + t.k_q * t.spin_direction * (-d)
        return B

    def nominal_allocation_matrix(self) -> np.ndarray:
        """
        Build the 6×n allocation matrix using nominal (undeflected) directions.

        Identical to control_allocation_matrix when all deflections are zero.
        """
        B = np.zeros((6, self.n))
        for i, t in enumerate(self.thrusters):
            d = t.nominal_direction
            B[:3, i] = d
            B[3:, i] = np.cross(t.pos_from_com, d) + t.k_q * t.spin_direction * (-d)
        return B

    def extended_allocation_matrix(self) -> np.ndarray:
        """
        Build the 6×(n + n_vec) extended allocation matrix for unified
        force + vectoring allocation.

        Columns 0..n-1 : wrench per unit force along nominal direction
        Columns n..n+n_vec-1 : wrench per unit force along swing direction

        The solver returns [f_n_0, ..., f_n_{n-1}, f_s_0, ..., f_s_{n_vec-1}]
        where f_n_i is the force component along the nominal direction and
        f_s_j is the force component along the swing direction for vectoring
        thruster j.  The deflection angle is delta = atan2(f_s, f_n) and the
        total force magnitude is sqrt(f_n^2 + f_s^2).
        """
        B_nom = self.nominal_allocation_matrix()
        if not self._has_vectoring:
            return B_nom

        B_swing = np.zeros((6, self.n_vec))
        for j, idx in enumerate(self._vec_indices):
            t = self.thrusters[idx]
            sd = t.swing_direction
            B_swing[:3, j] = sd
            B_swing[3:, j] = (np.cross(t.pos_from_com, sd)
                              + t.k_q * t.spin_direction * (-sd))

        return np.hstack([B_nom, B_swing])

    def saturate_scale_wrench(self, wrench: np.ndarray) -> np.ndarray:
        """
        Scale a wrench so the pseudoinverse allocation fits within actuator
        limits before clipping.

        For vectoring-enabled arrays, checks total force per thruster
        (sqrt(f_n^2 + f_s^2)) against force limits.  Deflection angles are
        invariant under uniform scaling so they need not be checked here.

        Parameters
        ----------
        wrench : (6,) desired body-frame wrench

        Returns
        -------
        scaled_wrench : (6,)
        """
        B_ext = self.extended_allocation_matrix()
        u_ext = np.linalg.pinv(B_ext) @ wrench

        scale = 1.0
        for i, thr in enumerate(self.thrusters):
            if thr.vectoring_enabled and i in self._vec_col:
                j = self._vec_col[i]
                f_total = np.sqrt(u_ext[i]**2 + u_ext[self.n + j]**2)
            else:
                f_total = u_ext[i]
            if f_total > thr.max_force:
                scale = min(scale, thr.max_force / f_total)
            elif f_total < thr.min_force:
                scale = min(scale, thr.min_force / f_total)
        return wrench * scale

    def _natural_patterns(self) -> np.ndarray:
        """
        4×4 matrix whose rows are the natural sum/difference command patterns
        for a symmetric 4-thruster corner layout.  Row i is the command pattern
        for virtual channel i:

          0 – sum    [ 1,  1,  1,  1]  → net Fz
          1 – x-diff [ 1,  1, -1, -1]  → Fx
          2 – y-diff [ 1, -1,  1, -1]  → Fy
          3 – yaw    [-1,  1,  1, -1]  → τz (pure, no lateral force coupling)

        Assumes diagonal spin assignment: T1(+x+y)=+1, T2(+x-y)=-1,
        T3(-x+y)=-1, T4(-x-y)=+1.

        The rows are mutually orthogonal: M @ M.T = 4·I, so the pseudoinverse
        is M.T / 4.  This means virtual_to_commands is exact with no lstsq.

        Only valid for the standard 4-thruster symmetric layout with no vectoring.
        """
        if self.n != 4:
            raise ValueError("_natural_patterns requires exactly 4 thrusters")
        return np.array([
            [ 1,  1,  1,  1],   # sum  → Fz
            [ 1,  1, -1, -1],   # Fx
            [ 1, -1,  1, -1],   # Fy
            [-1,  1,  1, -1],   # yaw → τz (pure)
        ], dtype=float)

    def virtual_input_basis(self):
        """
        Decompose the 4-D achievable wrench subspace into 4 independent virtual
        channels, following the book's reduced B-matrix approach (Ch. 8 §1.2.2).

        For the default 4-thruster layout the full 6×4 B has rank 4, so two
        wrench directions are structurally unachievable:
          • Pure τy (pitch) — always coupled to Fx and Fz via COM offset
          • Pure τz (yaw)   — always = k_q · Fy (reaction torque coupling)

        Returns
        -------
        B_v : (6, 4) ndarray
            Column j is the body-frame wrench [Fx,Fy,Fz,τx,τy,τz] produced by
            unit virtual input j.  Use this to understand side-effects per channel.
        M   : (4, 4) ndarray
            Pattern matrix; row j is the thruster command pattern for channel j.
            Thruster commands are u = M.T @ v / 4.

        Only valid for the standard 4-thruster symmetric layout.
        Uses nominal directions (ignores current deflection).
        """
        M = self._natural_patterns()
        B = self.nominal_allocation_matrix()
        # B_v[:,j] = wrench from virtual input j = B @ M[j,:]
        B_v = np.column_stack([B @ M[j] / self.n for j in range(self.n)])
        return B_v, M

    def virtual_to_commands(self, v: np.ndarray) -> np.ndarray:
        """
        Convert 4 virtual-channel scalars to clipped thruster commands.

        This is an exact inversion (no least-squares) because the natural
        patterns are orthogonal: u = M.T @ v / 4.

        Parameters
        ----------
        v : (4,) array
            v[0]  net thrust  → Fz
            v[1]  x-force    → Fx
            v[2]  y-force    → Fy
            v[3]  yaw        → τz (pure, no lateral coupling)

        Returns
        -------
        commands : (4,) array, clipped to per-thruster [min_force, max_force]
        """
        M = self._natural_patterns()
        u = M.T @ np.asarray(v, dtype=float) / self.n
        return np.array([
            np.clip(u[i], t.min_force, t.max_force)
            for i, t in enumerate(self.thrusters)
        ])

    def wrench_to_commands(self, wrench_body: np.ndarray,
                           return_residual: bool = False):
        """
        Convert a desired 6-DOF wrench to thruster force commands.

        When vectoring is enabled, uses the extended allocation matrix to
        jointly solve for force magnitudes and gimbal deflections.  The
        computed deflections are applied to each thruster via set_vector_command
        (with paired-opposite coupling enforced automatically).

        For non-vectoring layouts the original SVD pseudoinverse is used.

        Parameters
        ----------
        wrench_body    : (6,) [Fx, Fy, Fz, τx, τy, τz] in body frame
        return_residual: if True, also return the unachievable wrench component

        Returns
        -------
        commands : (n,) thruster force commands, clipped to per-thruster limits
        residual : (6,) unachievable wrench component (only if return_residual)
        """
        if self._has_vectoring:
            return self._wrench_to_commands_unified(wrench_body, return_residual)

        B = self.control_allocation_matrix()
        if self.n == 6:
            u = np.linalg.solve(B, wrench_body)
            residual = np.zeros(6)
        else:
            B_pinv = np.linalg.pinv(B)   # SVD-based; more stable than lstsq
            u = B_pinv @ wrench_body
            residual = wrench_body - B @ u
        commands = np.array([
            np.clip(u[i], t.min_force, t.max_force)
            for i, t in enumerate(self.thrusters)
        ])
        if return_residual:
            return commands, residual
        return commands

    def _wrench_to_commands_unified(self, wrench_body: np.ndarray,
                                     return_residual: bool = False):
        """
        Unified force + vectoring allocation.

        Decomposes each vectoring thruster's contribution into a nominal-direction
        component (f_n) and a swing-direction component (f_s).  The extended
        6×(n+n_vec) linear system is solved via pseudoinverse.

        After solving:
          delta_i = atan2(f_s_i, f_n_i)   (clamped to ±max_deflection)
          f_total_i = sqrt(f_n_i² + f_s_i²)

        For paired thrusters, deflections are averaged and coupled:
          'opposite' → δ_j = -δ_i   (lateral forces add, torques cancel)
          'same'     → δ_j =  δ_i   (lateral forces cancel, torques add)
        """
        B_ext = self.extended_allocation_matrix()
        B_ext_pinv = np.linalg.pinv(B_ext)
        u_ext = B_ext_pinv @ wrench_body

        # Split into nominal-force and swing-force components
        f_n = u_ext[:self.n]
        f_s = u_ext[self.n:]     # length n_vec

        # Compute deflection angles and total forces
        deflections = np.zeros(self.n)
        forces = np.copy(f_n)

        for j, idx in enumerate(self._vec_indices):
            t = self.thrusters[idx]
            fn_i = f_n[idx]
            fs_i = f_s[j]

            delta = np.arctan2(fs_i, fn_i) if abs(fn_i) > 1e-8 else 0.0
            delta = np.clip(delta, -t.max_deflection, t.max_deflection)
            deflections[idx] = delta

            # Total force magnitude
            forces[idx] = np.sqrt(fn_i**2 + fs_i**2)
            # Preserve sign: if fn_i < 0, total force is negative (reverse thrust)
            if fn_i < 0:
                forces[idx] = -forces[idx]

        # Enforce paired constraints: average the pair's deflections
        paired_done = set()
        for idx in self._vec_indices:
            t = self.thrusters[idx]
            if t.pair_index is not None and idx not in paired_done:
                pi = t.pair_index
                if t.pair_mode == 'opposite':
                    avg = (deflections[idx] - deflections[pi]) / 2.0
                    deflections[idx] = avg
                    deflections[pi] = -avg
                elif t.pair_mode == 'same':
                    avg = (deflections[idx] + deflections[pi]) / 2.0
                    deflections[idx] = avg
                    deflections[pi] = avg
                paired_done.add(idx)
                paired_done.add(pi)

        # Apply vectoring commands
        for idx in self._vec_indices:
            self.thrusters[idx].set_vector_command(deflections[idx])

        # Clip force commands
        commands = np.array([
            np.clip(forces[i], t.min_force, t.max_force)
            for i, t in enumerate(self.thrusters)
        ])

        if return_residual:
            residual = wrench_body - B_ext @ u_ext
            return commands, residual
        return commands
