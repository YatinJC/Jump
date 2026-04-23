"""Load and validate YAML configuration files."""

import yaml
import numpy as np
from pathlib import Path


def load_config(path: str) -> dict:
    """
    Load a YAML config file and return a validated dict.

    Normalises moment_of_inertia so callers always get a key 'matrix' (3x3 list).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(p, 'r') as f:
        cfg = yaml.safe_load(f)

    _validate_and_normalise(cfg)
    return cfg


def _validate_and_normalise(cfg: dict):
    """In-place normalisation and basic sanity checks."""

    # --- Moment of inertia: normalise to 3x3 matrix ---
    moi = cfg['ball']['moment_of_inertia']
    if 'diagonal' in moi and 'matrix' not in moi:
        d = moi['diagonal']
        moi['matrix'] = [[d[0], 0.0,  0.0 ],
                         [0.0,  d[1], 0.0 ],
                         [0.0,  0.0,  d[2]]]
    assert 'matrix' in moi, "moment_of_inertia must specify 'diagonal' or 'matrix'"
    I = np.array(moi['matrix'])
    assert I.shape == (3, 3), "moment_of_inertia matrix must be 3x3"
    assert np.allclose(I, I.T, atol=1e-9), "moment_of_inertia matrix must be symmetric"

    # --- Quaternion must be non-zero ---
    q = cfg['initial_conditions']['quaternion']
    assert np.linalg.norm(q) > 1e-9, "Initial quaternion must be non-zero"

    # --- throw_phase ---
    tp = cfg['throw_phase']
    assert float(tp['duration']) > 0, "throw_phase.duration must be positive"
    is_endstate = 'final_velocity' in tp or 'final_speed' in tp
    if not is_endstate:
        assert np.linalg.norm(tp['force']['direction']) > 1e-9, \
            "throw_phase.force.direction must be non-zero"
        assert np.linalg.norm(tp['torque']['axis']) > 1e-9, \
            "throw_phase.torque.axis must be non-zero"
    valid_profiles = {'constant', 'trapezoid', 'gaussian'}
    for section in ('force', 'torque'):
        if section in tp:
            prof = tp[section].get('profile', 'trapezoid')
            assert prof in valid_profiles, \
                f"throw_phase.{section}.profile must be one of {valid_profiles}, got '{prof}'"

    # --- Simulation dt and sensor rates ---
    sim_rate  = 1.0 / cfg['simulation']['dt']
    imu_rate  = cfg['sensors']['imu']['sample_rate']
    dist_rate = cfg['sensors']['distance_sensors']['sample_rate']

    for label, rate in [('IMU', imu_rate), ('Distance sensor', dist_rate)]:
        ratio = sim_rate / rate
        assert abs(ratio - round(ratio)) < 1e-9, (
            f"{label} sample rate {rate} Hz must divide evenly into "
            f"simulation rate {sim_rate} Hz (got ratio {ratio})"
        )

    # --- Wall normal must be non-zero ---
    n = np.array(cfg['environment']['wall']['normal'])
    assert np.linalg.norm(n) > 1e-9, "Wall normal must be non-zero"

    # --- Thruster spin_direction must be ±1 ---
    for i, t in enumerate(cfg['thrusters']):
        sd = t['spin_direction']
        assert sd in (1, -1, 1.0, -1.0), (
            f"Thruster {i}: spin_direction must be +1 or -1, got {sd}"
        )
        assert t['min_force'] <= t['max_force'], \
            f"Thruster {i}: min_force must be <= max_force"
        assert t['time_constant'] > 0, \
            f"Thruster {i}: time_constant must be positive"

        # --- Vectoring validation ---
        vcfg = t.get('vectoring', {})
        if vcfg.get('enabled', False):
            assert 'max_deflection' in vcfg, \
                f"Thruster {i}: vectoring.max_deflection is required when enabled"
            assert float(vcfg['max_deflection']) > 0, \
                f"Thruster {i}: vectoring.max_deflection must be positive"
            assert 'time_constant' in vcfg, \
                f"Thruster {i}: vectoring.time_constant is required when enabled"
            assert float(vcfg['time_constant']) > 0, \
                f"Thruster {i}: vectoring.time_constant must be positive"

            ga = vcfg.get('gimbal_axis', 'tangential')
            if isinstance(ga, str):
                assert ga in ('tangential', 'radial'), (
                    f"Thruster {i}: vectoring.gimbal_axis must be 'tangential', "
                    f"'radial', or an explicit [x,y,z] vector, got '{ga}'"
                )
            else:
                assert len(ga) == 3 and np.linalg.norm(ga) > 1e-9, (
                    f"Thruster {i}: vectoring.gimbal_axis vector must be "
                    f"a non-zero 3-element list"
                )

            pi = vcfg.get('pair_index')
            pm = vcfg.get('pair_mode')
            if pi is not None:
                assert isinstance(pi, int) and 0 <= pi < len(cfg['thrusters']), (
                    f"Thruster {i}: vectoring.pair_index must be a valid "
                    f"thruster index, got {pi}"
                )
                assert pi != i, \
                    f"Thruster {i}: vectoring.pair_index cannot reference itself"
                assert pm in ('opposite', 'same'), (
                    f"Thruster {i}: vectoring.pair_mode must be 'opposite' "
                    f"or 'same' when pair_index is set, got '{pm}'"
                )
