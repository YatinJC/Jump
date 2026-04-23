import h5py
import numpy as np

f = h5py.File('data/sim_output_test_quadrotor_vertical.h5', 'r')
gt = f['ground_truth']
t = gt['t'][:]
omega = gt['omega'][:]

t_throw = 0.3

yaw_settled_idx = -1
for i in range(len(t)):
    if t[i] < t_throw: continue
    if abs(omega[i, 2]) < 1.0:
        if np.all(np.abs(omega[i:i+50, 2]) < 1.0):
            yaw_settled_idx = i
            break

t_yaw_end = t[yaw_settled_idx]

tilt_settled_idx = -1
for i in range(yaw_settled_idx, len(t)):
    omega_perp = np.linalg.norm(omega[i, :2])
    if omega_perp < 1.0:
        if np.all(np.linalg.norm(omega[i:i+50, :2], axis=1) < 1.0):
            tilt_settled_idx = i
            break

t_tilt_end = t[tilt_settled_idx]

print(f"Phases Timeline:")
print(f"Throw ended at: {t_throw:.3f} s")
print(f"Yaw (Z) Despin: {t_throw:.3f}s to {t_yaw_end:.3f}s | Duration: {t_yaw_end - t_throw:.3f} s")
print(f"Tilt Despin:    {t_yaw_end:.3f}s to {t_tilt_end:.3f}s | Duration: {t_tilt_end - t_yaw_end:.3f} s")
print(f"Total Despin:   {t_tilt_end - t_throw:.3f} s (Terminal guidance begins at {t_tilt_end:.3f}s)")
