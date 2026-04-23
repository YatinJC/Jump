import h5py
import numpy as np
import math

f = h5py.File('data/sim_output_test_quadrotor_vertical.h5', 'r')
gt = f['ground_truth']
t = gt['t'][:]
q = gt['quaternion'][:]

tc = f['thrusters']['t'][:]
cmd = f['thrusters']['commanded'][:]

idx_trig = -1
for i in range(len(tc)):
    if np.mean(cmd[i]) > 1.0 and tc[i] > 2.0:
        idx_trig = i
        break

idx_gt = np.searchsorted(t, tc[idx_trig])
quat = q[idx_gt]

w, x, y, z = quat
thrust_dir = [
    2.0 * (x*z + w*y),
    2.0 * (y*z - w*x),
    w*w - x*x - y*y + z*z
]
tilt = math.acos(np.clip(thrust_dir[2], -1, 1)) * 180 / math.pi

print(f"Trigger Time: {tc[idx_trig]:.3f}s")
print(f"Quaternion: [{w:.4f}, {x:.4f}, {y:.4f}, {z:.4f}]")
print(f"Thrust Axis World: [{thrust_dir[0]:.3f}, {thrust_dir[1]:.3f}, {thrust_dir[2]:.3f}]")
print(f"Tilt Angle from Vertical: {tilt:.1f} degrees")
