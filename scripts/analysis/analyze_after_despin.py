import h5py
import numpy as np

f = h5py.File('data/sim_output_test_quadrotor_vertical.h5', 'r')
gt = f['ground_truth']
t = gt['t'][:]
p = gt['position'][:]
v = gt['velocity'][:]

tc = f['thrusters']['t'][:]
cmd = f['thrusters']['commanded'][:]
f.close()

print("Time | Z | Vz | Cmd0 | Cmd1 | Cmd2 | Cmd3")
for i in range(0, min(24000, len(t)), 1000): # every 0.125s approximately
    idx_tc = np.searchsorted(tc, t[i])
    if idx_tc < len(tc):
        print(f"{t[i]:.3f}s | {p[i,2]:.3f} | {v[i,2]:.3f} | {cmd[idx_tc,0]:.2f} {cmd[idx_tc,1]:.2f} {cmd[idx_tc,2]:.2f} {cmd[idx_tc,3]:.2f}")
