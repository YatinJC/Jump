import h5py
import numpy as np

f = h5py.File('data/sim_output_test_quadrotor_vertical.h5', 'r')
gt = f['ground_truth']
t = gt['t'][:]
p = gt['position'][:]
v = gt['velocity'][:]

print("Time   | Alt (m)  | X (m)   | Y (m)   | Vx (m/s) | Vy (m/s) | Vz (m/s)")
print("-" * 75)

# print the last 2 seconds leading up to impact
for i in range(len(t)):
    # only print when it drops below 5m and is falling, every 0.2s or so
    if p[i,2] < 5.0 and v[i,2] < -1.0 and i % 1000 == 0:
        print(f"{t[i]:6.3f} | {p[i,2]:8.3f} | {p[i,0]:7.3f} | {p[i,1]:7.3f} | {v[i,0]:8.3f} | {v[i,1]:8.3f} | {v[i,2]:8.3f}")

# Print the final moments right before landing at 8000Hz (every 0.05s)
print("-" * 75)
print("Final Hoverslam Approach:")
for i in range(len(t)):
    if p[i,2] < 0.5 and i % 200 == 0:
        print(f"{t[i]:6.3f} | {p[i,2]:8.3f} | {p[i,0]:7.3f} | {p[i,1]:7.3f} | {v[i,0]:8.3f} | {v[i,1]:8.3f} | {v[i,2]:8.3f}")

final_idx = len(t) - 1
print("-" * 75)
print(f"IMPACT | {p[final_idx,2]:8.3f} | {p[final_idx,0]:7.3f} | {p[final_idx,1]:7.3f} | {v[final_idx,0]:8.3f} | {v[final_idx,1]:8.3f} | {v[final_idx,2]:8.3f}")
