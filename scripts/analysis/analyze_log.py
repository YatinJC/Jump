import h5py
import matplotlib.pyplot as plt
import numpy as np

# Use an explicit backend to prevent Qt issues in conda
import matplotlib
matplotlib.use('Agg')

f = h5py.File('data/sim_output_test_quadrotor_vertical.h5', 'r')
gt = f['ground_truth']
t = gt['t'][:]
p = gt['position'][:]
v = gt['velocity'][:]

print("Simulation started at t=", t[0], " ended at t=", t[-1])
print("Initial Z:", p[0, 2], " Initial Vz:", v[0, 2])
print("Final Z:", p[-1, 2], "   Final Vz:", v[-1, 2])
print("Max Z:", np.max(p[:, 2]), " Min Vz:", np.min(v[:, 2]))

fig, ax = plt.subplots(3, 1, figsize=(10, 8))
ax[0].plot(t, p[:, 2], label='z position')
ax[0].legend()
ax[0].grid(True)
ax[1].plot(t, v[:, 2], label='z velocity')
ax[1].legend()
ax[1].grid(True)
ax[2].plot(t, p[:, 0], label='x')
ax[2].plot(t, p[:, 1], label='y')
ax[2].legend()
ax[2].grid(True)

plt.tight_layout()
plt.savefig('sim_plot_hoverslam.png')
print("Plot saved to sim_plot_hoverslam.png")
