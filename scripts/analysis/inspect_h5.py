import h5py
import matplotlib.pyplot as plt
import numpy as np

f = h5py.File('data/sim_output_test_quadrotor_vertical.h5', 'r')
t = f['t'][:]
p = f['state'][:, 0:3]
v = f['state'][:, 3:6]
q = f['state'][:, 6:10]

fig, ax = plt.subplots(3, 1, figsize=(10, 8))
ax[0].plot(t, p[:, 2], label='z')
ax[0].legend()
ax[1].plot(t, v[:, 2], label='vz')
ax[1].legend()
ax[2].plot(t, p[:, 0], label='x')
ax[2].plot(t, p[:, 1], label='y')
ax[2].legend()
plt.savefig('sim_plot.png')
print("Keys in h5:", list(f.keys()))
print("Final state (z, vz):", p[-1, 2], v[-1, 2])
print("Max z:", np.max(p[:, 2]))
