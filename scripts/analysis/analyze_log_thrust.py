import h5py
import numpy as np

f = h5py.File('data/sim_output_test_quadrotor_vertical.h5', 'r')
c = f['thrusters']['commanded'][:]
t = f['thrusters']['t'][:]

print("Commanded min/max bounds:", np.min(c), np.max(c))
print("First 10 commands:")
print(c[:10])
print("Middle commands:")
print(c[40000:40005])

f.close()
