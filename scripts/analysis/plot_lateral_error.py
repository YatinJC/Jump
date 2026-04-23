import h5py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('Agg')

f = h5py.File('data/sim_output_test_quadrotor_vertical.h5', 'r')
gt = f['ground_truth']
t = gt['t'][:]
p = gt['position'][:]

# Target is [0, 0]
lateral_error = np.sqrt(p[:, 0]**2 + p[:, 1]**2)

plt.figure(figsize=(10, 5))
plt.plot(t, lateral_error, label='Lateral Error (m)', color='red')
plt.title('Lateral Distance from Target Output vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Distance from Target Origin (m)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('lateral_error_plot.png')
print("Saved lateral_error_plot.png")
