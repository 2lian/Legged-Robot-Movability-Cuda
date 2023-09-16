import numpy as np
import matplotlib.pyplot as plt

# Specify the path to your binary file
binary_file_path = "build01/result.bin"

# Open the binary file and read its contents into a numpy array
data = np.fromfile(binary_file_path, dtype=np.float32)

width = 1920
height = 1080
reshaped_data = data.reshape(height, width, 3)
img = np.linalg.norm(reshaped_data, axis=2)

px = 1/plt.rcParams['figure.dpi'] /2  # pixel in inches
plt.figure(1, figsize=(width*px, width*px), layout="constrained")
plt.imshow(img, extent=(- width / 2, width / 2, - height / 2, height / 2), vmax=400)
plt.grid()
plt.xlim(-300, None)
# plt.axis('off')  # Turn off axis labels and ticks
plt.show()
