import numpy as np
import maps


def save_array_to_binary_file(array, filename):
    with open(filename, 'wb') as f:
        array.tofile(f)
    return
 
map = maps.obs_map
print(map, map.shape)
save_array_to_binary_file(map[:, 0].astype(np.float32), "numpy_input_tx.bin")
save_array_to_binary_file(map[:, 1].astype(np.float32), "numpy_input_ty.bin")
save_array_to_binary_file(map[:, 2].astype(np.float32), "numpy_input_tz.bin")

x_map2 = np.arange(map[:, 0].min(), map[:, 0].max(), 30)
y_map2 = np.arange(map[:, 1].min(), map[:, 1].max(), 30)
z_map2 = 100
X_map2, Y_map2, Z_map2 = np.meshgrid(x_map2, y_map2, z_map2)

body_map = np.concatenate([X_map2.flatten().reshape((len(X_map2.flatten()), 1)),
                           Y_map2.flatten().reshape((len(Y_map2.flatten()), 1)),
                           Z_map2.flatten().reshape((len(Z_map2.flatten()), 1))], axis=1).astype('float32')
save_array_to_binary_file(body_map[:, 0].astype(np.float32), "numpy_input_bx.bin")
save_array_to_binary_file(body_map[:, 1].astype(np.float32), "numpy_input_by.bin")
save_array_to_binary_file(body_map[:, 2].astype(np.float32), "numpy_input_bz.bin")
