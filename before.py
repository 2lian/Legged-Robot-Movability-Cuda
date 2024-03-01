import numpy as np
import maps


def save_array_to_binary_file(array, filename):
    with open(filename, 'wb') as f:
        array.tofile(f)
    return
 
map = maps.step_map
np.save("map.npy", map)
print("\nmap shape: ", map.shape)
save_array_to_binary_file(map[:, 0].astype(np.float32), "numpy_input_tx.bin")
save_array_to_binary_file(map[:, 1].astype(np.float32), "numpy_input_ty.bin")
save_array_to_binary_file(map[:, 2].astype(np.float32), "numpy_input_tz.bin")

x_map2 = np.arange(map[:, 0].min(), map[:, 0].max(), 25)
y_map2 = np.arange(map[:, 1].min(), map[:, 1].max(), 25)
z_map2 = np.arange(map[:, 2].min(), map[:, 2].max() + 300, 25)
# z_map2 = np.arange(50, 450, 25) 
X_map2, Y_map2, Z_map2 = np.meshgrid(x_map2, y_map2, z_map2)

body_map = np.concatenate([X_map2.flatten().reshape((len(X_map2.flatten()), 1)),
                           Y_map2.flatten().reshape((len(Y_map2.flatten()), 1)),
                           Z_map2.flatten().reshape((len(Z_map2.flatten()), 1))], axis=1).astype('float32')
save_array_to_binary_file(body_map[:, 0].astype(np.float32), "numpy_input_bx.bin")
save_array_to_binary_file(body_map[:, 1].astype(np.float32), "numpy_input_by.bin")
save_array_to_binary_file(body_map[:, 2].astype(np.float32), "numpy_input_bz.bin")

print("body samples shape: ", body_map.shape)

x_map_dist = np.arange(50,  601, 1)
y_map_dist = np.arange(-500, 500, 20) 
z_map_dist = np.arange(-300, 200, 1) - 35
X_map_dist, Y_map_dist, Z_map_dist = np.meshgrid(x_map_dist, y_map_dist, z_map_dist)

dist_map = np.concatenate([X_map_dist.flatten().reshape((len(X_map_dist.flatten()), 1)),
                           Y_map_dist.flatten().reshape((len(Y_map_dist.flatten()), 1)),
                           Z_map_dist.flatten().reshape((len(Z_map_dist.flatten()), 1))], axis=1).astype('float32')
save_array_to_binary_file(dist_map[:, 0].astype(np.float32), "dist_input_tx.bin")
save_array_to_binary_file(dist_map[:, 1].astype(np.float32), "dist_input_ty.bin")
save_array_to_binary_file(dist_map[:, 1].astype(np.float32), "test.bin")
save_array_to_binary_file(dist_map[:, 2].astype(np.float32), "dist_input_tz.bin")

print("reachability shape: ", dist_map.shape)
print("\n")