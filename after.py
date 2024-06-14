import open3d as o3d
import maps
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.cm as cm
from setting import *

matplotlib.use("Agg")


def bool_grid_image(
    grid: np.ndarray, data: np.ndarray, black_white=True, transparency=False
):
    """
    From a grid of coordinates corresponding to pixels positions and the corresponding bool value of that pixel
    draw a matplotlib image (then you have to plt.show() to show this result)

    :param grid: shape(N,2) - coordinates of pixels arranged as a flattened grid
    :param data: shape(N,1) - values of pixel
    :return: nothing, you have to plt.show() to show this result
    """
    coord = grid.copy()
    coord[:, 1] = coord[:, 1]
    size = (len(np.unique(coord[:, 0])) - 1, len(np.unique(coord[:, 1])) - 1)
    xmin, xmax, ymin, ymax = (
        coord[:, 0].min(),
        coord[:, 0].max(),
        coord[:, 1].min(),
        coord[:, 1].max(),
    )
    img = np.ones(size, dtype=float)

    coord_as_int = np.empty(coord.shape, dtype=int)
    coord_as_int[:, 0] = (coord[:, 0] - xmin) * (size[0] - 1) // (xmax - xmin)
    coord_as_int[:, 1] = -(coord[:, 1] - ymin) * (size[1] - 1) // (ymax - ymin)
    coord_as_int = coord_as_int.astype(int)

    img[coord_as_int[:, 0], coord_as_int[:, 1]] = data
    img = img.transpose()

    if black_white:
        if transparency:
            img_rgba = np.zeros((img.shape[0], img.shape[1], 4))
            img_rgba[..., -1] = img
            plt.imshow(img_rgba, extent=[xmin, xmax, ymin, ymax])
        else:
            plt.imshow(img, cmap="gray_r", extent=[xmin, xmax, ymin, ymax])
    else:
        plt.imshow(img, extent=[xmin, xmax, ymin, ymax])
    return


# Function to read binary file into numpy array along with length
def read_array_from_file_with_length(filename, dtype):
    with open(filename, "rb") as file:
        # Read array data
        array_data = np.fromfile(file, dtype=dtype)
    return array_data


filename = "cpp_array_xx.bin"
xx = read_array_from_file_with_length(filename, np.float32)
filename = "cpp_array_xy.bin"
xy = read_array_from_file_with_length(filename, np.float32)
filename = "cpp_array_xz.bin"
xz = read_array_from_file_with_length(filename, np.float32)

grid = np.empty(shape=(len(xy), 3))
grid[:, 0] = xx
grid[:, 1] = xy
grid[:, 2] = xz

filename = "cpp_array_y.bin"
reach_count = read_array_from_file_with_length(filename, np.int32)

bool_grid_image(
    grid[:, [0, 1]],
    np.clip(reach_count / (reach_count.max() * 3 / 4), 0, 1),
    black_white=False,
    transparency=False,
)

map = np.load("map.npy")
plt.scatter(map[:, 0], map[:, 1], c="red", s=5)

# plt.xlabel(f'Index shape: xx{xx.shape} xy{xy.shape}')
# plt.ylabel('Value')
# plt.title('Title')
# plt.grid(False)
# plt.axis("equal")

plt.savefig("graph1.png")
plt.clf()

filename = "out_dist_xx.bin"
xx = read_array_from_file_with_length(filename, np.float32)
filename = "out_dist_xy.bin"
xy = read_array_from_file_with_length(filename, np.float32)
filename = "out_dist_xz.bin"
xz = read_array_from_file_with_length(filename, np.float32)

dist = np.empty(shape=(len(xx), 3))
dist[:, 0] = xx
dist[:, 1] = xy
dist[:, 2] = xz

filename = "dist_input_tx.bin"
tx = read_array_from_file_with_length(filename, np.float32)
filename = "dist_input_ty.bin"
ty = read_array_from_file_with_length(filename, np.float32)
filename = "dist_input_tz.bin"
tz = read_array_from_file_with_length(filename, np.float32)

targets = np.empty(shape=(len(tx), 3))
targets[:, 0] = tx
targets[:, 1] = ty
targets[:, 2] = tz

closest_to_0 = min(targets[targets[:, 1] >= 0, 1])
selection = targets[:, 1] == closest_to_0

filename = "out_reachability.bin"
reach = read_array_from_file_with_length(filename, bool).astype(bool)

z_slice_cut = 200

if vertical_slice:
    closest_to_0 = min(targets[targets[:, 1] >= 0, 1])
    selection = targets[:, 1] == closest_to_0
    zero_plane = targets[selection][:, [0, 2]]
else:
    closest_to_0 = min(targets[(targets[:, 2] + z_slice_cut) >= 0, 2])
    # closest_to_0 = min(targets[(targets[:, 2] + 230) >= 0, 2])
    selection = targets[:, 2] == closest_to_0
    zero_plane = targets[targets[:, 2] == closest_to_0][:, [0, 1]]

plt.grid(True)
bool_grid_image(
    zero_plane,
    reach[selection],
    black_white=True,
    transparency=False,
)

plt.xlabel("x (mm)")
if vertical_slice:
    plt.title("Reachable area of Moonbot leg (y=0)")
    plt.ylabel("z (mm)")
else:
    plt.title(f"Reachable area of Moonbot leg (z={-z_slice_cut})")
    plt.ylabel("y (mm)")
plt.scatter(0, 0, s=0.01, c="black", marker="s", label="Reachable")
legend = plt.legend(loc="upper left")
for handle in legend.legendHandles:
    handle.set_sizes([50.0])
plt.savefig("reachability_result.png", bbox_inches="tight", dpi=300)
plt.clf()


plt.grid(True)
bool_grid_image(
    zero_plane,
    np.linalg.norm(dist[selection, :], axis=1),
    black_white=False,
    transparency=False,
)
sel2 =np.linalg.norm(dist[selection, :], axis=1) < 1
bool_grid_image(
    zero_plane,
    sel2,
    black_white=True,
    transparency=True,
)

plt.xlabel("x (mm)")
if vertical_slice:
    plt.title("Distance to reachablity's edge (y=0)")
    plt.ylabel("z (mm)")
else:
    plt.title(f"Distance to reachablity's edge (z={-z_slice_cut})")
    plt.ylabel("y (mm)")
plt.scatter(0, 0, s=0.01, c="black", marker="s", label="Reachability's edge")
legend = plt.legend(loc="upper left")
for handle in legend.legendHandles:
    handle.set_sizes([50.0])
plt.savefig("distance_result.png", bbox_inches="tight", dpi=300)

shaved = targets[reach, :]
np.save("leg0_reach.npy", shaved)

if False:
    r_pcd = o3d.geometry.PointCloud()
    r_pcd.points = o3d.utility.Vector3dVector(shaved)

    cmap = cm.get_cmap("viridis")
    norm = plt.Normalize(vmin=np.min(shaved[:, 2]), vmax=np.max(shaved[:, 2]))
    colors_rgb = np.array(cmap(norm(shaved[:, 2])))[:, :3]
    print(colors_rgb)

    r_pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        r_pcd, voxel_size=np.linalg.norm(targets[0, :] - targets[1, :])
    )

    o3d.visualization.draw_geometries([voxel_grid])

select = reach_count > 1
shaved = grid[select, :]
intensity = reach_count[select]
np.save("robot_reach.npy", shaved)
np.save("robot_reach_intens.npy", intensity)
print(f"robot reachable samples: {select.sum()}")
d = np.linalg.norm(grid[:-1, :] - grid[1:, :], axis=1)
d = np.min(d) / 1_000
# delta = max(abs(grid[0, :] - grid[1, :])) / 1_000
delta = d
print(f"detected voxel size: {delta}")
print(f"robot reachable m^3: {select.sum() * delta**3}")

print("python post process finished")

if False:
    map_pcd = o3d.geometry.PointCloud()
    map_pcd.points = o3d.utility.Vector3dVector(map)

    select = reach_count > 1
    shaved = grid[select, :]
    intensity = reach_count[select]
    np.save("robot_reach.npy", shaved)
    np.save("robot_reach_intens.npy", intensity)
    r_pcd = o3d.geometry.PointCloud()
    r_pcd.points = o3d.utility.Vector3dVector(shaved)

    cmap = cm.get_cmap("viridis")
    norm = plt.Normalize(vmin=np.min(intensity), vmax=np.max(intensity))
    colors_rgb = np.array(cmap(norm(intensity)))[:, :3]

    r_pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        r_pcd, voxel_size=np.linalg.norm(grid[0, :] - grid[1, :])
    )

    o3d.visualization.draw_geometries([map_pcd, voxel_grid])
