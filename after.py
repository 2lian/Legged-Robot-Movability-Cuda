from numpy.typing import NDArray
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import open3d as o3d
import maps
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.cm as cm
from setting import *

matplotlib.use("Agg")
if VERT_SLICE:
    postfix = "_side"
else:
    postfix = "_top"


def get_closest_vector(point_a: NDArray, points: NDArray, vectors: NDArray):
    distances = np.linalg.norm(points - point_a, axis=1)
    closest_index = np.argmin(distances)
    closest_vector = vectors[closest_index]

    return closest_vector


def bool_grid_image(
    grid: np.ndarray,
    data: np.ndarray,
    black_white=True,
    transparency=False,
    bool_color: NDArray = np.array([0, 0, 0]),
    background: NDArray = np.array([1, 1, 1]),
    true_alpha=1,
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
        img_rgba = np.zeros((img.shape[0], img.shape[1], 4))
        if transparency:
            img_rgba[..., -1] = img * true_alpha
        else:
            img_rgba[..., -1] = 1
        for i in range(3):
            img_rgba[..., i] = (1 - img) * background[i] - (0 - img) * bool_color[i]
        plt.imshow(img_rgba, extent=[xmin, xmax, ymin, ymax])
        # plt.imshow(img, cmap="gray_r", extent=[xmin, xmax, ymin, ymax])
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
reach_count = np.empty(shape=(len(xy),)) + 2
# filename = "cpp_array_y.bin"
# reach_count = read_array_from_file_with_length(filename, np.int32)

# bool_grid_image(
#     grid[:, [0, 1]],
#     np.clip(reach_count / (reach_count.max() * 3 / 4), 0, 1),
#     black_white=False,
#     transparency=False,
# )
#
map = np.load("map.npy")
# plt.scatter(map[:, 0], map[:, 1], c="red", s=5)

# plt.xlabel(f'Index shape: xx{xx.shape} xy{xy.shape}')
# plt.ylabel('Value')
# plt.title('Title')
# plt.grid(False)
# plt.axis("equal")

# plt.savefig("graph1.png")
# plt.clf()

filename = "out_dist_xx.bin"
# filename = "out_rec_xx.bin"
xx = read_array_from_file_with_length(filename, np.float32)
filename = "out_dist_xy.bin"
# filename = "out_rec_xy.bin"
xy = read_array_from_file_with_length(filename, np.float32)
filename = "out_dist_xz.bin"
# filename = "out_rec_xz.bin"
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

if VERT_SLICE:
    plane_sel = [0, 2]
    closest_to_0 = min(targets[targets[:, 1] >= 0, 1])
    selection = targets[:, 1] == closest_to_0
    zero_plane = targets[selection][:, [0, 2]]
else:
    plane_sel = [0, 1]
    closest_to_0 = min(targets[(targets[:, 2] - Z_CUT) >= 0, 2])
    # closest_to_0 = min(targets[(targets[:, 2] + 230) >= 0, 2])
    selection = targets[:, 2] == closest_to_0
    zero_plane = targets[targets[:, 2] == closest_to_0][:, [0, 1]]

color = np.array([0, 0.5, 0.3])
plt.grid(True)
plt.scatter(0, 0, s=0.0001, color=color, marker="s", label="Reachable")
bool_grid_image(
    zero_plane,
    reach[selection],
    black_white=True,
    transparency=False,
    bool_color=color,
)

plt.xlabel("x (mm)")
if VERT_SLICE:
    if TITLE:
        plt.title("Reachable area of Moonbot leg (y=0)")
    plt.ylabel("z (mm)")
else:
    if TITLE:
        plt.title(f"Reachable area of Moonbot leg (z={Z_CUT})")
    plt.ylabel("y (mm)")
legend = plt.legend(loc="upper left")
for handle in legend.legendHandles:
    handle.set_sizes([50.0])
plt.savefig(f"reachability_result{postfix}.jpg", bbox_inches="tight", dpi=150)
plt.savefig(f"reachability_result{postfix}.png", bbox_inches="tight", dpi=150)
plt.savefig(f"reachability_result{postfix}.pdf", bbox_inches="tight", dpi=150)
plt.savefig(f"reachability_result{postfix}.eps", bbox_inches="tight", dpi=150)
plt.clf()

# plt.grid(True)
sel1 = np.linalg.norm(dist[selection, :], axis=1)
sel1 = np.minimum(sel1, SATURATE)
if GRADIENT:
    bool_grid_image(
        zero_plane,
        sel1,
        black_white=False,
        transparency=False,
    )
    color = np.array([1, 1, 1])
else:
    color = np.array([0, 0, 0])
# sel2 = np.all(dist[selection, :] == [0,0,0], axis=1)
sel2 = sel1 < max(PIX_SIZE / 2, 1)
plt.scatter(0, 0, s=0.0, color=color, marker="_", label="Reachability edge")
plt.scatter(0, 0, s=0.0000, c="black", marker="$\leftarrow$", label="Vector to the edge")
bool_grid_image(
    zero_plane,
    sel2,
    black_white=True,
    transparency=True,
    bool_color=color,
)

# plt.xlabel(f"x (mm) <min = {min(sel1)}, max = {max(sel1)}>")
plt.xlabel(f"x (mm)")
if VERT_SLICE:
    if TITLE:
        plt.title("y=0 plane")
    plt.ylabel("z (mm)")
else:
    if TITLE:
        plt.title(f"z={Z_CUT} plane")
    plt.ylabel("y (mm)")

if LEGEND:
    legend = plt.legend(loc="upper left", facecolor=(0.5, 0.5, 0.5, 1))
    for handle in legend.legendHandles:
        handle.set_sizes([50.0])

if QUIVER:
    tailCount = 5
    x_map_dist = np.linspace(
        min(targets[:, plane_sel[0]]), max(targets[:, plane_sel[0]]), tailCount
    )
    z_map_dist = np.linspace(
        min(targets[:, plane_sel[1]]), max(targets[:, plane_sel[1]]) - 0, tailCount
    )
    X_map_dist, Z_map_dist = np.meshgrid(x_map_dist, z_map_dist)
    point = np.concatenate(
        [
            X_map_dist.flatten().reshape((len(X_map_dist.flatten()), 1)),
            Z_map_dist.flatten().reshape((len(Z_map_dist.flatten()), 1)),
        ],
        axis=1,
    ).astype("float32")
    endp = np.empty_like(point)
    for i in range(point.shape[0]):
        endp[i, :] = (
            get_closest_vector(point[i, :], targets[:, plane_sel], dist[:, plane_sel])
            * -1
        )
    plt.quiver(
        point[:, 0], point[:, 1], endp[:, 0], endp[:, 1], scale_units="xy", scale=1
    )

if COLORBAR:
    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=0, vmax=SATURATE)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # [left, bottom, width, height]
    cax = plt.axes((0.92, 0.2, 0.03, 0.59))
    colorbar = plt.colorbar(sm, cax=cax)
# Set the ticks and labels, replacing the last tick with '>= 400'
    ticks = np.linspace(0, SATURATE, 5)
# ticks = [0, 100, 200, 300, 400]
    tick_labels = np.round(ticks).astype(int).astype(str)
    tick_labels[-1] = f">{SATURATE}"
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(tick_labels)

    colorbar.set_label("Distance to reachability edge (mm)")

plt.savefig(f"distance_result{postfix}.jpg", bbox_inches="tight", dpi=150)
plt.savefig(f"distance_result{postfix}.png", bbox_inches="tight", dpi=150)
plt.savefig(f"distance_result{postfix}.pdf", bbox_inches="tight", dpi=150)
plt.savefig(f"distance_result{postfix}.eps", bbox_inches="tight", dpi=150)

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

if grid.shape[0] > 1:
    select = reach_count > 1
    shaved = grid[select, :]
    intensity = reach_count[select]
    np.save("robot_reach.npy", shaved)
    np.save("robot_reach_intens.npy", intensity)
    print(f"robot reachable samples: {select.sum()}")
    d = np.linalg.norm(grid[:-1, :] - grid[1:, :], axis=1)
    print(d)
    d = np.min(d) / 1_000
    # delta = max(abs(grid[0, :] - grid[1, :])) / 1_000
    delta = d
    print(f"detected voxel size: {delta}")
    print(f"robot reachable m^3: {select.sum() * delta**3}")
else:
    print("Warning: no points.")

print("python post process finished")

if False:
    map_pcd = o3d.geometry.PointCloud()
    map_pcd.points = o3d.utility.Vector3dVector(map)

    # select = reach_count > 1
    # select = True
    # shaved = grid[select, :]
    # intensity = reach_count[select]
    # np.save("robot_reach.npy", shaved)
    # np.save("robot_reach_intens.npy", intensity)
    print(np.max(grid))
    r_pcd = o3d.geometry.PointCloud()
    r_pcd.points = o3d.utility.Vector3dVector(grid[:12000, :])

    # cmap = cm.get_cmap("viridis")
    # norm = plt.Normalize(vmin=np.min(intensity), vmax=np.max(intensity))
    # colors_rgb = np.array(cmap(norm(intensity)))[:, :3]
    # colors_rgb = np.array(cmap(norm(intensity)))[:, :3]
    colors_rgb = np.zeros(shape=(grid.shape[0], 3))
    colors_rgb[:, 1] = 255

    r_pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
    # voxel_grid = r_pcd
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        r_pcd,
        voxel_size=100.1,
        # r_pcd, voxel_size=np.linalg.norm(grid[0, :] - grid[1, :])
    )

    o3d.visualization.draw_geometries([map_pcd, voxel_grid])
    # o3d.visualization.draw_geometries([r_pcd])
