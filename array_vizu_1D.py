import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def bool_grid_image(grid: np.ndarray, data: np.ndarray, black_white=True, transparency = True):
    """
    From a grid of coordinates corresponding to pixels positions and the corresponding bool value of that pixel
    draw a matplotlib image (then you have to plt.show() to show this result)

    :param grid: shape(N,2) - coordinates of pixels arranged as a flattened grid
    :param data: shape(N,1) - values of pixel
    :return: nothing, you have to plt.show() to show this result
    """
    coord = grid.copy()
    coord[:, 1] = -coord[:, 1]
    size = (len(np.unique(coord[:, 0])) - 1, len(np.unique(coord[:, 1])) - 1)
    xmin, xmax, ymin, ymax = coord[:, 0].min(), coord[:, 0].max(), coord[:, 1].min(), coord[:, 1].max()
    img = np.ones(size, dtype=float)

    coord_as_int = np.empty(coord.shape, dtype=int)
    coord_as_int[:, 0] = (coord[:, 0] - xmin) * (size[0] - 1) // (xmax - xmin)
    coord_as_int[:, 1] = (coord[:, 1] - ymin) * (size[1] - 1) // (ymax - ymin)
    coord_as_int = coord_as_int.astype(int)

    img[coord_as_int[:, 0], coord_as_int[:, 1]] = data
    img = img.transpose()

    if black_white:
        if transparency:
            img_rgba = np.zeros((img.shape[0], img.shape[1], 4))
            img_rgba[..., -1] = img
            plt.imshow(img_rgba, extent=[xmin, xmax, ymin, ymax])
        else:
            plt.imshow(img, cmap='gray_r', extent=[xmin, xmax, ymin, ymax])
    else:
        plt.imshow(img, extent=[xmin, xmax, ymin, ymax])
    return
 

# Function to read binary file into numpy array along with length
def read_array_from_file_with_length(filename, dtype):
    with open(filename, 'rb') as file:
        # Read array data
        array_data = np.fromfile(file, dtype=dtype)
    return array_data

filename = 'cpp_array_xx.bin'
xx = read_array_from_file_with_length(filename, np.float32) 
filename = 'cpp_array_xy.bin'
xy = read_array_from_file_with_length(filename, np.float32) 

grid = np.empty(shape=(len(xy), 2))
grid[:,0] = xx
grid[:,1] = xy

filename = 'cpp_array_y.bin'
y = read_array_from_file_with_length(filename, np.int32)

bool_grid_image(grid, y, black_white=False)

# plt.scatter(xx, xy, c=y, s=10)
# plt.xlabel(f'Index shape: xx{xx.shape} xy{xy.shape}')
# plt.ylabel('Value')
# plt.title('Title')
# plt.grid(False)
# plt.axis("equal")

plt.savefig("graph.png")
