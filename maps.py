# -*- coding: utf-8 -*-
"""
This creates several maps of foothold to be used by the Astar

@author: Elian NEPPEL
@laboratory: Space Robotic Lab, Tohoku University
"""
import numpy as np
import perlinnumpy2d as per
from perlinnumpy2d import generate_fractal_noise_2d, generate_perlin_noise_2d

density_in_pt_per_m2 = 16  # 10
x_min, x_max = -700, 4000
y_min, y_max = -700, 4000
point_density = density_in_pt_per_m2 * 1e-6

area = (x_max - x_min) * (y_max - y_min)
num_points = int(point_density * area)

np.random.seed(42)

x_coords = np.random.uniform(x_min, x_max, size=num_points)
y_coords = np.random.uniform(y_min, y_max, size=num_points)
z_coords = np.zeros(num_points)

random_map = np.column_stack((x_coords, y_coords, z_coords))

# plt.scatter(random_map[:, 0], random_map[:, 1])
# plt.axis("equal")
# plt.show()

x_map2 = np.arange(-700, 7000, 200)
y_map2 = np.arange(-500, 501, 200)
z_map2 = 0
X_map2, Y_map2, Z_map2 = np.meshgrid(x_map2, y_map2, z_map2)

flat_map = np.concatenate(
    [
        X_map2.flatten().reshape((len(X_map2.flatten()), 1)),
        Y_map2.flatten().reshape((len(Y_map2.flatten()), 1)),
        Z_map2.flatten().reshape((len(Z_map2.flatten()), 1)),
    ],
    axis=1,
).astype("float32")
x = flat_map[:, 0]
y = flat_map[:, 1]
obs_map = flat_map[~((x > 700) & (x < 4800) & (y > -500) & (y < 500))]
obs_map = obs_map * np.array([1, 1.15, 1])

x_map2 = np.arange(-700, 6800, 200)
y_map2 = np.arange(-500, 2001, 200)
z_map2 = 0
X_map2, Y_map2, Z_map2 = np.meshgrid(x_map2, y_map2, z_map2)

flat_map = np.concatenate(
    [
        X_map2.flatten().reshape((len(X_map2.flatten()), 1)),
        Y_map2.flatten().reshape((len(Y_map2.flatten()), 1)),
        Z_map2.flatten().reshape((len(Z_map2.flatten()), 1)),
    ],
    axis=1,
).astype("float32")
x = flat_map[:, 0]
y = flat_map[:, 1]
obs_map_large = flat_map[~((x > 700) & (x < 4800) & (y > -500) & (y < 500))]
obs_map_large[:, 1] = obs_map_large[:, 1] * 1.0

x_map2 = np.arange(0, 8000, 200)
y_map2 = np.arange(-1000, 6000, 200)
z_map2 = 0
X_map2, Y_map2, Z_map2 = np.meshgrid(x_map2, y_map2, z_map2)

flat_map = np.concatenate(
    [
        X_map2.flatten().reshape((len(X_map2.flatten()), 1)),
        Y_map2.flatten().reshape((len(Y_map2.flatten()), 1)),
        Z_map2.flatten().reshape((len(Z_map2.flatten()), 1)),
    ],
    axis=1,
).astype("float32")
minimap = flat_map[
    ~(
        (flat_map[:, 1] < 3500)
        & (flat_map[:, 1] > 1000)
        & (flat_map[:, 0] <= 6000)
        & (2000 <= flat_map[:, 0])
    )
]
minimap = minimap[
    ~((minimap[:, 1] <= 1000) & (minimap[:, 0] <= 6000) & (4000 <= minimap[:, 0]))
]
# plt.scatter(minimap[:, 0], minimap[:, 1])
# plt.axis("equal")
# plt.xlabel('x (mm)')
# plt.ylabel('y (mm)')
# plt.show()

x_map2 = np.arange(-1000, 8500, 200)
y_map2 = np.arange(-1500, 1500, 200)
z_map2 = 0
X_map2, Y_map2, Z_map2 = np.meshgrid(x_map2, y_map2, z_map2)

flat_map = np.concatenate(
    [
        X_map2.flatten().reshape((len(X_map2.flatten()), 1)),
        Y_map2.flatten().reshape((len(Y_map2.flatten()), 1)),
        Z_map2.flatten().reshape((len(Z_map2.flatten()), 1)),
    ],
    axis=1,
).astype("float32")

x_map2 = np.arange(-500, 501, 50)
y_map2 = np.arange(-500, 501, 50)
z_map2 = 0
X_map2, Y_map2, Z_map2 = np.meshgrid(x_map2, y_map2, z_map2)

step_map = np.concatenate(
    [
        X_map2.flatten().reshape((len(X_map2.flatten()), 1)),
        Y_map2.flatten().reshape((len(Y_map2.flatten()), 1)),
        Z_map2.flatten().reshape((len(Z_map2.flatten()), 1)),
    ],
    axis=1,
).astype("float32")
step_height = 300
step_map = np.concatenate(
    [
        step_map,
        step_map + np.array([1000, 0, step_height * 1]),
        step_map + np.array([1000, 1000, step_height * 2]),
        step_map + np.array([0, 1000, step_height * 3]),
    ]
)

x_map2 = np.arange(-500, 2001, 50)
y_map2 = np.arange(-400, 401, 50)
z_map2 = 0
X_map2, Y_map2, Z_map2 = np.meshgrid(x_map2, y_map2, z_map2)

fence_map = np.concatenate(
    [
        X_map2.flatten().reshape((len(X_map2.flatten()), 1)),
        Y_map2.flatten().reshape((len(Y_map2.flatten()), 1)),
        Z_map2.flatten().reshape((len(Z_map2.flatten()), 1)),
    ],
    axis=1,
).astype("float32")

x_map2 = 1000
y_map2 = y_map2
z_map2 = np.arange(200, 201, 50)
X_map2, Y_map2, Z_map2 = np.meshgrid(x_map2, y_map2, z_map2)

fence = np.concatenate(
    [
        X_map2.flatten().reshape((len(X_map2.flatten()), 1)),
        Y_map2.flatten().reshape((len(Y_map2.flatten()), 1)),
        Z_map2.flatten().reshape((len(Z_map2.flatten()), 1)),
    ],
    axis=1,
).astype("float32")
step_height = 300
fence_map = np.concatenate(
    [
        fence_map,
        fence,
    ]
)

# Moonmap ----------------------------------


def clip2sphere(
    center: np.ndarray, radius: float, map: np.ndarray, down: bool = True
) -> np.ndarray:
    inside_sphere = map[np.linalg.norm(map - center, axis=1) <= radius, :] - center
    xy_dist = np.linalg.norm(inside_sphere[:, [0, 1]], axis=1)
    z_height = inside_sphere[:, 2] if down else -inside_sphere[:, 2]
    delta = np.sqrt(radius**2 - xy_dist**2) + z_height
    delta3d = np.zeros(shape=(delta.shape[0], 3), dtype=float)
    delta3d[:, 2] = delta
    map[np.linalg.norm(map - center, axis=1) <= radius, :] -= (
        delta3d if down else -delta3d
    )
    return map


# x_map2 = np.arange(-2000, 2000, 15)
# y_map2 = np.arange(-2000, 2000, 15)
x_map2 = np.linspace(-2000, 2000, 2**8)
y_map2 = np.linspace(-6000, 2000, 2**8)
z_map2 = 0
X_map2, Y_map2, Z_map2 = np.meshgrid(x_map2, y_map2, z_map2)

ground = np.concatenate(
    [
        X_map2.flatten().reshape((len(X_map2.flatten()), 1)),
        Y_map2.flatten().reshape((len(Y_map2.flatten()), 1)),
        Z_map2.flatten().reshape((len(Z_map2.flatten()), 1)),
    ],
    axis=1,
).astype("float32")

np.random.seed(seed=42)


center_radius = np.random.random_sample(size=(50, 4))
b = 2000
a = -2000
center_radius[:, 0] = (b - a) * center_radius[:, 0] + a
b = 2000
a = -2000
center_radius[:, 1] = (b - a) * center_radius[:, 1] + a
b = -100
a = -400
center_radius[:, 2] = (b - a) * center_radius[:, 2] + a
b = 500
a = 200
center_radius[:, 3] = (b - a) * center_radius[:, 3] + a

for row in range(center_radius.shape[0]):
    ground = clip2sphere(
        center=center_radius[row, [0, 1, 2]],
        radius=center_radius[row, 3],
        map=ground,
        down=center_radius[row, 2] > 0,
    )

# center_radius = np.random.random_sample(size=(1600, 4))
# b = 2000
# a = -2000
# center_radius[:, 0] = (b - a) * center_radius[:, 0] + a
# b = 2000
# a = -2000
# center_radius[:, 1] = (b - a) * center_radius[:, 1] + a
# b = 200
# a = -200
# center_radius[:, 2] = (b - a) * center_radius[:, 2] + a
# b = 200
# a = 0
# center_radius[:, 3] = (b - a) * center_radius[:, 3] + a
#
# for row in range(center_radius.shape[0]):
#     ground = clip2sphere(
#         center=center_radius[row, [0, 1, 2]],
#         radius=center_radius[row, 3],
#         map=ground,
#         down=center_radius[row, 2] > 0,
#     )
#


ground = clip2sphere(
    center=np.array([-2000, -2000, 300], dtype=float),
    radius=2000,
    map=ground,
    down=True,
)

ground = clip2sphere(
    center=np.array([2000, 4000, -800], dtype=float),
    radius=4000,
    map=ground,
    down=False,
)

ground = clip2sphere(
    center=np.array([1500, 0, -320], dtype=float),
    radius=1000,
    map=ground,
    down=False,
)

ground = clip2sphere(
    center=np.array([1500, -1000, -150], dtype=float),
    radius=700,
    map=ground,
    down=False,
)


# noise = generate_perlin_noise_2d(shape=(x_map2.shape[0], y_map2.shape[0]), res=(8,8)) * 75
# noise += generate_perlin_noise_2d(shape=(x_map2.shape[0], y_map2.shape[0]), res=(16,16)) * 100
# noise += generate_perlin_noise_2d(shape=(x_map2.shape[0], y_map2.shape[0]), res=(4,4)) * 200
noise = generate_fractal_noise_2d(shape=(x_map2.shape[0], y_map2.shape[0]), res=(8,4), octaves=5, persistence=0.35, lacunarity=2) * 300
ground[:, 2] += noise.reshape(-1,) 

SAT = 1000
ground[ground[:, 2] > 1000, 2] = SAT

# noise = generate_perlin_noise_2d(shape=(x_map2.shape[0], y_map2.shape[0]), res=(16,16)) * 30
noise = generate_fractal_noise_2d(shape=(x_map2.shape[0], y_map2.shape[0]), res=(32,16), octaves=3, persistence=0.2, lacunarity=2) * 30
ground[:, 2] += noise.reshape(-1,) 
