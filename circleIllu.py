import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import PathPatch, Patch
from matplotlib.path import Path

body2coxa = 181
coxa_pitch = np.deg2rad(-45)
coxa2femur = 65.5
femur2tibia = 129
tibia2tip = 160

coxa_margin = 0.0
femur_margin = 0.0
tibia_margin = 0.0

coxa_angle = np.deg2rad(60.0)
femur_angle = np.deg2rad(90.0)
tibia_angle = np.deg2rad(120.0)

dist_margin = 0.0
aboveang = -5
tib_abs_pos = np.deg2rad(aboveang)
tib_abs_neg = np.deg2rad(-180 - (aboveang))
coxa2femur = coxa2femur * np.cos(coxa_pitch)
coxa2femurZ = coxa2femur * np.sin(coxa_pitch)
# Create a figure and axis
fig, ax = plt.subplots()
bkgColor = "red"
falseColor = "red"
almostColor = "grey"


# Create a rectangle that covers the entire plot area with the desired pattern
background = patches.Rectangle(
    (-1000, -1000),
    2000,
    2000,
    edgecolor=bkgColor,
    facecolor="none",
    hatch="//",
)

# Add the rectangle to the plot
ax.add_patch(background)

# Define the center and radius of the circle
center = (body2coxa + coxa2femur, coxa2femurZ)
radius = femur2tibia + tibia2tip

# Create a circle with the specified center and radius, and fill it with a pattern
circle = patches.Circle(
    center,
    radius,
    edgecolor="black",
    facecolor="white",
    hatch="",
)

# Add the circle to the plot
ax.add_patch(circle)

P0 = (body2coxa + coxa2femur) + (coxa2femurZ * 1j)
P1 = P0 + (femur2tibia + tibia2tip + 0j) * np.exp((-femur_angle + coxa_pitch) * 1j) * 2
P2 = P0 + (femur2tibia + tibia2tip + 0j) * np.exp(tib_abs_pos * 1j) * 2
plt.fill_between(
    [center[0], np.real(P1)],
    [center[1], np.imag(P1)],
    y2=1000,
    color="none",
    hatch="/",
    edgecolor=bkgColor,
    linewidth=0.0,
)

plt.fill_between(
    [center[0], np.real(P2)],
    [center[1], np.imag(P2)],
    y2=1000,
    color="none",
    hatch="/",
    edgecolor=bkgColor,
    linewidth=0.0,
)

A = np.array([P1, P0, P2])
plt.plot(np.real(A), np.imag(A), c="black", zorder=0)

center = (
    body2coxa + coxa2femur + femur2tibia * np.cos(femur_angle + coxa_pitch),
    coxa2femurZ + femur2tibia * np.sin(femur_angle + coxa_pitch),
)
radius = tibia2tip
circle = patches.Circle(
    center,
    radius,
    edgecolor="none",
    facecolor="white",
)
ax.add_patch(circle)

center = (
    body2coxa + coxa2femur + femur2tibia * np.cos(-femur_angle + coxa_pitch),
    coxa2femurZ + femur2tibia * np.sin(-femur_angle + coxa_pitch),
)
radius = tibia2tip
circle = patches.Circle(
    center,
    radius,
    edgecolor="none",
    facecolor="white",
)
ax.add_patch(circle)


center = (
    body2coxa + coxa2femur + tibia2tip * np.cos(tib_abs_pos),
    coxa2femurZ + tibia2tip * np.sin(tib_abs_pos),
)
radius = femur2tibia
circle = patches.Circle(
    center,
    radius,
    edgecolor="none",
    facecolor="white",
)
ax.add_patch(circle)

center = (
    body2coxa + coxa2femur + tibia2tip * np.cos(tib_abs_neg),
    coxa2femurZ + tibia2tip * np.sin(tib_abs_neg),
)
radius = femur2tibia
circle = patches.Circle(
    center,
    radius,
    edgecolor="none",
    facecolor="white",
)
ax.add_patch(circle)

center = (
    body2coxa + coxa2femur + femur2tibia * np.cos(-femur_angle + coxa_pitch),
    coxa2femurZ + femur2tibia * np.sin(-femur_angle + coxa_pitch),
)
radius = tibia2tip
circle = patches.Circle(
    center,
    radius,
    edgecolor=almostColor,
    facecolor="none",
    hatch=".",
)
ax.add_patch(circle)


center = (
    body2coxa + coxa2femur + femur2tibia * np.cos(femur_angle + coxa_pitch),
    coxa2femurZ + femur2tibia * np.sin(femur_angle + coxa_pitch),
)
radius = tibia2tip
circle = patches.Circle(
    center,
    radius,
    edgecolor=falseColor,
    facecolor="none",
    hatch=".",
)
ax.add_patch(circle)
circle = patches.Circle(
    center,
    radius,
    edgecolor="black",
    facecolor="none",
)
ax.add_patch(circle)

center = (
    body2coxa + coxa2femur + tibia2tip * np.cos(tib_abs_pos),
    coxa2femurZ + tibia2tip * np.sin(tib_abs_pos),
)
radius = femur2tibia
circle = patches.Circle(
    center,
    radius,
    edgecolor=almostColor,
    facecolor="none",
    hatch="o",
)
ax.add_patch(circle)

center = (
    body2coxa + coxa2femur + tibia2tip * np.cos(tib_abs_neg),
    coxa2femurZ + tibia2tip * np.sin(tib_abs_neg),
)
radius = femur2tibia
circle = patches.Circle(
    center,
    radius,
    edgecolor=falseColor,
    facecolor="none",
    hatch="o",
)
ax.add_patch(circle)
circle = patches.Circle(
    center,
    radius,
    edgecolor="black",
    facecolor="none",
)
ax.add_patch(circle)

center = (body2coxa + coxa2femur, coxa2femurZ + 0)
radius = np.abs(femur2tibia + (tibia2tip + 0j) * np.exp(tibia_angle * 1j))
circle = patches.Circle(
    center,
    radius,
    edgecolor=bkgColor,
    facecolor="none",
    hatch="//",
)
ax.add_patch(circle)
circle = patches.Circle(
    center,
    radius,
    edgecolor="black",
    facecolor="none",
    hatch="",
)
ax.add_patch(circle)


ax.set_xlim(
    -(femur2tibia + tibia2tip + 25) + body2coxa + coxa2femur,
    femur2tibia + tibia2tip + 25 + body2coxa + coxa2femur,
)
ax.set_ylim(
    coxa2femurZ - (femur2tibia + tibia2tip + 25),
    coxa2femurZ + femur2tibia + tibia2tip + 25,
)


# Create custom patches for the legend
handleList = []

handleList.append(
    Patch(facecolor="none", edgecolor="black", hatch="//", label="$C_{out}$ and $C_{in}$")
)
handleList.append(
    Patch(facecolor="none", edgecolor="black", hatch=".", label="""$C_{\\beta}^{\\pm}$""")
)
handleList.append(
    Patch(facecolor="none", edgecolor="black", hatch="o", label="""$C_{\\phi}^{\\pm}$""")
)
handleList.append(
    Patch(facecolor="none", edgecolor=falseColor, hatch="", label="""Un-reachable""")
)

# Add the legend
ax.legend(
    handles=handleList,
    loc="upper left",
    handleheight=3,
    handlelength=2.0,
    # fontsize="large",
)


plt.xlabel(f"x (mm)")
plt.ylabel(f"z (mm)")
ax.set_aspect("equal")
# Show the plot
plt.savefig(f"image/circleIllu.jpg", bbox_inches="tight", dpi=150)
plt.savefig(f"image/circleIllu.png", bbox_inches="tight", dpi=150)
plt.savefig(f"image/circleIllu.pdf", bbox_inches="tight", dpi=150)
plt.savefig(f"image/circleIllu.eps", bbox_inches="tight", dpi=150)
