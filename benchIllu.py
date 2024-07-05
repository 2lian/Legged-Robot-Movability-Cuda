import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

D = "dict"
N = "name"
PC_GPU = "GPU GTX1080Ti"
PC_CPU = "CPU i5-12600K"
JET_GPU = "GPU Jetson Orin NX"
JET_CPU = "CPU Jetson Orin NX"

faceSetting = {
    PC_GPU: {
        "marker": "o",
        "line": "-",
        "color": "black",
    },
    PC_CPU: {
        "marker": "o",
        "line": "--",
        "color": "firebrick",
    },
    JET_GPU: {
        "marker": "x",
        "line": "-",
        "color": "black",
    },
    JET_CPU: {
        "marker": "x",
        "line": "--",
        "color": "firebrick",
    },
}

file_list_r = [
    {
        D: "./bdata/pc/rgpu.csv",
        N: PC_GPU,
    },
    {
        D: "./bdata/pc/rcpu.csv",
        N: PC_CPU,
    },
    {
        D: "./bdata/pc/rbdl.csv",
        N: "rbdl",
    },
    {
        D: "./bdata/jetson/rgpu.csv",
        N: JET_GPU,
    },
    {
        D: "./bdata/jetson/rcpu.csv",
        N: JET_CPU,
    },
    {
        D: "./bdata/jetson/rbdl.csv",
        N: "rbdl",
    },
]

file_list_d = [
    {
        D: "./bdata/pc/dgpu.csv",
        N: PC_GPU,
    },
    {
        D: "./bdata/pc/dcpu.csv",
        N: PC_CPU,
    },
    {
        D: "./bdata/jetson/dgpu.csv",
        N: JET_GPU,
    },
    {
        D: "./bdata/jetson/dcpu.csv",
        N: JET_CPU,
    },
]
# Set global font properties
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 16,
        "font.serif": ["DejaVu Serif"],
        # 'axes.titlesize': 16,
        # 'axes.labelsize': 14,
        # 'legend.fontsize': 12,
        # 'xtick.labelsize': 12,
        # 'ytick.labelsize': 12,
        "lines.linewidth": 3,  # Default line width
        "lines.markersize": 12,  # Default marker size
    }
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

for file in file_list_r:
    path = file[D]
    name = file[N]
    if name == "rbdl":
        continue
    # Read the CSV file using numpy
    data = np.genfromtxt(path, delimiter=";")

    # Split the data into x and y values
    x_values = data[:, 0]
    y_values = data[:, 1]

    ax1.plot(
        x_values,
        y_values,
        marker=faceSetting[name]["marker"],
        color=faceSetting[name]["color"],
        linestyle=faceSetting[name]["line"],
        label=name,
    )

for file in file_list_d:
    path = file[D]
    name = file[N]
    if name == "rbdl":
        continue
    # Read the CSV file using numpy
    data = np.genfromtxt(path, delimiter=";")

    # Split the data into x and y values
    x_values = data[:, 0]
    y_values = data[:, 1]

    ax2.plot(
        x_values,
        y_values,
        marker=faceSetting[name]["marker"],
        color=faceSetting[name]["color"],
        linestyle=faceSetting[name]["line"],
        label=name,
    )

# Set plot labels and title
ax2.set_xlabel("Sample Count")
# ax1.legend()

for ind, ax in enumerate([ax1, ax2]):
    if ind == 0:
        ax.set_ylabel("Reachability Execution Time \n(ns/sample)")
    if ind == 1:
        ax.set_ylabel("Distance Execution Time \n(ns/sample)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", ls="-", linewidth=1, color="black", alpha=0.4)
    ax.grid(True, which="minor", ls="-", linewidth=0.4, color="black", alpha=0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

# Extract the legend handles and labels
handles, labels = ax1.get_legend_handles_labels()

# Create an axis for the legend
legend_ax = fig.add_axes([0.05, 0.93, 0.8, 0.04], frame_on=False)
legend_ax.axis("off")  # Hide the axis

# Add the legend to the custom axis
legend_ax.legend(
    handles=handles,
    labels=labels,
    loc="center",
    ncol=2,
    # handleheight=2.5,
    handlelength=4.0,
    fontsize=14,
    facecolor="None",  # Set background color
    edgecolor="None",  # Set border color
)

plt.subplots_adjust(top=0.90, hspace=0.02)


# Show the plot
# plt.show()
plt.savefig(f"image/benchIllu.jpg", bbox_inches="tight", dpi=150)
plt.savefig(f"image/benchIllu.png", bbox_inches="tight", dpi=150)
plt.savefig(f"image/benchIllu.pdf", bbox_inches="tight", dpi=150)
plt.savefig(f"image/benchIllu.eps", bbox_inches="tight", dpi=150)
