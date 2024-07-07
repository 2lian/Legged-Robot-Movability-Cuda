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
        "line": "-",
        "color": "firebrick",
    },
    JET_GPU: {
        "marker": "x",
        "line": "--",
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
        N: "pcrbdl",
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
        N: "jetrbdl",
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
        "font.size": 19,
        "font.serif": ["DejaVu Serif"],
        # 'axes.titlesize': 16,
        # 'axes.labelsize': 17,
        "legend.fontsize": 15,
        "xtick.labelsize": 16,
        "ytick.labelsize": 15,
        "lines.linewidth": 3,  # Default line width
        "lines.markersize": 12,  # Default marker size
    }
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)


for p in [0, 1]:
    if p == 0:
        filelist = file_list_r
        ax = ax1
    else:
        filelist = file_list_d
        ax = ax2
    for file in filelist:
        path = file[D]
        name = file[N]
        if "rbdl" in name:
            continue
        # Read the CSV file using numpy
        data = np.genfromtxt(path, delimiter=";")
        x = data[:, 0]
        y = data[:, 1]
        unique_x = np.unique(x)
        # average_y = np.array([np.median(y[x == ux]) for ux in unique_x])
        low_y = np.array([np.percentile(y[x == ux], 5) for ux in unique_x])
        high_y = np.array([np.percentile(y[x == ux], 95) for ux in unique_x])
        for i in range(len(unique_x)):
            ux = unique_x[i]
            mask = (x == ux) & (y > high_y[i])
            mask |= (x == ux) & (y < low_y[i])
            print(mask.sum())
            y[mask] = np.nan
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]
        average_y = np.array([np.mean(y[x == ux]) for ux in unique_x])
        std_y = np.array([np.std(y[x == ux]) for ux in unique_x])
        low_y = np.array([np.percentile(y[x == ux], 5) for ux in unique_x])
        high_y = np.array([np.percentile(y[x == ux], 95) for ux in unique_x])

        # Split the data into x and y values
        x_values = unique_x
        y_values = average_y

        # ax.plot(
        #     x_values,
        #     high_y,
        #     color="grey",
        # )
        # ax.plot(
        #     x_values,
        #     low_y,
        #     color="grey",
        # )
        ax.plot(
            x_values,
            y_values,
            marker=faceSetting[name]["marker"],
            color=faceSetting[name]["color"],
            linestyle=faceSetting[name]["line"],
            label=name,
        )
        ax.errorbar(
            x_values,
            average_y,
            yerr=std_y,  # Add the standard deviation as the error bar
            # (low_y + high_y) / 2,
            # yerr=(high_y - low_y) / 2,  # Add the standard deviation as the error bar
            # ecolor=faceSetting[name]["color"],
            ecolor="grey",
            capsize=2,  # Add caps to the error bars
            elinewidth=1,
            fmt="none",
            barsabove=True,
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
    # ax.spines['bottom'].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)

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
    # fontsize=14,
    facecolor="None",  # Set background color
    edgecolor="None",  # Set border color
)

ax2.xaxis.label.set_size(22)
plt.subplots_adjust(top=0.90, hspace=0.02)


# Show the plot
# plt.show()
plt.savefig(f"image/benchIllu.jpg", bbox_inches="tight", dpi=150)
plt.savefig(f"image/benchIllu.png", bbox_inches="tight", dpi=150)
plt.savefig(f"image/benchIllu.pdf", bbox_inches="tight", dpi=150)
plt.savefig(f"image/benchIllu.eps", bbox_inches="tight", dpi=150)

# Set global font properties
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 22,
        "font.serif": ["DejaVu Serif"],
        # 'axes.titlesize': 16,
        # 'axes.labelsize': 17,
        # 'legend.fontsize': 12,
        "xtick.labelsize": 18,
        "ytick.labelsize": 14,
        "lines.linewidth": 3,  # Default line width
        "lines.markersize": 12,  # Default marker size
    }
)

# Sample data
# categories = [PC_GPU, PC_CPU, JET_GPU, JET_CPU]
categories = [
    "GPU\n1080Ti",
    "CPU\ni5-12600K",
    "GPU Jetson\nOrin NX",
    "GPU Jetson\nOrin NX",
]
subcategories = ["Proposed\nReachability", "Proposed\nDistance", "Levenberg-Marquardt"]
values = np.array(
    [
        [4, 7, np.nan],
        [1, 8, np.nan],
        [5, 3, np.nan],
        [1, 2, np.nan],
    ]
)  # Values for each group and subcategory

for file in file_list_r:
    path = file[D]
    name = file[N]
    col = 0
    row = 0
    if name == PC_GPU:
        row = 0
    if name == PC_CPU:
        row = 1
    if name == JET_GPU:
        row = 2
    if name == JET_CPU:
        row = 3
    if name == "pcrbdl":
        row = 1
        col = 2
    if name == "jetrbdl":
        row = 3
        col = 2
    # Read the CSV file using numpy
    data = np.genfromtxt(path, delimiter=";")
    data = data[data[:, 0] > 10_000, :]
    x = data[:, 0]
    y = data[:, 1]
    unique_x = np.unique(x)
    # average_y = np.array([np.median(y[x == ux]) for ux in unique_x])
    low_y = np.array([np.percentile(y[x == ux], 5) for ux in unique_x])
    high_y = np.array([np.percentile(y[x == ux], 95) for ux in unique_x])
    for i in range(len(unique_x)):
        ux = unique_x[i]
        mask = (x == ux) & (y > high_y[i])
        mask |= (x == ux) & (y < low_y[i])
        print(mask.sum())
        y[mask] = np.nan
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    values[row, col] = np.mean(y)

for file in file_list_d:
    path = file[D]
    name = file[N]
    col = 1
    row = 0
    if name == PC_GPU:
        row = 0
    if name == PC_CPU:
        row = 1
    if name == JET_GPU:
        row = 2
    if name == JET_CPU:
        row = 3
    # Read the CSV file using numpy
    data = np.genfromtxt(path, delimiter=";")
    data = data[data[:, 0] > 10_000, :]
    x = data[:, 0]
    y = data[:, 1]
    unique_x = np.unique(x)
    # average_y = np.array([np.median(y[x == ux]) for ux in unique_x])
    low_y = np.array([np.percentile(y[x == ux], 5) for ux in unique_x])
    high_y = np.array([np.percentile(y[x == ux], 95) for ux in unique_x])
    for i in range(len(unique_x)):
        ux = unique_x[i]
        mask = (x == ux) & (y > high_y[i])
        mask |= (x == ux) & (y < low_y[i])
        print(mask.sum())
        y[mask] = np.nan
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    values[row, col] = np.mean(y)

# Number of groups and subcategories
n_groups = len(categories)
n_subcategories = len(subcategories)
colors = ["black", "grey", "firebrick"]

# Bar width
bar_width = 0.3
slim = 0.9

# X locations for the groups
index = np.arange(n_groups)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 5))

# Plot bars for each subcategory
for i in range(n_subcategories):
    bars = plt.bar(
        index + i * bar_width,
        values[:, i],
        bar_width * slim,
        label=subcategories[i],
        color=colors[i],
    )
    # Add labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        if yval >= 10:
            roundedval = int(round(yval, 0))
        elif yval >=1:
            roundedval = round(yval, 1)
        else:
            roundedval = round(yval, 2)
            roundedval = str(roundedval)
            # roundedval = roundedval[1:]
            if len(roundedval) < 4:
                roundedval += "0"
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.12,
            roundedval,
            ha="center",
            va="bottom",
            fontsize=14,
            bbox=dict(facecolor="white", edgecolor="none", pad=-1),
        )
# Add labels and title
plt.yscale("log")
# plt.xlabel("Categories")
plt.ylabel("Average time (ns/sample)")
# plt.title("Grouped Bar Chart")
plt.xticks(index + bar_width / 2 * (n_subcategories - 1), categories)
# plt.legend()

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# ax.spines['bottom'].set_visible(False)
ax.spines["left"].set_visible(False)
ax.set_axisbelow(True)
plt.grid(axis="y", linewidth=2, color="black", alpha=0.2)

# Extract the legend handles and labels
handles, labels = ax.get_legend_handles_labels()

# Create an axis for the legend
legend_ax = fig.add_axes([0.07, 0.98, 0.8, 0.04], frame_on=False)
legend_ax.axis("off")  # Hide the axis

# Add the legend to the custom axis
legend_ax.legend(
    handles=handles,
    labels=labels,
    loc="center",
    ncol=3,
    handleheight=1.5,
    # handlelength=4.0,
    fontsize=17,
    facecolor="None",  # Set background color
    edgecolor="None",  # Set border color
)

plt.subplots_adjust(top=0.90, hspace=0.02)

plt.savefig(f"image/benchBar.jpg", bbox_inches="tight", dpi=150)
plt.savefig(f"image/benchBar.png", bbox_inches="tight", dpi=150)
plt.savefig(f"image/benchBar.pdf", bbox_inches="tight", dpi=150)
plt.savefig(f"image/benchBar.eps", bbox_inches="tight", dpi=150)
