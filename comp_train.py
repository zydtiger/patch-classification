import matplotlib.pyplot as plt
import numpy as np

# Data from NOTE.md (times in minutes)
models = ["ResNet-50", "VGG-16", "PatchMIL", "MulticlassMLP"]
training_times = [61, 79, 8.3, 16.7]  # Model training times
embedding_time = 9.5  # Embedding preprocessing time

# Total times including embedding for PatchMIL and MulticlassMLP
total_times = [
    61,  # ResNet-50 (no embedding needed)
    79,  # VGG-16 (no embedding needed)
    8.3 + embedding_time,  # PatchMIL + embedding
    16.7 + embedding_time,  # MulticlassMLP + embedding
]

# Create figure and axis
plt.figure(figsize=(12, 8))
ax = plt.gca()

# Use colorblind-friendly matte colors (some lightened)
colors_training = [
    "#7BA7CC",  # Lighter matte blue (ResNet-50)
    "#F28E2B",  # Matte orange
    "#E87979",  # Lighter matte red (PatchMIL)
    "#AF7AA1",  # Matte purple (MulticlassMLP)
]  # Colorblind-friendly palette
colors_embedding = ["#79BC89", "#79BC89"]  # Lighter matte green for embedding stacks

# Plot bars
x_pos = np.arange(len(models))
bar_width = 0.6

# Bars for ResNet-50 and VGG-16 (only training)
ax.bar(
    x_pos[0],
    training_times[0],
    bar_width,
    color=colors_training[0],
    label="ResNet-50 Training",
    edgecolor="black",
    linewidth=1,
)
ax.bar(
    x_pos[1],
    training_times[1],
    bar_width,
    color=colors_training[1],
    label="VGG-16 Training",
    edgecolor="black",
    linewidth=1,
)

# Stacked bars for PatchMIL (training + embedding)
ax.bar(
    x_pos[2],
    embedding_time,
    bar_width,
    color=colors_embedding[0],
    label="Embedding Time",
    edgecolor="black",
    linewidth=1,
)
ax.bar(
    x_pos[2],
    training_times[2],
    bar_width,
    bottom=embedding_time,
    color=colors_training[2],
    label="PatchMIL Training",
    edgecolor="black",
    linewidth=1,
)

# Stacked bars for MulticlassMLP (training + embedding)
ax.bar(
    x_pos[3],
    embedding_time,
    bar_width,
    color=colors_embedding[1],
    edgecolor="black",
    linewidth=1,
)
ax.bar(
    x_pos[3],
    training_times[3],
    bar_width,
    bottom=embedding_time,
    color=colors_training[3],
    label="MulticlassMLP Training",
    edgecolor="black",
    linewidth=1,
)

# Customize the plot
ax.set_xlabel("Architecture", fontsize=14, fontweight="bold")
ax.set_ylabel("Training Time (minutes)", fontsize=14, fontweight="bold")
ax.set_title(
    "Model Training Time Comparison\n(with embedding preprocessing for PatchMIL and MulticlassMLP)",
    fontsize=16,
    fontweight="bold",
    pad=20,
)

# Set x-axis labels
ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=12)


# Add value labels on bars
def add_value_labels(bars, offset: float = 0):
    for bar in bars:
        height = bar.get_height()
        if offset > 0:
            # For stacked bars - center in the middle of each segment
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                offset + height / 2,
                f"{height:.1f}min",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="black",
            )
        else:
            # For regular bars - center in the middle of the bar
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 2,
                f"{height:.1f}min",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )


# Add labels for all bars
bars = ax.patches
add_value_labels([bars[0]])  # ResNet-50
add_value_labels([bars[1]])  # VGG-16
add_value_labels([bars[2]], 0.01)  # PatchMIL embedding
add_value_labels([bars[3]], embedding_time)  # PatchMIL training
add_value_labels([bars[4]], 0.01)  # MulticlassMLP embedding
add_value_labels([bars[5]], embedding_time)  # MulticlassMLP training

# Add total time labels above stacked bars
ax.text(
    x_pos[2],
    total_times[2] + 1,
    f"Total: {total_times[2]:.1f}min",
    ha="center",
    va="bottom",
    fontsize=10,
    fontweight="bold",
    color="black",
)
ax.text(
    x_pos[3],
    total_times[3] + 1,
    f"Total: {total_times[3]:.1f}min",
    ha="center",
    va="bottom",
    fontsize=10,
    fontweight="bold",
    color="black",
)

# Customize grid and appearance
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

# Set y-axis to start from 0
ax.set_ylim(0, max(total_times) + 10)

# Add legend
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), framealpha=0.9)


# Improve layout
plt.tight_layout()

# Save the plot
plt.savefig(
    "training_time_comparison.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)

# Show the plot
plt.show()

# Print summary statistics
print("Training Time Summary (minutes):")
print("-" * 40)
print(f"ResNet-50:        {training_times[0]:.1f} min")
print(f"VGG-16:           {training_times[1]:.1f} min")
print(f"Embedding:        {embedding_time:.1f} min")
print(
    f"PatchMIL:         {training_times[2]:.1f} min (+ {embedding_time:.1f} min embedding = {total_times[2]:.1f} min total)"
)
print(
    f"MulticlassMLP:    {training_times[3]:.1f} min (+ {embedding_time:.1f} min embedding = {total_times[3]:.1f} min total)"
)
print("-" * 40)
print(f"Fastest total time: PatchMIL ({total_times[2]:.1f} min)")
print(f"Slowest total time: VGG-16 ({total_times[1]:.1f} min)")
