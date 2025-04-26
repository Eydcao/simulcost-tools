import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

output_dir = "./output/heat_1d"

# Get all result files and sort by frame number
result_files = sorted(
    [f for f in os.listdir(output_dir) if f.startswith("res_") and f.endswith(".h5")],
    key=lambda x: int(x.split("_")[1].split(".")[0]),
)

if not result_files:
    raise FileNotFoundError(f"No result files found in {output_dir}")

# First pass: Find global min/max temperature values
t_min = np.inf
t_max = -np.inf
for f in result_files:
    with h5py.File(os.path.join(output_dir, f), "r") as hf:
        T = hf["T"][:]
        t_min = min(t_min, T.min())
        t_max = max(t_max, T.max())

# Add 5% margin
margin = 0.05 * (t_max - t_min)
y_min = t_min - margin
y_max = t_max + margin

# Second pass: Create plots
for i, f in enumerate(result_files):
    with h5py.File(os.path.join(output_dir, f), "r") as hf:
        x = hf["x"][:]
        T = hf["T"][:]
        time = hf["time"][()]

    plt.figure(figsize=(10, 6))
    plt.plot(x, T, "b-", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("T")
    plt.ylim(y_min, y_max)
    plt.title(f"Time = {time:.3f} (Frame {i})")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"c_{i}.png"))
    plt.close()

# Create video (using ffmpeg)
os.system(
    f"ffmpeg -y -framerate 10 -i {output_dir}/c_%d.png "
    f"-c:v libx264 -r 30 -pix_fmt yuv420p "
    f"{output_dir}/heat_transfer.mp4"
)

print(f"Animation saved to {output_dir}/heat_transfer.mp4")
