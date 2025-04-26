import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from glob import glob

output_dir = "./output/heat_1d"


def read_results(output_dir):
    """Read all result files and extract data"""
    files = sorted(glob(os.path.join(output_dir, "res_*.h5")))
    if not files:
        raise FileNotFoundError(f"No result files found in {output_dir}")

    # Read all data
    results = []
    for f in files:
        with h5py.File(f, "r") as hf:
            results.append({"x": hf["x"][:], "T": hf["T"][:], "time": hf["time"][()]})

    return results


def create_plots(output_dir):
    """Create individual frame plots"""
    results = read_results(output_dir)

    # Find global min/max for consistent y-axis
    t_min = min(r["T"].min() for r in results)
    t_max = max(r["T"].max() for r in results)
    margin = 0.05 * (t_max - t_min)  # 5% margin

    # Create plots
    os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)

    for i, r in enumerate(results):
        plt.figure(figsize=(10, 6))
        plt.plot(r["x"], r["T"], "b-", linewidth=2)
        plt.xlabel("Position (m)", fontsize=12)
        plt.ylabel("Temperature (K)", fontsize=12)
        plt.ylim(t_min - margin, t_max + margin)
        plt.title(f"Heat Transfer Simulation\nTime = {r['time']:.3f} s", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        frame_path = os.path.join(output_dir, "frames", f"frame_{i:04d}.png")
        plt.savefig(frame_path, dpi=100, bbox_inches="tight")
        plt.close()

    return len(results)


def create_animation(output_dir, num_frames):
    """Create MP4 animation from frames"""
    output_video = os.path.join(output_dir, "heat_transfer.mp4")

    # Use ffmpeg to create video
    cmd = (
        f"ffmpeg -y -framerate 10 -i {output_dir}/frames/frame_%04d.png "
        f"-c:v libx264 -r 30 -pix_fmt yuv420p -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' "
        f"{output_video}"
    )
    os.system(cmd)
    print(f"Animation saved to {output_video}")


def create_interactive_plot(output_dir):
    """Create interactive matplotlib animation"""
    results = read_results(output_dir)
    fig, ax = plt.subplots(figsize=(10, 6))
    (line,) = ax.plot([], [], "b-", linewidth=2)

    # Set up axes
    ax.set_xlim(results[0]["x"].min(), results[0]["x"].max())
    t_min = min(r["T"].min() for r in results)
    t_max = max(r["T"].max() for r in results)
    margin = 0.05 * (t_max - t_min)
    ax.set_ylim(t_min - margin, t_max + margin)
    ax.set_xlabel("Position (m)", fontsize=12)
    ax.set_ylabel("Temperature (K)", fontsize=12)
    ax.grid(True, alpha=0.3)
    title = ax.set_title("", fontsize=14)

    def init():
        line.set_data([], [])
        title.set_text("")
        return line, title

    def update(i):
        r = results[i]
        line.set_data(r["x"], r["T"])
        title.set_text(f"Heat Transfer Simulation\nTime = {r['time']:.3f} s")
        return line, title

    ani = FuncAnimation(fig, update, frames=len(results), init_func=init, blit=True, interval=100)

    plt.tight_layout()
    return ani


if __name__ == "__main__":
    # Option 1: Create individual frames and video
    num_frames = create_plots(output_dir)
    create_animation(output_dir, num_frames)

    # Option 2: Create interactive plot (uncomment to use)
    # ani = create_interactive_plot(output_dir)
    # plt.show()
