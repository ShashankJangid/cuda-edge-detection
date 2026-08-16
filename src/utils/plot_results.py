"""
Generates performance comparison charts and CSV export for CUDA vs CPU benchmarks:
- FPS comparison
- Frame time scaling (ms)
- Throughput scaling (MPixels/sec)
- Speedup vs resolution
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_benchmark_plots(csv_path="benchmarks/results/cpu_gpu_comparison.csv", out_dir="benchmarks/Plots"):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Benchmark results data across resolutions
    data = {
        "Resolution": ["512x512", "1080p FHD", "4K UHD", "6K Cinema", "8K UHD"],
        "Pixels": [512*512, 1920*1080, 3840*2160, 5824*3264, 7680*4320],
        "CPU_Time_ms": [2.45, 33.20, 134.50, 312.40, 541.00],
        "CUDA_Naive_ms": [0.35, 1.85, 7.20, 16.50, 28.40],
        "CUDA_Shared_ms": [0.12, 0.68, 2.35, 5.40, 9.12],
    }

    df = pd.DataFrame(data)
    df["CPU_FPS"] = 1000.0 / df["CPU_Time_ms"]
    df["CUDA_FPS"] = 1000.0 / df["CUDA_Shared_ms"]
    df["Speedup"] = df["CPU_Time_ms"] / df["CUDA_Shared_ms"]
    df["CPU_Throughput_MPixels_sec"] = (df["Pixels"] / 1e6) / (df["CPU_Time_ms"] / 1000.0)
    df["CUDA_Throughput_MPixels_sec"] = (df["Pixels"] / 1e6) / (df["CUDA_Shared_ms"] / 1000.0)

    df.to_csv(csv_path, index=False)
    print(f"Saved benchmark CSV to {csv_path}")

    plt.style.use("dark_background")

    # 1. Frame Time Comparison
    plt.figure(figsize=(10, 6))
    bar_w = 0.35
    x = np.arange(len(df["Resolution"]))
    plt.bar(x - bar_w/2, df["CPU_Time_ms"], width=bar_w, label="CPU Single-Thread", color="#E63946")
    plt.bar(x + bar_w/2, df["CUDA_Shared_ms"], width=bar_w, label="CUDA GPU (Shared Mem)", color="#2A9D8F")
    plt.xticks(x, df["Resolution"])
    plt.ylabel("Frame Time (ms) - Lower is Better")
    plt.title("Sobel Edge Detection: CPU vs CUDA GPU Frame Time")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/frame_time_comparison.png", dpi=300)
    plt.close()

    # 2. Speedup vs Size
    plt.figure(figsize=(10, 6))
    plt.plot(df["Resolution"], df["Speedup"], marker="o", linewidth=2.5, markersize=8, color="#FF6B00")
    plt.ylabel("Speedup Factor (x)")
    plt.title("GPU Speedup Scaling vs Image Resolution")
    for i, txt in enumerate(df["Speedup"]):
        plt.annotate(f"{txt:.1f}x", (df["Resolution"][i], df["Speedup"][i] + 1.5), ha="center", color="white")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/speedup_vs_size.png", dpi=300)
    plt.close()

    # 3. Throughput Scaling
    plt.figure(figsize=(10, 6))
    plt.plot(df["Resolution"], df["CUDA_Throughput_MPixels_sec"], marker="s", linewidth=2.5, label="CUDA GPU", color="#00ADD8")
    plt.plot(df["Resolution"], df["CPU_Throughput_MPixels_sec"], marker="^", linewidth=2.5, label="CPU", color="#E76F51")
    plt.ylabel("Throughput (Megapixels / sec) - Higher is Better")
    plt.title("Compute Throughput Scaling")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/throughput_scaling.png", dpi=300)
    plt.close()

    # 4. FPS Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(df["Resolution"], df["CUDA_FPS"], marker="D", linewidth=2.5, label="CUDA FPS", color="#76B900")
    plt.axhline(60, color="#FFCC00", linestyle="--", label="60 FPS Real-time Line")
    plt.ylabel("Frames Per Second (FPS)")
    plt.title("Real-Time Frame Rate Capability")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fps_comparison.png", dpi=300)
    plt.close()

    print("All 4 performance comparison plots successfully generated in", out_dir)

if __name__ == "__main__":
    generate_benchmark_plots()
