import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# --- Configuration ---
METHODS = [
    ("Sequential CPU",  "Sequential.csv"),
    ("OpenMP Parallel", "OpenMP.csv"),
    ("CUDA GPU",        "CUDA.csv"),     
    ("OpenCL GPU",      "OpenCL.csv"),   
]

def get_file_path(filename):
    """Helper to find the csv file whether running from root or results dir"""
    if os.path.exists(filename):
        return filename
    if os.path.exists(os.path.join("results", filename)):
        return os.path.join("results", filename)
    return None

def analyze_benchmarks():
    print("--- Starting HOG Benchmark Analysis ---")
    
    all_data = [] 
    summary_stats = []
    
    # Use a nice style for plots if available
    try:
        plt.style.use('ggplot')
    except:
        pass

    # 1. Load Data
    for name, filename in METHODS:
        filepath = get_file_path(filename)
        
        if not filepath:
            if "Sequential" in name:
                print(f"[Warning] Could not find {filename}. Run the C++ app first.")
            continue
            
        print(f"Loading {name} from {filepath}...")
        
        try:
            df = pd.read_csv(filepath)
            
            # Validation
            if 'Width' not in df.columns:
                print(f"[Error] {filename} is old format. Re-run C++ app.")
                continue

            # Add Metadata
            df['Method'] = name
            df['Pixels'] = df['Width'] * df['Height']
            df['Resolution'] = df['Width'].astype(str) + "x" + df['Height'].astype(str)
            
            # --- NEW: Calculate Cumulative Time ---
            # cumulative sum of ms converted to seconds
            df['Cumulative_Time_s'] = df['Time_ms'].cumsum() / 1000.0
            
            all_data.append(df)
            
            # --- Metrics ---
            total_time_s = df['Time_ms'].sum() / 1000.0
            avg_time_ms = df['Time_ms'].mean()
            std_dev = df['Time_ms'].std()
            real_fps = len(df) / total_time_s if total_time_s > 0 else 0.0
            
            summary_stats.append({
                "Method": name,
                "Total Time (s)": round(total_time_s, 2),
                "Avg Time (ms)": round(avg_time_ms, 2),
                "FPS": round(real_fps, 2),
                "Stability (StdDev)": round(std_dev, 2)
            })
            
        except Exception as e:
            print(f"[Error] Reading {filepath}: {e}")

    if not all_data:
        print("No data found.")
        return

    full_df = pd.concat(all_data)
    df_summary = pd.DataFrame(summary_stats)
    
    # Determine output directory
    out_prefix = "results/" if not os.path.exists("Sequential.csv") and os.path.exists("results") else ""

    # 2. Print Summary Table
    print("\n" + "="*40)
    print("       BENCHMARK RESULTS")
    print("="*40)
    print(df_summary.to_string(index=False))
    print("="*40 + "\n")

    # --- PLOT 1: Performance Scalability (Resolution vs Time) ---
    # Best for: Comparing Folders of Mixed Images
    plt.figure(figsize=(10, 6))
    for name in full_df['Method'].unique():
        subset = full_df[full_df['Method'] == name]
        grouped = subset.groupby('Pixels').agg({'Time_ms': 'mean', 'Resolution': 'first'}).reset_index().sort_values('Pixels')
        plt.plot(grouped['Pixels'], grouped['Time_ms'], marker='o', linewidth=2, label=name)
        
        # Label points
        for _, row in grouped.iterrows():
            plt.text(row['Pixels'], row['Time_ms'], f" {row['Resolution']}", fontsize=8)

    plt.title("Performance vs Resolution (Scalability)")
    plt.xlabel("Image Size (Pixels)")
    plt.ylabel("Processing Time (ms)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(out_prefix + "benchmark_scalability.png")
    print(f"[Saved] benchmark_scalability.png")

    # --- PLOT 2: Cumulative Time (Linear Progression) ---
    # Best for: Videos (Slope indicates speed)
    plt.figure(figsize=(10, 6))
    for name in full_df['Method'].unique():
        subset = full_df[full_df['Method'] == name]
        plt.plot(subset['Frame'], subset['Cumulative_Time_s'], linewidth=2, label=name)
        
    plt.title("Cumulative Processing Time (Video Progress)")
    plt.xlabel("Frame Number")
    plt.ylabel("Total Time Elapsed (Seconds)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(out_prefix + "benchmark_cumulative.png")
    print(f"[Saved] benchmark_cumulative.png")

    # --- PLOT 3: Total Execution Time (Bar) ---
    # Best for: Comparing Total Workload
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df_summary["Method"], df_summary["Total Time (s)"], color='#e74c3c', alpha=0.8)
    plt.title("Total Execution Time (Lower is Better)")
    plt.ylabel("Time (Seconds)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height()}s", ha='center', va='bottom', fontweight='bold')
    plt.savefig(out_prefix + "benchmark_total_time.png")
    print(f"[Saved] benchmark_total_time.png")

    # --- PLOT 4: Average FPS (Bar) ---
    # Best for: Real-time capability check
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df_summary["Method"], df_summary["FPS"], color='#2ecc71', alpha=0.8)
    plt.title("Average FPS (Higher is Better)")
    plt.ylabel("FPS")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height()}", ha='center', va='bottom', fontweight='bold')
    plt.savefig(out_prefix + "benchmark_fps.png")
    print(f"[Saved] benchmark_fps.png")

    # --- PLOT 5: Frame Stability (Timeline) ---
    # Best for: Checking Jitters/Spikes
    plt.figure(figsize=(12, 5))
    for name in full_df['Method'].unique():
        subset = full_df[full_df['Method'] == name]
        plt.plot(subset['Frame'], subset['Time_ms'], label=name, alpha=0.6, linewidth=1)
    plt.title("Per-Frame Processing Time (Stability)")
    plt.xlabel("Frame Number")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_prefix + "benchmark_timeline.png")
    print(f"[Saved] benchmark_timeline.png")

if __name__ == "__main__":
    analyze_benchmarks()
