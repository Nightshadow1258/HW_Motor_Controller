import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import sys

def print_usage():
    print("""
Usage:
    python plot_single.py <config.yaml>

Description:
    This script extracts forward and backward segments, averages their profiles,
    calculates 32 linearization parameters, applies simulated piecewise linear 
    hardware correction, and plots all phases of the data.
""")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def clean_column_names(columns, cleanup_rules):
    cleaned = []
    for col in columns:
        for rule in cleanup_rules:
            col = col.replace(rule, "")
        cleaned.append(col.strip())
    return cleaned

def get_input_files(config):
    if "input_files" in config:
        return config["input_files"]
    elif "input_file" in config:
        return [config["input_file"]]
    else:
        return [os.path.join("input", f) for f in os.listdir("input") if f.endswith(".csv")]

def apply_plot_formatting(config, ax, xlabel, ylabel, title):
    plot_format = config.get("plot_format", {})
    fonts = plot_format.get("font_sizes", {})
    
    ax.set_xlabel(xlabel, fontsize=fonts.get("axis", 14))
    ax.set_ylabel(ylabel, fontsize=fonts.get("axis", 14))
    ax.set_title(title, fontsize=fonts.get("title", 16))
    ax.legend(fontsize=fonts.get("legend", 12))

    major = plot_format.get("grid", {}).get("major", {})
    minor = plot_format.get("grid", {}).get("minor", {})
    
    ax.grid(True, which='major', axis='both', color=major.get("color", "#D3D3D3"), linestyle=major.get("linestyle", ":"), linewidth=0.8)
    ax.grid(True, which='minor', axis='both', color=minor.get("color", "#A8A8A8C8"), linestyle=minor.get("linestyle", ":"), linewidth=0.5)
    ax.tick_params(axis='both', which='both', labelsize=fonts.get("ticks", 12))

def main():
    if len(sys.argv) != 2 or sys.argv[1] in ("--help", "-h"):
        print_usage()
        return

    config_file = sys.argv[1]
    config = load_config(config_file)
    input_files = get_input_files(config)
    batch_mode = len(input_files) > 1

    angle_col = config.get("target_angle_column", "Angle from ABI [°]")

    for csv_file in input_files:
        print(f"Processing: {csv_file}")
        df = pd.read_csv(csv_file)
        
        cleanup_rules = config.get("header_cleanup", [])
        df.columns = clean_column_names(df.columns, cleanup_rules)
        
        x_col_index = config.get("x_axis_col", 1)
        time_col = df.columns[x_col_index]

        # Detect wrap around transitions
        df["diff"] = df[angle_col].diff()
        forward_wraps = df[df["diff"] < -300].index.tolist()
        backward_wraps = df[df["diff"] > 300].index.tolist()

        if len(forward_wraps) < 2 or len(backward_wraps) < 2:
            print(f"Error: Could not isolate sufficient wrap points in {csv_file}. Skipping.")
            continue

        # Extract strict motion profiles from boundary indices
        forward_motion = df.iloc[forward_wraps[0]:forward_wraps[1]].copy()
        backward_motion = df.iloc[backward_wraps[0]:backward_wraps[1]].copy()

        # Generate continuous ideal references
        forward_motion["ideal_position"] = np.linspace(0, 360, len(forward_motion))
        backward_motion["ideal_position"] = np.linspace(360, 0, len(backward_motion))

        # Compute raw differences and wrap to symmetric window (-180, 180)
        f_err = forward_motion[angle_col].values - forward_motion["ideal_position"].values
        f_err = (f_err + 180) % 360 - 180

        b_err = backward_motion[angle_col].values - backward_motion["ideal_position"].values
        b_err = (b_err + 180) % 360 - 180

        # Build shared coordinate system to execute vector averaging 
        common_angles = np.linspace(0, 360, 360, endpoint=False)
        
        # Interpolate spatial distributions
        f_err_interp = np.interp(common_angles, forward_motion[angle_col].values, f_err)
        b_sort_idx = np.argsort(backward_motion[angle_col].values)
        b_err_interp = np.interp(common_angles, backward_motion[angle_col].values[b_sort_idx], b_err[b_sort_idx])

        # Core averaging to collapse hysteresis error signatures
        averaged_error = (f_err_interp + b_err_interp) / 2.0

        # Map to fixed Allegro physical register index configuration
        allegro_nodes = np.linspace(0, 360, 32, endpoint=False)
        
        # Smooth with a 5th degree polynomial fit
        poly_coeffs = np.polyfit(common_angles, averaged_error, 5)
        hardware_corrections = -np.polyval(poly_coeffs, allegro_nodes)

        # Generate structural output configuration table
        allegro_eeprom_df = pd.DataFrame({
            "Register_Index": range(32),
            "Sensor_Angle_Deg": allegro_nodes,
            "EEPROM_Correction_Deg": hardware_corrections
        })

        # Simulate hardware implementation using linear interpolation across the 32 nodes
        # The sensor adds the correction value based on its current raw reading
        forward_motion["simulated_correction"] = np.interp(forward_motion[angle_col], allegro_nodes, hardware_corrections, period=360)
        forward_motion["corrected_position"] = (forward_motion[angle_col] + forward_motion["simulated_correction"]) % 360

        backward_motion["simulated_correction"] = np.interp(backward_motion[angle_col], allegro_nodes, hardware_corrections, period=360)
        backward_motion["corrected_position"] = (backward_motion[angle_col] + backward_motion["simulated_correction"]) % 360

        # Visualization block setup (3 subplots)
        plot_format = config.get("plot_format", {})
        plot_size = plot_format.get("size", {"width": 12, "height": 14})
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=[plot_size.get("width", 12), plot_size.get("height", 14)])
        time_label = config.get("plot_format", {}).get("axis_labels", {}).get("x", "Time [ms]")

        # Subplot 1: Absolute Forward Motion with Correction Applied
        ax1.plot(forward_motion[time_col], forward_motion[angle_col], label="Raw Measured Position (ABI)", color="tab:red", alpha=0.6)
        ax1.plot(forward_motion[time_col], forward_motion["corrected_position"], label="Calibrated Output (Simulated)", color="tab:green", linewidth=2)
        ax1.plot(forward_motion[time_col], forward_motion["ideal_position"], label="Perfect Linear Reference", color="black", linestyle="--", alpha=0.7)
        apply_plot_formatting(config, ax1, time_label, "Angle [deg]", "Extracted Forward Motion: Baseline vs Correction")
        ax1.set_xlim(forward_motion[time_col].iloc[0], forward_motion[time_col].iloc[-1])
        ax1.set_ylim(-5, 365)

        # Subplot 2: Absolute Backward Motion with Correction Applied
        ax2.plot(backward_motion[time_col], backward_motion[angle_col], label="Raw Measured Position (ABI)", color="tab:red", alpha=0.6)
        ax2.plot(backward_motion[time_col], backward_motion["corrected_position"], label="Calibrated Output (Simulated)", color="tab:green", linewidth=2)
        ax2.plot(backward_motion[time_col], backward_motion["ideal_position"], label="Perfect Linear Reference", color="black", linestyle="--", alpha=0.7)
        apply_plot_formatting(config, ax2, time_label, "Angle [deg]", "Extracted Backward Motion: Baseline vs Correction")
        ax2.set_xlim(backward_motion[time_col].iloc[0], backward_motion[time_col].iloc[-1])
        ax2.set_ylim(-5, 365)

        # Subplot 3: Spatial Errors and Register Mapping Points
        ax3.plot(forward_motion[angle_col], f_err, label="Extracted Forward Error", alpha=0.4)
        ax3.plot(backward_motion[angle_col], b_err, label="Extracted Backward Error", alpha=0.4)
        ax3.plot(common_angles, averaged_error, label="Calculated Average Error Curve", color="black", linewidth=2)
        ax3.scatter(allegro_nodes, -hardware_corrections, color="red", zorder=5, label="Allegro Register Support Nodes")
        apply_plot_formatting(config, ax3, "Sensor Angle [deg]", "Error Deviation [deg]", "Extracted Angle Spatial Errors & Supporting Nodes")
        ax3.set_xlim(0, 360)

        plt.tight_layout()
        
        # IO Serialization
        filename = os.path.splitext(os.path.basename(csv_file))[0]
        output_dir = config.get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        
        allegro_eeprom_df.to_csv(f"{output_dir}/{filename}_allegro_32_eeprom_registers.csv", index=False)
        plt.savefig(f"{output_dir}/{filename}_unified_linearization.png", format="png")
        
        if not batch_mode:
            plt.show()
        else:
            plt.close()

if __name__ == "__main__":
    main()