import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import sys
from scipy.signal import savgol_filter


def print_usage():
    print("""
Usage:
    python plot_single.py <config.yaml>

Description:
    This script extracts forward and backward segments, averages them,
    optimizes a harmonic model (Fourier series) to represent the periodic error,
    maps results onto the 32 Allegro registers, and saves the parameters.
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

def fit_harmonics(angles_deg, errors, num_harmonics=4):
    angles_rad = np.radians(angles_deg)
    matrix_columns = [np.ones_like(angles_rad)]
    
    for i in range(1, num_harmonics + 1):
        matrix_columns.append(np.sin(i * angles_rad))
        matrix_columns.append(np.cos(i * angles_rad))
        
    A = np.column_stack(matrix_columns)
    coeffs, _, _, _ = np.linalg.lstsq(A, errors, rcond=None)
    return coeffs

def evaluate_harmonics(angles_deg, coeffs, num_harmonics=4):
    angles_rad = np.radians(angles_deg)
    result = coeffs[0] * np.ones_like(angles_rad)
    idx = 1
    
    for i in range(1, num_harmonics + 1):
        result += coeffs[idx] * np.sin(i * angles_rad)
        result += coeffs[idx+1] * np.cos(i * angles_rad)
        idx += 2
        
    return result

def main():
    if len(sys.argv) != 2 or sys.argv[1] in ("--help", "-h"):
        print_usage()
        return

    config_file = sys.argv[1]
    config = load_config(config_file)
    input_files = get_input_files(config)
    batch_mode = len(input_files) > 1

    # Switch data source column based on configuration flag
    use_spi = config.get("use_spi", True)
    if use_spi:
        angle_col = config.get("spi_angle_column", "Angle from SPI [°]")
    else:
        angle_col = config.get("abi_angle_column", "Angle from ABI [°]")

    num_harmonics = config.get("num_harmonics", 4)

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
        forward_df = df.iloc[forward_wraps[0]+1:forward_wraps[1]-1].copy()
        backward_df = df.iloc[backward_wraps[0]:backward_wraps[1]].copy()

        
        # Generate continuous ideal references
        forward_df["ideal_position"] = np.linspace(0, 360, len(forward_df))
        backward_df["ideal_position"] = np.linspace(360, 0, len(backward_df))
        
        # Compute raw differences and wrap to symmetric window (-180, 180)
        forward_df["f_err"] = forward_df[angle_col].values - forward_df["ideal_position"].values
        forward_df["f_err"] = (forward_df["f_err"] + 180) % 360 - 180

        backward_df["b_err"] = backward_df[angle_col].values - backward_df["ideal_position"].values
        backward_df["b_err"] = (backward_df["b_err"] + 180) % 360 - 180

        # Build shared coordinate system to execute vector averaging 
        common_angles = np.linspace(0, 360, 360, endpoint=False)

        # Interpolate spatial distributions using corrected dataframes
        f_err_interp = np.interp(common_angles, forward_df[angle_col].values, forward_df["f_err"].values)
        b_sort_idx = np.argsort(backward_df[angle_col].values)
        b_err_interp = np.interp(common_angles, backward_df[angle_col].values[b_sort_idx], backward_df["b_err"].values[b_sort_idx])

        # Core averaging to collapse hysteresis error signatures
        averaged_error = (f_err_interp + b_err_interp) / 2.0

        # Run Harmonic Least-Squares Optimization
        harmonic_coeffs = fit_harmonics(common_angles, averaged_error, num_harmonics)

        # Parse harmonic coefficients into an unrolled parameters dataset
        param_records = [{"Parameter": "DC_Offset", "Harmonic": 0, "Value": harmonic_coeffs[0]}]
        idx = 1
        for h in range(1, num_harmonics + 1):
            param_records.append({"Parameter": "Sin_Amplitude", "Harmonic": h, "Value": harmonic_coeffs[idx]})
            param_records.append({"Parameter": "Cos_Amplitude", "Harmonic": h, "Value": harmonic_coeffs[idx+1]})
            idx += 2
        harmonics_df = pd.DataFrame(param_records).round(6)

        # Map to fixed Allegro physical register index configuration (32 nodes)
        allegro_nodes = np.linspace(0, 360, 32, endpoint=False)
        hardware_corrections = -evaluate_harmonics(allegro_nodes, harmonic_coeffs, num_harmonics)

        # Generate structural output configuration table
        allegro_eeprom_df = pd.DataFrame({
            "Register_Index": range(32),
            "Sensor_Angle_Deg": allegro_nodes,
            "EEPROM_Correction_Deg": hardware_corrections
        }).round(6)

        # Simulate hardware correction implementation using 32 segment lookup nodes
        forward_df["simulated_correction"] = np.interp(forward_df[angle_col], allegro_nodes, hardware_corrections, period=360)
        forward_df["corrected_position"] = (forward_df[angle_col] + forward_df["simulated_correction"]) % 360
    
        backward_df["simulated_correction"] = np.interp(backward_df[angle_col], allegro_nodes, hardware_corrections, period=360)
        backward_df["corrected_position"] = (backward_df[angle_col] + backward_df["simulated_correction"]) % 360

        # Compute post-calibration residual errors
        f_err_calibrated = forward_df["corrected_position"].values - forward_df["ideal_position"].values
        forward_df["corrected_error"] = (f_err_calibrated + 180) % 360 - 180

        b_err_calibrated = backward_df["corrected_position"].values - backward_df["ideal_position"].values
        backward_df["corrected_error"] = (b_err_calibrated + 180) % 360 - 180

        # Visualization block setup (6 subplots organized in a 3x2 grid)
        plot_format = config.get("plot_format", {})
        plot_size = plot_format.get("size", {"width": 16, "height": 14})
        
        fig, axes = plt.subplots(3, 2, figsize=[plot_size.get("width", 16), plot_size.get("height", 14)])
        time_label = plot_format.get("axis_labels", {}).get("x", "Time [ms]")
        y_label = "Angle [deg]"

        # Calculate a continuous timeline for extracted data to eliminate the gap
        # Shift backward time so it starts right after forward time ends
        f_time = forward_df[time_col].values
        b_time_original = backward_df[time_col].values
        
        f_duration = f_time[-1] - f_time[0]
        # Assume a small nominal gap (e.g., 10 ms) or 0 to join them directly
        time_gap = 10 
        b_time_shifted = b_time_original - b_time_original[0] + f_time[-1] + time_gap

        # Row 1, Column 1: Original Data (Whole dataset)
        axes[0, 0].plot(df[time_col], df[config.get("spi_angle_column", "Angle from SPI [°]")], label=f"Original Data Angle from SPI [°]", color="tab:blue")
        axes[0, 0].plot(df[time_col], df[config.get("abi_angle_column", "Angle from ABI [°]")], label=f"Original Data Angle from ABI [°]", color="tab:orange")

        apply_plot_formatting(config, axes[0, 0], time_label, y_label, "1. Original Sensor Dataset")
        axes[0, 0].set_xlim(df[time_col].iloc[0], 5200)
        
        # Row 1, Column 2: Extracted Profiles (Continuous Time)
        axes[0, 1].plot(f_time, forward_df[angle_col], label="Forward Profile", color="tab:orange")
        axes[0, 1].plot(b_time_shifted, backward_df[angle_col], label="Backward Profile", color="tab:purple")
        apply_plot_formatting(config, axes[0, 1], time_label, y_label, "2. Extracted Processing Profiles")
        axes[0, 1].set_xlim(f_time[0], b_time_shifted[-1])

        # Row 2, Column 1: Error from Ideal Linear Curve (Continuous Time)
        axes[1, 0].plot(f_time, forward_df["f_err"], label="Forward Error", color="tab:red", alpha=0.7)
        axes[1, 0].plot(b_time_shifted, backward_df["b_err"], label="Backward Error", color="orange", alpha=0.7)
        apply_plot_formatting(config, axes[1, 0], time_label, "Error [deg]", "3. Raw Error Deviation")
        axes[1, 0].set_xlim(f_time[0], b_time_shifted[-1])

        # Row 2, Column 2: Original vs Corrected Data (Continuous Time)
        axes[1, 1].plot(f_time, forward_df[angle_col], label="Orig Forward", color="tab:blue", alpha=0.5)
        axes[1, 1].plot(f_time, forward_df["corrected_position"], label="Corr Forward", color="tab:green")
        axes[1, 1].plot(b_time_shifted, backward_df[angle_col], label="Orig Backward", color="tab:purple", alpha=0.5)
        axes[1, 1].plot(b_time_shifted, backward_df["corrected_position"], label="Corr Backward", color="lime")
        apply_plot_formatting(config, axes[1, 1], time_label, y_label, "4. Original vs Harmonic Corrected")
        axes[1, 1].set_xlim(f_time[0], b_time_shifted[-1])

        # Row 3, Column 1: Error of Corrected Data vs Ideal Data (Forward Only)
        axes[2, 0].plot(f_time, forward_df["f_err"], label="Original Error", color="tab:red", alpha=0.4)
        axes[2, 0].plot(f_time, forward_df["corrected_error"], label="Residual Error", color="tab:green")
        apply_plot_formatting(config, axes[2, 0], time_label, "Error [deg]", "5. Residual Performance: Forward")
        axes[2, 0].set_xlim(f_time[0], f_time[-1])

        # Row 3, Column 2: Error of Corrected Data vs Ideal Data (Backward Only)
        # Uses original backward time window since it is an isolated single-segment plot
        axes[2, 1].plot(b_time_original, backward_df["b_err"], label="Original Error", color="orange", alpha=0.4)
        axes[2, 1].plot(b_time_original, backward_df["corrected_error"], label="Residual Error", color="tab:purple")
        apply_plot_formatting(config, axes[2, 1], time_label, "Error [deg]", "6. Residual Performance: Backward")
        axes[2, 1].set_xlim(b_time_original[0], b_time_original[-1])

        # Use tight_layout with explicit padding and add structural hspace to eliminate title/axis collisions
        plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=3.0)
        
        # IO Serialization
        filename = os.path.splitext(os.path.basename(csv_file))[0]
        output_dir = config.get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        
        allegro_eeprom_df.to_csv(f"{output_dir}/{filename}_allegro_32_eeprom_registers.csv", index=False)
        harmonics_df.to_csv(f"{output_dir}/{filename}_harmonic_optimization_parameters.csv", index=False)
        plt.savefig(f"{output_dir}/{filename}_harmonic_linearization.png", format="png")


                # Row 1, Column 1: Original Data (Whole dataset)
        axes[0, 0].plot(df[time_col], df[config.get("spi_angle_column", "Angle from SPI [°]")], label=f"Original Data Angle from SPI [°]", color="tab:blue")
        axes[0, 0].plot(df[time_col], df[config.get("abi_angle_column", "Angle from ABI [°]")], label=f"Original Data Angle from ABI [°]", color="tab:orange")

        apply_plot_formatting(config, axes[0, 0], time_label, y_label, "1. Original Sensor Dataset")
        axes[0, 0].set_xlim(df[time_col].iloc[0], 5200)

        if not batch_mode:
            plt.show()
        else:
            plt.close()


        spi_col = config.get("spi_angle_column", "Angle from SPI [°]")
        abi_col = config.get("abi_angle_column", "Angle from ABI [°]")

        # 1. Convert degrees to radians for the unwrap function
        spi_rad = np.radians(df[spi_col].values)
        abi_rad = np.radians(df[abi_col].values)

        # 2. Unwrap the phase jumps (removes the 360 to 0 transitions)
        spi_unwrapped = np.unwrap(spi_rad)
        abi_unwrapped = np.unwrap(abi_rad)

        # 3. Apply the smoothing filter on the continuous unwrapped data
        window_len = 15
        poly_order = 3
        spi_smooth_unwrapped = savgol_filter(spi_unwrapped, window_length=window_len, polyorder=poly_order)
        abi_smooth_unwrapped = savgol_filter(abi_unwrapped, window_length=window_len, polyorder=poly_order)

        # 4. Convert back to degrees and wrap back into the 0 to 360 range
        spi_smoothed = np.degrees(spi_smooth_unwrapped) % 360
        abi_smoothed = np.degrees(abi_smooth_unwrapped) % 360
        
        plot_format_size = {'width': 12, 'height': 7.5}
        fig2, ax2 = plt.subplots(figsize=[plot_format_size.get("width", 12), plot_format_size.get("height", 7.5)])        
        ax2.plot(df[time_col], df[config.get("spi_angle_column", "Angle from SPI [°]")], label=f"Original Data Angle from SPI [°]", color="tab:blue")
        ax2.plot(df[time_col], df[config.get("abi_angle_column", "Angle from ABI [°]")], label=f"Original Data Angle from ABI [°]", color="tab:orange")

        ax2.plot(df[time_col], spi_smoothed, label=f"Smoothed Data Angle from SPI [°]", color="tab:blue", linestyle="--")
        ax2.plot(df[time_col], abi_smoothed, label=f"Smoothed Data Angle from ABI [°]", color="tab:orange", linestyle="--")

        apply_plot_formatting(config, ax2, time_label, y_label, "1. Original Sensor Dataset")
        ax2.set_xlim(df[time_col].iloc[0], 5200)
        plt.show()


if __name__ == "__main__":
    main()