import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import sys
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error, r2_score

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

    ax.minorticks_on()
    major = plot_format.get("grid", {}).get("major", {})
    minor = plot_format.get("grid", {}).get("minor", {})
    
    ax.grid(True, which='major', axis='both', color=major.get("color", "#D3D3D3"), linestyle=major.get("linestyle", ":"), linewidth=0.8)
    ax.grid(True, which='minor', axis='both', color=minor.get("color", "#A8A8A8C8"), linestyle=minor.get("linestyle", ":"), linewidth=0.5)
    ax.tick_params(axis='both', which='both', labelsize=fonts.get("ticks", 12))

def fit_harmonics(angles_deg, errors, num_harmonics=2):
    angles_rad = np.radians(angles_deg)
    matrix_columns = [np.ones_like(angles_rad)]
    
    for i in range(1, num_harmonics + 1):
        matrix_columns.append(np.sin(i * angles_rad))
        matrix_columns.append(np.cos(i * angles_rad))
        
    A = np.column_stack(matrix_columns)
    coeffs, _, _, _ = np.linalg.lstsq(A, errors, rcond=None)
    return coeffs

def evaluate_harmonics(angles_deg, coeffs, num_harmonics=2):
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

    num_harmonics = config.get("num_harmonics", 2)

    for csv_file in input_files:
        print(f"Processing: {csv_file}")
        df = pd.read_csv(csv_file)
        
        cleanup_rules = config.get("header_cleanup", [])
        df.columns = clean_column_names(df.columns, cleanup_rules)
        
        x_col_index = config.get("x_axis_col", 1)
        time_col = df.columns[x_col_index]

        # clean and filter input data due to mechanical properies of the bldc motor the test setup creates oscialtions
        # get column names
        spi_col = config.get("spi_angle_column", "Angle from SPI [°]")
        abi_col = config.get("abi_angle_column", "Angle from ABI [°]")


        plot_format = config.get("plot_format", {})
        plot_format_size = plot_format.get("size", {})
        plot_format_width = plot_format_size.get("width",12)
        plot_format_height = plot_format_size.get("height",7.5)
        
        x_label = plot_format.get("axis_labels", {}).get("x", "Time [ms]")
        y_label = plot_format.get("axis_labels", {}).get("y", {})

        # 1. Convert degrees to radians for the unwrap function
        spi_rad = np.radians(df[spi_col].values)
        abi_rad = np.radians(df[abi_col].values)

        # 2. Unwrap the phase jumps (removes the 360 to 0 transitions)
        spi_unwrapped = np.unwrap(spi_rad)
        abi_unwrapped = np.unwrap(abi_rad)

        # 3. Apply the filtereding filter on the continuous unwrapped data
        window_len = 20
        poly_order = 1

        spi_filtered_unwrapped = savgol_filter(spi_unwrapped, window_length=window_len, polyorder=poly_order)
        abi_filtered_unwrapped = savgol_filter(abi_unwrapped, window_length=window_len, polyorder=poly_order)

        # 4. Convert back to degrees and wrap back into the 0 to 360 range
        spi_filtered = np.degrees(spi_filtered_unwrapped) % 360
        abi_filtered = np.degrees(abi_filtered_unwrapped) % 360
        

        fig, axes = plt.subplots(1,2,figsize=[plot_format_width,plot_format_height])        
        axes[0].plot(df[time_col], df[config.get("spi_angle_column", "Angle from SPI [°]")], label=f"Original Data Angle from SPI [°]", color="tab:blue")
        axes[0].plot(df[time_col], df[config.get("abi_angle_column", "Angle from ABI [°]")], label=f"Original Data Angle from ABI [°]", color="tab:orange")

        axes[0].plot(df[time_col], spi_filtered, label=f"filtered Data Angle from SPI [°]", color="tab:blue", linestyle="--")
        axes[0].plot(df[time_col], abi_filtered, label=f"filtered Data Angle from ABI [°]", color="tab:orange", linestyle="--")

        apply_plot_formatting(config, axes[0], x_label, y_label, "1. Original Sensor Dataset")
        axes[0].set_xlim(df[time_col].iloc[0], 5200)

        axes[1].plot(df[time_col], df[config.get("spi_angle_column", "Angle from SPI [°]")], label=f"Original Data Angle from SPI [°]", color="tab:blue")
        axes[1].plot(df[time_col], df[config.get("abi_angle_column", "Angle from ABI [°]")], label=f"Original Data Angle from ABI [°]", color="tab:orange")

        axes[1].plot(df[time_col], spi_filtered, label=f"filtered Data Angle from SPI [°]", color="tab:blue", linestyle="--")
        axes[1].plot(df[time_col], abi_filtered, label=f"filtered Data Angle from ABI [°]", color="tab:orange", linestyle="--")

        apply_plot_formatting(config, axes[0], x_label, y_label, "Orignal vs filtered Data")
        apply_plot_formatting(config, axes[1], x_label, y_label, "Orignal vs filtered Data (zoomed)")
        zoom_start = 760
        zoom_end = 840
        axes[1].set_xlim(zoom_start, zoom_end)

        # Filter the dataframe to get data points within the zoom window
        zoom_mask = (df[time_col] >= zoom_start) & (df[time_col] <= zoom_end)
        df_zoomed = df[zoom_mask]

        # Find the absolute min and max across all plotted series in this window
        y_min = min(df_zoomed[spi_col].min(), df_zoomed[abi_col].min())
        y_max = max(df_zoomed[spi_col].max(), df_zoomed[abi_col].max())

        # Add a small margin (e.g., 5%) so the lines do not touch the edges
        y_margin = (y_max - y_min) * 0.05
        axes[1].set_ylim(y_min - y_margin, y_max + y_margin)

        filename = os.path.splitext(os.path.basename(csv_file))[0]
        output_dir = config.get("output_dir", "output")
        plt.savefig(f"{output_dir}/{filename}_filtered.png", format="png")

        plt.tight_layout()
        plt.show()
        
        # use the filtered data from here on
        df[abi_col] = abi_filtered
        df[spi_col] = spi_filtered

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
        common_angles = np.linspace(0, 360, 8192, endpoint=False)

        # Interpolate spatial distributions using corrected dataframes
        f_err_interp = np.interp(common_angles, forward_df[angle_col].values, forward_df["f_err"].values)
        b_sort_idx = np.argsort(backward_df[angle_col].values)
        b_err_interp = np.interp(common_angles, backward_df[angle_col].values[b_sort_idx], backward_df["b_err"].values[b_sort_idx])

        # Core averaging to collapse hysteresis error signatures
        averaged_error = (f_err_interp + b_err_interp) / 2.0

        # Run Harmonic Least-Squares Optimization
        harmonic_coeffs = fit_harmonics(common_angles, averaged_error, num_harmonics)
        print(f"Harmonic Coefficients for {csv_file}: {harmonic_coeffs}")
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
        # still not optimal because of the hard boundary conditions at the edges of the 32 node segments, but it is a good starting point for the Allegro register mapping
        poly_coeffs = np.polyfit(common_angles, averaged_error, 15)
        poly_fit_continuous = np.polyval(poly_coeffs, common_angles)
        poly_fit_nodes = np.polyval(poly_coeffs, allegro_nodes)

        fig3, ax3 = plt.subplots(figsize=[plot_format_width, plot_format_height])
        # 1. Plot the original averaged error signature baseline
        ax3.plot(common_angles, averaged_error, label="Averaged Error Baseline", color="tab:gray", alpha=0.6)
        # 2. Plot the continuous 5th-degree polynomial fit curve
        ax3.plot(common_angles, poly_fit_continuous, label="15th-Degree Poly Fit Trend", color="tab:red", linewidth=2)
        # 3. Plot the specific discrete sample points at the 32 Allegro register locations
        ax3.scatter(allegro_nodes, poly_fit_nodes, label="Allegro Register Nodes (32)", color="tab:blue", zorder=3, marker="o")
        # Format and display the chart
        apply_plot_formatting(config, ax3, "Mechanical Angle [°]", "Error Deviation [°]", "Polynomial Fit vs Allegro Nodes")
        ax3.set_xlim(0, 360)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{filename}_linear.png", format="png")

        hardware_corrections = -np.polyval(poly_coeffs, allegro_nodes)


        # Generate structural output configuration table
        allegro_eeprom_df = pd.DataFrame({
            "Register_Index": range(32),
            "Sensor_Angle_Deg": allegro_nodes,
            "EEPROM_Correction_Deg": hardware_corrections
        }).round(6)

        # Verification for linearization and correction implementation using the Allegro 32 node register mapping
        # Simulate hardware correction implementation using 32 segment lookup nodes
        forward_df["linear_simulated_correction"] = np.interp(forward_df[angle_col], allegro_nodes, hardware_corrections, period=360)
        forward_df["linear_corrected_position"] = (forward_df[angle_col] + forward_df["linear_simulated_correction"]) % 360
    
        backward_df["linear_simulated_correction"] = np.interp(backward_df[angle_col], allegro_nodes, hardware_corrections, period=360)
        backward_df["linear_corrected_position"] = (backward_df[angle_col] + backward_df["linear_simulated_correction"]) % 360

        # Compute post-calibration residual errors
        f_err_calibrated = forward_df["linear_corrected_position"].values - forward_df["ideal_position"].values
        forward_df["linear_corrected_error"] = (f_err_calibrated + 180) % 360 - 180

        b_err_calibrated = backward_df["linear_corrected_position"].values - backward_df["ideal_position"].values
        backward_df["linear_corrected_error"] = (b_err_calibrated + 180) % 360 - 180


        # Verifiaciton of the harmonic model fit against the continuous extracted profiles
        # Evaluate the optimized Fourier series across the continuous extracted profiles
        forward_df["harmonic_corrected_position"] = forward_df[angle_col] - evaluate_harmonics(forward_df[angle_col].values, harmonic_coeffs, num_harmonics)
        backward_df["harmonic_corrected_position"] = backward_df[angle_col] - evaluate_harmonics(backward_df[angle_col].values, harmonic_coeffs, num_harmonics)
        
        # Calculate residual error if using the raw mathematical model directly (before 32-node quantization)
        forward_df["math_residual_error"] = forward_df["f_err"].values - forward_df["harmonic_corrected_position"].values
        f_err_calibrated = forward_df["linear_corrected_position"].values - forward_df["ideal_position"].values
        backward_df["harmonic_corrected_error"] = (b_err_calibrated + 180) % 360 - 180
        backward_df["math_residual_error"] = backward_df["b_err"].values - backward_df["harmonic_corrected_position"].values

        # Compute post-calibration residual errors
        f_err_calibrated = forward_df["harmonic_corrected_position"].values - forward_df["ideal_position"].values
        forward_df["harmonic_corrected_error"] = (f_err_calibrated + 180) % 360 - 180

        b_err_calibrated = backward_df["harmonic_corrected_position"].values - backward_df["ideal_position"].values
        backward_df["harmonic_corrected_error"] = (b_err_calibrated + 180) % 360 - 180



        # Visualization block setup (6 subplots organized in a 3x2 grid)
        plot_format = config.get("plot_format", {})
        plot_size = plot_format.get("size", {"width": 16, "height": 14})
        
        fig, axes = plt.subplots(3, 2, figsize=[10,16])
        x_label = plot_format.get("axis_labels", {}).get("x", "Time [ms]")
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

        apply_plot_formatting(config, axes[0, 0], x_label, y_label, "1. Original Sensor Dataset")
        axes[0, 0].set_xlim(df[time_col].iloc[0], 5200)
        
        # Row 1, Column 2: Extracted Profiles (Continuous Time)
        axes[0, 1].plot(f_time, forward_df["ideal_position"], label="Ideal Profile", color="tab:purple", linestyle="--")
        axes[0, 1].plot(f_time, forward_df[angle_col], label="Extracted Forward Profile", color="tab:blue")

        axes[0, 1].plot(b_time_shifted, backward_df["ideal_position"], label="Ideal Profile", color="tab:purple", linestyle="--")
        axes[0, 1].plot(b_time_shifted, backward_df[angle_col], label="Extracted Backward Profile", color="tab:blue")
        apply_plot_formatting(config, axes[0, 1], x_label, y_label, "2. Extracted Processing Profiles")
        axes[0, 1].set_xlim(f_time[0], b_time_shifted[-1])

        # Row 2, Column 1: Error from Ideal Linear Curve (Continuous Time)
        axes[1, 0].plot(f_time, forward_df["f_err"], label="Forward Error", color="tab:red", alpha=0.7)
        axes[1, 0].plot(b_time_shifted, backward_df["b_err"], label="Backward Error", color="orange", alpha=0.7)
        apply_plot_formatting(config, axes[1, 0], x_label, "Error [deg]", "3. Raw Error Deviation")
        axes[1, 0].set_xlim(f_time[0], b_time_shifted[-1])

        # Row 2, Column 2: Original vs Corrected Data (Continuous Time)
        axes[1, 1].plot(f_time, forward_df[angle_col], label="Original Forward", color="tab:blue", alpha=0.5)
        axes[1, 1].plot(f_time, forward_df["linear_corrected_position"], label="Linear Corrected Forward", color="tab:green", linestyle="--")
        axes[1, 1].plot(f_time, forward_df["harmonic_corrected_position"], label="Harmonic Corrected Forward", color="tab:red", linestyle="-.")
        axes[1, 1].plot(b_time_shifted, backward_df[angle_col], label="Original Backward", color="tab:purple", alpha=0.5)
        axes[1, 1].plot(b_time_shifted, backward_df["linear_corrected_position"], label="Linear Corrected Backward", color="lime", linestyle="--")
        axes[1, 1].plot(b_time_shifted, backward_df["harmonic_corrected_position"], label="Harmonic Corrected Backward", color="tab:orange", linestyle="-.")


        apply_plot_formatting(config, axes[1, 1], x_label, y_label, "4. Original vs. Linear and Harmonic Correction")
        axes[1, 1].set_xlim(f_time[0], b_time_shifted[-1])

        # Row 3, Column 1: Error of Corrected Data vs Ideal Data (Forward Only)
        # axes[2, 0].plot(f_time, forward_df["f_err"], label="Original Error", color="tab:orange", alpha=0.4)
        axes[2, 0].plot(f_time, forward_df["linear_corrected_error"], label="Linear Residual Error", color="tab:green")
        axes[2, 0].plot(f_time, forward_df["harmonic_corrected_error"], label="Harmonic Residual Error", color="tab:red")
        apply_plot_formatting(config, axes[2, 0], x_label, "Error [deg]", "5. Residual Performance: Forward")
        axes[2, 0].set_xlim(f_time[0], f_time[-1])

        # Row 3, Column 2: Error of Corrected Data vs Ideal Data (Backward Only)
        # Uses original backward time window since it is an isolated single-segment plot
        # axes[2, 1].plot(b_time_original, backward_df["b_err"], label="Original Error", color="orange", alpha=0.4)
        axes[2, 1].plot(b_time_original, backward_df["linear_corrected_error"], label="Linear Residual Error", color="tab:green")
        axes[2, 1].plot(b_time_original, backward_df["harmonic_corrected_error"], label="Harmonic Residual Error", color="tab:red")
        apply_plot_formatting(config, axes[2, 1], x_label, "Error [deg]", "6. Residual Performance: Backward")
        axes[2, 1].set_xlim(b_time_original[0], b_time_original[-1])

        # Use tight_layout with explicit padding and add structural hspace to eliminate title/axis collisions
        plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=1.0)
        
        # IO Serialization

        os.makedirs(output_dir, exist_ok=True)
        
        allegro_eeprom_df.to_csv(f"{output_dir}/{filename}_allegro_32_eeprom_registers.csv", index=False)
        harmonics_df.to_csv(f"{output_dir}/{filename}_harmonic_optimization_parameters.csv", index=False)
        plt.savefig(f"{output_dir}/{filename}_harmonic_linearization.png", format="png")


                # Row 1, Column 1: Original Data (Whole dataset)
        axes[0, 0].plot(df[time_col], df[config.get("spi_angle_column", "Angle from SPI [°]")], label=f"Original Data Angle from SPI [°]", color="tab:blue")
        axes[0, 0].plot(df[time_col], df[config.get("abi_angle_column", "Angle from ABI [°]")], label=f"Original Data Angle from ABI [°]", color="tab:orange")

        apply_plot_formatting(config, axes[0, 0], x_label, y_label, "1. Original Sensor Dataset")
        axes[0, 0].set_xlim(df[time_col].iloc[0], 5200)

        if not batch_mode:
            plt.show()
        else:
            plt.close()

        linear_rmse = np.sqrt(mean_squared_error(forward_df["linear_corrected_error"], np.zeros_like(forward_df["linear_corrected_error"])))
        linear_min_err = np.min(np.abs(forward_df["linear_corrected_error"]))
        linear_max_err = np.max(np.abs(forward_df["linear_corrected_error"]))

        harmonic_rmse = np.sqrt(mean_squared_error(forward_df["harmonic_corrected_error"], np.zeros_like(forward_df["harmonic_corrected_error"])))
        harmonic_min_err = np.min(np.abs(forward_df["harmonic_corrected_error"]))
        harmonic_max_err = np.max(np.abs(forward_df["harmonic_corrected_error"]))

        print(f"\nLinear Correction vs Harmonic Correction for {csv_file}:")
        print(f"Linear RMSE: {linear_rmse:.4f}")
        print(f"Linear Min Error: {linear_min_err:.4f}")
        print(f"Linear Max Error: {linear_max_err:.4f}")
        print("")
        print(f"Harmonic RMSE: {harmonic_rmse:.4f}")
        print(f"Harmonic Min Error: {harmonic_min_err:.4f}")
        print(f"Harmonic Max Error: {harmonic_max_err:.4f}")


if __name__ == "__main__":
    main()