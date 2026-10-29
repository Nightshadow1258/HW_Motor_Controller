import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import yaml
import os
import sys
from scipy.signal import savgol_filter

def print_usage():
    print("""
Usage:
    python plot_single.py <config.yaml>

Description:
    This script reads one or more CSV files and plots selected columns based on settings in a YAML config file.

Input File Options (choose one):
    input_file:           Path to a single CSV file (e.g., input/data.csv)
    input_files:          List of multiple CSV files to process (e.g., ["input/data1.csv", "input/data2.csv"])
                          If neither is specified, all CSV files in the 'input/' folder will be processed automatically.

Available Configuration Options in config.yaml:
    plot_size:            Plot dimensions in inches [width, height] (e.g., [10, 6])
    font_sizes:
        title:            Font size for the plot title
        legend:           Font size for the legend
        axis:             Font size for axis labels
    columns_to_plot:      List of column indices to plot (excluding time/sample column).
                          Use [] or omit to plot all available columns.
    x_unit:               Optional x-axis unit: "auto" (default), "s", "ms", "us", "ns".
    x_step:               Optional step size for x-axis ticks in the selected x_unit (e.g. 1, 10, 100).
    x_window:             limit x-axis range to display [start, end].
                          Leave empty or omit to plot full range.
    header_cleanup:       List of substrings to remove from column headers (e.g., ["Ave. (C)", "(C)"])
    save_formats:         Output formats to save the plot (e.g., ["jpeg", "pdf"] or [] to skip saving)
    output_dir:           Optional. Directory to save output files. Defaults to "output/"
    show_plot:            plot will be displayed interactively if only one file is processed.

Example:
    python plot_single.py config.yaml
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

def auto_scale_x(x_seconds, preferred_unit="auto"):
    if len(x_seconds) > 0:
        max_val = abs(x_seconds).max()
    else:
        max_val = 0

    unit = preferred_unit.lower()
    if unit == "auto":
        if max_val < 1e-6:
            unit = "ns"
        elif max_val < 1e-3:
            unit = "us"
        elif max_val < 1.0:
            unit = "ms"
        else:
            unit = "s"

    factors = {
        "ns": (1e9, "Zeit (ns)"),
        "us": (1e6, "Zeit (µs)"),
        "ms": (1e3, "Zeit (ms)"),
        "s":  (1.0, "Zeit (s)")
    }

    factor, default_label = factors.get(unit, (1.0, "Zeit (s)"))
    return x_seconds * factor, default_label, unit

def apply_plot_formatting(fig, ax1, ax2, config, x_label_auto="", x_step=None):
    plot_format = config.get("plot_format", {})
    fonts = plot_format.get("font_sizes", {})

    # X axis label
    x_label = plot_format.get("axis_labels", {}).get("x", x_label_auto)
    ax1.set_xlabel(x_label, fontsize=fonts.get("axis", 14))

    # Y axis labels
    y_label_v = plot_format.get("axis_labels", {}).get("y_voltage", "Voltage [V]")
    y_label_i = plot_format.get("axis_labels", {}).get("y_current", "Current [A]")
    
    ax1.set_ylabel(y_label_v, fontsize=fonts.get("axis", 14), color="tab:orange")
    ax2.set_ylabel(y_label_i, fontsize=fonts.get("axis", 14), color="tab:blue")

    ax1.set_title(plot_format.get("plot_title", "TITLE"), fontsize=fonts.get("title", 16))

    # X axis locator
    if x_step is not None and x_step > 0:
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(x_step))
    else:
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins='auto', steps=[1, 2, 5, 10]))

    # Grid applied to primary axis
    major = plot_format.get("grid", {}).get("major", {})
    minor = plot_format.get("grid", {}).get("minor", {})
    
    ax1.grid(True, which='major', axis='both', 
             color=major.get("color", "#CCCCCC"),
             linestyle=major.get("linestyle", ":"),
             linewidth=major.get("linewidth", 0.8))
    ax1.grid(True, which='minor', axis='both', 
             color=minor.get("color", "#EEEEEE"),
             linestyle=minor.get("linestyle", ":"),
             linewidth=minor.get("linewidth", 0.8))

    ax1.minorticks_on()
    
    # Tick formatting
    ax1.tick_params(axis='x', labelsize=fonts.get("ticks", 12))
    ax1.tick_params(axis='y', labelsize=fonts.get("ticks", 12), labelcolor="tab:orange")
    ax2.tick_params(axis='y', labelsize=fonts.get("ticks", 12), labelcolor="tab:blue")

    fig.tight_layout()

def save_formats(plt, df, csv_file, config, output_tag=""):
    filename = os.path.splitext(os.path.basename(csv_file))[0]
    output_dir = config.get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)
    save_fmt = config.get("save_formats", {})

    for fmt_img in save_fmt.get("image", []) or []:
        plt.savefig(f"{output_dir}/{filename}{output_tag}.{fmt_img}", format=fmt_img)

    for fmt_data in save_fmt.get("data", []) or []:
        if fmt_data == "csv":
            df.to_csv(f"{output_dir}/{filename}{output_tag}_cleaned.csv", index=False)
        elif fmt_data == "xlsx":
            df.to_excel(f"{output_dir}/{filename}{output_tag}_cleaned.xlsx", index=False)

def main():
    if len(sys.argv) != 2 or sys.argv[1] in ("--help", "-h"):
        print_usage()
        return

    config_file = sys.argv[1]
    config = load_config(config_file)
    input_files = get_input_files(config)

    batch_mode = len(input_files) > 1

    for csv_file in input_files:
        print(f"Processing: {csv_file}")
        df = pd.read_csv(csv_file)
        cleanup_rules = config.get("header_cleanup", [])
        df.columns = clean_column_names(df.columns, cleanup_rules)

        x_col_index = config.get("x_axis_col", 0)
        x_raw = df.iloc[:, x_col_index].values

        x_window = config.get("x_window", [])
        if x_window:
            start_idx, end_idx = x_window
        else:
            start_idx = 0
            end_idx = len(x_raw)
        print(f"x_window: {x_window}, start_idx: {start_idx}, end_idx: {end_idx}") 

        x_sliced = x_raw[start_idx:end_idx]

        preferred_unit = config.get("x_unit", "auto")
        x_windowed, x_label_auto, unit_used = auto_scale_x(x_sliced, preferred_unit)

        all_columns = df.columns[1:]
        selected_indices = config.get("columns_to_plot", [])
        if not selected_indices:
            selected_indices = list(range(len(all_columns)))
        selected_columns = [all_columns[i] for i in selected_indices]

        plot_format = config.get("plot_format", {})
        plot_format_size = plot_format.get("size", {})
        
        fig, ax1 = plt.subplots(figsize=[plot_format_size.get("width", 12), plot_format_size.get("height", 7.5)])
        ax2 = ax1.twinx()

        lines = []

        for col in selected_columns:
            voltage_sliced = df[col][start_idx:end_idx]
            filtered_voltage = savgol_filter(voltage_sliced, 20, 3)

            # Calculation: (V - 1.65) / 0.04
            current_data = (filtered_voltage - 1.65) / 0.04

            # Visibility technique: Solid thick line behind, thinner dashed line in front
            line1 = ax1.plot(
                x_windowed, filtered_voltage, 
                label=f"{col} (Voltage)", 
                color="tab:orange", 
                linestyle="-", 
                linewidth=2.5, 
                alpha=0.6
            )
            
            line2 = ax2.plot(
                x_windowed, current_data, 
                label=f"{col} (Current)", 
                color="tab:blue", 
                linestyle="--", 
                linewidth=1.5
            )

            lines.extend(line1 + line2)

        # Combined Legend
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right", fontsize=plot_format.get("font_sizes", {}).get("legend", 12))

        x_step = config.get("x_step", None)
        
        # Pass explicit fig, ax1, ax2 references
        apply_plot_formatting(fig, ax1, ax2, config, x_label_auto=x_label_auto, x_step=x_step)
        ax1.set_xlim(x_windowed[0], x_windowed[-1])
        
        save_formats(plt, df, csv_file, config)

        if not batch_mode:
            plt.show()
        else:
            plt.close()

if __name__ == "__main__":
    main()