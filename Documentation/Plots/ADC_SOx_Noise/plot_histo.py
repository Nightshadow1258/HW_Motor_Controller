import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
import sys

 
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
    sampling_rate_ms:     Sampling interval in milliseconds (e.g., 100)
    time_window:          Time range to display [start_s, end_s] in seconds.
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


def ms_to_seconds(ms):
    return ms / 1000.0

def get_input_files(config):
    # Priority: input_files > input_file > all CSVs in input/
    if "input_files" in config:
        return config["input_files"]
    elif "input_file" in config:
        return [config["input_file"]]
    else:
        return [os.path.join("input", f) for f in os.listdir("input") if f.endswith(".csv")]

def apply_plot_formatting(config):
    plot_format = config.get("plot_format", {})
    fonts = plot_format.get("font_sizes", {})
    
    plt.xlabel(plot_format.get("axis_labels", {}).get("x", "X-AXIS NAME PLACEHOLDER"), fontsize=fonts.get("axis", 14))
    plt.ylabel(plot_format.get("axis_labels", {}).get("y", "Y-AXIS NAME PLACEHOLDER"), fontsize=fonts.get("axis", 14))
    plt.title(plot_format.get("plot_title", "TITLE"), fontsize=fonts.get("title", 16))
    plt.legend(fontsize=fonts.get("legend", 12))

    major = plot_format.get("grid", {}).get("major", {})
    minor = plot_format.get("grid", {}).get("minor", {})
    plt.grid(True, which='major', axis='both', 
            color=major.get("color", "#00000000"),
            linestyle=major.get("linestyle", ":"),
            linewidth=major.get("linewidth", 0.8))
    plt.grid(True, which='minor', axis='both', 
        color=minor.get("color", "#00000000"),
        linestyle=minor.get("linestyle", ":"),
        linewidth=minor.get("linewidth", 0.8))
    
    plt.minorticks_on()
    plt.xticks(fontsize=fonts.get("ticks", 12))
    plt.yticks(fontsize=fonts.get("ticks", 12))

    plt.tight_layout()


def apply_subplot_formatting(ax, config):
    plot_format = config.get("plot_format", {})
    fonts = plot_format.get("font_sizes", {})
    ax.set_xlabel(config.get("axis_labels", {}).get("x", "Interger [LSB]"), fontsize=fonts.get("axis", 14))
    ax.set_ylabel(config.get("axis_labels", {}).get("y", "Frequency [#]"), fontsize=fonts.get("axis", 14))
    major = plot_format.get("grid", {}).get("major", {})
    minor = plot_format.get("grid", {}).get("minor", {})
    ax.grid(True, which='major', axis='both', 
            color=major.get("color", "#00000000"),
            linestyle=major.get("linestyle", ":"),
            linewidth=major.get("linewidth", 0.8))
    ax.grid(True, which='minor', axis='both', 
        color=minor.get("color", "#00000000"),
        linestyle=minor.get("linestyle", ":"),
        linewidth=minor.get("linewidth", 0.8))

    ax.minorticks_on()
    ax.tick_params(axis='both', labelsize=fonts.get("ticks", 12))
    
def save_formats(plt, df, csv_file, config, output_tag=""):
    """
    Saves plot and cleaned data in multiple formats as specified in the config.

    Parameters:
    - plt (matplotlib.pyplot): The plotting context containing the figure to be saved.
    - df (pandas.DataFrame): The cleaned DataFrame to be exported.
    - csv_file (str): Path to the original CSV file, used to derive the output filename.
    - config (dict): Configuration dictionary containing:
        - 'output_dir' (str): Directory where outputs will be saved. Defaults to 'output'.
        - 'save_formats' (dict): Specifies formats to save:
            - 'image' (list[str]): List of image formats (e.g., ['jpeg', 'pdf']) for plot export.
            - 'data' (list[str]): List of data formats (e.g., ['csv', 'xlsx']) for DataFrame export.

    Behavior:
    - Creates the output directory if it doesn't exist.
    - Saves the plot in each specified image format using `plt.savefig`.
    - Saves the DataFrame in each specified data format using `df.to_csv` or `df.to_excel`.

    Example config:
    save_formats:
      image: ["jpeg", "pdf"]
      data: ["csv", "xlsx"]
    """

    filename = os.path.splitext(os.path.basename(csv_file))[0]
    output_dir = config.get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)
    save_formats = config.get("save_formats", [])
    for fmt_img in save_formats.get("image", []) or []:
        plt.savefig(f"{output_dir}/{filename}{output_tag}.{fmt_img}", format=fmt_img)

    for fmt_data in save_formats.get("data", []) or []:
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

    # determine if batch mode
    batch_mode = len(input_files) > 1

    for csv_file in input_files:
        print(f"Processing: {csv_file}")
        # Load CSV
        df = pd.read_csv(csv_file, header=0)
        cleanup_rules = config.get("header_cleanup", [])
        df.columns = clean_column_names(df.columns, cleanup_rules)

        # data preprocessing
        # conversion from hex to int
        # df_int = df.map(lambda x: int(x, 16))
        df_int = df # hex conversion is commented out, assuming data is already in integer format

        # Flatten all values into one list
        all_values = df_int.values.flatten()

        # Plot setup
        fonts = config.get("font_sizes", {})
        filename = os.path.splitext(os.path.basename(csv_file))[0]
        output_dir = config.get("output_dir", "output")

        # Single Histogramm
        plot_format = config.get("plot_format", {})
        plot_format_size = plot_format.get("size")
        plt.figure(figsize=[plot_format_size.get("width",12), plot_format_size.get("height",7.5)])
        for i, column in enumerate(df_int.columns):
            plt.hist(df_int[column], bins=30, alpha=0.6, label=column, edgecolor="black")

        apply_plot_formatting(config)

        save_formats(plt, df, csv_file, config, "_all")

        if not batch_mode:
            plt.show()
        else:
            plt.close()

        # Separated Hsitogram

        fig, axes = plt.subplots(1, 3, figsize=[plot_format_size.get("width",12), plot_format_size.get("height",7.5)])
        axes = axes.flatten()
        plot_format = config.get("plot_format", {})
        fonts = plot_format.get("font_sizes", {})

        # Plot each column in its own subplot
        for i, column in enumerate(df_int.columns):

            data = df_int[column]
            mean_val = data.mean()
            std_dev = data.std()
            data_min = data.min()
            data_max = data.max()
            delta = data_max - data_min

            axes[i].hist(df_int[column], bins=30, edgecolor="black", alpha=0.75)
            axes[i].set_xlabel("Integer Value", fontsize=fonts.get("axis", 12))
            axes[i].set_ylabel("Frequency", fontsize=fonts.get("axis", 12))
            axes[i].set_title(f"ADC Histogram of {column}\nMean={mean_val:.2f}, Std={std_dev:.2f}\nMin={data_min:.2f}, Max={data_max:.2f}, delta={delta}", fontsize=fonts.get("title", 14))
            
            axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}', linewidth=1.5)
            axes[i].axvline(mean_val + std_dev, color='green', linestyle=':', label=f'+1 Std: {mean_val + std_dev:.2f}', linewidth=1.5)
            axes[i].axvline(mean_val - std_dev, color='green', linestyle=':', label=f'-1 Std: {mean_val - std_dev:.2f}', linewidth=1.5)
            axes[i].minorticks_on()

        
            # apply formatting
            apply_subplot_formatting(axes[i],config)

        plt.tight_layout()

        # Save outputs

        os.makedirs(output_dir, exist_ok=True)

        save_formats(plt, df, csv_file, config)

        if not batch_mode:
            plt.show()
        else:
            plt.close()

if __name__ == "__main__":
    main()
