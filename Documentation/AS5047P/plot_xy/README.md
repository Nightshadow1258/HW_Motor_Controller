# CSV Plotting Script single

This Python script reads a CSV file and generates a time-based plot using configuration options defined in a `config.yaml` file. It supports flexible column selection, custom styling, header cleanup, and export to multiple formats.

---

## Usage
```
python3 plot_single.py config.yaml
```
show help/usage with
```
python3 plot_single.py --help
```

## Features
- Automatically extracts and cleans column headers
- Time axis calculated from sampling rate
- Optional time window slicing
- Gridlines and tight axis bounds
- Customizable fonts and plot size
- Saves to JPEG, PDF, or both
- batch processing of all csv files in input folder (disables interactive preview of matplotlib - only if one file is processed)

## Output
Plots are saved to the output/ directory using the input filename as the base.