# CSV Plotting Script single

This Python script reads a CSV file and generates a histogram using the configuration options defined in a `config.yaml` file. It supports single or multi historgrams, custom styling, header cleanup, and export to multiple formats.

---

## Usage
```
python3 plot_histo.py config.yaml
```
show help/usage with
```
python3 plot_histo.py --help
```

## Features
- Automatically extracts and cleans column headers
- show mean, min, max and delta for each column (multi mode only)
- Gridlines and tight axis bounds
- Customizable fonts and plot size
- Saves to JPEG, PDF, or both

## Output
Plots are saved to the output/ directory using the input filename as the base.