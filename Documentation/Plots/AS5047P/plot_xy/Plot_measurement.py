import pandas as pd
import matplotlib.pyplot as plt

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



# Load the data directly from the CSV file
file_path = 'input/'
file_name = 'AS5047P_4096_BLDC_Mot_calc_corr_2'
df = pd.read_csv(file_path + file_name + '.csv')

# Strip leading and trailing whitespace from column names
df.columns = df.columns.str.strip()

# Define the x axis variable
x_axis = 'ts [ms]'

# Identify the angle columns for the y axis (excludes index and time columns)
y_columns = [col for col in df.columns if col not in ['idx', x_axis]]

# Initialize the plot layout using the object oriented interface
fig, ax = plt.subplots(figsize=(10, 6))

# Iterate through each angle column to generate lines
for column in y_columns:
    ax.plot(df[x_axis], df[column], label=column, marker='')

# Enable minor ticks so minor grid lines can be drawn
ax.minorticks_on()

# Configure major and minor grid lines
ax.grid(True, which='major', axis='both', color="#000000", linestyle=":", linewidth=0.8)
ax.grid(True, which='minor', axis='both', color="#A8A8A8", linestyle=":", linewidth=0.5)

# Apply axis labels, limits, and title directly to the ax object
ax.set_xlabel('Time [ms]', fontsize=14)
ax.set_ylabel('Angle [degrees]', fontsize=14)
ax.set_title('Angle Measurements over Time', fontsize=16)
ax.set_xlim(0, 5200)

# Configure ticks and legend
ax.tick_params(axis='both', which='both', labelsize=12)
ax.legend(fontsize=12)

# Save the resulting visualization
plt.savefig(f"output/{file_name}_plot.png", format="png", bbox_inches='tight')

# Display the plot
plt.show()


