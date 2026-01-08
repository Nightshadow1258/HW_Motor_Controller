import numpy as np
import matplotlib.pyplot as plt

# Define the function: f(x) = x^3 - 6x^2 + 4x + 12
def function_to_plot(x):
    return 21.82*x**(-1.167)

# Define the minimum and maximum x values
x_min = 0.1
x_max = 1.3

# Generate x values, starting and ending at the defined limits
x_values = np.linspace(x_min, x_max, 500)

# Calculate y values
y_values = function_to_plot(x_values)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label='$I_o$')
plt.title('Plot of a Cubic Function with Custom X Limits')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.grid(True)
plt.legend()

# Ensure the plot limits strictly adhere to the provided min and max values
plt.xlim(x_min, x_max)

plt.savefig('PWM.png')
plt.close()