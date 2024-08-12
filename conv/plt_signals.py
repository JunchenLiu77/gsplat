import matplotlib.pyplot as plt
import numpy as np
from signals import SIGNALS_2D

# Set up a grid for the x-axis
x = np.linspace(0, 4 * np.pi, 1000)

# Create a figure with subplots
fig, axs = plt.subplots(4, 3, figsize=(18, 18))

# Iterate over the SIGNALS_1D dictionary and plot each signal
for i, (signal_name, signal_func) in enumerate(SIGNALS_2D.items()):
    row, col = divmod(i, 3)
    y = signal_func(x)
    axs[row, col].plot(x, y)
    axs[row, col].set_title(signal_name.replace("_", " ").title())

# Adjust layout and remove empty subplot
plt.tight_layout()

# Save the figure as an image
image_path = "signals_2d.png"
plt.savefig(image_path)
plt.close(fig)
