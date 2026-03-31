#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

"""
Build a custom 850 hPa temperature colormap from discrete temperature/color anchors.
"""
temps = np.array([
    -40, -35, -30, -25, -20, -15, -10, -5,
    0, 1,  5,  10,  15,  20,  25,  30,  35,  40
])
colors_hex = [
    "#b3ece0", "#a8b7d2", "#a08ac6", "#975eba", "#8f31ae",
    "#b478c4", "#d5e2e8", "#799ed4", "#1e51b5", "#16505f",  "#8faa87",
    "#f8eea2", "#caa46f", "#9c593e", "#6d0f0a", "#7c2130",
    "#bfa098", "#c8c0ba"
]
# Normalize the anchor temperatures to the [0, 1] interval expected by Matplotlib.
t_norm = (temps - temps[0]) / (temps[-1] - temps[0])
points = list(zip(t_norm, colors_hex))

colormap = mcolors.LinearSegmentedColormap.from_list("temp_map", points)

def demo_gradual():
    # Create a simple synthetic field so the colormap can be previewed by itself.
    x = np.linspace(-40, 40, 200)
    y = np.linspace(-40, 40, 200)
    xx, yy = np.meshgrid(x, y)
    data = xx

    cmap = colormap

    plt.figure(figsize=(6, 5))
    im = plt.imshow(data, cmap=cmap, origin='lower', extent=(-40,40,-40,40),
                    vmin=-40, vmax=40)
    plt.title("Custom Gradual Colormap: -40 to 40")
    plt.xlabel("X axis (temp range)")
    plt.ylabel("Y axis (temp range)")

    cb = plt.colorbar(im, shrink=0.8, ticks=range(-40, 41, 5))
    cb.set_label("Temperature (°C)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo_gradual()

