#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

"""
Build a custom 500 hPa temperature colormap from discrete temperature/color anchors.
"""
temps = np.array([
    -50, -45, -40, -35, -30, -25, -19, -20, -15, -10, -5, 0
])
colors_hex = [
    "#b3ebe0", #-50
    "#9c76c0", #-45
    "#951aa4", #-40
    "#bf98d0", #-35
    "#bf98d0", #-30
    "#6e95d0", #-25
    "#6e95d0", #-20
    "#1a505f", #-19
    "#a7bc8f", #-15
    "#f8eea2", #-10
    "#a66a48", #-5
    "#540100", #-0
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

