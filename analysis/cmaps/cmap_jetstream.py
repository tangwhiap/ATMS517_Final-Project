#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

"""
Build a custom jet-stream wind-speed colormap from discrete speed/color anchors.
"""
temps = np.arange(50, 171, 5)

colors_hex = [
    "#f1f8ff",
    "#bde3fd",
    "#87cefa",
    "#7995e5",
    "#6a5acd",
    "#a878d5",
    "#e796dc", #80
    "#d779cd", #85
    "#c85abf", #90
    "#b437ab", #95
    "#a01497", #100
    "#b5095f", #105
    "#c80028", #110
    "#d21432", #115
    "#dd283d", #120
    "#e73c46", #125
    "#f15150", #130
    "#f5a05a", #135
    "#faf065", #140
    "#ecd756", #145
    "#dcbf46", #150
    "#dcbf46", #155
    "#be8c28", #160
    "#b07319", #165
    "#a15a0a", #170
]
# Normalize the anchor speeds to the [0, 1] interval expected by Matplotlib.
t_norm = (temps - temps[0]) / (temps[-1] - temps[0])
points = list(zip(t_norm, colors_hex))

colormap = mcolors.LinearSegmentedColormap.from_list("temp_map", points)

def demo_gradual():
    # Create a simple synthetic field so the colormap can be previewed by itself.
    x = np.linspace(50, 170, 200)
    y = np.linspace(50, 170, 200)
    xx, yy = np.meshgrid(x, y)
    data = xx

    cmap = colormap

    plt.figure(figsize=(6, 5))
    im = plt.imshow(data, cmap=cmap, origin='lower', extent=(50,170,50,170),
                    vmin=50, vmax=170)
    plt.title("Custom Gradual Colormap: 50 to 170")
    plt.xlabel("X axis (wind-speed range)")
    plt.ylabel("Y axis (wind-speed range)")

    cb = plt.colorbar(im, shrink=0.8, ticks=range(50, 170, 5))
    cb.set_label("Wind speed (m/s)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo_gradual()

