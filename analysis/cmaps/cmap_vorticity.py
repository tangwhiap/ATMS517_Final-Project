#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

"""
Build a custom relative-vorticity colormap from discrete value/color anchors.
"""
# Relative-vorticity anchor values.
temps = np.array([
    -40, -35, -30, -25, -20, -15, -10, -5,
    0,   5,  10,  15,  20,  25,  30,  35,
    40,  45,  50,  55
])

# Matching HEX colors for each anchor value above.
colors_hex = [
   '#343434', '#414141', '#515151', '#616161', '#717171',
   '#818181', '#919191', '#B1B1B1', '#FFFFFF', '#FFE600',
   '#FFB600', '#FF7A00', '#FF1E00', '#FF0050', '#F700FF',
   '#B700FF', '#6000FF', '#0000FF', '#0080FF', '#00FFFF'
]

# Normalize the anchor values to the [0, 1] interval expected by Matplotlib.
t_normalized = (temps + 40) / 95.0  # 55 - (-40) = 95

# Pack the anchor/value pairs for LinearSegmentedColormap.from_list.
points = list(zip(t_normalized, colors_hex))

# Create the actual segmented colormap.
colormap = mcolors.LinearSegmentedColormap.from_list("temp_map", points)

def main():
    # Create a small synthetic field to preview the colormap by itself.
    mycmap = colormap

    x = np.linspace(-40, 55, 200)
    y = np.linspace(-40, 55, 200)
    xx, yy = np.meshgrid(x, y)
    data = xx

    plt.figure(figsize=(10, 5))

    im = plt.imshow(data, cmap=mycmap, vmin=-40, vmax=55, origin='lower',
                    extent=[-40, 55, -40, 55])
    plt.title("Custom Temp Colormap (from -40 to 55)")
    plt.xlabel("X axis (temp range)")
    plt.ylabel("Y axis (temp range)")

    cb = plt.colorbar(im, shrink=0.85)
    cb.set_label("Temperature (°C)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

