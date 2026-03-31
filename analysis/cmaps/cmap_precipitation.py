#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

"""
Build custom precipitation colormaps from hand-picked rate/color anchor pairs.
"""
temps = np.array([
    0.01, 0.05, 0.1, 0.19, 0.2, 0.5])

colors_hex = [
    "#bce6bd", # 0.01
    "#559056", # 0.05
    "#3b8044", # 0.1
    "#136a28", # 0.19
    "#f4f06f", # 0.2
    "#fc9440", # 0.5
]
t_norm = (temps - temps[0]) / (temps[-1] - temps[0])
points = list(zip(t_norm, colors_hex))

colormap_rain = mcolors.LinearSegmentedColormap.from_list("temp_map", points)

colors_hex = [
    "#bde1f3", # 0.01
    "#5aa1c7", # 0.05
    "#1b6689", # 0.1
    "#154b62", # 0.19
    "#b1248b", # 0.2
    "#f7d1e0", # 0.5
]

t_norm = (temps - temps[0]) / (temps[-1] - temps[0])
points = list(zip(t_norm, colors_hex))

colormap_snow = mcolors.LinearSegmentedColormap.from_list("temp_map", points)

colors_hex = [
    "#ecccdb", # 0.01
    "#f38381", # 0.05
    "#e84822", # 0.1
    "#d13e2a", # 0.19
    "#a63f3c", # 0.2
    "#783043", # 0.5
]

t_norm = (temps - temps[0]) / (temps[-1] - temps[0])
points = list(zip(t_norm, colors_hex))

colormap_freezing_rain = mcolors.LinearSegmentedColormap.from_list("temp_map", points)

colors_hex = [
    "#dec7ee", # 0.01
    "#c283d7", # 0.05
    "#9b34b4", # 0.1
    "#79268a", # 0.19
    "#6f2478", # 0.2
    "#491b2c", # 0.5
]

t_norm = (temps - temps[0]) / (temps[-1] - temps[0])
points = list(zip(t_norm, colors_hex))

colormap_mixture = mcolors.LinearSegmentedColormap.from_list("temp_map", points)

temps = np.array([
    0.01, 0.05, 0.1, 0.19, 0.2, 0.5]) * 50

colors_hex = [
    "#bce6bd", # 0.01
    "#559056", # 0.05
    "#3b8044", # 0.1
    "#136a28", # 0.19
    "#f4f06f", # 0.2
    "#fc9440", # 0.5
]
t_norm = (temps - temps[0]) / (temps[-1] - temps[0])
points = list(zip(t_norm, colors_hex))

colormap_tp = mcolors.LinearSegmentedColormap.from_list("temp_map", points)

colors_hex = [
    "#bde1f3", # 0.01
    "#5aa1c7", # 0.05
    "#1b6689", # 0.1
    "#154b62", # 0.19
    "#b1248b", # 0.2
    "#f7d1e0", # 0.5
]

t_norm = (temps - temps[0]) / (temps[-1] - temps[0])  # 即 (T - (-40)) / (40 - (-40)) = (T+40)/80
points = list(zip(t_norm, colors_hex))

colormap_tsd = mcolors.LinearSegmentedColormap.from_list("temp_map", points)

