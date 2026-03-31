# ATMS517 Final Project: ERA5 Winter Storm Analysis 

![Surface weather animation](demo/demo_surface.gif)

This project visualizes the evolution of a winter storm over the contiguous United States using ERA5 reanalysis data. It includes data-access utilities, reusable map-drawing functions for multiple pressure levels, a notebook for manual testing, and a script that combines generated PNG frames into an animated GIF.

## What This Project Does

- Reads ERA5 daily NetCDF files for geopotential height, wind, temperature, vorticity, humidity, surface pressure, precipitation, and CAPE.
- Produces synoptic maps such as `200/300 hPa` jet-level wind and height, `500 hPa` vorticity and height, `850 hPa` temperature/moisture diagnostics, and surface weather analyses.
- Supports batch rendering across a time range with multiprocessing.
- Builds an animation from the rendered PNG files for presentation and quick inspection.

## Main Files

- `analysis/global_info.py`: global configuration such as the ERA5 data directory and local time offset.
- `analysis/data_API.py`: loads one time step from the ERA5 archive, performs unit conversions, and subsets data to the requested region.
- `analysis/visual_lib.py`: the core plotting library, including all map products and the `parallel_plot(...)` batch renderer.
- `analysis/cmaps/`: custom colormaps used by the meteorological figures.
- `visual_test.ipynb`: notebook for interactively testing the plotting functions.
- `make_animation.py`: creates PNG frames and/or converts the PNG sequence into `demo/demo_surface.gif`.

## Example Products

- `surface`: surface pressure, thickness, precipitation type, and 10 m wind barbs
- `surfaceT`: surface pressure, 2 m temperature, and 10 m wind barbs
- `map500`: 500 hPa height and relative vorticity
- `map500T`: 500 hPa height, temperature, and wind barbs
- `map500Va`: 500 hPa height and vorticity advection
- `map850`: 850 hPa height, temperature, and wind barbs
- `map850H`: 850 hPa height and specific humidity
- `map850F`: 850 hPa height, vapor flux, and humidity contours
- `lakeEffect`: CAPE and near-surface wind over the Great Lakes region

## Quick Start

Run the notebook for interactive testing:

```bash
jupyter notebook visual_test.ipynb
```

Generate figures and then build an animation:

```bash
python make_animation.py
```

## Output

- Intermediate PNG frames are written to a temporary figure directory such as `demo/.tmp_surface/`.
- The final animation is written to `demo/demo_surface.gif`.

## Notes

- The plotting code assumes the ERA5 files already exist under the path configured in `analysis/global_info.py`.
- Time labels are shown in both UTC and local standard time.
- The GIF generation step rescales frames in memory, so the source PNG files remain unchanged.