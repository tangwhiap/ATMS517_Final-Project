#!/usr/bin/env python
"""
make_data.py – Build the feature matrix and target for the snow-dominance
logistic-regression classifier.

Reads hourly ERA5 reanalysis from 2022-12-22 through 2022-12-26 using
``analysis.data_API.data_access``.  Each (date, hour) pair is processed in
parallel, two pressure levels (850 and 500 hPa) are queried per time step,
and the results are merged into a single DataFrame saved as a compressed CSV.

ERA5 precipitation-type codes:
    0 = no precipitation, 1 = rain, 3 = freezing rain,
    5 = snow (wet), 6 = snow (dry), 7 = mixed / ice pellets
"""

import os
import sys
import time as _time
import numpy as np
import pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis.data_API import data_access

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats_data")
OUT_FILE = os.path.join(OUT_DIR, "logistic_regression_data.csv.gz")

DATES = ["20221222", "20221223", "20221224", "20221225", "20221226"]
HOURS = list(range(24))
N_WORKERS = 8

REGION_US = {
    "lon_s": -129,
    "lon_e": -64,
    "lat_s": 20,
    "lat_e": 55,
}


def _process_one_timestep(args):
    """Worker: read one (date, hour) and return a DataFrame of precipitating
    grid cells with features from 850 hPa, 500 hPa, and the surface."""

    date_Ymd, i_time, region = args

    d850 = data_access(i_time, 850, date_Ymd, extend=region, get_thickness=False)
    d500 = data_access(i_time, 500, date_Ymd, extend=region, get_thickness=True)

    lon = d850["lon"]
    lat = d850["lat"]
    tp = d850["tp"]            # in/hr (converted inside data_access)
    ptype = d850["ptype"]

    precip_mask = tp > 0
    if not precip_mask.any():
        return None

    lon2d, lat2d = np.meshgrid(lon, lat)
    idx = precip_mask.ravel()
    n = int(idx.sum())
    time_str = str(d850["time"])

    ws10 = np.sqrt(d850["u10"]**2 + d850["v10"]**2)
    ws850 = np.sqrt(d850["u"]**2 + d850["v"]**2)
    dewpoint_depression = d850["t2m"] - d850["d2m"]
    vapor_flux_850 = ws850 * d850["spechum"]

    rec = {
        "time":                np.full(n, time_str),
        "lon":                 lon2d.ravel()[idx],
        "lat":                 lat2d.ravel()[idx],
        # ── target-related ──
        "tp":                  tp.ravel()[idx],
        "ptype":               ptype.ravel()[idx],
        # ── surface features ──
        "t2m":                 d850["t2m"].ravel()[idx],
        "d2m":                 d850["d2m"].ravel()[idx],
        "dewpoint_depression": dewpoint_depression.ravel()[idx],
        "mslp":                d850["P_sfc"].ravel()[idx],
        "u10":                 d850["u10"].ravel()[idx],
        "v10":                 d850["v10"].ravel()[idx],
        "wind_speed_10m":      ws10.ravel()[idx],
        "cape":                d850["cape"].ravel()[idx],
        "cin":                 d850["cin"].ravel()[idx],
        # ── 850 hPa features ──
        "T_850":               d850["T"].ravel()[idx],
        "q_850":               d850["spechum"].ravel()[idx],
        "rh_850":              d850["rh"].ravel()[idx],
        "u_850":               d850["u"].ravel()[idx],
        "v_850":               d850["v"].ravel()[idx],
        "wind_speed_850":      ws850.ravel()[idx],
        "vor_850":             d850["vor"].ravel()[idx],
        "w_850":               d850["w"].ravel()[idx],
        "vapor_flux_850":      vapor_flux_850.ravel()[idx],
        # ── 500 hPa features ──
        "Z_500":               d500["z"].ravel()[idx],
        "T_500":               d500["T"].ravel()[idx],
        "vor_500":             d500["vor"].ravel()[idx],
        "w_500":               d500["w"].ravel()[idx],
        # ── cross-level thermodynamic ──
        "thickness_1000_500":  d500["thickness"].ravel()[idx],
    }

    return pd.DataFrame(rec)


def _progress_bar(done, total, width=40, elapsed=0.0):
    frac = done / total
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    eta_str = ""
    if done > 0:
        eta = elapsed / done * (total - done)
        eta_str = f"  ETA {eta:5.0f}s"
    sys.stderr.write(f"\r  [{bar}] {done:>3d}/{total}  ({frac:6.1%}){eta_str}  ")
    sys.stderr.flush()
    if done == total:
        sys.stderr.write("\n")


if __name__ == "__main__":
    tasks = [(date, hour, REGION_US) for date in DATES for hour in HOURS]
    total = len(tasks)
    print(f"Processing {total} time steps with {N_WORKERS} parallel workers …\n")

    t0 = _time.time()
    dfs = []
    done = 0
    with Pool(N_WORKERS) as pool:
        for result in pool.imap_unordered(_process_one_timestep, tasks):
            done += 1
            if result is not None:
                dfs.append(result)
            _progress_bar(done, total, elapsed=_time.time() - t0)

    print(f"\n  Parallel read finished in {_time.time() - t0:.1f}s")
    print(f"  Concatenating {len(dfs)} DataFrames …")
    df = pd.concat(dfs, ignore_index=True)

    df["is_snow"] = df["ptype"].isin([5, 6]).astype(np.int8)

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"  Writing {OUT_FILE} …")
    df.to_csv(OUT_FILE, index=False, compression="gzip")
    #df.to_csv(OUT_FILE, index=False)

