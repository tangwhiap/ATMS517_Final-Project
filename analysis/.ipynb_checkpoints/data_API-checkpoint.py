#!/usr/bin/env python

from .global_info import DataDir

import xarray as xr
import netCDF4 as nc
import numpy as np

def data_access(i_time, P_level, date_Ymd, extend = None, get_thickness = False):

    ncf = nc.Dataset(DataDir + "/ERA5_geopotential_" + date_Ymd + ".nc")
    time_int = int(ncf.variables["valid_time"][i_time].filled(np.nan))
    time_units = ncf.variables["valid_time"].units
    time_calendar = ncf.variables["valid_time"].calendar
    time = nc.num2date(time_int, units = time_units, calendar = time_calendar)
    time_string = time.strftime("%Y-%m-%d %H:%M")

    lon = ncf.variables["longitude"][:].filled(np.nan)
    lat = ncf.variables["latitude"][::-1].filled(np.nan)
    lev = ncf.variables["pressure_level"][:].filled(np.nan)
    i_lev = np.argmin(np.abs(P_level - lev))

    Z = ncf.variables["z"][i_time, i_lev, ::-1, :].filled(np.nan) / 98
    ncf.close()

    ncf = nc.Dataset(DataDir + "/ERA5_u_component_of_wind_" + date_Ymd + ".nc")
    u = ncf.variables["u"][i_time, i_lev, ::-1, :].filled(np.nan)
    ncf.close()

    ncf = nc.Dataset(DataDir + "/ERA5_v_component_of_wind_" + date_Ymd + ".nc")
    v = ncf.variables["v"][i_time, i_lev, ::-1, :].filled(np.nan)
    ncf.close()

    ncf = nc.Dataset(DataDir + "/ERA5_vorticity_" + date_Ymd + ".nc")
    vor = ncf.variables["vo"][i_time, i_lev, ::-1, :].filled(np.nan) * 1e5
    ncf.close()

    ncf = nc.Dataset(DataDir + "/ERA5_temperature_" + date_Ymd + ".nc")
    T = ncf.variables["t"][i_time, i_lev, ::-1, :].filled(np.nan)

    if get_thickness:
        i1000_lev = np.argmin(np.abs(lev-1000)) 
        i500_lev = np.argmin(np.abs(lev-500))
        T_1000_500 = ncf.variables["t"][i_time, i1000_lev : (i500_lev + 1), ::-1, :].filled(np.nan)
        P_1000_500 = lev[i1000_lev : (i500_lev + 1)]
        T_mean = (T_1000_500[1:, :, :] + T_1000_500[:-1, :, :]) / 2
        R = 287
        g = 9.89
        dZ = R * T_mean / g * np.log( P_1000_500[:-1] / P_1000_500[1:])[:, np.newaxis, np.newaxis]
        thickness = dZ.sum(axis = 0) / 10
    ncf.close()

    ncf = nc.Dataset(DataDir + "/ERA5_specific_humidity_" + date_Ymd + ".nc")
    spechum = ncf.variables["q"][i_time, i_lev, ::-1, :].filled(np.nan) * 1000
    ncf.close()

    ncf = nc.Dataset(DataDir + "/ERA5_mean_sea_level_pressure_" + date_Ymd + ".nc")
    P_sfc = ncf.variables["msl"][i_time, ::-1, :].filled(np.nan) / 100
    ncf.close()

    ncf = nc.Dataset(DataDir + "/ERA5_total_precipitation_" + date_Ymd + ".nc")
    tp = ncf.variables["tp"][i_time, ::-1, :].filled(np.nan) * 39.3701 # m/hr -> in/hr
    ncf.close()

    ncf = nc.Dataset(DataDir + "/ERA5_precipitation_type_" + date_Ymd + ".nc")
    ptype = ncf.variables["ptype"][i_time, ::-1, :].filled(np.nan).astype("int") 
    ncf.close()

    ncf = nc.Dataset(DataDir + "/ERA5_10m_u_component_of_wind_" + date_Ymd + ".nc")
    u10 = ncf.variables["u10"][i_time, ::-1, :].filled(np.nan)
    ncf.close()

    ncf = nc.Dataset(DataDir + "/ERA5_10m_v_component_of_wind_" + date_Ymd + ".nc")
    v10 = ncf.variables["v10"][i_time, ::-1, :].filled(np.nan)
    ncf.close()

    ncf = nc.Dataset(DataDir + "/ERA5_2m_temperature_" + date_Ymd + ".nc")
    t2m = ncf.variables["t2m"][i_time, ::-1, :].filled(np.nan)
    ncf.close()

    ncf = nc.Dataset(DataDir + "/ERA5_convective_available_potential_energy_" + date_Ymd + ".nc")
    cape = ncf.variables["cape"][i_time, ::-1, :].filled(np.nan)
    ncf.close()

    if not(extend is None):
        lon_s = extend["lon_s"]
        lon_e = extend["lon_e"]
        lat_s = extend["lat_s"]
        lat_e = extend["lat_e"]

        index_lon = np.where( (lon >= lon_s) & (lon <= lon_e) )[0]
        index_lat = np.where( (lat >= lat_s) & (lat <= lat_e) )[0]

        lon = lon[index_lon]
        lat = lat[index_lat]
        Z = Z[index_lat, :][:, index_lon]
        u = u[index_lat, :][:, index_lon]
        v = v[index_lat, :][:, index_lon]
        vor = vor[index_lat, :][:, index_lon]
        T = T[index_lat, :][:, index_lon]
        spechum = spechum[index_lat, :][:, index_lon]
        P_sfc = P_sfc[index_lat, :][:, index_lon]
        tp = tp[index_lat, :][:, index_lon]
        ptype = ptype[index_lat, :][:, index_lon]
        u10 = u10[index_lat, :][:, index_lon]
        v10 = v10[index_lat, :][:, index_lon]
        t2m = t2m[index_lat, :][:, index_lon]
        cape = cape[index_lat, :][:, index_lon]
        if get_thickness:
            thickness = thickness[index_lat, :][:, index_lon]

    data_accessed = {
        "time": time,
        "lon": lon,
        "lat": lat,
        "lev": lev,
        "z": Z,
        "u": u,
        "v": v,
        "vor": vor,
        "T": T,
        "spechum": spechum,
        "P_sfc": P_sfc,
        "tp": tp,
        "ptype": ptype,
        "u10": u10,
        "v10": v10,
        "t2m": t2m,
        "cape": cape,
    }
    if get_thickness:
        data_accessed["thickness"] = thickness

    return data_accessed