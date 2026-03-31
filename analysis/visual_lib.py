#!/usr/bin/env python

from .global_info import LST
from .data_API import data_access

import xarray as xr
import netCDF4 as nc
import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter, label
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime as dtm
import os
import multiprocessing as mtp
from pdb import set_trace

# matplotlib.use('Agg')


# Reuse one timezone offset everywhere so each figure shows both UTC and local time consistently.
dt_UTC2LST = dtm.timedelta(hours = LST)

states_provinces = cfeature.NaturalEarthFeature(
    category='cultural', 
    name='admin_1_states_provinces_lines', 
    scale='50m',
    facecolor='none'
)


def draw_vor(fig, ax, lon, lat, vor, P_level):
    from .cmaps.cmap_vorticity import colormap as cmap_vorticity
    levels_vor = np.arange(-40, 51, 1)
    cf = ax.contourf(lon, lat, vor,
                        levels=levels_vor,
                        cmap=cmap_vorticity,
                        transform=ccrs.PlateCarree()
                    )
    return fig, ax, cf

def draw_temperature(fig, ax, lon, lat, temperature, P_level):
    T0 = 273.15
    if P_level == 850:
        from .cmaps.cmap_temp850 import colormap
        levels_temperature = np.arange(-40, 41, 1)
    elif P_level == 500:
        from .cmaps.cmap_temp500 import colormap
        levels_temperature = np.arange(-50, 1, 1)
    elif P_level == 1000:
        from .cmaps.cmap_temp850 import colormap
        levels_temperature = np.arange(-40, 41, 1)
    else:
        assert False, "No height range registered at " + str(P_level) + " hPa level."


    cf = ax.contourf(lon, lat, temperature - T0,
                        levels=levels_temperature,
                        cmap=colormap,
                        transform=ccrs.PlateCarree()
                    )
    return fig, ax, cf

def draw_cape(fig,ax, lon, lat, cape):
    colors = matplotlib.colormaps.get_cmap("Reds")
    colormap = colors(np.linspace(0, 1, colors.N))
    colormap[:, -1] = np.linspace(0, 1, colors.N)
    colormap = ListedColormap(colormap)
    levels_cape = np.arange(0, 61, 2)
    cf = ax.contourf(lon, lat, cape,
                        levels=levels_cape,
                        cmap=colormap,
                        transform=ccrs.PlateCarree(),
                        extend = "max",
                    )
    return fig, ax, cf


def draw_specific_humidity(fig, ax, lon, lat, spechum, P_level, draw_contour = False):
    T0 = 273.15
    if P_level == 850:
        levels_spechum = np.arange(0, 13, 1)
    else:
        assert False, "No height range registered at " + str(P_level) + " hPa level."

    colormap = "YlGn"
    if draw_contour:
        cf = ax.contour(lon, lat, spechum,
                        levels=levels_spechum,
                        #cmap=colormap,
                        #color = "red",
                        linestyles = "dashed",
                        linewidths = 0.5,
                        transform=ccrs.PlateCarree()
                    )
        
        ax.clabel(cf, inline=True, fontsize=5, fmt='%d')
    else:

        cf = ax.contourf(lon, lat, spechum,
                            levels=levels_spechum,
                            cmap=colormap,
                            transform=ccrs.PlateCarree(),
                            extend = "max",
                        )
    return fig, ax, cf

def draw_vapor_flux(fig, ax, lon, lat, spechum, u, v, P_level):
    if P_level == 850:
        levels_spechum = np.arange(0, 150, 10)
    else:
        assert False, "No height range registered at " + str(P_level) + " hPa level."

    colormap = "YlGn"
    vapor_flux = np.sqrt(u**2 + v**2) * spechum
    cf = ax.contourf(lon, lat, vapor_flux,
                        levels=levels_spechum,
                        cmap=colormap,
                        transform=ccrs.PlateCarree(),
                        extend = "max",
                    )
    return fig, ax, cf


def calculate_dx_dy(lon, lat):
    # Approximate grid spacing on a spherical Earth for centered-difference diagnostics.
    Re = 6371000
    dlon = lon[1:] - lon[:-1]
    dlat = lat[1:] - lat[:-1]
    nLon = len(lon)
    nLat = len(lat)
    DX = np.matmul((Re * np.cos(np.deg2rad(lat))).reshape(nLat, 1), np.deg2rad(dlon).reshape(1, nLon - 1))
    dy = Re * np.deg2rad(dlat)
    return DX, dy

def draw_temperature_advection(fig, ax, lon, lat, temperature, u, v, P_level):
    levels_Ta = np.arange(-2, 2.1, 0.1)
    colormap = "coolwarm"
    DX, dy = calculate_dx_dy(lon, lat)
    # Convert meters to kilometers so the shaded values match the colorbar label.
    DX = DX / 1000
    dy = dy / 1000
    Ta = np.full( temperature.shape, np.nan )
    # Use centered differences in both directions and apply the standard -V dot grad(T) form.
    Ta[1:-1, 1:-1] = -u[1:-1, 1:-1] * (temperature[1:-1, 2:] - temperature[1:-1, :-2]) / (DX[1:-1, 1:] + DX[1:-1, :-1]) \
        -v[1:-1, 1:-1] * (temperature[2:, 1:-1] - temperature[:-2, 1:-1]) / (dy[1:] + dy[:-1])[:, np.newaxis]
    
    cf = ax.contourf(lon, lat, Ta,
                        levels=levels_Ta,
                        extend = "both",
                        cmap=colormap,
                        transform=ccrs.PlateCarree()
                    )
    return fig, ax, cf

def draw_vorticity_advection(fig, ax, lon, lat, vor, u, v, P_level):
    colormap = "coolwarm"
    DX, dy = calculate_dx_dy(lon, lat)
    # Convert meters to kilometers so the shaded values match the colorbar label.
    DX = DX / 1000
    dy = dy / 1000
    Va = np.full( vor.shape, np.nan )
    # Use centered differences in both directions and apply the standard -V dot grad(zeta) form.
    Va[1:-1, 1:-1] = -u[1:-1, 1:-1] * (vor[1:-1, 2:] - vor[1:-1, :-2]) / (DX[1:-1, 1:] + DX[1:-1, :-1]) \
        -v[1:-1, 1:-1] * (vor[2:, 1:-1] - vor[:-2, 1:-1]) / (dy[1:] + dy[:-1])[:, np.newaxis]
    vmin = -4
    vmax = 4
    levels_Va =  np.arange(vmin, vmax+0.1, 0.1)
    cf = ax.contourf(lon, lat, Va, vmin = vmin, vmax = vmax,
                        levels=levels_Va,
                        extend = "both",
                        cmap=colormap,
                        transform=ccrs.PlateCarree()
                    )
    return fig, ax, cf


def draw_height(fig, ax, lon, lat, Z, P_level):
    if P_level == 200:
        levels_z = range(1002, 1262, 4)
    elif P_level == 300:
        levels_z = range(844, 986, 4)
    elif P_level == 500:
        levels_z = range(500, 600, 4)
    elif P_level == 850:
        levels_z = range(107, 193, 4)
    else:
        assert False, "No height range registered at " + str(P_level) + " hPa level."
    cs = ax.contour(lon, lat, Z,
                        levels=levels_z,
                        colors='black',
                        linewidths=1,
                        transform=ccrs.PlateCarree()
                    )
    ax.clabel(cs, inline=True, fontsize=9, fmt='%d')
    return fig, ax, cs

def draw_precipitation(fig, ax, lon, lat, tp, ptype):
    levels_prec = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.125, 0.15, 0.175, 0.2, 0.3, 0.4, 0.5, 1])

    from .cmaps.cmap_precipitation import colormap_rain, colormap_snow, colormap_freezing_rain, colormap_mixture
    # ERA5 precipitation-type IDs are grouped here into the categories used by the surface plot.
    prec_type_dict = {
        "rain": [[1], colormap_rain],
        "snow": [[5, 6], colormap_snow],
        "freezing_rain": [[3], colormap_freezing_rain],
        "mixture": [[7], colormap_mixture],
    }
    cf_dict = {}
    # Rain
    for prec_type_name in prec_type_dict:
        region_selected = np.full( (ptype.shape), False)
        for type_id in prec_type_dict[prec_type_name][0]:
            region_selected = region_selected | (ptype == type_id)
        tp_draw = np.where(region_selected, tp, np.nan)
        if prec_type_name == "snow":
            # Apply a fixed snow-to-liquid ratio so snow shades are easier to compare with rain rates.
            tp_draw = tp_draw * 10 # ice-water ratio
        cf = ax.contourf(lon, lat, tp_draw,
                            levels=levels_prec,
                            cmap=prec_type_dict[prec_type_name][1],
                            extend = "max",
                            transform=ccrs.PlateCarree()
                        )
        cf_dict[prec_type_name] = cf
    return fig, ax, cf_dict

def draw_P_sfc(fig, ax, lon, lat, P_sfc):
    levels_P_sfc = range(940, 1080, 4)
    cs = ax.contour(lon, lat, P_sfc,
                        levels=levels_P_sfc,
                        colors='black',
                        linewidths=1,
                        transform=ccrs.PlateCarree()
                    )
    ax.clabel(cs, inline=True, fontsize=9, fmt='%d')
    return fig, ax, cs

def draw_thickness(fig, ax, lon, lat, thickness):
    levels_thickness = range(500, 1000, 4)
    # 540-541 dam is a common rain/snow rule-of-thumb, so split the line colors around that threshold.
    thickness_draw = np.where(thickness <= 541, thickness, np.nan)
    cs = ax.contour(lon, lat, thickness_draw,
                        levels=levels_thickness,
                        colors='blue',
                        linestyles = "dashed",
                        linewidths = 0.5,
                        transform=ccrs.PlateCarree()
                    )
    ax.clabel(cs, inline=True, fontsize=9, fmt='%d')

    thickness_draw = np.where(thickness > 541, thickness, np.nan)
    cs = ax.contour(lon, lat, thickness_draw,
                        levels=levels_thickness,
                        colors='red',
                        linestyles = "dashed",
                        linewidths = 0.5,
                        transform=ccrs.PlateCarree()
                    )
    ax.clabel(cs, inline=True, fontsize=9, fmt='%d')
    return fig, ax, cs




def draw_wind_speed(fig, ax, lon, lat, u, v, P_level):
    if P_level <= 500:
        from .cmaps.cmap_jetstream import colormap
    from .cmaps.cmap_jetstream import colormap
    levels_ws = np.arange(50, 171, 5)
    ws = np.sqrt(u**2 + v**2)
    cf = ax.contourf(lon, lat, ws, levels = levels_ws, cmap = colormap, transform = ccrs.PlateCarree())
    return fig, ax, cf

def draw_wind_barb(fig, ax, lon, lat, u, v, wind_skip = 30):
    ax.barbs(lon[::wind_skip], lat[::wind_skip], u[::wind_skip, ::wind_skip], v[::wind_skip, ::wind_skip],
        length=5,
        linewidth=0.5,
        transform=ccrs.PlateCarree()
    )
    return fig, ax

def draw_200(date_Ymd, i_time, region):

    P_level = 200
    data = data_access(i_time, P_level, date_Ymd, extend = region)
    time = data["time"]
    lon = data["lon"]
    lat = data["lat"]
    lev = data["lev"]
    vor = data["vor"]
    Z = data["z"]
    u = data["u"]
    v = data["v"]

    fig, ax = plt.subplots(figsize=(20, 12),
                           subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.8)
    ax.add_feature(states_provinces, edgecolor='black', linewidth=0.4)

    fig, ax, cf = draw_wind_speed(fig, ax, lon, lat, u, v, P_level)
    cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=50, shrink = 0.6)
    cbar.set_label("Wind speed (m/s)")
    fig, ax, cs = draw_height(fig, ax, lon, lat, Z, P_level) 
    fig, ax = draw_wind_barb(fig, ax, lon, lat, u, v, wind_skip = 20)
    title_string = str(P_level) + " hPa Height (Contours), Wind speed (Shaded) & Wind (Barbs)"
    title_string += "\n" + time.strftime("%Y-%m-%d %H:%M") + " (UTC); " + (time + dt_UTC2LST).strftime("%Y-%m-%d %H:%M") + " (CST)"
    ax.set_title(title_string, fontsize=14)
    return fig

def draw_300(date_Ymd, i_time, region):

    P_level = 300
    data = data_access(i_time, P_level, date_Ymd, extend = region)
    time = data["time"]
    lon = data["lon"]
    lat = data["lat"]
    lev = data["lev"]
    vor = data["vor"]
    Z = data["z"]
    u = data["u"]
    v = data["v"]

    fig, ax = plt.subplots(figsize=(20, 12),
                           subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.8)
    ax.add_feature(states_provinces, edgecolor='black', linewidth=0.4)


    fig, ax, cf = draw_wind_speed(fig, ax, lon, lat, u, v, P_level)
    cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=50, shrink = 0.6)
    cbar.set_label("Wind speed (m/s)")
    fig, ax, cs = draw_height(fig, ax, lon, lat, Z, P_level) 
    fig, ax = draw_wind_barb(fig, ax, lon, lat, u, v, wind_skip = 20)
    title_string = str(P_level) + " hPa Height (Contours), Wind speed (Shaded) & Wind (Barbs)"
    title_string += "\n" + time.strftime("%Y-%m-%d %H:%M") + " (UTC); " + (time + dt_UTC2LST).strftime("%Y-%m-%d %H:%M") + " (CST)"
    ax.set_title(title_string, fontsize=14)
    #plt.show()
    return fig






def draw_500(date_Ymd, i_time, region):

    P_level = 500
    data = data_access(i_time, P_level, date_Ymd, extend = region)
    time = data["time"]
    lon = data["lon"]
    lat = data["lat"]
    lev = data["lev"]
    vor = data["vor"]
    Z = data["z"]
    u = data["u"]
    v = data["v"]

    fig, ax = plt.subplots(figsize=(20, 12),
                           subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.8)
    ax.add_feature(states_provinces, edgecolor='black', linewidth=0.4)

    fig, ax, cf = draw_vor(fig, ax, lon, lat, vor, P_level) 
    cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=50, shrink = 0.6)
    cbar.set_label("Relative Vorticity (10$^{-5}$ s$^{-1}$)")
    fig, ax, cs = draw_height(fig, ax, lon, lat, Z, P_level) 
    title_string = str(P_level) + " hPa Height (Contours) & Vorticity (Shaded)"
    title_string += "\n" + time.strftime("%Y-%m-%d %H:%M") + " (UTC); " + (time + dt_UTC2LST).strftime("%Y-%m-%d %H:%M") + " (CST)"
    ax.set_title(title_string, fontsize=14)
    return fig

def draw_500Va(date_Ymd, i_time, region):

    P_level = 500
    data = data_access(i_time, P_level, date_Ymd, extend = region)
    time = data["time"]
    lon = data["lon"]
    lat = data["lat"]
    lev = data["lev"]
    vor = data["vor"]
    Z = data["z"]
    u = data["u"]
    v = data["v"]

    fig, ax = plt.subplots(figsize=(20, 12),
                           subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.8)
    ax.add_feature(states_provinces, edgecolor='black', linewidth=0.4)

    fig, ax, cf = draw_vorticity_advection(fig, ax, lon, lat, vor, u, v, P_level)
    cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=50, shrink = 0.6)
    cbar.set_label("Relative Vorticity advection (10$^{-5}$ s$^{-1}/km$)")
    fig, ax, cs = draw_height(fig, ax, lon, lat, Z, P_level) 
    title_string = str(P_level) + " hPa Height (Contours) & Vorticity advection (Shaded)"
    title_string += "\n" + time.strftime("%Y-%m-%d %H:%M") + " (UTC); " + (time + dt_UTC2LST).strftime("%Y-%m-%d %H:%M") + " (CST)"
    ax.set_title(title_string, fontsize=14)
    #plt.show()
    return fig

def draw_500T(date_Ymd, i_time, region):

    P_level = 500
    data = data_access(i_time, P_level, date_Ymd, extend = region)
    time = data["time"]
    lon = data["lon"]
    lat = data["lat"]
    lev = data["lev"]
    T = data["T"]
    Z = data["z"]
    u = data["u"]
    v = data["v"]

    fig, ax = plt.subplots(figsize=(20, 12),
                           subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.8)
    ax.add_feature(states_provinces, edgecolor='black', linewidth=0.4)

    fig, ax, cf = draw_temperature(fig, ax, lon, lat, T, P_level) 
    cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=50, shrink = 0.6)
    cbar.set_label("Temperature (˚C)")
    fig, ax, cs = draw_height(fig, ax, lon, lat, Z, P_level) 
    fig, ax = draw_wind_barb(fig, ax, lon, lat, u, v, wind_skip = 20)
    title_string = str(P_level) + " hPa Height (Contours), Temperature (Shaded) & Wind (Barbs)"
    title_string += "\n" + time.strftime("%Y-%m-%d %H:%M") + " (UTC); " + (time + dt_UTC2LST).strftime("%Y-%m-%d %H:%M") + " (CST)"
    ax.set_title(title_string, fontsize=14)
    return fig

def draw_850(date_Ymd, i_time, region):

    P_level = 850
    data = data_access(i_time, P_level, date_Ymd, extend = region)
    time = data["time"]
    lon = data["lon"]
    lat = data["lat"]
    lev = data["lev"]
    vor = data["vor"]
    Z = data["z"]
    u = data["u"]
    v = data["v"]
    T = data["T"]

    fig, ax = plt.subplots(figsize=(20, 12),
                           subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.8)
    ax.add_feature(states_provinces, edgecolor='black', linewidth=0.4)

    fig, ax, cf = draw_temperature(fig, ax, lon, lat, T, P_level) 
    cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=50, shrink = 0.6)
    cbar.set_label("Temperature (˚C)")
    fig, ax, cs = draw_height(fig, ax, lon, lat, Z, P_level) 
    fig, ax = draw_wind_barb(fig, ax, lon, lat, u, v, wind_skip = 10)
    title_string = str(P_level) + " hPa Height (Contours), Temperature (Shaded) & Wind (Barbs)"
    title_string += "\n" + time.strftime("%Y-%m-%d %H:%M") + " (UTC); " + (time + dt_UTC2LST).strftime("%Y-%m-%d %H:%M") + " (CST)"
    ax.set_title(title_string, fontsize=14)
    return fig

def draw_850_hum(date_Ymd, i_time, region):

    P_level = 850
    data = data_access(i_time, P_level, date_Ymd, extend = region)
    time = data["time"]
    lon = data["lon"]
    lat = data["lat"]
    lev = data["lev"]
    vor = data["vor"]
    Z = data["z"]
    u = data["u"]
    v = data["v"]
    T = data["T"]
    spechum = data["spechum"]

    fig, ax = plt.subplots(figsize=(20, 12),
                           subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.8)
    ax.add_feature(states_provinces, edgecolor='black', linewidth=0.4)

    fig, ax, cf = draw_specific_humidity(fig, ax, lon, lat, spechum, P_level)
    cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=50, shrink = 0.6)
    cbar.set_label("Specific humidity (g/kg)")
    fig, ax, cs = draw_height(fig, ax, lon, lat, Z, P_level) 
    fig, ax = draw_wind_barb(fig, ax, lon, lat, u, v, wind_skip = 10)
    title_string = str(P_level) + " hPa Height (Contours), Specific humidity (Shaded) & Wind (Barbs)"
    title_string += "\n" + time.strftime("%Y-%m-%d %H:%M") + " (UTC); " + (time + dt_UTC2LST).strftime("%Y-%m-%d %H:%M") + " (CST)"
    ax.set_title(title_string, fontsize=14)

    return fig


def draw_850_flux(date_Ymd, i_time, region):

    P_level = 850
    data = data_access(i_time, P_level, date_Ymd, extend = region)
    time = data["time"]
    lon = data["lon"]
    lat = data["lat"]
    lev = data["lev"]
    vor = data["vor"]
    Z = data["z"]
    u = data["u"]
    v = data["v"]
    T = data["T"]
    spechum = data["spechum"]

    fig, ax = plt.subplots(figsize=(20, 12),
                           subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.8)
    ax.add_feature(states_provinces, edgecolor='black', linewidth=0.4)

    fig, ax, cf = draw_vapor_flux(fig, ax, lon, lat, spechum, u, v, P_level)
    cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=50, shrink = 0.6)
    fig, ax, cf = draw_specific_humidity(fig, ax, lon, lat, spechum, P_level, draw_contour = True) 
    cbar.set_label("Vapor flux (g m kg$^{-1}$s$^{-1}$)")
    fig, ax, cs = draw_height(fig, ax, lon, lat, Z, P_level) 
    fig, ax = draw_wind_barb(fig, ax, lon, lat, u, v, wind_skip = 10)
    title_string = str(P_level) + " hPa Height (Contours), Specific humidity (Dashed contours), Vapor flux (Shaded) & Wind (Barbs)"
    title_string += "\n" + time.strftime("%Y-%m-%d %H:%M") + " (UTC); " + (time + dt_UTC2LST).strftime("%Y-%m-%d %H:%M") + " (CST)"
    ax.set_title(title_string, fontsize=14)

    return fig


def draw_850Ta(date_Ymd, i_time):

    P_level = 850
    # NOTE: this function still depends on a `region` variable that is not passed in explicitly.
    data = data_access(i_time, P_level, date_Ymd, extend = region)
    time = data["time"]
    lon = data["lon"]
    lat = data["lat"]
    lev = data["lev"]
    vor = data["vor"]
    Z = data["z"]
    u = data["u"]
    v = data["v"]
    T = data["T"]

    fig, ax = plt.subplots(figsize=(20, 12),
                           subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.8)
    ax.add_feature(states_provinces, edgecolor='black', linewidth=0.4)

    fig, ax, cf = draw_temperature_advection(fig, ax, lon, lat, T, u, v, P_level) 
    cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=50, shrink = 0.6)
    cbar.set_label("Temperature advection (˚C/km)")
    fig, ax, cs = draw_height(fig, ax, lon, lat, Z, P_level) 
    fig, ax = draw_wind_barb(fig, ax, lon, lat, u, v, wind_skip = 10)
    title_string = str(P_level) + " hPa Height (Contours), Temperature advection (Shaded) & Wind (Barbs)"
    title_string += "\n" + time.strftime("%Y-%m-%d %H:%M") + " (UTC); " + (time + dt_UTC2LST).strftime("%Y-%m-%d %H:%M") + " (CST)"
    ax.set_title(title_string, fontsize=14)
    return fig

def find_extrema(data, mode='min', size=50):
    # A moving-window extrema filter is used to place only the broadest H/L centers on the map.

    if mode == 'min':
        filtered = minimum_filter(data, size=size, mode='constant')
        extrema = (data == filtered)
    elif mode == 'max':
        filtered = maximum_filter(data, size=size, mode='constant')
        extrema = (data == filtered)
    else:
        raise ValueError("mode must be 'min' or 'max'")
    return extrema

def draw_surface(date_Ymd, i_time, region):

    P_level = 1000
    data = data_access(i_time, P_level, date_Ymd, extend = region, get_thickness = True)
    time = data["time"]
    lon = data["lon"]
    lat = data["lat"]
    lev = data["lev"]
    P_sfc = data["P_sfc"]
    tp = data["tp"]
    ptype = data["ptype"]
    u10 = data["u10"]
    v10 = data["v10"]
    thickness = data["thickness"]

    fig = plt.figure(figsize = (20, 12))
    #fig, ax = plt.subplots(figsize=(20, 12),
    #                       subplot_kw={'projection': ccrs.PlateCarree()})
    # Reserve the last row of the grid for one colorbar per precipitation type.
    gs = gridspec.GridSpec(50, 4, figure = fig)
    ax = fig.add_subplot(gs[0:49, :], projection = ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.8)
    ax.add_feature(states_provinces, edgecolor='black', linewidth=0.4)

    fig, ax, cs_thk = draw_thickness(fig, ax, lon, lat, thickness)
    fig, ax, cs_sfc = draw_P_sfc(fig, ax, lon, lat, P_sfc)
    fig, ax = draw_wind_barb(fig, ax, lon, lat, u10, v10, wind_skip = 10)
    fig, ax, cf_dict = draw_precipitation(fig, ax, lon, lat, tp, ptype)


    for ind, prec_type_name in enumerate(cf_dict):
        ax_cbar = fig.add_subplot(gs[-1, ind])
        cbar = fig.colorbar(cf_dict[prec_type_name], cax=ax_cbar, orientation='horizontal', pad=0.05, aspect=50, shrink = 0.6)
        cbar.set_ticks([0.05, 0.1, 0.2, 0.5])
        cbar.set_label(prec_type_name.replace("_", " ") + " (in/hr)")

    low_centers = find_extrema(P_sfc, mode = "min", size = 30)
    high_centers = find_extrema(P_sfc, mode = "max", size = 30)

    bnd_length = 5
    sign_word_dis = 1
    for i, j in zip(*np.where(low_centers)):
        if (i <= bnd_length) or (i >= len(lat) - bnd_length) or (j <= bnd_length) or (j >= len(lon) - bnd_length):
            continue
        ax.text(lon[j], lat[i], 'L', color='red', fontsize=16,
                fontweight='bold', ha='center', va='center')
        ax.text(lon[j], lat[i] - sign_word_dis, '%.0f' % (P_sfc[i, j]), color='red', fontsize=9,
                fontweight='bold', ha='center', va='top')

    for i, j in zip(*np.where(high_centers)):
        if (i <= bnd_length) or (i >= len(lat) - bnd_length) or (j <= bnd_length) or (j >= len(lon) - bnd_length):
            continue
        ax.text(lon[j], lat[i], 'H', color='blue', fontsize=16,
                fontweight='bold', ha='center', va='center')
        ax.text(lon[j], lat[i] - sign_word_dis, '%.0f' % (P_sfc[i, j]), color='blue', fontsize=9,
                fontweight='bold', ha='center', va='top')

    title_string = " Surface Pressure (Contours), Thickness (Dashed contours) & Precipitation rate (Shaded)"
    title_string += "\n" + time.strftime("%Y-%m-%d %H:%M") + " (UTC); " + (time + dt_UTC2LST).strftime("%Y-%m-%d %H:%M") + " (CST)"
    ax.set_title(title_string, fontsize=14)
    #plt.show()
    return fig

def draw_surfaceT(date_Ymd, i_time, region):

    P_level = 1000
    data = data_access(i_time, P_level, date_Ymd, extend = region, get_thickness = False)
    time = data["time"]
    lon = data["lon"]
    lat = data["lat"]
    lev = data["lev"]
    P_sfc = data["P_sfc"]
    tp = data["tp"]
    ptype = data["ptype"]
    u10 = data["u10"]
    v10 = data["v10"]
    t2m = data["t2m"]

    fig, ax = plt.subplots(figsize=(20, 12),
                           subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.8)
    ax.add_feature(states_provinces, edgecolor='black', linewidth=0.4)

    fig, ax, cf = draw_temperature(fig, ax, lon, lat, t2m, P_level) 
    cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=50, shrink = 0.6)
    cbar.set_label("Temperature (˚C)")
    fig, ax, cs_sfc = draw_P_sfc(fig, ax, lon, lat, P_sfc)
    fig, ax = draw_wind_barb(fig, ax, lon, lat, u10, v10, wind_skip = 10)


    title_string = " Surface Pressure (Contours), 2m temperature (Shaded) & Wind (Barbs)"
    title_string += "\n" + time.strftime("%Y-%m-%d %H:%M") + " (UTC); " + (time + dt_UTC2LST).strftime("%Y-%m-%d %H:%M") + " (CST)"
    ax.set_title(title_string, fontsize=14)
    #plt.show()
    return fig

def draw_lake_effect(date_Ymd, i_time, region):

    P_level = 1000
    data = data_access(i_time, P_level, date_Ymd, extend = region)
    time = data["time"]
    lon = data["lon"]
    lat = data["lat"]
    lev = data["lev"]
    P_sfc = data["P_sfc"]
    tp = data["tp"]
    ptype = data["ptype"]
    u10 = data["u10"]
    v10 = data["v10"]
    cape = data["cape"]

    fig = plt.figure(figsize = (20, 12))
    gs = gridspec.GridSpec(50, 4, figure = fig)
    ax = fig.add_subplot(gs[0:49, :], projection = ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.8)
    ax.add_feature(states_provinces, edgecolor='black', linewidth=0.4)
    lakes = cfeature.NaturalEarthFeature(
        category='physical',
        name='lakes',
        scale='110m',
        facecolor='none'
    )

    ax.add_feature(lakes, edgecolor='royalblue', linewidth=1)
    fig, ax, cf = draw_cape(fig, ax, lon, lat, cape)
    cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=50, shrink = 0.6)
    cbar.set_label("CAPE (J/kg)")
    fig, ax = draw_wind_barb(fig, ax, lon, lat, u10, v10, wind_skip = 3)


    title_string = "CAPE (Shaded) & 2m surface wind (Barbs)"
    title_string += "\n" + time.strftime("%Y-%m-%d %H:%M") + " (UTC); " + (time + dt_UTC2LST).strftime("%Y-%m-%d %H:%M") + " (CST)"
    ax.set_title(title_string, fontsize=14)
    #plt.show()
    return fig

visual_lib_dict = {
    "map200": draw_200,
    "map300": draw_300,
    "map500": draw_500,
    "map500T": draw_500T,
    "map500Va": draw_500Va,
    "map850": draw_850,
    "map850H": draw_850_hum,
    "map850F": draw_850_flux,
    "surface": draw_surface,
    "surfaceT": draw_surfaceT,
    "map850Ta": draw_850Ta,
    "lakeEffect": draw_lake_effect,
}

def draw_figure(args):
    varName, time, hour, region, kwargs, figure_name, dpi = args
    print(f"Making {figure_name:s} ...")
    # `kwargs` lets the batch driver pass optional keyword arguments through to any plot function.
    fig = visual_lib_dict[varName](time, hour, region=region, **kwargs)
    plt.savefig(figure_name, dpi=dpi)
    plt.close(fig)
    
def parallel_plot(StartTime_str, EndTime_str, dt_draw_int, region, varName, figDir, n_core, dpi, **kwargs):
    StartTime = dtm.datetime.strptime(StartTime_str, "%Y-%m-%d_%H:%M")
    EndTime = dtm.datetime.strptime(EndTime_str, "%Y-%m-%d_%H:%M")
    dt_draw = dtm.timedelta(hours = dt_draw_int)
    time_draw = StartTime
    os.makedirs(figDir, exist_ok=True)
    argsList = []
    while time_draw <= EndTime:       
        figure_name = figDir + "/" + varName + "_" + time_draw.strftime("%Y-%m-%d_%H:%M") + ".png"
        # Each tuple contains everything one worker needs to render one timestamp independently.
        argsList.append( (
            varName,
            time_draw.strftime("%Y%m%d"),
            time_draw.hour,
            region,
            kwargs,
            figure_name,
            dpi
        ) )

        time_draw += dt_draw
    pool = mtp.Pool(n_core)
    pool.map(draw_figure, argsList)
    pool.close()
    pool.join()

if __name__ == "__main__":
    draw_surfaceT("20221224", 12)
    #draw_850_hum("20221223", 6)
    #draw_500Va("20221221", 20)
    #fig = draw_lake_effect("20221224", 12, {"lon_s": -97.664, "lon_e": -64.310, "lat_s": 38.204, "lat_e": 51.591})
    plt.show()

