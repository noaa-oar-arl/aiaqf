import glob
import numpy as np
import pandas as pd
import xarray as xr
from gfs2aqm_regridder import gfs2aqm_regridder

def gfs_netcdf_loader(filename, varname):
    ds = xr.open_dataset(filename)

    if varname in list(ds.coords):
        return ds.coords[varname].data
    else:
        return np.squeeze(ds[varname].data)


def gfs_grib2_loader(filename, varname, typeoflevel):
    ds = xr.open_dataset(
        filename,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": typeoflevel}}
    )

    if varname in list(ds.coords):
        return ds.coords[varname].data
    else:
        return ds[varname].data


def gfs_met_loader(filename, format_option="netcdf"):
    varlist = ["ugrd10m", "vgrd10m", "hpbl", "spfh2m", "dswrf_ave", "tmp2m", "veg", "tcdc_aveclm", "tprcp", "lhtfl_ave", "fricv", "pressfc", "cnwat", "snowc_ave"]
    offset = [     0,        0,      0,       0,        0,     200,    0,            0,     0,          0,     0,      5e+4,     0,   0]
    scaler = [    10,       10,   1e+3,    1e-2,     1e+3,     100,   100,         100,     1,       1e+3,     1,      1e+5,  1e-3,   100]

    if format_option == "netcdf":
        ds = xr.open_dataset(filename)
        lat = ds["grid_yt"].values
        lon = ds["grid_xt"].values
        regridder = gfs2aqm_regridder([lat, lon])
    
        daily_data = []
        for i in range(len(varlist)):
            data = ds[varlist[i]].values[0, :, :]
            data = regridder.main_driver(data)
            daily_data += [data]
        ds.close()

    # wind rotation
    # rotate earth-relative geographic GFS U/V components into target grid X/Y
    u = daily_data[varlist.index("ugrd10m")]
    v = daily_data[varlist.index("vgrd10m")]
    u_rot, v_rot = regridder.wind_rotation(u, v)
    daily_data[varlist.index("ugrd10m")] = u_rot
    daily_data[varlist.index("vgrd10m")] = v_rot

    # scaling
    for i in range(len(varlist)):
        daily_data[i] = (daily_data[i] - offset[i]) / scaler[i]
    daily_data = np.stack(daily_data, 0)
    return daily_data
