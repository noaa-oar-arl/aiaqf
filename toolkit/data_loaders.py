
import glob
import numpy as np
import pandas as pd
import xarray as xr

def aqm_loader(filename, varname):
    ds = xr.open_dataset(filename)
    return ds[varname].data


def aqm_bndy_loader(filename, varname, layer_idx, halo_idx):
    ds = xr.open_dataset(filename)
    halo = len(ds["halo"].data)
    nx = len(ds["lon"].data) - (halo * 2)
    ny = len(ds["lat"].data)

    if varname == "pm25_tot":
        varlist = ["aothrj", "asoil", "aecj", "aorgcj"]
        top = np.zeros(nx)
        bottom = np.zeros(nx)
        left = np.zeros(ny)
        right = np.zeros(ny)
        for var in varlist:
            top = top + ds[var + "_top"].data[layer_idx, halo_idx, halo:-1*halo]
            bottom = bottom + ds[var + "_bottom"].data[layer_idx, halo_idx, halo:-1*halo]
            left = left + ds[var + "_left"].data[layer_idx, :, halo_idx]
            right = right + ds[var + "_right"].data[layer_idx, :, halo_idx]
    else:
        top = ds[varname + "_top"].data[layer_idx, halo_idx, halo:-1*halo]
        bottom = ds[varname + "_bottom"].data[layer_idx, halo_idx, halo:-1*halo]
        left = ds[varname + "_left"].data[layer_idx, :, halo_idx]
        right = ds[varname + "_right"].data[layer_idx, :, halo_idx]
    return top, bottom, left, right


def nexus_emi_loader(filename, varname):
    ds = xr.open_dataset(filename)

    if varname == "NOx":
        data = ds["NO"].data + ds["NO2"].data

    elif (varname == "SO2") or (varname == "NH3"):
        data = ds[varname].data

    elif varname == "VOC":
        species = ["AACD", "ACET", "ACROLEIN", "ALD2", "ALDX", "APIN", "BENZ", "BUTADIENE13", "ETH", "ETHA", "ETHY", "ETOH", "FACD", "FORM", "IOLE", "ISOP", "KET", "MEOH", "NAPH", "OLE", "PAR", "PRPA", "SESQ", "TERP", "TOL", "XYLMN"]
        ds = ds[species]
        data = ds.to_array().sum(dim="variable").data

    elif varname == "PM25":
        species = ["PAL", "PCA", "PCL", "PEC", "PFE", "PK", "PMG", "PMN", "PMOTHR", "PNA", "PNCOM", "PNH4", "PNO3", "POC", "PSI", "PSO4", "PTI"]
        ds = ds[species]
        data = ds.to_array().sum(dim="variable").data
    return data[0, :]


def plume_emi_loader(filename, varname, time_length, aqm_delz_files):
    ## AQM dimension
    grid_lat = aqm_loader("./fix/grid_spec.nc", "grid_latt")
    ny = grid_lat.shape[0]
    nx = grid_lat.shape[1]

    ## plume location index
    plume_stack = aqm_loader("./fix/pt2aqm_grid_idx.nc", "grid_idx")  # axis x point
    emi_row = plume_stack[0, :]
    emi_col = plume_stack[1, :]

    ## plume emission
    ds = xr.open_dataset(filename)
    pheight = ds["STKHT"].data
    nt = len(ds["TIME"].data)

    ## AQM layer heights
    #delz = []
    #for delz_file in aqm_delz_files:
    #    delz += [aqm_loader(delz_file, "delz")]
    #delz = np.concatenate(delz, 0)
    #delz = delz[:, ::-1]
    #ght = np.cumsum(abs(delz),1)

    #nt = delz.shape[0]
    #nl = delz.shape[1]
    #ny = delz.shape[2]
    #nx = delz.shape[3]
    #emi_layer = np.argmin(abs(pheight / ght[:, :, emi_row, emi_col] - 1), axis=1)
    #emi_layer_index = emi_layer * (ny * nx) + (emi_row * nx) + emi_col  # (time, n_stack)

    #def accumulate_fast(nt, nl, nx, ny, varlist):
    #    data_flat = np.zeros((nt, nl*ny*nx))

    #    for v in varlist:
    #        for t in range(nt):
    #            data_flat[t, :] += np.bincount(
    #                emi_layer_index[t],
    #                weights=ds[v].data[t, :],
    #                minlength=nl*ny*nx
    #            )

    #    data = data_flat.reshape(nt, nl, ny, nx)
    #    return data

    def point_alloc(nt, nx, ny, varlist):
        data = np.zeros((nt, ny, nx))
        for v in varlist:
            vals = ds[v].data
            for t in range(nt):
                np.add.at(data[t], (emi_row, emi_col), vals[t])
        return data

    ## get plume emissions
    if varname == "NOx":
        species = ["NO", "NO2"]
    elif (varname == "SO2") or (varname == "NH3"):
        species = [varname]
    elif varname == "VOC":
        species = ["ACET", "ALD2", "ALDX", "BENZ", "ETH", "ETHA", "ETHY", "ETOH", "FORM", "IOLE", "ISOP", "KET", "MEOH", "NAPH", "OLE", "PAR", "PRPA", "TERP", "TOL", "XYLMN"]
    elif varname == "PM25":
        species = ["PAL", "PCA", "PCL", "PEC", "PFE", "PK", "PMG", "PMN", "PMOTHR", "PNA", "PNCOM", "PNH4", "PNO3", "POC", "PSI", "PSO4", "PTI"]

    ## spatial + plume rise allocation 
    #data = accumulate_fast(nt, nl, nx, ny, species)
    #data = data.sum(1)

    ## spatial allocation only
    data = point_alloc(nt, nx, ny, species)

    ## use daily emission for now, remove following two lines if switching to hourly emission
    data = data.sum(axis=0, keepdims=True)
    data = np.repeat(data, nt, axis=0)
    return data[1:1+time_length, :]


def fire_emi_loader(filename, varname, time_length):
    emission_factor = {
        "NOx": 0.05,
        "SO2": 0.08,
        "NH3": 0.015,
        "VOC": 0.0425
    }

    ds = xr.open_dataset(filename)
    if varname == "PM25":
        data = ds["PM2.5"].data
    else:
        data = ds["CO"].data * emission_factor[varname]
    return data[1:1+time_length, :]
