# Generate grid index for quick plume emission mapping
# Required when changing emission versions (e.g. AQMv7 -> AQMv8)
# This script is not included in the forecast workflow and should be called before running the workflow

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree


# ========================
# 1. Load plume stack data
# ========================
print("\n---- Loading PT.nc stack data...")

pt_sample = "/scratch3/NCEPDEV/stmp/Youhua.Tang/naqfc-data/aqm.20260823/12/aqm.t12z.PT.nc"
ds = xr.open_dataset(pt_sample)
plat = ds["LATITUDE"].values   
plon = ds["LONGITUDE"].values + 360  
pN = len(plat)
ds.close()

print(f"Loaded {pN} points.")


# ===========================
# 2. Load AQM_NA_13km grid
# ===========================
print("\n---- Loading grid edge coordinates...")

ds = xr.open_dataset("./fix/grid_spec.nc")
lat_edges = ds["grid_lat"].values
lon_edges = ds["grid_lon"].values
ds.close()

ny = lat_edges.shape[0] - 1   # 544
nx = lat_edges.shape[1] - 1   # 800

print(f"Grid size: {ny} x {nx}")


# ==========================================
# 3. Compute grid cell centroid coordinates
# ==========================================
print("\n---- Computing grid centroids...")

lat_c = 0.25 * (
    lat_edges[:-1, :-1] +
    lat_edges[1:, :-1] +
    lat_edges[:-1, 1:] +
    lat_edges[1:, 1:]
)

lon_c = 0.25 * (
    lon_edges[:-1, :-1] +
    lon_edges[1:, :-1] +
    lon_edges[:-1, 1:] +
    lon_edges[1:, 1:]
)

# flatten to (Ngrid, 2)
centers = np.column_stack([lat_c.ravel(), lon_c.ravel()])

print(f"Centroids shape: {centers.shape}")


# ===========================
# 4. Build KD-tree
# ===========================
print("\n---- Building KD-tree...")

tree = cKDTree(centers)


# ===================================================
# 5. Query (row, col) for each plume stack location
# ===================================================
print("\n---- Querying nearest grid cell...")

# input points shape = (N, 2)
pts = np.column_stack([plat, plon])

dist, idx_nearest = tree.query(pts, k=1)

# convert flat index -> (row, col)
rows = idx_nearest // nx
cols = idx_nearest % nx

pRC = np.vstack([rows, cols])   # shape = (2, N)

print("Done. Example row/col:", pRC[:, :100])


# ======================
# 6. Save output
# ======================
f_out = "./pt2aqm_grid_idx.nc"

ds = xr.Dataset()
ds["grid_idx"] = xr.DataArray(pRC, dims=["axis", "point"])
ds.to_netcdf(f_out, format="NETCDF4_CLASSIC")
print(f"Saved to {f_out}")

