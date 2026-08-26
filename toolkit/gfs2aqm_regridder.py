
import numpy as np
import xarray as xr
import xesmf as xe

class gfs2aqm_regridder():
    def __init__(self, GFS_COORDS):
        # original GFS coords
        self.gfs_coords = {
            "lat": GFS_COORDS[0],
            "lon": GFS_COORDS[1]
        }

        # target AQM coords
        ds = xr.open_dataset("./fix/grid_spec.nc")
        self.aqm_coords = {
            "grid_lat": ds["grid_latt"].values,
            "grid_lon": ds["grid_lont"].values
        }
        ds.close()
    
        # create regridder
        ds_src = xr.Dataset({
            "lat": (["lat"], self.gfs_coords["lat"]),
            "lon": (["lon"], self.gfs_coords["lon"]),
        })
        ds_tgt = xr.Dataset({
            "lat": (["y", "x"], self.aqm_coords["grid_lat"]),
            "lon": (["y", "x"], self.aqm_coords["grid_lon"]),
        })
        self.regridder = xe.Regridder(
            ds_src,
            ds_tgt,
            method="bilinear",
            filename="./fix/regrid_gfs_bilinear_1536x3072_544x800.nc",
            reuse_weights=True
        )

    
    def wind_rotation(self, U, V):
        lat_rad = np.deg2rad(self.aqm_coords["grid_lat"])
        lon_rad = np.deg2rad(self.aqm_coords["grid_lon"])

        # --------------------------------------------------------
        # Estimate derivative in target-grid X direction
        # using centered differences
        # --------------------------------------------------------
    
        dlat = np.empty_like(lat_rad)
        dlon = np.empty_like(lon_rad)
    
        # Interior points
        dlat[:, 1:-1] = lat_rad[:, 2:] - lat_rad[:, :-2]
        dlon[:, 1:-1] = lon_rad[:, 2:] - lon_rad[:, :-2]
    
        # Left boundary
        dlat[:, 0] = lat_rad[:, 1] - lat_rad[:, 0]
        dlon[:, 0] = lon_rad[:, 1] - lon_rad[:, 0]
    
        # Right boundary
        dlat[:, -1] = lat_rad[:, -1] - lat_rad[:, -2]
        dlon[:, -1] = lon_rad[:, -1] - lon_rad[:, -2]

        # --------------------------------------------------------
        # Correct longitude differences across +/-180 degrees
        # --------------------------------------------------------
    
        dlon = np.arctan2(
            np.sin(dlon),
            np.cos(dlon)
        )
    
        # --------------------------------------------------------
        # Local geographic displacement:
        #
        # east  ~ dlon*cos(lat)
        # north ~ dlat
        # --------------------------------------------------------
    
        east = dlon * np.cos(lat_rad)
        north = dlat
    
        alpha = np.arctan2(
            north,
            east
        )

        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        
        #print(
        #    "Target-grid rotation angle range:",
        #    np.rad2deg(np.nanmin(alpha)),
        #    "to",
        #    np.rad2deg(np.nanmax(alpha)),
        #    "degrees"
        #)

        U_rot = cos_alpha * U + sin_alpha * V
        V_rot = -1 * sin_alpha * U + cos_alpha * V
        return U_rot, V_rot


    def main_driver(self, DATA):     # gfs-to-aqm regridder + wind rotation
        da_src = xr.DataArray(
            DATA,
            dims=("lat", "lon"),
            coords={"lat": self.gfs_coords["lat"], "lon": self.gfs_coords["lon"]}
            )
        data_regrid = self.regridder(da_src).values
        return data_regrid
