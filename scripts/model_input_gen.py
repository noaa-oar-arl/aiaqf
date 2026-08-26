
import os, sys, glob
import numpy as np
import xarray as xr
import pandas as pd

sys.path.insert(0, "./toolkit")
from aqm_loaders import aqm_loader, aqm_met_loader, aqm_bndy_loader, nexus_emi_loader, plume_emi_loader, fire_emi_loader
from gfs_loaders import gfs_met_loader

class input_generator():
    def namelist_config(self):
        namelist = pd.read_csv("./namelist", header=None, delimiter="=")
        namelist = namelist.rename(columns={0: "key", 1: "value"}).set_index("key")
        namelist = namelist.to_dict()["value"]

        AQM_DATE = namelist["AQM_DATE"]
        AQM_CYCLE = namelist["AQM_CYCLE"]
        MET_OPTION = namelist["MET_OPTION"]
        MET_PATH = namelist["MET_PATH"]
        AQM_PATH = namelist["AQM_PATH"]
        EMI_PATH = namelist["EMI_PATH"]
        OUTPUT_PATH = namelist["OUTPUT_PATH"]
        OUTPUT_PATH = f"{OUTPUT_PATH}/{AQM_DATE}/{AQM_CYCLE}"
       
        FCST_LENGTH = int(namelist["FCST_LENGTH"])
        BNDY_LENGTH = int(namelist["BNDY_LENGTH"])
        FCST_LAYER = int(namelist["FCST_LAYER"])  # for AQM, FCST_LAYER=64 for surface
        BNDY_OPTION = namelist["BNDY_OPTION"]  # retro uses AQM dyn files, fcst uses LBC files

        RESTART = namelist["RESTART"] in ["True", "true"]

        FCST_SPECIES = ["NO2", "NH3", "HCHO", "OZONE", "PM25"]
        FCST_OPTION = [
            namelist["NO2"] in ["True", "true"],
            namelist["NH3"] in ["True", "true"],
            namelist["HCHO"] in ["True", "true"],
            namelist["OZONE"] in ["True", "true"],
            namelist["PM25"] in ["True", "true"]
        ]
        FCST_SPECIES = [species for species, opt in zip(FCST_SPECIES, FCST_OPTION) if opt]
        return AQM_DATE, AQM_CYCLE, MET_OPTION, MET_PATH, AQM_PATH, EMI_PATH, OUTPUT_PATH, FCST_LENGTH, BNDY_LENGTH, FCST_SPECIES, FCST_LAYER, BNDY_OPTION, RESTART
    
    
    def read_static_fields(self, AQM_PATH):
        varlist = ["grid_latt", "grid_lont", "area", "zsurf", "land"]
        offset = [    0,    180,     0,    0,     0]
        scaler = [  100,    100,  1e+8, 1000,     1]
    
        print('Read grid specs from:')
        print("./fix/grid_spec.nc")
        print("./fix/atmos_static.nc")

        total_data = []
        for i in range(len(varlist)):
            if varlist[i] in ["grid_latt", "grid_lont", "area"]:
                data = aqm_loader("./fix/grid_spec.nc", varlist[i])
            elif varlist[i] in ["zsurf", "land"]:
                data = np.squeeze(aqm_loader("./fix/atmos_static.nc", varlist[i]))
            data = (data - offset[i]) / scaler[i]
            total_data += [data]
            del data
        total_data = np.stack(total_data, 0)
        total_data = np.expand_dims(total_data, axis=0)
        print('static', total_data.shape, pd.Timestamp("now"))
        return total_data
    
    
    def read_met_fields(self, MET_OPTION, MET_PATH, AQM_DATE, AQM_CYCLE, FCST_TIME):
        #varlist = ["ugrd10m", "vgrd10m", "hpbl", "spfh2m", "dswrf_ave", "tmp2m", "veg", "tcdc_aveclm", "tprcp", "lhtfl_ave", "fricv", "pressfc", "cnwat", "snowc_ave"]
        #offset = [     0,        0,      0,       0,        0,     200,    0,            0,     0,          0,     0,      5e+4,     0,   0]
        #scaler = [    10,       10,   1e+3,    1e-2,     1e+3,     100,   100,         100,     1,       1e+3,     1,      1e+5,  1e-3,   100]
        
        total_data = []
        if MET_OPTION in ["AQM", "aqm"]:
            for time in FCST_TIME:
                aqm_timestep = pd.to_datetime(time, format="%Y%m%d%H") - pd.Timedelta(hours=int(AQM_CYCLE))
                if aqm_timestep.strftime("%H%M") == "0000":
                    aqm_date = (aqm_timestep - pd.Timedelta(days=1)).strftime("%Y%m%d")
                    aqm_hour = "24"
                else:
                    aqm_date = aqm_timestep.strftime("%Y%m%d")
                    aqm_hour = aqm_timestep.strftime("%H")

                phy_file = f"{AQM_PATH}/{aqm_date}{AQM_CYCLE}/phyf0{aqm_hour}.nc"  
                print(f"Read met from {phy_file}")
                daily_data = aqm_met_loader(phy_file)
                total_data += [daily_data]

        elif MET_OPTION in ["GFS", "gfs"]:
            # use forecast from previous cycle
            aqm_timestep = pd.to_datetime(AQM_DATE, format="%Y%m%d")
            cycle_dict = {
                "00": "18",
                "06": "00",
                "12": "06",
                "18": "12"
            }
            gfs_cycle = cycle_dict[AQM_CYCLE]
            if AQM_CYCLE == "00":
                gfs_date = (aqm_timestep - pd.Timedelta(days=1)).strftime("%Y%m%d")
            else:
                gfs_date = aqm_timestep.strftime("%Y%m%d")

            for hour in range(len(FCST_TIME)):
                gfs_hour = hour + 1 + 6   # account for time differece from previous cycle
                phy_file = f"{MET_PATH}/gfs.{gfs_date}/{gfs_cycle}/atmos/gfs.t{gfs_cycle}z.sfcf0{gfs_hour:02d}.nc"
                print(f"Read met from {phy_file}")
                daily_data = gfs_met_loader(phy_file)
                total_data += [daily_data]

        else:
            print("Met source is not identified. Terminated.")
            sys.exit(1)

        total_data = np.stack(total_data, 0)
        print('met', total_data.shape, pd.Timestamp("now"))
        return total_data
    
    
    def read_emi_fields(self, EMI_PATH, AQM_DATE, AQM_CYCLE, FCST_TIME):
        varlist = ["NOx", "SO2", "NH3", "VOC", "PM25"]
        offset = [    0,    0,    0,    0,     0]
        scaler = [ 1e-9, 1e-9, 1e-9, 1e-9,  1e-9]
        emspt = [46, 96, 17, 72, 1]
        time_length = len(FCST_TIME)

        emi_path = f"{EMI_PATH}/aqm.{AQM_DATE}/{AQM_CYCLE}"
        nexus_emi_file = f"{emi_path}/aqm.t{AQM_CYCLE}z.NEXUS_Expt.nc"
        plume_emi_file = f"{emi_path}/aqm.t{AQM_CYCLE}z.PT.nc"
        fire_emi_file = glob.glob(emi_path + "/FIRE_EMISSION/Hourly_Emissions_*.nc")[0]

        print("Read emissions from:")
        print(nexus_emi_file)
        print(plume_emi_file)
        print(fire_emi_file)

        ## NEXUS emission
        nexus = []
        for i in range(len(varlist)):
            data = nexus_emi_loader(nexus_emi_file, varlist[i], time_length)
            data = (data - offset[i]) / scaler[i]
            nexus += [data]
        nexus = np.stack(nexus, 1)
        print('nexus', nexus.shape, pd.Timestamp("now"))
        
        ## plume emission
        plume = []
        area = aqm_loader("./fix/grid_spec.nc", "area")
        for i in range(len(varlist)):
            data = plume_emi_loader(
                emi_path + "/aqm.t12z.PT.nc",
                varlist[i],
                time_length,
            )
            data = data * emspt[i] / area * 1e-3  # convert unit from moles/s to kg /m2/s
            data = (data - offset[i]) / scaler[i]
            plume += [data]
        plume = np.stack(plume, 1)
        print('plume', plume.shape, pd.Timestamp("now"))
        
        ## fire emission
        fire = []
        for i in range(len(varlist)):
            data = fire_emi_loader(fire_emi_file, varlist[i], time_length)
            data = (data - offset[i]) / scaler[i]
            fire += [data]
        fire = np.stack(fire, 1)
        print('fire', fire.shape, pd.Timestamp("now"))

        total_emi = np.concatenate((nexus, plume, fire), 1)
        print('emi', total_emi.shape, pd.Timestamp("now"))
        return total_emi
    
    
    def read_icbc_fields(self, EMI_PATH, AQM_CYCLE, FCST_TIME, BNDY_LENGTH, FCST_SPECIES, FCST_LAYER, BNDY_OPTION):
        init_date = (pd.to_datetime(FCST_TIME[0], format="%Y%m%d%H") - pd.Timedelta(hours=int(AQM_CYCLE))).strftime("%Y%m%d")
        species_fulllist = ["NO2", "NH3", "HCHO", "OZONE", "PM25"]
        varlist = ["no2", "nh3", "form", "o3", "pm25_tot"]
        dynlist = ["no2", "nh3", "form", "o3_ave", "pm25_ave"]
        offset = [    0,    0,    0,    0,    0]
        scaler = [ 1e-2, 1e-2, 1e-2, 1e-2,   10]

        total_data = []
        for species in FCST_SPECIES:
            try:
                idx = species_fulllist.index(species)
            except Exception as e:
                print(f"IC/LBC loading error: {e}")
       
            # IC
            init_file = f"{EMI_PATH}/aqm.{init_date}/{AQM_CYCLE}/aqm.t{AQM_CYCLE}z.gfs_data.tile7.halo0.nc"
            print(f"Read IC from {init_file}")
            data = aqm_loader(init_file, varlist[idx])
            data = np.squeeze(data[FCST_LAYER, :, :])
            data = np.repeat(np.expand_dims(data, axis=0), len(FCST_TIME), axis=0)

            # LBC
            if BNDY_OPTION == "retro":
                # Current retro settings are based on 24h AQMv8 products
                for i in range(len(FCST_TIME)):
                    time = FCST_TIME[i]
                    aqm_timestep = pd.to_datetime(time, format="%Y%m%d%H") - pd.Timedelta(hours=int(AQM_CYCLE))
                    if (time != FCST_TIME[0]) & (aqm_timestep.strftime("%H%M") == "0000"):
                        aqm_date = (aqm_timestep - pd.Timedelta(days=1)).strftime("%Y%m%d")
                        aqm_hour = "24"
                    else:
                        aqm_date = aqm_timestep.strftime("%Y%m%d")
                        aqm_hour = aqm_timestep.strftime("%H")
            
                    dyn_file = f"{EMI_PATH}/aqm.{aqm_date}/{AQM_CYCLE}/dynf0{aqm_hour}.nc"
                    print(f"Read LBC from {dyn_file}")
    
                    bc = aqm_loader(dyn_file, dynlist[idx])
                    bc = np.squeeze(bc[0, FCST_LAYER - 1, :, :])
                    if species == "OZONE":
                        bc = bc / 1000
    
                    data[i, 0, :] = bc[0, :]  # top
                    data[i, -1, :] = bc[-1, :]  # bottom
                    data[i, :, 0] = bc[:, 0]  # left
                    data[i, :, -1] = bc[:, -1]  # right
                    del [time, aqm_date, aqm_hour, dyn_file, bc]

            elif BNDY_OPTION == "fcst":
                halo = 0  # inner layer
                bndy_hour = np.arange(0, len(FCST_TIME), BNDY_LENGTH)
                bndy_path = f"{EMI_PATH}/aqm.{init_date}/{AQM_CYCLE}"

                pm25ac = aqm_loader(init_file, "pm25ac")
                pm25ac = np.squeeze(pm25ac[FCST_LAYER, :, :])
                pm25co = aqm_loader(init_file, "pm25co")
                pm25co = np.squeeze(pm25co[FCST_LAYER, :, :])

                for i in range(len(bndy_hour)):
                    bndy_file = f"{bndy_path}/aqm.t{AQM_CYCLE}z.gfs_bndy.tile7.f{bndy_hour[i]:03d}.nc"
                    print(f"Read LBC from {bndy_file}")
                    top, bottom, left, right = aqm_bndy_loader(bndy_file, varlist[idx], FCST_LAYER, halo, pm25ac, pm25co)
                    ll = data[bndy_hour[i]:bndy_hour[i]+BNDY_LENGTH, :].shape[0]
                    data[bndy_hour[i]:bndy_hour[i]+BNDY_LENGTH, 0, :] = np.repeat(np.expand_dims(top, axis=0), ll, axis=0)
                    data[bndy_hour[i]:bndy_hour[i]+BNDY_LENGTH, -1, :] = np.repeat(np.expand_dims(bottom, axis=0), ll, axis=0)
                    data[bndy_hour[i]:bndy_hour[i]+BNDY_LENGTH, :, 0] = np.repeat(np.expand_dims(left, axis=0), ll, axis=0)
                    data[bndy_hour[i]:bndy_hour[i]+BNDY_LENGTH, :, -1] = np.repeat(np.expand_dims(right, axis=0), ll, axis=0)

            else:
                print(f"LBC mode is not identified. Terminated")
                sys.exit(1)

            data = (data - offset[idx]) / scaler[idx]
            data = np.expand_dims(data, 1)
            total_data += [data]
            print(f'{species} icbc', data.shape, pd.Timestamp("now"))
        total_data = np.concatenate(total_data, 1)
        print('icbc', total_data.shape, pd.Timestamp("now"))
        return total_data


    def save_restart(self, FCST_TIME, INPUT_DATA, FCST_SPECIES, FCST_LAYER, OUTPUT_PATH):
        fcst_hour = np.arange(len(FCST_TIME))
        grid_yt = np.arange(INPUT_DATA[0].shape[2]) + 1
        grid_xt = np.arange(INPUT_DATA[0].shape[3]) + 1

        emi_variable = ["nexus_NOx", "nexus_SO2", "nexus_NH3", "nexus_VOC", "nexus_PM25", \
                       "pt_NOx", "pt_SO2", "pt_NH3", "pt_VOC", "pt_PM25", \
                       "fire_NOx", "fire_SO2", "fire_NH3", "fire_VOC", "fire_PM25"]
        static_variable = ["grid_latt", "grid_lont", "area", "zsurf", "land"]
        met_variable = ["ugrd10m", "vgrd10m", "hpbl", "spfh2m", "dswrf_ave", \
                       "tmp2m", "veg", "tcdc_aveclm", "tprcp", "lhtfl_ave", \
                       "fricv", "pressfc", "cnwat", "snowc_ave"]
        fcst_variable = FCST_SPECIES

        output_file = f"{OUTPUT_PATH}/deepaqm_restart_layer{FCST_LAYER}_f{(len(FCST_TIME)-1):03d}.nc"

        ds = xr.Dataset(
            coords={
                "static_time": ("static_time", [fcst_hour[0]]),
                "met_time": ("met_time", fcst_hour[1:]),
                "fcst_time": ("fcst_time", fcst_hour),
                "yt": ("grid_yt", grid_yt),
                "xt": ("grid_xt", grid_xt),
                "emi_variable": ("emi_variable", emi_variable),
                "static_variable": ("static_variable", static_variable),
                "met_variable": ("met_variable", met_variable),
                "fcst_variable": ("fcst_variable", fcst_variable),
            }
        )
    
        ds["time_utc"] = xr.DataArray(FCST_TIME, dims=["fcst_time"], coords=[fcst_hour])
        ds["emi"] = xr.DataArray(INPUT_DATA[0], dims=["met_time", "emi_variable", "yt", "xt"], coords=[fcst_hour[1:], emi_variable, grid_yt, grid_xt])
        ds["static"] = xr.DataArray(INPUT_DATA[1], dims=["static_time", "static_variable", "yt", "xt"], coords=[[fcst_hour[0]], static_variable, grid_yt, grid_xt])
        ds["met"] = xr.DataArray(INPUT_DATA[2], dims=["met_time", "met_variable", "yt", "xt"], coords=[fcst_hour[1:], met_variable, grid_yt, grid_xt])
        ds["fcst_species"] = xr.DataArray(INPUT_DATA[3], dims=["fcst_time", "fcst_variable", "yt", "xt"], coords=[fcst_hour, fcst_variable, grid_yt, grid_xt])

        ds["emi"].attrs["description"] = emi_variable
        ds["static"].attrs["description"] = static_variable
        ds["met"].attrs["description"] = met_variable
        ds["fcst_species"].attrs["description"] = fcst_variable

        ds.to_netcdf(output_file)
        print(f"Restart file saved to {output_file}")
    
    
    def main_driver(self):
        start_time = pd.Timestamp("now")
    
        AQM_DATE, AQM_CYCLE, MET_OPTION, MET_PATH, AQM_PATH, EMI_PATH, OUTPUT_PATH, FCST_LENGTH, BNDY_LENGTH, FCST_SPECIES, FCST_LAYER, BNDY_OPTION, RESTART = self.namelist_config()
    
        if RESTART:
            RESTART_FILE = f"{OUTPUT_PATH}/deepaqm_restart_layer{FCST_LAYER}_f{FCST_LENGTH:03d}.nc"
            print(f"Reading input data from restart file {RESTART_FILE}")
            STATIC = aqm_loader(RESTART_FILE, "static")
            MET = aqm_loader(RESTART_FILE, "met")
            EMI = aqm_loader(RESTART_FILE, "emi")
            ICBC = aqm_loader(RESTART_FILE, "fcst_species")
        else:
            print("Reading input data from initial AQM files...")
            FCST_TIME = pd.date_range(
                start=pd.to_datetime(AQM_DATE + AQM_CYCLE, format="%Y%m%d%H"),
                periods=FCST_LENGTH + 1,
                freq="h"
            ).strftime("%Y%m%d%H")
    
            STATIC = self.read_static_fields(f"{AQM_PATH}/{AQM_DATE}{AQM_CYCLE}")
            MET = self.read_met_fields(MET_OPTION, MET_PATH, AQM_DATE, AQM_CYCLE, FCST_TIME[1:])
            EMI = self.read_emi_fields(EMI_PATH, AQM_DATE, AQM_CYCLE, FCST_TIME[1:])
            ICBC = self.read_icbc_fields(EMI_PATH, AQM_CYCLE, FCST_TIME, BNDY_LENGTH, FCST_SPECIES, FCST_LAYER, BNDY_OPTION)
            self.save_restart(FCST_TIME, (EMI, STATIC, MET, ICBC), FCST_SPECIES, FCST_LAYER, OUTPUT_PATH)

        # Input shape = (batch, time, channel, height, width)
        STATIC = np.expand_dims(STATIC, 0)
        MET = np.expand_dims(MET, 0)
        EMI = np.expand_dims(EMI, 0)
        ICBC = np.expand_dims(ICBC, 0)

        self.data_set = (EMI, STATIC, MET, ICBC)
