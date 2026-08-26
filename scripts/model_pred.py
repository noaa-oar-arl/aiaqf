
import os, sys
import torch
import torch.nn as nn
import numpy as np
import xarray as xr
import pandas as pd
import multiprocessing
from functools import partial

sys.path.insert(0, "./toolkit")
sys.path.insert(0, "./models")
from aqm_loaders import aqm_loader
from model_input_gen import input_generator
from seq2seq_ConvLSTM import EncoderDecoderConvLSTM

def namelist_config():
    namelist = pd.read_csv("./namelist", header=None, delimiter="=")
    namelist = namelist.rename(columns={0: "key", 1: "value"}).set_index("key")
    namelist = namelist.to_dict()["value"]

    AQM_DATE = namelist["AQM_DATE"]
    AQM_CYCLE = namelist["AQM_CYCLE"]
    OUTPUT_PATH = namelist["OUTPUT_PATH"]
    OUTPUT_PATH = f"{OUTPUT_PATH}/{AQM_DATE}/{AQM_CYCLE}"

    FCST_LENGTH = int(namelist["FCST_LENGTH"])
    BNDY_LENGTH = int(namelist["BNDY_LENGTH"])
    FCST_LAYER = int(namelist["FCST_LAYER"])

    MET_OPTION = namelist["MET_OPTION"]
    BNDY_OPTION = namelist["BNDY_OPTION"]

    FCST_OPTION = [
        namelist["NO2"] in ["True", "true"],
        namelist["NH3"] in ["True", "true"],
        namelist["HCHO"] in ["True", "true"],
        namelist["OZONE"] in ["True", "true"],
        namelist["PM25"] in ["True", "true"]
        ]
    FCST_MODELS = [
        namelist["NO2_MODEL_PATH"],
        namelist["NH3_MODEL_PATH"],
        namelist["HCHO_MODEL_PATH"],
        namelist["OZONE_MODEL_PATH"],
        namelist["PM25_MODEL_PATH"]
        ]
    FCST_SPECIES = ["NO2", "NH3", "HCHO", "OZONE", "PM25"]
    FCST_SPECIES = [species for species, opt in zip(FCST_SPECIES, FCST_OPTION) if opt]
    FCST_MODELS = [models for models, opt in zip(FCST_MODELS, FCST_OPTION) if opt]
    return AQM_DATE, AQM_CYCLE, OUTPUT_PATH, FCST_MODELS, FCST_LENGTH, BNDY_LENGTH, FCST_SPECIES, FCST_LAYER, MET_OPTION, BNDY_OPTION

def fcst_task(idx, FCST_MODELS, FCST_SPECIES, INPUT_SET, FCST_LENGTH, species_fulllist, species_scaler, species_chan, species_emidx):
    model_path = FCST_MODELS[idx]
    species = FCST_SPECIES[idx]
    scaler = species_scaler[species_fulllist.index(species)]
    N_chan = species_chan[species_fulllist.index(species)]
    em_idx = species_emidx[species_fulllist.index(species)]
    input_em = INPUT_SET[0][:, :, em_idx, :, :]
    input_label = np.expand_dims(INPUT_SET[3][:, :, idx, :, :], 2)

    learningrate = 0.00001
    model = EncoderDecoderConvLSTM(in_chan=N_chan, out_chan=1).to(device=device)
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"{species} fcst model loaded: {model_path}", pd.Timestamp("now"))

    # tensor size: batch, time, channel, height, width
    tensor_em = torch.Tensor(input_em)
    tensor_ic = torch.Tensor(INPUT_SET[1])
    tensor_met = torch.Tensor(INPUT_SET[2])
    tensor_label = torch.Tensor(input_label)
    tensor_input = torch.utils.data.TensorDataset(tensor_em, tensor_ic, tensor_met, tensor_label)
    data_loader = torch.utils.data.DataLoader(tensor_input, batch_size=1, shuffle=False, num_workers=0)
    print("Data loader created.")

    model.eval()
    with torch.no_grad():
        for batch_idx, (em, gd, met, label) in enumerate(data_loader):
            tmet = torch.cat((em, met), 2)
            tmet = tmet.to(device=device)
            gd = gd.to(device=device)
            label = label.to(device=device)
            out = model(tmet, label, gd[:, 0, :, :, :], FCST_LENGTH)
    out = np.squeeze(out.cpu().detach().numpy())
    out = out * scaler
    print(f"{species} fcst generated", pd.Timestamp("now"))
    return out

def run_fcst_model(FCST_MODELS, INPUT_SET, FCST_SPECIES, FCST_LENGTH):
    species_fulllist = ["NO2", "NH3", "HCHO", "OZONE", "PM25"]
    species_scaler = [1e-2, 1e-2, 1e-2, (1e-2)*(1e+3),   10]
    species_chan = [5+14+15+1, 5+14+15+1, 5+14+15+1, 5+14+6+1, 5+14+15+1]
    species_emidx = [
        np.arange(15),   # full list
        np.arange(15),   # full list
        np.arange(15),   # full list
        [0, 3, 5, 8, 10, 13],  # NOx, VOC
        np.arange(15)    # full list
    ]

    #worker = partial(
    #    fcst_task,
    #    FCST_MODELS=FCST_MODELS,
    #    FCST_SPECIES=FCST_SPECIES,
    #    INPUT_SET=INPUT_SET,
    #    FCST_LENGTH=FCST_LENGTH,
    #    species_fulllist=species_fulllist,
    #    species_scaler=species_scaler,
    #    species_chan=species_chan,
    #    species_emidx=species_emidx
    #)
    #with multiprocessing.Pool() as pool:
    #    total_out = pool.map(worker, range(len(FCST_MODELS)))

    total_out = []
    for i in range(len(FCST_MODELS)):
        predic = fcst_task(i, FCST_MODELS, FCST_SPECIES, INPUT_SET, FCST_LENGTH, species_fulllist, species_scaler, species_chan, species_emidx)
        total_out += [predic]
    total_out = np.stack(total_out, 1)
    return total_out

def save_prediction(TIMESTAMP, FCST_LENGTH, INIT, OUTPUT, FCST_SPECIES, FCST_LAYER, OUTPUT_PATH):
    var_fulllist = ["NO2", "NH3", "HCHO", "OZONE", "PM25"]
    unit_fulllist = ["ppm", "ppm", "ppm", "ppb", "ug/m3"]

    # INIT is f000 initial condition, shape=(batch=1, channel=N, height, width)
    OUTPUT = np.append(INIT, OUTPUT, axis=0)

    fcst_hour = np.arange(FCST_LENGTH +1)
    fcst_time_pd = pd.date_range(start=f"{TIMESTAMP}00", periods=FCST_LENGTH + 1, freq="h")  # f000 - f00N
    fcst_time = fcst_time_pd.strftime("%Y-%m-%dT%H:%M:%SZ")

    grid_lon = aqm_loader("./fix/grid_spec.nc", "grid_lont")
    grid_lat = aqm_loader("./fix/grid_spec.nc", "grid_latt")
    grid_lon[grid_lon >= 180] = grid_lon[grid_lon >= 180] - 360
    grid_yt = np.arange(grid_lat.shape[0]) + 1
    grid_xt = np.arange(grid_lat.shape[1]) + 1

    for tt in fcst_hour:
        output_file = f"{OUTPUT_PATH}/deepaqm_fcst_layer{FCST_LAYER}_f{tt:03d}.nc"
    
        ds = xr.Dataset(
            coords={
                "time": ("time", [np.float64(tt)]),
                "grid_yt": ("grid_yt", np.float64(grid_yt)),
                "grid_xt": ("grid_xt", np.float64(grid_xt)),
                "z": ("z", [np.float64(FCST_LAYER)]),
                "latitude": (("grid_yt", "grid_xt"), grid_lat),
                "longitude": (("grid_yt", "grid_xt"), grid_lon)
            }
        )

        time_iso = np.array(fcst_time[tt])
        ds["time_iso"] = xr.DataArray([time_iso], dims=["time"], coords=[[np.float64(tt)]])
        ds["time_iso"].attrs["long_name"] = "valid time"
        ds["time_iso"].attrs["description"] = "ISO 8601 datetime string"
    
        ds["latitude"].attrs["_FillValue"] = 9.99e+20
        ds["latitude"].attrs["long_name"] = "T-cell latitude"
        ds["latitude"].attrs["cartesian_axis"] = "Y"
        ds["latitude"].attrs["units"] = "degrees_N"
    
        ds["longitude"].attrs["_FillValue"] = 9.99e+20
        ds["longitude"].attrs["long_name"] = "T-cell longitude"
        ds["longitude"].attrs["cartesian_axis"] = "X"
        ds["longitude"].attrs["units"] = "degrees_E"
    
        for i in range(len(FCST_SPECIES)):
            species = FCST_SPECIES[i]
            unit = unit_fulllist[var_fulllist.index(species)]
            ds[species] = xr.DataArray(
                np.expand_dims(OUTPUT[tt, i, :, :], axis=(0, 1)),
                dims=["time", "z", "grid_yt", "grid_xt"],
                coords=[[np.float64(tt)], [np.float64(FCST_LAYER)], np.float64(grid_yt), np.float64(grid_xt)]
            )
            ds[species].attrs["_FillValue"] = 9.99e+20
            ds[species].attrs["cell_methods"] = "time: point"
            ds[species].attrs["long_name"] = f"hourly averaged {species}"
            #ds[species].attrs["missing_value"] = 9.99e+20
            ds[species].attrs["output_file"] = "deepaqm"
            ds[species].attrs["units"] = unit

        ds["time"].attrs["calendar"] = "JULIAN"
        ds["time"].attrs["calendar_type"] = "JULIAN"
        ds["time"].attrs["cartesian_axis"] = "T"
        ds["time"].attrs["long_name"] = "time"
        ds["time"].attrs["units"] = f"hours since {pd.to_datetime(TIMESTAMP, format='%Y%m%d%H').strftime('%Y-%m-%d %H:00:00')}"

        ds.to_netcdf(output_file, format="NETCDF4_CLASSIC")
        print(f"Forecast saved to {output_file}")
        del [output_file, ds, time_iso]


# ---- Load configs ----
AQM_DATE, AQM_CYCLE, OUTPUT_PATH, FCST_MODELS, FCST_LENGTH, BNDY_LENGTH, FCST_SPECIES, FCST_LAYER, BNDY_OPTION, MET_OPTION = namelist_config()
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ---- Log saving ----
class StdoutLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class StderrLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stderr
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

f_log = (f"{OUTPUT_PATH}/deepaqm_job")
sys.stdout = StdoutLogger(f_log + ".log")
sys.stderr = StderrLogger(f_log + ".err")


print("---- Initializaing DeepAQM...", pd.Timestamp("now"))

device = torch.device("cpu")
if(torch.cuda.is_available()):
    device = torch.device("cuda")


print(f"AQM initial time: {AQM_DATE}{AQM_CYCLE}")
print(f"Forecast length: {FCST_LENGTH}")
print(f"LBC update timestep: {np.arange(0, FCST_LENGTH + 1, BNDY_LENGTH)}")
print(f"Met source: {MET_OPTION}, LBC source: {BNDY_OPTION}")
print(f"Forecast species: {FCST_SPECIES} at layer {FCST_LAYER}")
print(f"Work directory: {OUTPUT_PATH}")

print("---- Generating input data...", pd.Timestamp("now"))

input_gen = input_generator()
input_gen.main_driver()
INPUT_DATA = input_gen.data_set   # (EMI, STATIC, MET, LABEL)

print("---- Input data ready!", pd.Timestamp("now"))
print("Shape = [batch, time, channel, height, width]")
print(f"Emissions: {INPUT_DATA[0].shape}")
print(f"Static fields: {INPUT_DATA[1].shape}")
print(f"Meteorology: {INPUT_DATA[2].shape}")
print(f"Predicted chemical: {INPUT_DATA[3].shape}")

print("---- Running forecast...", pd.Timestamp("now"))

FCST_OUTPUT = run_fcst_model(FCST_MODELS, INPUT_DATA, FCST_SPECIES, FCST_LENGTH)

print("---- Forecast complete!", pd.Timestamp("now"))
print(f"Output min: {FCST_OUTPUT.min()}, max: {FCST_OUTPUT.max()}")
print(f"Output shape: {FCST_OUTPUT.shape}")

save_prediction(AQM_DATE + AQM_CYCLE, FCST_LENGTH, INPUT_DATA[3][:, 0, :, :, :], FCST_OUTPUT, FCST_SPECIES, FCST_LAYER, OUTPUT_PATH)

open(f"{OUTPUT_PATH}/complete_checkpoint", "a").close()
