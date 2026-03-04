
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
from data_loaders import aqm_loader
from model_input_gen import input_generator
from seq2seq_ConvLSTM import EncoderDecoderConvLSTM

def namelist_config():
    namelist = pd.read_csv("./namelist", header=None, delimiter="=")
    namelist = namelist.rename(columns={0: "key", 1: "value"}).set_index("key")
    namelist = namelist.to_dict()["value"]

    AQM_DATE = namelist["AQM_DATE"]
    AQM_CYCLE = namelist["AQM_CYCLE"]
    OUTPUT_PATH = namelist["OUTPUT_PATH"]

    FCST_LENGTH = int(namelist["FCST_LENGTH"])
    BNDY_LENGTH = int(namelist["BNDY_LENGTH"])
    FCST_LAYER = namelist["FCST_LAYER"]

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
    return AQM_DATE, AQM_CYCLE, OUTPUT_PATH, FCST_MODELS, FCST_LENGTH, BNDY_LENGTH, FCST_SPECIES, FCST_LAYER

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
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"{species} fcst model loaded: {model_path}", pd.Timestamp("now"))

    # tensor size: batch, time, channel, height, width
    tensor_em = torch.Tensor(input_em)
    tensor_ic = torch.Tensor(INPUT_SET[1])
    tensor_met = torch.Tensor(INPUT_SET[2])
    tensor_label = torch.Tensor(input_label)
    tensor_input = torch.utils.data.TensorDataset(tensor_em, tensor_ic, tensor_met, tensor_label)
    data_loader = torch.utils.data.DataLoader(tensor_input, batch_size=1, shuffle=False, num_workers=0)

    model.eval()
    with torch.no_grad():
        for batch_idx, (em, gd, met, label) in enumerate(data_loader):
            tmet = torch.cat((em, met), 2)
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

    worker = partial(
        fcst_task,
        FCST_MODELS=FCST_MODELS,
        FCST_SPECIES=FCST_SPECIES,
        INPUT_SET=INPUT_SET,
        FCST_LENGTH=FCST_LENGTH,
        species_fulllist=species_fulllist,
        species_scaler=species_scaler,
        species_chan=species_chan,
        species_emidx=species_emidx
    )

    with multiprocessing.Pool() as pool:
        total_out = pool.map(worker, range(len(FCST_MODELS)))
    total_out = np.stack(total_out, 1)
    return total_out

def save_prediction(TIMESTAMP, FCST_LENGTH, INIT, OUTPUT, FCST_SPECIES, FCST_LAYER, OUTPUT_PATH):
    var_fulllist = ["NO2", "NH3", "HCHO", "OZONE", "PM25"]
    unit_fulllist = ["ppm", "ppm", "ppm", "ppb", "ug/m3"]

    # INIT is f000 initial condition, shape=(batch=1, channel=N, height, width)
    OUTPUT = np.append(INIT, OUTPUT, axis=0)

    fcst_hour = np.arange(FCST_LENGTH +1)
    fcst_time = pd.date_range(start=f"{TIMESTAMP}00", periods=FCST_LENGTH + 1, freq="h")  # f000 - f00N
    fcst_time = fcst_time.strftime("%Y%m%d%H%M").astype("int")

    grid_lon = aqm_loader("./fix/grid_spec.nc", "grid_lont")
    grid_lat = aqm_loader("./fix/grid_spec.nc", "grid_latt")
    grid_yt = np.arange(grid_lat.shape[0]) + 1
    grid_xt = np.arange(grid_lat.shape[1]) + 1

    for tt in fcst_hour:
        output_file = f"{OUTPUT_PATH}/deepctm_fcst_layer{FCST_LAYER}_f{tt:03d}.nc"
    
        ds = xr.Dataset(
            coords={
                "time": ("time", [tt]),
                "yt": ("grid_yt", grid_yt),
                "xt": ("grid_xt", grid_xt),
                "grid_lat": (("yt", "xt"), grid_lat),
                "grid_lon": (("yt", "xt"), grid_lon)
            }
        )
    
        ds["time_utc"] = xr.DataArray(fcst_time[tt], dims=["time"], coords=[[tt]])
        for i in range(len(FCST_SPECIES)):
            species = FCST_SPECIES[i]
            unit = unit_fulllist[var_fulllist.index(species)]
            ds[species] = xr.DataArray(
                np.expand_dims(OUTPUT[tt, i, :, :], axis=0),
                dims=["time", "yt", "xt"],
                coords=[[tt], grid_yt, grid_xt]
            )
            ds[species].attrs["unit"] = unit
        ds.to_netcdf(output_file)
        del ds
        print(f"Forecast saved to {output_file}")


print("---- Initializaing DeepCTM...", pd.Timestamp("now"))

device = torch.device("cpu")
if(torch.cuda.is_available()):
    device = torch.device("cuda")


AQM_DATE, AQM_CYCLE, OUTPUT_PATH, FCST_MODELS, FCST_LENGTH, BNDY_LENGTH, FCST_SPECIES, FCST_LAYER = namelist_config()
os.makedirs(OUTPUT_PATH, exist_ok=True)

print(f"AQM initial time: {AQM_DATE}{AQM_CYCLE}")
print(f"Forecast length: {FCST_LENGTH}")
print(f"BC update timestep: {np.arange(0, FCST_LENGTH + 1, BNDY_LENGTH)}")
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
