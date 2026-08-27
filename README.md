# aiaqf
AI-based model to provide air quality forecasting.

The repository provide the research-to-operation transition for implementing the UFS-DeepAQM, which uses UFS-AQM's inputs to makes air quality forecasting. Two major python scripts are included: the main model driver (`model_pred.py`) and the input generator (`model_input_gen.py`).

### Namelist options
The UFS-AQM run used as input sources, forecasting period and chmiecal species, and input and output directories can be specified in the namelist. Five chmiecal species (NO2, NH3, HCHO, O3 and PM2.5) are set as default, but currently only O3 and PM2.5 are available (updated on Feb 25 2026).

| Option           | Description                                             |
| ---------------- | ------------------------------------------------------- |
| AQM_DATE         | Date of UFS-AQM run, format=YYYYMMDD                    |
| AQM_CYCLE        | Time cycle of UFS-AQM run, format=HH                    |
| FCST_LENGTH      | Length of forecast timestep, unit=hour                  |
| BNDY_LENGTH      | Time interval of UFS-AQM boundary conditions, unit=hour |
| FCST_LAYER       | The UFS-AQM vertical layer where forecasting chmiecal species at. FCST_LAYER=64 represents the surface based on UFS-AQM configuration. |
| MET_OPTION       | Source of met data (GFS/AQM)                            |
| BNDY_OPTION      | Source of LBC data (retro/fcst). For retro, LBC will be generated from AQM dyn outputs. For fcst, LBC will be loaded from AQM LBC files. |
| NO2              | Option to run NO2 forecasting (True/False)              |
| NH3              | Option to run NH3 forecasting (True/False)              |
| HCHO             | Option to run HCHO forecasting (True/False)             |
| OZONE            | Option to run O3 forecasting (True/False)               |
| PM25             | Option to run PM2.5 forecasting (True/False)            |
| RESTART          | Option to use generated restart file instead of collecting inputs from UFS-AQM run (True/False). When `RESTART=Flase`, the input generator will read and process inputs from the specified UFS-AQM run. All processed inputs will be saved in a restart file. When `RESTART=True`, the model driver will directly read inputs from the restart file and the input processes will NOT be triggered. |
| OUTPUT_PATH      | Path of the work/output directory. A subdirectory based on AQM_DATE and AQM_CYLCE will be created under this path.               |
| AQM_PATH         | Path of UFS_AQM runs. The input generator will search for the subdirectory based on AQM_DATE and AQM_CYLCE under this path.      |
| EMI_PATH         | Path of UFS_AQM emissions and IC/BC. The input generator will search for the subdirectory based on AQM_DATE and AQM_CYLCE under this path. |
| NO2_MODEL_PATH   | Path of the NO2 AI model. Use `None` if no model available.   |
| NH3_MODEL_PATH   | Path of the NH3 AI model. Use `None` if no model available.   |
| HCHO_MODEL_PATH  | Path of the HCHO AI model. Use `None` if no model available.  |
| OZONE_MODEL_PATH | Path of the OZONE AI model. Use `None` if no model available. |
| PM25_MODEL_PATH  | Path of the PM25 AI model. Use `None` if no model available.  |

### Required UFS-AQM inputs
| Application                          | Filename                                                            |
| ------------------------------------ | ------------------------------------------------------------------- |
| Model Coordinate / land properties   | `grid_spec.nc` `atmos_static.nc` (included in `fix/`)               |
| Meteorology                          | Operational GFS or `phyf*.nc` from AQM                              |
| Emission                             | `aqm.t*z.NEXUS_Expt.nc`, `aqm.t*z.PT.nc`, `Hourly_Emissions_*.nc` |
| Initial/Boundary                     | `aqm.t*z.gfs_data.tile7.halo0.nc`, `aqm.t*z.gfs_bndy.tile7.f*.nc` |​

### Run the models
AI models are running parallelly in `model_pred.py`. 

[Update on Aug 26 2026] Parallel process is currently disabled for GPU. See line 108-120 in `model_pred.py`.

On GMU Hopper (1 node 12 core), the input generation process takes ~ 4 min and AI models take ~ 13 min to complete 72 h forecast for two chemical species. Recommended slurm settings to run 72 h forecast for two chemical species on Hopper:
```
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem=72G
#SBATCH --time=0-00:20:00
```
On NOAA Gaea C6 (1 node 2 core), the input generation process takes ~ 6 min and AI models take ~ 2 min to complete 72 h forecast for two chemical species. Recommended slurm settings to run 72 h forecast for two chemical species on Gaea C6:
```
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=0-00:10:00
```
On NOAA Ursa, if running on CPU (1 node 20 core), the input generation process takes ~ 11 min and AI models take ~ 1 hour to complete 72 h forecast for two chemical species. If running forecasting on 1 GPU, AI models take ~ 2 min to complete 72 h forecast for two chemical species. Recommended slurm settings to run 72 h forecast for two chemical species on Ursa:

[CPU]
```
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=0-01:20:00
```
[GPU]
```
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --gpus=h100:1
#SBATCH --time=0-00:20:00
```

### AI model version log
| Date        | Version (species_version#_epoch)       |
| ----------- | -------------------------------------- |
| Feb 18 2026 | pm25_l1_19, ozone_l2_19                |
