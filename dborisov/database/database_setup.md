## Project's database setup

This project database is presented with set of binary files (.bin). 
Files contain meteorological gridded data, simulated by INMCM subseasonal NWP model (files with 'inm_' prefix), and ERA5 reanalysis data (files with no prefix). 
Both INMCM and ERA5 grids cover Northern hemisphere, INMCM full grid size is (360, 91), 1° spatial resolution, ERA5 grid size is (1440, 361), 0.25° spatial resolution. 
Files are named by the meteorological parameter. 

**The database is requested through** custom "patcher" library, created in order to compress data and to speed up loading to the training batches. Also this "patcher" data setup allows to request and load custom patches - samples derived from the full meteo variable field: spatial samples, timespan samples, spatially- or time- aggregated samples.   

Date requested throught ```pathcer.Request()``` and ```patcher.load()``` methods: 

```
request = patcher.Request(element:   variable name in the database
                          x0, y0 :   patch center (in grid pixels)
                          t0     :   date string YYYYMMDD
                          xSize  :   patch width  in downsampled cells
                          ySize  :   patch height in downsampled cells
                          tSize  :   number of time steps
                          xyStep :   spatial scale (1 = 0.25° for ERA5, 4 = 1° for INM-CM)
                          tStep  :   temporal stride in days
                          tag    :   forecast start month (INM-CM only)
)
patcher.load(patcher_context: database pointer (directory) 
            [request]       
            ) --> list of torch.Tensor
```
 - ERA5 variable  request loads [T, H, W]  tensor, float32
 - INM-CM variable request loads [E, T, H, W] tensor, float32  (E = ensemble size)

, where T is timespan dimension, **E is ensemble dimension. For every variable and date INMCM data contain simulations from several ensemble members (10 or 30 members)**. Ensemble members derived from different initializations of INMCM dynamis model for the same date forecast. 
**All tensors contain meteo variables anomalies** - values normalized by 1991-2021 climate for this date and grid cell. To derive absolute values: ```patcher.load([request]) + pathcer.load_climate([request])```. 

**Also INMCM data requires "tag" argument**, which is forecast's initialization month. Maximum lead time of the INMCM forecast is 4 month, the foreacsts are launched once a month, at the start of the month. 
So, database request for particular date might return 4 different tensors, depending on "tag" argument. For example: 
* request for '20200401', tag=1: INMCM forecast for April, 1st 2025 initialized on January, 1st 2025 (3 month lead time). 
* request for '20200401', tag=2: INMCM forecast for April, 1st 2025 initialized on February, 1st 2025 (2 month lead time).
* request for '20200401', tag=3: INMCM forecast for April, 1st 2025 initialized on March, 1st 2025 (1 month lead time).
* request for '20200401', tag=2: INMCM forecast for April, 1st 2025 initialized on April, 1st 2025 (1 day lead-time).

### Variables
*Notes for the distribution features derived from EDA, near-to-normal distribution where there is no note*

ERA5: 
- t2m: air temperature at 2 meters, K 
- tp: Total precipitation, mm. *Right-skewed, needs log-transform*
- sd: Snow water equivalent, m. *Mask values: 5000 for glaicer cells (always snow), 0 for sea cells (0 as wel for land cells if there's no snow). Right-skewed, needs log-transform*. **Target variable**
- sden: Snow density, kg/m^-3. *Mask values: 100 for the no-snow cells, 300 for glaicer cells (always snow). Normal distribution with masked cells excluded*  
- pt: Precipitation type, categorical
- snow_cover: Fraction of grid cell covered by snow {0-1}. *No masked values, need to mask by glacier. Close to moving (seasonal) binary distribution (0,1). Maybe make binary with a minimum threshhold*  
- sst: Sea surface temperature, K. *Masked with near-zeros for land cells, cut manually with <= 0.1. Distribution something like bimodal, skewed to the left (273.0, ice) and to the rigth (tropics, warm)*  

INMCM: 
- h500: Heopotential 500 gPa, m
- hlt: Latent heat flux, W/m**2
- olr: Outgoing longwave radiation, W/m**2
- tp: Total precipitation, mm. *Right-skewed, needs log-transform*
- mslp: Sea-level pressure, mb.
- ts: Surface temperature, K                        
- swe: Snow water equivalent, m. *Right-skewed, needs log-transform ? Outliers manual cut ~ 1000 mm SWE* 
- snow_cover: Fraction of grid cell covered by snow {0-1}. *Close to binary distribution (0,1). Close to moving (seasonal) binary distribution (0,1). Maybe make binary with a minimum threshhold*  
- ww: 100 sm. soil water, mm. *Normal-bimodal, manually cut to >= 0.1*      
- u850: U-wind 850 gPa, m/s
- v850: V-wind 850 gPa, m/s
- t2m: air temperature at 2 meters, K 

Time-invariant variables: 
- sdor: Standard deviation of orography, m (ERA5)
- z: surface height, m (ERA5)
- glacier: glacier mask, {0,1}
- lsm: land-sea mask, {0-1}


### Data timespan and temporary resolution
ERA5: daily fields of all ERA5 variables from February 1st, 1991 to June 27th, 2026. 35.5 years, no missing data. 
INMCM: daily fields of all INMCM variables from February 1st, 1991 to April 30th, 2021, which is 30 years retrospective simulations (hindcast), containing 10 ensemble members. February 1st, 1991 is first initialized hincast (tag=2 only for this February month, tag={2,3} for the next month, etc.) and April 30th, 2021 is a last lead time day of the foreacst initialized on January 1st, 2021 (tag=1 only for this April month). 
The second peace of INMCM data starts on Semptember 1st, 2024 (first day of the foreacst, tag=9 only) and ends on October 31th, 2026 (last day of the forecast initialized on July 1st, 2026, tag=7 only). These are almost 2 years of operative forecasts that are being released to this day.  

### Data storage
All heavy data (.bin files) is stored on external SSD, directory: /Volumes/portable/data. 
