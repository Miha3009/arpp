# DL model for forecasting snow water equivalent up to 4 month ahead (resarch and developing stage)

## General context
Here we will try to develop a deep learning model that predicts snow water equivalent (SWE). 
- Target domain is the territory of Russia, but maybe worth trying to predict it globaly. 
- The model will be given information about previous atmospheric state with long time window (up to several month) from ERA5 reanalizys, for the forward time step (t0+n) the model will be given atmosperic states simulated with numerical model (INMCM6, maximum lead time 4 month), as predictors. Target variable are SWE values from ERA5 reanalizys for the target forecasting period. 
- So the model setup is something between forecasting and bias-correction. Training dataset covers daily atmospheric variables from 1991 almost up to date (16 june 2026). 
- Testing metrics are RMSE ans ACC (anomaly correlation coefficient) of snow water equivalent. 

## Final goal 
Final goal for this project is to develop full training and testing pipeline. There will be a lot of experiments so the system must be reproducible. We need to develop a stable experiment config structure, so all experiment settings - from exact predictors to model architecture and training hyperparameters - goes to one json and the system starts training model from this json. 

## Technical setup
All the techical details derived so far you can get from README.md and data_pipline_desc.md files in the root of this project. 
We work on this project tigether with the colleague of mine, so you can see his commits in this repo. 
For now we are on data preparation stage, so I'll briefly describe data we have so far: 
1. ERA5: 1991-2016 daily meteorologiacal variables - Temperature at 2 meters, Sea-level pressure, Precipitation, etc. Globally, on 0.25° grid.
1. INMCM6: Simulated 1991-2016 daily meteorologiacal variables - Temperature at 2 meters, Sea-level pressure, Precipitation, etc. Globally, on 1° grid.
There are many additional meteorological variables avaliable, from ERA5 and INMCM6 both. Exact predictors (features) set is being discussed. 

*Note: INMCM6 data presented in the **ensemble**, meaning there are several simulated versions of any variable of any date. This gives additional E tensor dimension - [E,T,H,W]. And that's the challenge in some way, because E dimension varies depending on the INMCM version - some years have 10 ensemble members and some have 30. Also in the future inference the number of ensemble members may change. We must decide how to procees this varying E dimension - just aggregate, that gives inconsistence in data, or feed them all in the model and let the model aggregate (embed) it. But how the model can aggregate inconsisten dimension? That's the question to answer to*

## Next steps
Some ideas of the model architecture: 
- U-net with attention heads inside of it. SWE is slowly changing variable that depends on global processes of subseasonal and seasonal scales. So the receptive field of the model must be wide, that's what U-net gives. Attention head must be in the bottleneck. Also there are some long timeseries in the inputs (meteovariable timeseries), so maybe aggregate this series through attention too. Spatial attention, channel attention are for consideration as well.  

## Pointers
~data/data_pipeline.md: describes how data (ERA5 meteorological parameters fields and INMCM forecasts) are prepared from raw fields in netCDF files to tensors that PatherDataset class returns. Just technical description for reminder. 
~data/database_setup.md: all details about database described from EDA.  
~/model_sketch.md: the first idea of the model - general structure, dimensions handling, some things to decide. Raw, need discussion and tests. 
~/details.md: deep-dive companion to model_sketch.md on axis handling - T (time) collapse options (flatten/conv3D/PMA, with PMA mechanics expanded) and E (ensemble) handling (pooling options + members-as-augmentation).
~/options.md: shortened version of details.md, created for the project group discussion. 
~/notes.md: some things to remember 

