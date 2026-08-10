import numpy as np

scales = {
    "inm_h500": 100,
    "inm_hlt": 100,
    "inm_mslp": 10,
    "inm_olr": 25,
    "inm_swe": 1,
    "inm_t2m": 10,
    "inm_tp": 1,
    "inm_ts": 10,
    "inm_u850": 10,
    "inm_v850": 10,
    "inm_ww": 0.1,
    "sd": 1,
    "sden": 100,
    "sdor": 5,
    "sst": 5,
    "t2m": 10,
    "tp": 1,
    "z": 2,
    "inm_lead_time": 120
}

bias = {
    "inm_t2m": 273.15,
    "t2m": 273.15,
    "inm_ts": 273.15,
    "sst": 273.15,
    "z": np.log(129)
}

def normalize(data, element):
    return (data - bias.get(element, 0)) / scales.get(element, 1)

def denormalize(data, element):
    return (data * scales.get(element, 1)) + bias.get(element, 0)
