import cdsapi
import os
import requests
from urllib.parse import urljoin

output_dir = "../../data/raw/time_invariant"
os.makedirs(output_dir, exist_ok=True)

def download_era5():
    output_file = os.path.join(output_dir, f"era5_time_invariant.nc")
    if os.path.exists(output_file):
        return

    client = cdsapi.Client()
    dataset = "derived-era5-single-levels-daily-statistics"
    request = {
        "product_type": "reanalysis",
        "variable": [
            "lake_cover",
            "soil_type",
            "high_vegetation_cover",
            "low_vegetation_cover",
            "type_of_high_vegetation",
            "type_of_low_vegetation",
            "geopotential",
            "land_sea_mask",
            "standard_deviation_of_orography"
        ],
        "year": "2025",
        "month": ["01"],
        "day": ["01"],
        "daily_statistic": "daily_mean",
        "time_zone": "utc+00:00",
        "frequency": "6_hourly",
        "area": [90, -180, 0, 180],
    }
    try:
        client.retrieve(
            dataset,
            request,
            output_file
        )
    except Exception as e:
           print(f"Error while downloading: {e}")

def download_natural_earth():
    url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries_rus.zip"
    filename = os.path.join(output_dir, os.path.basename(url))
    if os.path.exists(filename):
        return

    print(f"Downloading: {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

download_era5()
download_natural_earth()
