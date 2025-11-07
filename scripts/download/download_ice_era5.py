import cdsapi
import os

output_dir = "../../data/raw/ice_era5"
os.makedirs(output_dir, exist_ok=True)

dataset = "derived-era5-single-levels-daily-statistics"
years = list(range(2018, 2026))
months = [f"{m:02d}" for m in range(1, 13)]
days = [f"{d:02d}" for d in range(1, 32)]

client = cdsapi.Client()

for year in years:
    for month in months:
        output_file = os.path.join(output_dir, f"era5_ice_cover_{year}_{month}.nc")
        if os.path.exists(output_file):
            continue

        print(f"Download {year}-{month}")

        try:
            client.retrieve(
                dataset,
                {
                    "product_type": "reanalysis",
                    "variable": ["sea_ice_cover"],
                    "year": str(year),
                    "month": [month],
                    "day": days,
                    "daily_statistic": "daily_mean",
                    "time_zone": "utc+00:00",
                    "frequency": "6_hourly",
                    "area": [90, -180, 0, 180],
                },
                output_file
            )
        except Exception as e:
            print(f"Error while downloading {year}-{month}: {e}")

