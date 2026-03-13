import cdsapi
import os

output_dir = "../../data/raw/swe_seas"
os.makedirs(output_dir, exist_ok=True)

dataset = "seasonal-original-single-levels"
years = list(range(2024, 2026))
months = [f"{m:02d}" for m in range(1, 13)]
leadtime_hour = [str(i) for i in range(24, 3000, 24)]

client = cdsapi.Client(sleep_max=5)

for year in years:
    for month in months:
        output_file = os.path.join(output_dir, f"seas_snow_depth_{year}_{month}.nc")
        if os.path.exists(output_file):
            continue

        print(f"Download {year}-{month}")

        try:
            client.retrieve(
                dataset,
                {
                    "originating_centre": "ecmwf",
                    "system": "51",
                    "variable": ["snow_depth"],
                    "year": str(year),
                    "month": [month],
                    "day": ["01"],
                    "leadtime_hour": leadtime_hour,
                    "area": [50, 27, 65, 56]
                },
                output_file
            )
        except Exception as e:
            print(f"Error while downloading {year}-{month}: {e}")

