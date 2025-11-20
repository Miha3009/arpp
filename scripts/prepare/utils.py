import pandas as pd
from datetime import datetime

def get_weeks():
    dates = pd.date_range(f"2001-01-01", f"2001-12-30", freq="D")
    weeks = {}

    week_num = 1
    day_count = 0
    for ts in dates:
        key = f"{ts.month:02d}{ts.day:02d}"
        weeks[key] = week_num
        day_count += 1
        if day_count == 7:
            week_num += 1
            day_count = 0
    return weeks

def get_week_dates(year, week):
    result = []
    for date, week_num in weeks.items():
        if week_num == week:
            result.append(datetime(year, int(date[:2]), int(date[2:])))
    return result

weeks = get_weeks()

month_elements = ["aice", "cld", "h500", "hice", "olr", "prec", "ps", "rq2", "sst",
                  "t2", "t2max", "t2min", "t850", "u850", "uv10", "v850", "ws", "ww"]
week_elements = ["swe", "cld", "h500", "olr", "prec", "ps", "rq2", "t2", "t2max", "t2min", "t850",
                 "u850", "uv10", "v850", "ws", "ww"]


