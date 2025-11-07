import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import shutil

base_url = "https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/"
station_url = "https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.csv"
output_dir = "../../data/raw/ghcn"
os.makedirs(output_dir, exist_ok=True).

def get_file_links(base_url, start_year=1991):
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "html.parser")
    links = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href.endswith(".csv.gz"):
            year = href.split(".")[0]
            if year.isdigit() and int(year) >= start_year:
                links.append(urljoin(base_url, href))
    return links

def download_and_extract(url):
    filename = os.path.basename(url)
    path = os.path.join(output_dir, filename)

    if os.path.exists(path):
        return

    print(f"Download: {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

file_links = get_file_links(base_url)
print(f"Found: {len(file_links)}")
for url in file_links:
    download_and_extract(url)
download_and_extract(station_url)
