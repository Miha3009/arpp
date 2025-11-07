import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

base_url = "https://noaadata.apps.nsidc.org/NOAA/G02202_V6/north/daily/"
output_dir = "../../data/raw/noaa_cdr_ice"
os.makedirs(output_dir, exist_ok=True)

def get_file_links(year_url):
    response = requests.get(year_url)
    soup = BeautifulSoup(response.text, "html.parser")
    file_links = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href.endswith('.nc'):
            file_links.append(urljoin(year_url, href))
    return file_links

def download_file(url):
    filename = os.path.join(output_dir, os.path.basename(url))
    if os.path.exists(filename):
        return
    print(f"Download: {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

for year in range(1991, 2026):
    file_links = get_file_links(base_url + str(year) + '/')
    for url in file_links:
        download_file(url)
    print(f"Year {year} completed")
