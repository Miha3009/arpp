import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

base_url = "https://www.globsnow.info/swe/archive_v3.0/L3A_daily_SWE/NetCDF4/"
output_dir = "../../data/raw/globsnow_archive_v3"
os.makedirs(output_dir, exist_ok=True)

def get_file_links(base_url):
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "html.parser")
    links = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href.endswith(".nc"):
            links.append(urljoin(base_url, href))
    return links

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

file_links = get_file_links(base_url)
print(f"Found: {len(file_links)}")
for url in file_links:
    download_file(url)

