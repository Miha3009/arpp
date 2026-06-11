#!/usr/bin/env python3
import os
import glob
import sys
import requests
from pathlib import Path
from tqdm import tqdm
from content import CONFIG, ELEMENTS, ELEMENTS_INFO

def download_file(file_id, output):
    resp = requests.get(f"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=gc{file_id}")
    resp.raise_for_status()
    download_url = resp.json()["href"]
    
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(output, 'wb') as f:
            with tqdm(total=total, unit='B', unit_scale=True, desc=output, ncols=80) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

def check_damaged():
    idx_files = set(Path(p).stem for p in glob.glob("*.idx"))
    bin_files = set(Path(p).stem for p in glob.glob("*.bin"))
    complete = [e for e in ELEMENTS.keys() if e in idx_files and e in bin_files]
    missing = [e for e in ELEMENTS.keys() if e not in idx_files and e not in bin_files]
    damaged = [e for e in ELEMENTS.keys() if e not in complete and e not in missing]
    return complete, damaged, missing

def format_elements_list(elements):
    return '\n'.join(f"  - {element}: {ELEMENTS_INFO[element]['name']} ({ELEMENTS_INFO[element]['units']})" for element in elements)

def download_elements():
    if not os.path.exists("config.bin"):
        download_file(CONFIG, "config.bin")

    complete, damaged, missing = check_damaged()
    if complete:
        print(f"Complete:\n{format_elements_list(complete)}")
    if not damaged and not missing:
        return
    if damaged:
        print(f"Damaged:\n{format_elements_list(damaged)}")
    if missing:
        print(f"Missing:\n{format_elements_list(missing)}")

    choice = input("\nEnter elements to download (space-separated, empty for all missing): ").strip()
    if choice:
        to_download = [e for e in choice.split() if e in ELEMENTS]
    else:
        to_download = missing + damaged
    if len(to_download) == 0:
        return

    for element in to_download:
        bin_path = f"{element}.bin"
        idx_path = f"{element}.idx"
        link_bin, link_idx = ELEMENTS[element]

        download_file(link_bin, bin_path)
        download_file(link_idx, idx_path)

if __name__ == "__main__":
    download_elements()
