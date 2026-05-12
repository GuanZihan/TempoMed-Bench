import os
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
# BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/xml/"
# DOWNLOAD_DIR = "./pmc_oa_noncomm_xml_2026"

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"
DOWNLOAD_DIR = "pubmed_updatefiles_xml_2026"

MAX_WORKERS = 4  # number of parallel downloads

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def list_tar_files(base_url):
    """Parse PMC FTP directory HTML listing and extract all .tar.gz links"""
    resp = requests.get(base_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    files = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and (href.endswith(".tar.gz") or href.endswith(".gz")):
            files.append(base_url + href)
    return files

def download_file(url):
    """Download one file with resume support"""
    local_path = os.path.join(DOWNLOAD_DIR, os.path.basename(url))
    temp_path = local_path + ".part"
    headers = {}

    if os.path.exists(temp_path):
        downloaded = os.path.getsize(temp_path)
        headers["Range"] = f"bytes={downloaded}-"
    else:
        downloaded = 0

    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        mode = "ab" if downloaded > 0 else "wb"
        with open(temp_path, mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    os.rename(temp_path, local_path)
    print(f"Finished: {local_path}")

def main():
    files = list_tar_files(BASE_URL)
    # print(files)
    print(f"Found {len(files)} files to download.")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(download_file, files)

if __name__ == "__main__":
    main()

