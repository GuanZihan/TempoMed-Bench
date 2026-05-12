import os
import tarfile
from tqdm import tqdm
from lxml import etree
import requests
import time
import json


DOWNLOAD_DIR = "./pmc_oa_comm_xml_2026"
EXTRACT_DIR = "./pmc_oa_comm_extracted_2026_relaxed"
DELETE_AFTER_EXTRACT = False

count_review = 0
count_guideline = 0
count_total_articles = 0
with open("pmid_guideline_mapping.json", "r") as f:
    pmid_guideline_mapping = json.load(f)

print(len(pmid_guideline_mapping))

def pmcid_to_pmid(pmcid, max_retries=5, backoff_factor=1.5, session=None):
    """
    Convert PMCID to PMID using NCBI API
    with retry and exponential backoff.
    """

    url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    params = {
        "ids": pmcid,
        "format": "json"
    }

    for attempt in range(max_retries):
        try:
            if session:
                response = session.get(url, params=params, timeout=10)
            else:
                response = requests.get(url, params=params, timeout=10)

            response.raise_for_status()

            data = response.json()
            records = data.get("records", [])

            if records and "pmid" in records[0]:
                return records[0]["pmid"]

            return None

        except (requests.exceptions.RequestException, ValueError) as e:
            wait_time = backoff_factor ** attempt
            print(f"⚠️ Retry {attempt+1}/{max_retries} for {pmcid}, waiting {wait_time:.2f}s")

            time.sleep(wait_time)

    print(f"❌ Failed to convert {pmcid} after {max_retries} retries.")
    return None


def is_guideline_article(xml_path):
    global count_review
    global count_guideline

    try:
        tree = etree.parse(xml_path)
        root = tree.getroot()

        article_type = root.attrib.get("article-type", "").lower()

        title_node = tree.find(".//front/article-meta/title-group/article-title")
        title = ""
        if title_node is not None:
            title = "".join(title_node.itertext()).strip()

        if "guideline" in article_type:
            count_guideline += 1
            return True

        if "guideline" in title.lower():
            count_review += 1
            return True

        subjects = tree.findall(".//article-categories//subject")
        for subj in subjects:
            text = "".join(subj.itertext()).lower()
            if "guideline" in text:
                count_guideline += 1
                return True

        pmid_node = tree.find(".//article-id[@pub-id-type='pmid']")
        pmid = None

        if pmid_node is not None:
            pmid = pmid_node.text.strip()
        else:
            pmcid_node = tree.find(".//article-id[@pub-id-type='pmc']")
            if pmcid_node is not None:
                pmcid = pmcid_node.text.strip()
                pmid = pmcid_to_pmid(pmcid)
                # print(pmid, pmcid)
        if pmid and pmid in pmid_guideline_mapping:
            count_guideline += 1
            print('!')
            return True

        return False

    except Exception as e:
        print(f"⚠️ {xml_path} Parsing Fail: {e}")
        return False


def extract_and_filter_tar(tar_path, extract_dir):
    global count_total_articles
    kept = 0
    skipped = 0
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
            members_to_extract = []

            for member in tqdm(members, desc=f"Unzip {os.path.basename(tar_path)}", leave=False):
                if not member.name.endswith(".xml"):
                    print(member.name)
                    continue
                count_total_articles += 1

                file_obj = tar.extractfile(member)
                
                if file_obj is None:
                    print("!!!!")
                    skipped += 1
                    continue

                try:
                    if is_guideline_article(file_obj):
                        pmc_dir = member.name.split("/")[0]
                        members_to_extract.append(member)
                        kept += 1
                    else:
                        skipped += 1
                except Exception as e:
                    print(e)
                    skipped += 1
                    continue
            tar.extractall(path=extract_dir, members=members_to_extract)

        print(f"✅ {os.path.basename(tar_path)}: Keep {kept} papers, Skip {skipped} papers")
        return kept

    except Exception as e:
        print(f"❌ Unzip Failing: {tar_path} - {e}")
        return 0


def main():
    tar_files = sorted([f for f in os.listdir(DOWNLOAD_DIR) if f.endswith(".tar.gz")], reverse=True)
    print(f"Found {len(tar_files)} gz files.")

    total_kept = 0
    for tar_name in tar_files:
        # if "incr" in tar_name or "PMC012" in tar_name or "PMC011" in tar_name:
        #     continue
        tar_path = os.path.join(DOWNLOAD_DIR, tar_name)
        print(f"\n🔍 Processing: {tar_name}")
        kept = extract_and_filter_tar(tar_path, EXTRACT_DIR)
        total_kept += kept

        if DELETE_AFTER_EXTRACT:
            os.remove(tar_path)
            print(f"🗑️ Deleted Original File: {tar_name}")

    print("\n📊 Extraction summary")
    print(f"   Total articles scanned : {count_total_articles}")
    print(f"   Guideline-title matches: {count_review}")
    print(f"   Guideline-type matches : {count_guideline}")
    print(f"   Total kept papers      : {total_kept}")

if __name__ == "__main__":
    main()
