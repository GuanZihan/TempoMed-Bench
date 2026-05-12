import os
from lxml import etree
from tqdm import tqdm
import json


def extract_guidelines_from_file(xml_file):
    """
    Parse ONE XML file (which contains many PubmedArticle entries)
    and return PMID -> publication types mapping
    ONLY if 'guideline' appears.
    """

    pmid_to_types = {}

    # iterate per PubmedArticle (NOT per file)
    context = etree.iterparse(
        xml_file,
        events=("end",),
        tag="PubmedArticle"
    )

    for _, article in context:

        # print(article)
        # input()

        # 1️⃣ Extract PMID (Version="1")
        pmid_elem = article.find(".//PMID[@Version='1']")
        if pmid_elem is None:
            article.clear()
            continue

        pmid = pmid_elem.text.strip()

        # 2️⃣ Extract publication types
        publication_types = [
            pt.text.strip()
            for pt in article.findall(".//PublicationTypeList/PublicationType")
            if pt.text
        ]

        # 3️⃣ Check if contains guideline
        if any("guideline" in pt.lower() for pt in publication_types):
            pmid_to_types[pmid] = publication_types

        # 4️⃣ Free memory
        article.clear()
        while article.getprevious() is not None:
            del article.getparent()[0]
        
        

    del context
    return pmid_to_types


def extract_guidelines_from_directory(directory):
    all_results = {}

    xml_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".xml")
    ]

    for xml_file in tqdm(xml_files, desc=f"Processing {directory}"):
        result = extract_guidelines_from_file(xml_file)
        all_results.update(result)

    return all_results


# ---- MAIN ----
baseline_dir = "pubmed_baseline_xml_extracted_2026"
update_dir = "pubmed_updatefiles_xml_extracted_2026"

baseline_results = extract_guidelines_from_directory(baseline_dir)
update_results = extract_guidelines_from_directory(update_dir)

pmid_guideline_mapping = {**baseline_results, **update_results}

print("Total guideline articles:", len(pmid_guideline_mapping))

with open("pmid_guideline_mapping.json", "w") as f:
    json.dump(pmid_guideline_mapping, f)

print("Saved to pmid_guideline_mapping.json")
