import pickle
import os
import re
from data_utils.parse_pdf.parse_pdf import parse_xml_to_dict
from grobid_client.grobid_client import GrobidClient
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
# from data_utils.collection.multi_column import column_boxes


def generate_xml(pdf_path: str, out_dir: str, processed_dir: str):
    """
    Generate XML files from PDFs using Grobid.

    Args:
        pdf_path (str): The path to the directory containing the PDF files.
        out_dir (str): The output directory where the generated XML files will be saved.

    Returns:
        None
    """

    if pdf_path[-1] == "/":
        pdf_path = pdf_path[:-1]
    if out_dir[-1] == "/":
        out_dir = out_dir[:-1]

    raw_files = os.listdir(pdf_path)
    xmls = [filename.split(".")[0] for filename in os.listdir(out_dir) if filename.split(".")[-1] == "xml"]

    for file in raw_files:
        id = file.split(".")[0]
        if id in xmls:
            os.rename(pdf_path + "/" + file, processed_dir + "/" + file)

    print("Parsing PDFs in: ", pdf_path)

    client = GrobidClient(config_path="./config.json")
    client.process(
        "processFulltextDocument",
        pdf_path,
        output=out_dir,
        consolidate_citations=True,
        tei_coordinates=True,
        force=True
    )


def xmls_to_dict(in_dir: str, out_dir:str):
    """
    Convert a directory of XML files to a dictionary and save them as pickle files.

    Args:
        in_dir (str): The input directory containing XML files.
        out_dir (str): The output directory to save the pickle files.

    Returns:
        None
    """
    for file in tqdm(os.listdir(in_dir), desc="Converting XMLs to dicts"):
        if file.endswith(".tei.xml"):
            # print("Parsing: ", file)
            paper_dict = parse_xml_to_dict(in_dir + file)
            pickle.dump(paper_dict, open(out_dir + file[:-15] + ".pkl", "wb"))



