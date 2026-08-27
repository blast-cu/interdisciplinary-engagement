import argparse
from data_utils.parse_pdf.utils import generate_xml, xmls_to_dict

def dir_parse(args):

    base_dir = args.dir
    raw_dir = f"{base_dir}/raw/"
    xml_out_dir = f"{base_dir}/xml/"
    processed_dir = f"{base_dir}/processed/"
    dict_out_dir = f"{base_dir}/dict/"

    # generate_xml(
    #     pdf_path=raw_dir,
    #     out_dir=xml_out_dir,
    #     processed_dir=processed_dir
    # )

    xmls_to_dict(xml_out_dir, dict_out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse PDF files in specified directory.")
    parser.add_argument("--dir", type=str, help="Input PDF file or directory")
    args = parser.parse_args()
    dir_parse(args)