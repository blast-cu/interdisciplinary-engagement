import re
import os
import os.path as op
from glob import glob
import urllib
import subprocess
import requests
from bs4 import BeautifulSoup, NavigableString
from tqdm import tqdm, tqdm_notebook



DIR_PATH = op.dirname(op.abspath(__file__))
PDF_FIGURES_JAR_PATH = op.join(
    DIR_PATH, "pdffigures2", "pdffigures2-assembly-0.0.12-SNAPSHOT.jar"
)


def list_pdf_paths(pdf_folder: str):
    """
    list of pdf paths in pdf folder
    """
    return glob(op.join(pdf_folder, "*", "*", "*.pdf"))


def parse_xml(
    xml_path: str,
    fulltext: bool = True,
    soup: bool = False,
    return_coordinates: bool = True
):
    """
    TODO
    """
    file = open(xml_path, "r")
    parsed_article = file.read()
    xml_article = ""
    if soup and parsed_article is not None:
        lxml_parsed_article = BeautifulSoup(parsed_article, "lxml")
        xml_article = BeautifulSoup(parsed_article, "xml")
    

    return lxml_parsed_article, xml_article


def parse_authors(article):
    """
    Parse authors from a given BeautifulSoup of an article
    """
    author_names = article.find("sourcedesc").findAll("persname")
    authors = []
    for author in author_names:
        firstname = author.find("forename", {"type": "first"})
        firstname = firstname.text.strip() if firstname is not None else ""
        middlename = author.find("forename", {"type": "middle"})
        middlename = middlename.text.strip() if middlename is not None else ""
        lastname = author.find("surname")
        lastname = lastname.text.strip() if lastname is not None else ""
        if middlename != "":
            authors.append(firstname + " " + middlename + " " + lastname)
        else:
            authors.append(firstname + " " + lastname)
    authors = "; ".join(authors)
    return authors


def parse_date(article):
    """
    Parse date from a given BeautifulSoup of an article
    """
    pub_date = article.find("publicationstmt")
    year = pub_date.find("date")
    year = year.attrs.get("when") if year is not None else ""
    return year


def parse_abstract(article):
    """
    Parse abstract from a given BeautifulSoup of an article
    """
    div = article.find("abstract")
    abstract = ""
    for p in list(div.children):
        if not isinstance(p, NavigableString) and len(list(p)) > 0:
            abstract += " ".join(
                [elem.text for elem in p if not isinstance(elem, NavigableString)]
            )
    return abstract


def calculate_number_of_references(div):
    """
    For a given section, calculate number of references made in the section
    """
    refs = [ref for ref in div.find_all("ref") if ref.attrs.get("type") == "bibr" if ref.text.strip() != ""]

    clean_refs = []
    context = []
    p_context = []
    for ref in refs:
        # get context
        temp_refs = [ref.text for ref in refs]
        parent_paragraph = ref.parent.text  # paragraph
        char_index = parent_paragraph.find(ref.text)
        parent = parent_paragraph[:char_index]
        for t in temp_refs:
            parent = parent.replace(t, "")
        parent = parent.strip()
        if parent.endswith("."):
            parent = parent[:-1]

        parent_split = parent.split(".")
        parent = parent_split[-1]
        parent_split = parent_split[:-1]
        parent = parent.strip()
        while parent == "":
            try:
                parent = parent_split[-1]
                parent_split = parent_split[:-1]
                parent = parent.strip()
            except:
                # print(ref.text)
                # print(ref.parent.text)
                parent = ""
                break

        if parent != "":
            clean_refs.append(ref)
            context.append(parent)
            p_context.append(parent_paragraph)
    
    refs = [ref.text for ref in clean_refs]
    # refs = [re.sub('[();]', '', ref) for ref in refs]
    ref_nav_ids = []
    for r in clean_refs:
        target = r.get("target")
        if target:
            target = target.replace("#", "")
        ref_nav_ids.append(target)

    assert len(refs) == len(ref_nav_ids), "incorrect match between number of refs and ref_nav_ids"
    n_publication_ref = len(
        refs
    )
    n_figure_ref = len(
        [ref for ref in div.find_all("ref") if ref.attrs.get("type") == "figure"]
    )
    return {"n_publication_ref": n_publication_ref, "n_figure_ref": n_figure_ref, "refs": refs, "contexts": context, "p_contexts": p_context, "ref_nav_id": ref_nav_ids}


def parse_sections(article, xml_article, as_list: bool = False):
    """
    Parse list of sections from a given BeautifulSoup of an article

    Parameters
    ==========
    as_list: bool, if True, output text as a list of paragraph instead
        of joining it together as one single text
    """
    id_dict = {}
    xml_article = xml_article.find("text")
    id_divs = xml_article.find_all("div", attrs={"xmlns": "http://www.tei-c.org/ns/1.0"})
    for div in id_divs:
        id_div = div.find("head")
        if id_div is not None and 'n' in id_div.attrs:
            id_text = id_div.text
            div_id = id_div.attrs['n']
            id_dict[id_text] = div_id

    article_text = article.find("text")
    divs = article_text.find_all("div", attrs={"xmlns": "http://www.tei-c.org/ns/1.0"})
    sections = []
    for div in divs:
    
        div_list = list(div.children)

        if len(div_list) == 0:
            heading = ""
            text = ""
        elif len(div_list) == 1:
            if isinstance(div_list[0], NavigableString):
                heading = str(div_list[0])
                text = ""
            else:
                heading = ""
                text = div_list[0].text
        else:
            text = []
            
            heading = div_list[0]
            # exit()
            if isinstance(heading, NavigableString):
                
                heading = str(heading)
                p_all = list(div.children)[1:]
            else:
                heading = ""
                p_all = list(div.children)
            for p in p_all:
                if p is not None:
                    try:
                        text.append(p.text)
                    except:
                        pass
            if not as_list:
                text = "\n".join(text)
        if heading in id_dict:
            section_id = id_dict[heading]
            # print(section_id)
        else:
            section_id = ""

        if heading != "" or text != "":
            ref_dict = calculate_number_of_references(div)
            sections.append(
                {
                    "heading": heading,
                    "id": section_id,
                    "text": text,
                    "n_publication_ref": ref_dict["n_publication_ref"],
                    "n_figure_ref": ref_dict["n_figure_ref"],
                    "refs": ref_dict['refs'],
                    "contexts": ref_dict['contexts'],
                    "p_contexts": ref_dict['p_contexts'],
                    "nav_id": ref_dict["ref_nav_id"]
                }
            )
    return sections


def parse_references(article):
    """
    Parse list of references from a given BeautifulSoup of an article
    """
    reference_list = []
    references = article.find("text").find("div", attrs={"type": "references"})
    references = references.find_all("biblstruct") if references is not None else []
    reference_list = []
    for reference in references:
        title = reference.find("title", attrs={"level": "a"})
        if title is None:
            title = reference.find("title", attrs={"level": "m"})
        title = title.text if title is not None else ""

        journal = reference.find("title", attrs={"level": "j"})
        journal = journal.text if journal is not None else ""


        meeting = reference.find("meeting").text if reference.find("meeting") is not None else ""
        if meeting == "":
            meeting = reference.find("title", attrs={"level": "m"})
            meeting = meeting.text if meeting is not None else ""

        year = reference.find("date")
        year = year.attrs.get("when") if year is not None else ""
        authors = []
        first = True
        first_author = ""
        for author in reference.find_all("author"):
            
            firstname = author.find("forename", {"type": "first"})
            firstname = firstname.text.strip() if firstname is not None else ""
            middlename = author.find("forename", {"type": "middle"})
            middlename = middlename.text.strip() if middlename is not None else ""
            lastname = author.find("surname")
            lastname = lastname.text.strip() if lastname is not None else ""

            if middlename is not "":
                full_name = firstname + " " + middlename + " " + lastname
            else:
                full_name = firstname + " " + lastname
            if first:
                first_author = full_name
                first = False
            authors.append(full_name)

        authors = "; ".join(authors)

        # get nav id used to nav from in-text citation to ref section
        nav_id = reference.get("xml:id")

        # get id
        id_type = ""
        id = ""
        analytic = reference.find("analytic")
        if analytic is not None:
            id_div = analytic.find("idno", recursive=False)
            if id_div is not None:
                id_type = id_div.attrs.get("type")
                if id_type is not None:
                    id_type = re.sub('[^0-9a-zA-Z]+', '', id_type)
                    id_type = id_type.strip()
                else:
                    id_type = ""
                id = id_div.text
                if id is None:
                    id = ""
            else:
                id_type = ""
                id = ""
    
        reference_list.append(
            {"title": title, "journal": journal, "meeting": meeting, "year": year, "authors": authors, "id_type": id_type, "id": id, "nav_id": nav_id}
        )
    return reference_list


def parse_figure_caption(article):
    """
    Parse list of figures/tables from a given BeautifulSoup of an article
    """
    figures_list = []
    figures = article.find_all("figure")
    for figure in figures:
        figure_type = figure.attrs.get("type") or ""
        figure_id = figure.attrs.get("xml:id") or ""
        label = figure.find("label").text
        if figure_type == "table":
            caption = figure.find("figdesc").text
            data = figure.table.text
        else:
            caption = figure.text
            data = ""
        figures_list.append(
            {
                "figure_label": label,
                "figure_type": figure_type,
                "figure_id": figure_id,
                "figure_caption": caption,
                "figure_data": data,
            }
        )
    return figures_list


def parse_formulas(article):
    """
    Parse list of formulas from a given BeautifulSoup of an article
    """
    formulas_list = []
    formulas = article.find_all("formula")
    for formula in formulas:
        formula_id = formula.attrs["xml:id"] or ""
        formula_text = formula.text
        formula_coordinates = formula.attrs.get("coords") or ""
        if formula_coordinates != "":
            formula_coordinates = [float(x) for x in formula_coordinates.split(",")]
            formulas_list.append(
                {
                    "formula_id": formula_id,
                    "formula_text": formula_text,
                    "formula_coordinates": formula_coordinates,
                }
            )
    return formulas_list


def convert_article_soup_to_dict(article, xml_article, as_list: bool = False):
    """
    Function to convert BeautifulSoup to JSON format
    similar to the output from https://github.com/allenai/science-parse/

    Parameters
    ==========
    article: BeautifulSoup

    Output
    ======
    article_json: dict, parsed dictionary of a given article in the following format
        {
            'title': ...,
            'abstract': ...,
            'sections': [
                {'heading': ..., 'text': ...},
                {'heading': ..., 'text': ...},
                ...
            ],
            'references': [
                {'title': ..., 'journal': ..., 'year': ..., 'authors': ...},
                {'title': ..., 'journal': ..., 'year': ..., 'authors': ...},
                ...
            ],
            'figures': [
                {'figure_label': ..., 'figure_type': ..., 'figure_id': ..., 'figure_caption': ..., 'figure_data': ...},
                ...
            ]
        }
    """
    article_dict = {}
    if article is not None:
        title = article.find("title", attrs={"type": "main"})
        title = title.text.strip() if title is not None else ""

        article_dict["title"] = title
        article_dict["authors"] = parse_authors(article)
        article_dict["pub_date"] = parse_date(article)
        article_dict["abstract"] = parse_abstract(article)
        article_dict["sections"] = parse_sections(article, xml_article, as_list=as_list)
        article_dict["references"] = parse_references(article)
        article_dict["figures"] = parse_figure_caption(article)
        article_dict["formulas"] = parse_formulas(article)

        doi = article.find("idno", attrs={"type": "DOI"})
        doi = doi.text if doi is not None else ""
        article_dict["doi"] = doi

        return article_dict
    else:
        return None


def parse_xml_to_dict(
    pdf_path: str,
    fulltext: bool = True,
    soup: bool = True,
    as_list: bool = False,
    return_coordinates: bool = True
):
    """
    Parse the given PDF and return dictionary of the parsed article

    Parameters
    ==========
    pdf_path: str, path to publication or article
    fulltext: bool, whether to extract fulltext or not
    soup: bool, whether to return BeautifulSoup or not
    as_list: bool, whether to return list of sections or not

    Ouput
    =====
    article_dict: dict, dictionary of an article
    """
    lxml_parsed_article, xml_article = parse_xml(
        pdf_path,
        fulltext=fulltext,
        soup=soup,
        return_coordinates=return_coordinates
    )
    article_dict = convert_article_soup_to_dict(lxml_parsed_article, xml_article, as_list=as_list)
    return article_dict


def parse_figures(
    pdf_folder: str,
    jar_path: str = PDF_FIGURES_JAR_PATH,
    resolution: int = 300,
    output_folder: str = "figures",
):
    """
    Parse figures from the given scientific PDF using pdffigures2

    Parameters
    ==========
    pdf_folder: str, path to a folder that contains PDF files. A folder must contains only PDF files
    jar_path: str, default path to pdffigures2-assembly-0.0.12-SNAPSHOT.jar file
    resolution: int, resolution of the output figures
    output_folder: str, path to folder that we want to save parsed data (related to figures) and figures

    Output
    ======
    folder: making a folder of output_folder/data and output_folder/figures of parsed data and figures relatively
    """
    if not op.isdir(output_folder):
        os.makedirs(output_folder)

    # create ``data`` and ``figures`` subfolder within ``output_folder``
    data_path = op.join(output_folder, "data")
    figure_path = op.join(output_folder, "figures")
    if not op.exists(data_path):
        os.makedirs(data_path)
    if not op.exists(figure_path):
        os.makedirs(figure_path)

    if op.isdir(data_path) and op.isdir(figure_path):
        args = [
            "java",
            "-jar",
            jar_path,
            pdf_folder,
            "-i",
            str(resolution),
            "-d",
            op.join(op.abspath(data_path), ""),
            "-m",
            op.join(op.abspath(figure_path), ""),  # end path with "/"
        ]
        _ = subprocess.run(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20
        )
        print("Done parsing figures from PDFs!")
    else:
        print(
            "You may have to check of ``data`` and ``figures`` in the the output folder path."
        )
