
import pickle
import os
from tqdm import tqdm

import fitz
import requests
from bs4 import BeautifulSoup

from src.db_utils import clean_title, create_connection

def update_nlp_css_label(db_path: str, title: str, venue: str, year: int, nlp_css: int):
    """
    Update the nlp_css label for a given title in the database.
    """
    con, cur = create_connection(db_path)
    title_clean = clean_title(title)
    update_query = f"UPDATE article SET nlp_css = {nlp_css} WHERE clean_title = '{title_clean}' AND venue = '{venue}' AND year = {year}"
    cur.execute(update_query)
    con.commit()
    con.close()
    return

def get_paper_link(venue, year, venue_url, paper_title):
    clean_paper_title = clean_title(paper_title)

    # Requests URL and get response object
    response = requests.get(venue_url)
    
    # Parse text obtained
    og_soup = BeautifulSoup(response.text, 'html.parser')
    main_soup = og_soup.find(attrs={'id': 'main'})

    # this method works for acl 2014 - 2020
    # if year <= 2020:
    soup = main_soup.find_all('a', attrs={'class': 'align-middle'})
    if soup is None:
        raise AttributeError
    
    for s in soup:
        # first check to see if there are any 'acl-fixed-case' divs
        # fc_divs = s.find_all('span', attrs={'class': 'acl-fixed-case'})
        # if len(fc_divs) > 0:
        #     print(s.text)
        #     print(fc_divs)
        #     exit()
        # else:
        div_title = s.text
        div_clean_title = clean_title(div_title)

        if div_clean_title == clean_paper_title:
            link_end = s['href']
            pdf_link = "https://aclanthology.org" + link_end[:-1] + ".pdf"
            return pdf_link
    
    # else:

        # soup = main_soup.find_all('p', attrs={'class': 'd-sm-flex align-items-stretch'})
        
        # # Find all hyperlinks present on webpage
        # for s in soup:
        #     href_div = s.find('a', attrs={'class': 'align-middle', 'data-toggle': True})
        #     title_div = s.find('a', attrs={'class': 'align-middle', 'data-toggle': False})
            
        #     if href_div and title_div:
        #         curr_title = title_div.text
        #         curr_title = clean_title(curr_title)

        #         if curr_title == clean_paper_title: 
        #             return href_div['href']
        

    return None

def download_paper_by_title(
        urls: list,
        paper_title: str,
        article_id: int,
        out_dir:str,
        venue: str,
        year: int,
        logger
):
    """
    Downloads a paper by its title from a given URL and saves it as a PDF file.

    Args:
        url (str): The URL of the webpage containing the paper.
        title (str): The title of the paper to be downloaded.
        id (int): The unique identifier of the paper.
        out_dir (str): The directory where the downloaded PDF file will be saved.

    Raises:
        ValueError: If the paper with the given title is not found on the webpage.

    Returns:
        None
    """
    for venue_url in urls:
        paper_link = get_paper_link(venue, year, venue_url, paper_title)
        if paper_link:
            break

    if paper_link is None:
        raise ValueError(f"Paper with title '{paper_title}' not found in the conference '{venue} {year}'")

    response = requests.get(paper_link)

    # Write content in pdf file
    filename = out_dir + str(article_id) + ".pdf"
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
    else:
        logger.error(f"Could not download pdf with title '{paper_title}' at link ({paper_link})")
        raise IOError


def download_papers(
        url: str,
        out_dir:str,
        download: bool = True
):
    """
    Downloads PDF papers from a given URL and saves them to the specified output directory.

    Args:
        url (str): The URL of the webpage containing the papers.
        out_dir (str): The directory where the downloaded papers will be saved.

    Returns:
        None
    """
    return_dict = {}

    # Requests URL and get response object
    response = requests.get(url)
    
    # Parse text obtained
    soup = BeautifulSoup(response.text, 'html.parser')
    soup = soup.find(attrs={'id': 'main'})
    soup = soup.find_all('p', attrs={'class': 'd-sm-flex align-items-stretch'})
    
    # Find all hyperlinks present on webpage
    papers = {}
    curr_id = 0
    for s in soup:
        href_div = s.find('a', attrs={'class': 'align-middle', 'data-toggle': True})
        title_div = s.find('a', attrs={'class': 'align-middle', 'data-toggle': False})
        
        if href_div and title_div:
            text = title_div.text
            if "Proceedings of the" not in text:
                papers[curr_id] = {}
                papers[curr_id]['link'] = href_div['href']
                papers[curr_id]['title'] = text
                curr_id += 1

    # From all links check for pdf link and
    # if present download file
    for id in tqdm(papers.keys(), desc="Downloading papers"):
        
        link = papers[id]['link']
        title = papers[id]['title']


        if ('.pdf' in link):
            if download:
                # print("Downloading file: ", title)

                # Get response object for link
                response = requests.get(link)

                # Write content in pdf file
                pdf = open(out_dir + str(id) + ".pdf", 'wb')
                pdf.write(response.content)
                pdf.close()

            return_dict[id] = {}
            return_dict[id]['title'] = title
    
    return return_dict


def parse_pdf_abstract(
        pdf_path: str
):
    """
    Parse the abstract from a PDF file.

    Args:
        file (str): The path to the PDF file.

    Returns:
        str: The abstract of the paper.
    """

    paper_dict = {}
    id = 0
    paper_dict[id] = {}

    # Open the PDF
    doc = fitz.open(pdf_path)
    page = doc[0]

    bboxes = column_boxes(page, footer_margin=50, no_image_text=True)
    rects = bboxes
    text = [page.get_text(clip=rect, sort=True).strip() for rect in rects]
    text = "\n".join(text)

    try:
        text = text.split("\nAbstract")[1]
    except Exception as e:
        # print(e)
        # print("Abstract not found in PDF")
        # print(text)
        pass


    text = text.replace("-\n", "")
    text = " ".join(text.split())

    try: 
        text = text.split("1 Introduction")[0]
    except Exception as e:
        # print(e)
        # print("Introduction not found in PDF")
        # print(text)
        pass
        
    text = text.strip()
    doc.close()

    return text


def get_abstracts(
        in_dir: str,
        out_dir: str,
        abstract_dict: dict
):
    abstract_file = f"{out_dir}/abstracts.pkl"
    parse_failed = 0
    for file in tqdm(os.listdir(in_dir), desc="Parsing abstracts"):
        if file.endswith(".pdf"):
            # print("Parsing abstract from file: ", file)
            index = file.split(".")[0]
            index = int(index)
            abstract = parse_pdf_abstract(in_dir + file)
            if abstract == "":
                parse_failed += 1
            abstract_dict[index]['abstract'] = abstract

    print(f"Failed to parse: {parse_failed}/{len(abstract_dict)}")
    pickle.dump(abstract_dict, open(abstract_file, "wb"))


def column_boxes(page, footer_margin=50, header_margin=50, no_image_text=True):
    """Determine bboxes which wrap a column."""
    paths = page.get_drawings()
    bboxes = []

    # path rectangles
    path_rects = []

    # image bboxes
    img_bboxes = []

    # bboxes of non-horizontal text
    # avoid when expanding horizontal text boxes
    vert_bboxes = []

    # compute relevant page area
    clip = +page.rect
    clip.y1 -= footer_margin  # Remove footer area
    clip.y0 += header_margin  # Remove header area

    def can_extend(temp, bb, bboxlist):
        """Determines whether rectangle 'temp' can be extended by 'bb'
        without intersecting any of the rectangles contained in 'bboxlist'.

        Items of bboxlist may be None if they have been removed.

        Returns:
            True if 'temp' has no intersections with items of 'bboxlist'.
        """
        for b in bboxlist:
            if not intersects_bboxes(temp, vert_bboxes) and (
                b == None or b == bb or (temp & b).is_empty
            ):
                continue
            return False

        return True

    def in_bbox(bb, bboxes):
        """Return 1-based number if a bbox contains bb, else return 0."""
        for i, bbox in enumerate(bboxes):
            if bb in bbox:
                return i + 1
        return 0

    def intersects_bboxes(bb, bboxes):
        """Return True if a bbox intersects bb, else return False."""
        for bbox in bboxes:
            if not (bb & bbox).is_empty:
                return True
        return False

    def extend_right(bboxes, width, path_bboxes, vert_bboxes, img_bboxes):
        """Extend a bbox to the right page border.

        Whenever there is no text to the right of a bbox, enlarge it up
        to the right page border.

        Args:
            bboxes: (list[IRect]) bboxes to check
            width: (int) page width
            path_bboxes: (list[IRect]) bboxes with a background color
            vert_bboxes: (list[IRect]) bboxes with vertical text
            img_bboxes: (list[IRect]) bboxes of images
        Returns:
            Potentially modified bboxes.
        """
        for i, bb in enumerate(bboxes):
            # do not extend text with background color
            if in_bbox(bb, path_bboxes):
                continue

            # do not extend text in images
            if in_bbox(bb, img_bboxes):
                continue

            # temp extends bb to the right page border
            temp = +bb
            temp.x1 = width

            # do not cut through colored background or images
            if intersects_bboxes(temp, path_bboxes + vert_bboxes + img_bboxes):
                continue

            # also, do not intersect other text bboxes
            check = can_extend(temp, bb, bboxes)
            if check:
                bboxes[i] = temp  # replace with enlarged bbox

        return [b for b in bboxes if b != None]

    def clean_nblocks(nblocks):
        """Do some elementary cleaning."""

        # 1. remove any duplicate blocks.
        blen = len(nblocks)
        if blen < 2:
            return nblocks
        start = blen - 1
        for i in range(start, -1, -1):
            bb1 = nblocks[i]
            bb0 = nblocks[i - 1]
            if bb0 == bb1:
                del nblocks[i]

        # 2. repair sequence in special cases:
        # consecutive bboxes with almost same bottom value are sorted ascending
        # by x-coordinate.
        y1 = nblocks[0].y1  # first bottom coordinate
        i0 = 0  # its index
        i1 = -1  # index of last bbox with same bottom

        # Iterate over bboxes, identifying segments with approx. same bottom value.
        # Replace every segment by its sorted version.
        for i in range(1, len(nblocks)):
            b1 = nblocks[i]
            if abs(b1.y1 - y1) > 10:  # different bottom
                if i1 > i0:  # segment length > 1? Sort it!
                    nblocks[i0 : i1 + 1] = sorted(
                        nblocks[i0 : i1 + 1], key=lambda b: b.x0
                    )
                y1 = b1.y1  # store new bottom value
                i0 = i  # store its start index
            i1 = i  # store current index
        if i1 > i0:  # segment waiting to be sorted
            nblocks[i0 : i1 + 1] = sorted(nblocks[i0 : i1 + 1], key=lambda b: b.x0)
        return nblocks

    # extract vector graphics
    for p in paths:
        path_rects.append(p["rect"].irect)
    path_bboxes = path_rects

    # sort path bboxes by ascending top, then left coordinates
    path_bboxes.sort(key=lambda b: (b.y0, b.x0))

    # bboxes of images on page, no need to sort them
    for item in page.get_images():
        img_bboxes.extend(page.get_image_rects(item[0]))

    # blocks of text on page
    blocks = page.get_text(
        "dict",
        flags=fitz.TEXTFLAGS_TEXT,
        clip=clip,
    )["blocks"]

    # Make block rectangles, ignoring non-horizontal text
    for b in blocks:
        bbox = fitz.IRect(b["bbox"])  # bbox of the block

        # ignore text written upon images
        if no_image_text and in_bbox(bbox, img_bboxes):
            continue

        if len(b["lines"]) == 0:  # no text lines
            # print("No text lines in block")
            continue

        # confirm first line to be horizontal
        line0 = b["lines"][0]  # get first line
        if line0["dir"] != (1, 0):  # only accept horizontal text
            vert_bboxes.append(bbox)
            continue

        srect = fitz.EMPTY_IRECT()
        for line in b["lines"]:
            lbbox = fitz.IRect(line["bbox"])
            text = "".join([s["text"].strip() for s in line["spans"]])
            if len(text) > 1:
                srect |= lbbox
        bbox = +srect

        if not bbox.is_empty:
            bboxes.append(bbox)

    # Sort text bboxes by ascending background, top, then left coordinates
    bboxes.sort(key=lambda k: (in_bbox(k, path_bboxes), k.y0, k.x0))

    # Extend bboxes to the right where possible
    bboxes = extend_right(
        bboxes, int(page.rect.width), path_bboxes, vert_bboxes, img_bboxes
    )

    # immediately return of no text found
    if bboxes == []:
        return []

    # --------------------------------------------------------------------
    # Join bboxes to establish some column structure
    # --------------------------------------------------------------------
    # the final block bboxes on page
    nblocks = [bboxes[0]]  # pre-fill with first bbox
    bboxes = bboxes[1:]  # remaining old bboxes

    for i, bb in enumerate(bboxes):  # iterate old bboxes
        check = False  # indicates unwanted joins

        # check if bb can extend one of the new blocks
        for j in range(len(nblocks)):
            nbb = nblocks[j]  # a new block

            # never join across columns
            if bb == None or nbb.x1 < bb.x0 or bb.x1 < nbb.x0:
                continue

            # never join across different background colors
            if in_bbox(nbb, path_bboxes) != in_bbox(bb, path_bboxes):
                continue

            temp = bb | nbb  # temporary extension of new block
            check = can_extend(temp, nbb, nblocks)
            if check == True:
                break

        if not check:  # bb cannot be used to extend any of the new bboxes
            nblocks.append(bb)  # so add it to the list
            j = len(nblocks) - 1  # index of it
            temp = nblocks[j]  # new bbox added

        # check if some remaining bbox is contained in temp
        check = can_extend(temp, bb, bboxes)
        if check == False:
            nblocks.append(bb)
        else:
            nblocks[j] = temp
        bboxes[i] = None

    # do some elementary cleaning
    nblocks = clean_nblocks(nblocks)

    # return identified text bboxes
    return nblocks