import os
import re
import logging
import sqlite3
import pickle
from nameparser import HumanName

from src.db_utils import clean_title

def clean_string(s: str) -> str:
    if s is None:
        return "NULL"

    s = s.replace("\'", "'")
    s = s.replace("`", "'")
    s = clean_db_string(s) # remove pre-existing double '
    s = s.strip()
    return s

def clean_db_string(s:str) -> str:
    """
    remove '' from string in database
    """
    s = re.sub(r"'{2,}", "'", s)
    return s

def handle_none(value):
    if value is None:
        return 'NULL'
    return value

def fix_nulls(con, cur):
    """
    if any column contains 'NULL' strings, replace with actual NULL values
    """

    # get all columns in citations table
    cur.execute("PRAGMA table_info(citations)")
    columns = cur.fetchall()
    column_names = [col[1] for col in columns]

    for col in column_names:
        update_query = f"UPDATE citations SET {col} = NULL WHERE {col} = 'NULL' or {col} = 'None'"
        cur.execute(update_query)
    
    con.commit()


# def get_local_article_path(venue_data_dir, venue, year, id, format, logger: None, label=1):
    # if label == 1:
    #     path = os.path.join(venue_data_dir, venue, str(year), format, f"{pdf_id}.pkl")
    # else:  # unlabelled example
    #     path = os.path.join(venue_data_dir, venue, str(year), "predicted_pos", format, f"{pdf_id}.pkl")
                            

def get_local_article_path(venue_data_dir, venue, year, id, logger: None):

    # path = os.path.join(venue_data_dir, venue, str(year), "pos", "dict", f"{id}.pkl")
    path = os.path.join(venue_data_dir, "dict", f"{id}.pkl")
    if os.path.exists(path):
            return path
    else:
        if logger is not None:
            logger.error(f"Local article path does not exist: {path}")
        
        return None

SECTIONS = {
    'introduction': ['introduction'],
    'related work': ['related work', 'background', 'related research'],
    'method': ['method', 'methodology', 'approach', 'notation', 'technique', 'algorithm', 'architecture', 'design', 'solution', 'method', 'approach', 'methodology', 'model', 'corpus', 'data'],
    'experiments': ['experiment', 'evaluation', 'test', 'analysis', 'vs.', 'compare', 'accuracy', 'scores', 'our method vs.', 'state-of-the-art', 'baseline', 'results', 'performance', 'evaluation'],
    'conclusion': ['conclusion', 'future work', 'discussion', 'limitation', 'ethical consideration', 'ethics'],
    'appendix': ['appendix', 'appendices']

}

def combine_sections(prev, curr):

    prev_sec_heading = prev[1]
    prev_sec_refs = prev[2]
    prev_sec_id = prev[0]
    prev_sec_contexts = prev[3]
    prev_sec_p_contexts = prev[4]
    prev_sec_nav_ids = prev[5]
    prev_sec_text = prev[6]

    sec_title = prev_sec_heading + curr[1]
    sec_refs = prev_sec_refs + curr[2]
    sec_contexts = prev_sec_contexts + curr[3]
    sec_p_contexts = prev_sec_p_contexts + curr[4]
    sec_nav_ids = prev_sec_nav_ids + curr[5]
    sec_text = prev_sec_text + curr[6]
    
    
    return (prev_sec_id, sec_title, sec_refs, sec_contexts, sec_p_contexts, sec_nav_ids, sec_text)

def clean_sections(sections, logger):

    clean_list = []
    for i, s in enumerate(sections):
        sec_id = s['id']
        sec_heading = [s['heading'].lower().strip()]
        sec_refs = [s['refs']]

        sec_contexts = [s['contexts']]
        sec_p_contexts = [s['p_contexts']]
        sec_nav_ids = [s['nav_id']]
        sec_text = s['text']


        if 'appendix' in sec_heading[0] or 'appendices' in sec_heading[0]:
            appendix_heading = ['appendix']
            appendix_id = 'a'
            appendix_refs = []
            appendix_contexts = []
            appendix_p_contexts = []
            appendix_nav_ids = []
    
            for a in range(i, len(sections)):
                appendix_refs += [sections[a]['refs']]
                appendix_contexts += [sections[a]['contexts']]
                appendix_p_contexts += [sections[a]['p_contexts']]
                appendix_nav_ids += [sections[a]['nav_id']]

            if len(appendix_refs) != len(appendix_contexts):
                logger.error(f"Appendix refs and contexts not the same length: {len(appendix_refs)}, {len(appendix_contexts)}")
                exit()
            clean_list.append((appendix_id, appendix_heading, appendix_refs, appendix_contexts, appendix_p_contexts, appendix_nav_ids, sec_text))
            break

        elif sec_heading == "" and sec_refs == "":
            if sec_id != "":
                clean_list.append((sec_id, sec_heading, sec_refs, sec_contexts, sec_p_contexts, sec_nav_ids, sec_text))
            else:
                continue

        if i == 0:  # first section
            if sec_heading == "":
                sec_heading = "introduction"
            #     print("First section has no heading, setting to introduction")
            # elif sec_heading[0] != "introduction":
            #     print("First section is not introduction")
            if len(sec_refs) != len(sec_contexts):
                logger.error(f"Line 76 Refs and contexts not the same length: {len(sec_refs)}, {len(sec_contexts)}")
                exit()  
            clean_list.append((sec_id, sec_heading, sec_refs, sec_contexts, sec_p_contexts, sec_nav_ids, sec_text))

        elif sec_id == "":  # likely a paragraph?
            if re.findall(r"^[a-g].", sec_heading[0].strip()):
                if sec_refs != []:
                    sec_heading[0] = 'appendix ' + sec_heading[0]
                    clean_list.append((sec_id, sec_heading, sec_refs, sec_contexts, sec_p_contexts, sec_nav_ids, sec_text))
            else:
                clean_list[-1] = combine_sections(clean_list[-1], (sec_id, sec_heading, sec_refs, sec_contexts, sec_p_contexts, sec_nav_ids, sec_text))
    
        elif clean_list != []:  # not the first section
            split_id = sec_id.split('.')
            if len(split_id) == 1:  # first level section
                if split_id != clean_list[-1][0]:  # not the same section as prev
                    clean_list.append((sec_id, sec_heading, sec_refs, sec_contexts, sec_p_contexts, sec_nav_ids, sec_text))
                else:
                    logger.debug("detected top level but not same section as previous")
            elif split_id[0] == clean_list[-1][0]:
                clean_list[-1] = combine_sections(clean_list[-1], (sec_id, sec_heading, sec_refs, sec_contexts, sec_p_contexts, sec_nav_ids, sec_text))
            else:
                try:
                    prev_sec_int = int(clean_list[-1][0])
                except ValueError as e:
                    prev_sec_int = -1

                curr_sec_int = int(split_id[0])
                if prev_sec_int == curr_sec_int or prev_sec_int + 1 == curr_sec_int:  # off by one
                    clean_list.append((split_id[0], sec_heading, sec_refs, sec_contexts, sec_p_contexts, sec_nav_ids, sec_text))
                elif len(sec_refs) == 0:  # no use making a fuss about it
                    continue
                else:  # likely the same section as previous
                    prev_sec_id = clean_list[-1][0]
                    prev_sec_heading = clean_list[-1][1]
                    prev_sec_refs = clean_list[-1][2]
                    sec_refs = prev_sec_refs + sec_refs
                    sec_contexts = clean_list[-1][3] + sec_contexts
                    sec_p_contexts = clean_list[-1][4] + sec_p_contexts
                    sec_nav_ids = clean_list[-1][5] + sec_nav_ids
                    sec_text = clean_list[-1][6] + sec_text

                    clean_list[-1] = (prev_sec_id, prev_sec_heading, sec_refs, sec_contexts, sec_p_contexts, sec_nav_ids, sec_text)

        else:
            logger.error("weird shit happening")
            exit()

    clean_dict = {}
    for k in SECTIONS.keys():
        clean_dict[k] = {}
        clean_dict[k]['refs'] = []
        clean_dict[k]['contexts'] = []
        clean_dict[k]['paragraph_contexts'] = []
        clean_dict[k]['nav_ids'] = []
        clean_dict[k]['text'] = ""

    clean_section_options = SECTIONS
    likely_labels = list(clean_section_options.keys())
    for c in clean_list:
        likely_label = likely_labels[0]
        refs = c[2]
        contexts = c[3]
        p_contexts = c[4]
        nav_ids = c[5]

        headings = c[1]
        text = c[6]
        found = []
        for h in headings:
            heading_found = []
            if h != '':
                for clean_heading, keywords in clean_section_options.items():
                    for k in keywords:
                        if k in h:
                            heading_found.append(clean_heading)
            if heading_found == []:
                heading_found.append(likely_label)
            else: 
                likely_label = heading_found[0]
            found.append(heading_found)
            try:
                i = likely_labels.index(likely_label)
                likely_labels = likely_labels[i:]   
            except:
                pass

        for pred_idx, pred_list in enumerate(found):

            max_pred = max(set(pred_list), key = pred_list.count)

            try:
                clean_dict[max_pred]['text'] += text
            except TypeError as e:
                raise e
            if refs != []:
                clean_dict[max_pred]['refs'] += refs[pred_idx]
                clean_dict[max_pred]['contexts'] += contexts[pred_idx]
                clean_dict[max_pred]['paragraph_contexts'] += p_contexts[pred_idx]
                clean_dict[max_pred]['nav_ids'] += nav_ids[pred_idx]

    return clean_dict

def add_citations_to_db(cur, article_path, article_id, year, logger: logging.Logger):
    successful_count = 0

    # load parsed data
    with open(article_path, 'rb') as f:
        parsed_data = pickle.load(f)
    
    # get citations
    for r in parsed_data['references']:

        title = clean_string(r['title'])
        title_clean = clean_title(title)
        id_type = clean_string(r['id_type'])
        external_id = r['id']
        authors = r['authors']
        journal = clean_string(r['journal'])
        meeting = clean_string(r['meeting'])
        

        year = r['year']
        if year == '' or year is None:
            year = -1
        elif '-' in year:
            year = year.split('-')[0]
        year = int(year)

        if title == '':
            if external_id != '' and authors != "":
                first_author = authors.split(';')[0]
                first_author = authors.strip()
                title = f"untitled_{first_author}_{journal}"
                title_clean = clean_title(title)
            else:
                continue
        
        f_author_ln = HumanName(authors.split(';')[0]).last.strip()

        try: # all citations in all articles
            insert_query = f"""
                INSERT INTO citations (clean_title, title, id_type, external_id, authors, f_author_ln, year, journal, meeting)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            cur.execute(insert_query, (title_clean, title, id_type, external_id, authors, f_author_ln, year, journal, meeting,))
            id = cur.lastrowid

        except sqlite3.IntegrityError as e:
            # WE EXPECT DUPLICATES -> if a unique constraint fails, get the existing citation id
            if "UNIQUE constraint failed" in str(e):
                get_id_query = f"SELECT id FROM citations WHERE clean_title='{title_clean}' AND year={year}"
                cur.execute(get_id_query)
                id = cur.fetchone()[0]
            else:
                raise e

        # entry specific to article
        try:
            insert_query = \
                f"INSERT INTO article_citations (article_id, citation_id, count) VALUES(?, ?, ?)"
            cur.execute(insert_query, (article_id, id, 0,))

            successful_count += 1

            nav_id = r['nav_id'] # add id for navigating from in-text to ref section
            if nav_id:
                insert_nav_query = f"UPDATE article_citations SET nav_id = ? WHERE article_id = ? and citation_id = ?;"
                cur.execute(insert_nav_query, (nav_id, article_id, id,))

            # TODO: add alpha index 
        
        except sqlite3.IntegrityError as e:
            # logger.error(f"article_id = '{article_id}', citation_id = '{id}', citation_title = '{title_clean}'")
            # for whatever reason, sometimes there are duplicates in the xml, ignore
            if "UNIQUE constraint failed" in str(e):
                pass
            else:
                raise e

    return successful_count
