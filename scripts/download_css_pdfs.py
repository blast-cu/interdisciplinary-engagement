import pandas as pd
import logging
from tqdm import tqdm
import numpy as np

from src.utils import get_logger, load_env, create_out_dirs
from src.collection import download_paper_by_title
from src.db_utils import create_connection, get_column_names

"""
script to download all nlp+css pdfs
"""

def main(env_vars: dict, logger: logging.Logger):
    
    logger.info("starting script to download pdfs for all nlp+css articles (labeled or predicted positive)")

    # CREATE DB CONNECTION
    conn, cur = create_connection(env_vars["OLD_DB_PATH"])

    # GET SET OF ARTICLES WE NEED TO DOWNLOAD
    # get predicted positive
    select_pred_pos_ids = "SELECT article_id FROM predictions WHERE nlp_css_prediction = 1;"
    cur.execute(select_pred_pos_ids)
    pred_pos_ids = [entry[0] for entry in cur.fetchall()]
    
    # get labeled positive
    select_labeled_pos_ids = "SELECT id FROM article WHERE nlp_css = 1;"
    cur.execute(select_labeled_pos_ids)
    labeled_pos_ids = [entry[0] for entry in cur.fetchall()]

    # combine, ensure all are in the table
    nlp_css_ids = list(set(pred_pos_ids + labeled_pos_ids))
    id_placeholder = ', '.join(['?'] * len(nlp_css_ids))
    select_nlp_css_articles = f"SELECT * FROM article WHERE id IN ({id_placeholder});"
    cur.execute(select_nlp_css_articles, nlp_css_ids)

    # load the venue info
    venue_info_df = pd.read_csv(env_vars["VENUE_INFO"])
    venue_info_df = venue_info_df.sort_values(by=["venue", "year"])
    logger.info(f"getting data for {len(cur.fetchall())} nlp+css articles in {len(venue_info_df)} conferences")

    for _, row in tqdm(venue_info_df.iterrows(), desc="downloading articles", total=len(venue_info_df)):
        venue = row['venue']
        year = row['year']
        urls = [url for url in [row['url'], row['short_url'], row['findings_url'], row['workshop0']] if url is not np.nan]
        
        # CREATE THE FILE STRUCTURE NEEDED
        out_dir, raw_dir, processed_dir, xml_out_dir, dict_out_dir = \
            create_out_dirs(env_vars["VENUE_DATA_PATH"], venue, year, "pos")
        
        get_current_articles_query = f"SELECT id, title FROM article WHERE id IN ({id_placeholder}) AND venue = '{venue}' AND year = {year};"
        cur.execute(get_current_articles_query, nlp_css_ids)

        # DOWNLOAD EACH PDF
        for article_id, title in cur.fetchall():

            # there are some inconsistencies in db titles versus proceedings titles
            if title == "Changes in European Solidarity Before and During COVID-19: Evidence from a Large Crowd-and Expert-Annotated Twitter Dataset":
                title = "Changes in European Solidarity Before and During COVID-19: Evidence from a Large Crowd- and Expert-Annotated Twitter Dataset"

            if title == "Suum Cuique: Studying Bias in Taboo Detection with a Community Perspective ⋆":
                title = "Suum Cuique: Studying Bias in Taboo Detection with a Community Perspective"

            if title == "Grounding Characters and Places in Narrative Texts":
                title = "Grounding Characters and Places in Narrative Text"

            if title == "RESPER : Computationally Modelling Resisting Strategies in Persuasive Conversations":
                title = "ResPer: Computationally Modelling Resisting Strategies in Persuasive Conversations"

            if title == "Implicitly Abusive Comparisons -A New Dataset and Linguistic Analysis":
                title = "Implicitly Abusive Comparisons – A New Dataset and Linguistic Analysis"

            if title == "Fine-Grained Analysis of Propaganda in News Article":
                title = "Fine-Grained Analysis of Propaganda in News Articles"

            if title == "SAFETEXT: A Benchmark for Exploring Physical Safety in Language Models Warning: This paper contains examples of potentially dangerous and harmful text":
                title = "SafeText: A Benchmark for Exploring Physical Safety in Language Models"
            
            if title == "Exploring the Impact of Personality Traits on LLM Toxicity and Bias":
                title = "Exploring the Impact of Personality Traits on LLM Bias and Toxicity"

            if title == "News Sentiment and Cross-Country Fluctuations *":
                title = "News Sentiment and Cross-Country Fluctuations"

            if title == "Uphill from here: Sentiment patterns in videos from left-and right-wing YouTube news channels":
                title = "Uphill from here: Sentiment patterns in videos from left- and right-wing YouTube news channels"

            if title == "News Déjà Vu: Connecting Past and Present with Semantic Search":
                title = "News Deja Vu: Connecting Past and Present with Semantic Search"

            if title == "One Size Fits All? A simple LSTM for Non-literal Token-and Construction-level Classification":
                title = "One Size Fits All? A simple LSTM for non-literal token and construction-level classification"

            if title == "Cliché Expressions in Literary and Genre Novels":
                title = "Cliche Expressions in Literary and Genre Novels"
            
            if title == "Sign Clustering and Topic Extraction in Proto-Elamite *":
                title = "Sign Clustering and Topic Extraction in Proto-Elamite"

            if title == '"Shakespeare in the Vectorian Age" -An evaluation of different word embeddings and NLP parameters for the detection of Shakespeare quotes':
                title = '“Shakespeare in the Vectorian Age” – An evaluation of different word embeddings and NLP parameters for the detection of Shakespeare quotes'

            if title == "The diffusion of scientific terms -tracing individuals` influence in the history of science for English":
                title = "The diffusion of scientific terms – tracing individuals’ influence in the history of science for English"

            if title == "The FairyNet Corpus -Character Networks for German Fairy Tales":
                title = "The FairyNet Corpus - Character Networks for German Fairy Tales"

            if title == "Estimating Overreporting in the Creditor Reporting System on Climate Adaptation Finance Using Text Classification and Bayesian Correction":
                title = "Constructing a Credible Estimation for Overreporting of Climate Adaptation Funds in the Creditor Reporting System"

            if title == '"Who is the Madonna of Italian-American Literature?": Extracting and Analyzing Target Entities of Vossian Antonomasia':
                title = "“Who is the Madonna of Italian-American Literature?”: Target Entity Extraction and Analysis of Vossian Antonomasia"

            if title == "OdyCy -A general-purpose NLP pipeline for Ancient Greek":
                title = "OdyCy – A general-purpose NLP pipeline for Ancient Greek"

            if title == "Large Bibliographies as a Source of Data for the Humanities -NLP in the Analysis of Gender of Book Authors in German Countries and in Poland (1801-2021)":
                title = "Great Bibliographies as a Source of Data for the Humanities – NLP in the Analysis of Gender of Book Authors in German Countries and in Poland (1801-2021)"
            
            if title == "People and Places of the Past -Named Entity Recognition in Swedish Labour Movement Documents from Historical Sources":
                title = "People and Places of the Past - Named Entity Recognition in Swedish Labour Movement Documents from Historical Sources"

            if title == "Perplexing Canon: A study on GPT-based perplexity for canonical and non-canonical literary works":
                title = "Perplexing Canon: A study on GPT-based perplexity of canonical and non-canonical literary works"

            if title == "Direct Speech Identification in Swedish Literature -an Exploration of Training Data Type, Typographical Markers, and Evaluation Granularity":
                title = "Direct Speech Identification in Swedish Literature and an Exploration of Training Data Type, Typographical Markers, and Evaluation Granularity"

            download_paper_by_title(urls, title, article_id, raw_dir, venue, year, logger)


    conn.close()
    return

if __name__ == "__main__":
    env_vars = load_env()
    logger = get_logger("download nlp+css articles")

    main(env_vars, logger)