# A Framework to Study Meaningful Interdisciplinary Discourse in Scholarly Publications

Code for the paper [How Do We Engage with Other Disciplines? A Framework to Study Meaningful Interdisciplinary Discourse in Scholarly Publications](https://arxiv.org/abs/2601.17020).

## Abstract 
With the rising popularity of interdisciplinary work and increasing institutional incentives in this direction, there is a growing need to understand how resulting publications incorporate ideas from multiple disciplines. Existing computational approaches, such as affiliation diversity, keywords, and citation patterns, do not account for how individual citations are used to advance the citing work. Although, in line with addressing this gap, prior studies have proposed taxonomies to classify citation purpose, these frameworks are not well-suited to interdisciplinary research and do not provide quantitative measures of citation engagement quality. To address these limitations, we propose a framework for the evaluation of citation engagement in interdisciplinary Natural Language Processing (NLP) publications. Our approach introduces a citation purpose taxonomy tailored to interdisciplinary work, supported by an annotation study. We demonstrate the utility of this framework through a thorough analysis of publications at the intersection of NLP and Computational Social Science. 


## Data

Data can be found [here](https://o365coloradoedu-my.sharepoint.com/:f:/g/personal/basu9216_colorado_edu/IgDqB_VZqpgYQZMZwaVhYDHRARoKgFyWm51I-lugihyno5E?e=wWmYER). The citation purpose codebook is included [here](https://docs.google.com/document/d/1iepRLpcOYWA9K5AC0o1gmuRfwGLeb6N2WjWJerLEjBg/edit?usp=sharing).

## Code Structure 
- `data_utils`
    - `collection`: scripts for scraping pdfs and parsing abstracts from ACL anthology
    - `parse_pdf`: scripts for fully parsing pdfs with grobid
    - `db_creation`: scripts for adding all information to database
    - `visualization`: scripts to conduct analysis on completed database, generate figs and tables for paper
- `classifiers`: preprocessing, training, testing for track and argument classifiers
- `runners`: bash runner files for use with rc

## Research Process
General order of research process described in paper.

### Download necessary .csv files
To get started, create a `data/csv_data` directory. Then, download and add the [venue_info.csv](), [workshop_info.csv]() and [css_track_papers.csv]() csv files to the directory.

### Collect Data

First, collect all pdf files and export abstracts of long papers from the desired venues and years:
```bash
python3 data_utils/collection/unlabelled_collector.py
```

Add the workshop papers:
```bash
python3 data_utils/collection/workshop_collector.py
```

Add labelled data from regular conferences:
```bash
python3 data_utils/collection/track_labelled_collector.py
```

Then, create and populate the `article` table in the database:

```bash
python3 data_utils/db_creation/create_article_table.py
```

### Predict Track

Add CSS/NON-CSS labels to hand-annotations articles in the database:
```bash
python3 data_utils/db_creation/add_supervision.py
```

Train and test the model (add the `--test` argument if needed): 
```bash
python3 classifiers/track/train.py
```

Get predictions across the unlabelled dataset:
```bash
python3 classifiers/track/predict.py
```


### Process PDFs for Case Study
Fully process the papers that were predicted CSS:
```bash
bash data_utils/parse_pdf/run_xml_gen.sh
```
Note this step takes time and each venue/year is done individually.

### Citation Mapping
To run the full process of identifying citations, adding their f.o.s., etc:

```bash
bash data_utils/db_creation/run_analysis.sh
```
