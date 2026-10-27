# A Framework to Study Meaningful Interdisciplinary Discourse in Scholarly Publications

Code for the paper [How Do We Engage with Other Disciplines? A Framework to Study Meaningful Interdisciplinary Discourse in Scholarly Publications](https://aclanthology.org/2026.findings-emnlp.2290/).

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

