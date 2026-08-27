import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
import pandas as pd
import sqlite3

# from data_utils.visualization.plot_hand_anns import get_year_bucket

MATPLOT_STYLE = 'seaborn-v0_8-darkgrid'
COLORS = ['#92B4A7', '#D5573B', '#FFC15E', '#A9C7EE', '#26532B', '#FDA172', '#235789']

def get_year_bucket(year):
        buckets = [(2014, 2016, "1"),
        (2017, 2019, "2"),
        (2020, 2020, "3"),
        (2021, 2021, "4"),
        (2022, 2023, "5"),
        (2024, 2025, "6")]

        for start_year, end_year, num in buckets:
            if (year >= start_year and year <= end_year) or (year == start_year and start_year == end_year):
                if start_year == end_year: return str(start_year)
                else: return f"{start_year} - {end_year}"

            if num == 5:
                print("NO BUCKET")

def get_article_year(id, cur):
    cur.execute(f"SELECT year FROM article WHERE id = {id};")
    return cur.fetchone()[0]


def get_article_venue(id, cur):
    cur.execute(f"SELECT venue FROM article WHERE id = {id};")
    return cur.fetchone()[0]

def fix_venue_names(df):
    df = df.replace("nlp_css", "NLP+CSS")
    df = df.replace("acl", "ACL")
    df = df.replace("eacl", "EACL")
    df = df.replace("naacl", "NAACL")
    df = df.replace("aacl", "AACL")
    df = df.replace("emnlp", "EMNLP")
    df = df.replace("sighum", "SIGHUM")

    return df

def generate_pos_label_plot(con: sqlite3.Connection, filename: str):
    """
    Generate a stacked bar plot showing the number of NLP+CSS publications by year and venue.
    This function queries a database for articles labeled as NLP+CSS (either manually or through predictions),
    creates a stacked bar visualization of publication counts grouped by year and venue,
    and saves the resulting plot to a file.
    Args:
        con: Database connection object (sqlite3.Connection)
        filename (str): Path where the generated plot should be saved
    The plot shows:
        - X-axis: Years
        - Y-axis: Number of papers
        - Each bar represents the number of NLP+CSS papers for that year and venue
    Note:
        The function first updates the nlp_css_prediction values in the article table
        from the predictions table before generating the visualization.
    """

    cur = con.cursor()
    select_query = "SELECT article_id, nlp_css_prediction FROM predictions;"
    cur.execute(select_query)

    # create nlp_css_prediction column 
    try:
        cur.execute("ALTER TABLE article ADD COLUMN nlp_css_prediction INTEGER;")
    except:
        pass

    create_column_query = "UPDATE article SET nlp_css_prediction = 1 WHERE id IN (SELECT article_id FROM predictions WHERE nlp_css_prediction = 1)"
    cur.execute(create_column_query)

    con.commit()

    select_query = "SELECT year, venue FROM article WHERE (nlp_css = 1 OR nlp_css_prediction = 1) AND id IN (SELECT article_id from article_citations_context);"

    df = pd.read_sql(select_query, con)
    df = fix_venue_names(df)
    df['year'] = df['year'].apply(get_year_bucket)

    matplotlib.style.use(MATPLOT_STYLE)
    rcParams.update({'figure.autolayout': True})
    df = df.groupby(['year', 'venue']).size().unstack().plot(kind='bar', stacked=True, color=COLORS)

    # add labels and save.
    plt.title('Number of Publications Identified as NLP+CSS by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Papers')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)


def generate_css_per_year_plot(con: sqlite3.Connection, filename: str):
    """
    Generates and saves a bar plot showing the percentage of NLP publications identified as CSS-related per year.
    The function queries a SQLite database for articles' CSS classification data, processes it to calculate
    yearly percentages, and creates a bar plot visualization.
    Args:
        con (sqlite3.Connection): Connection object to the SQLite database
        filename (str): Path where the resulting plot image will be saved
    The plot shows:
        - X-axis: Years
        - Y-axis: Percentage of papers classified as CSS-related
        - Each bar represents the combined percentage of manually labeled and predicted CSS papers for that year
    """

    unlabelled_select_query = "SELECT id, year, nlp_css, nlp_css_prediction FROM article;"
    df = pd.read_sql(unlabelled_select_query, con)

    df['total_count'] = 1

    df = df.groupby(['year']).sum()
    df['percent pos'] = (df['nlp_css_prediction'] + df['nlp_css']) / df['total_count']
    df = df.drop(columns=['nlp_css', 'nlp_css_prediction', 'total_count', 'id'])
    df['percent pos'] = df['percent pos'].apply(lambda x: x * 100)
    df['percent pos'] = df['percent pos'].apply(lambda x: round(x, 2))

    matplotlib.style.use(MATPLOT_STYLE)
    rcParams.update({'figure.autolayout': True})
    df = df.plot(kind='bar', y='percent pos', color=['#235789'])

    # add labels and save.
    plt.title('Percentage of NLP Publications Identified as NLP+CSS by Year')
    plt.legend().remove()
    plt.xlabel('Year')
    plt.ylabel('Percentage of Papers')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)


def generate_ood_boxplot_year(con: sqlite3.Connection, filename: str):
    """
    Generate a box plot showing the distribution of out-of-discipline citations by year.
    This function creates a box plot visualization showing how the percentage of
    interdisciplinary citations in NLP/CSS publications varies across different years.
    The data is retrieved from a SQLite database containing publication and citation information.
    Args:
        con (sqlite3.Connection): Connection object to the SQLite database
        filename (str): Path where the resulting plot image will be saved
    The plot shows:
        - X-axis: Years
        - Y-axis: Average percentage of interdisciplinary citations per publication
        - Each box represents the distribution of interdisciplinary citations
    """

    get_ood_query = "SELECT article.id, citations.social_science, citations.interdisciplinary FROM article \
        JOIN article_citations ON article.id = article_citations.article_id \
            JOIN citations ON article_citations.citation_id = citations.id \
                WHERE (article.nlp_css = 1 OR article.nlp_css_prediction = 1) AND citations.field_of_study IS NOT NULL ;"
    cur = con.cursor()
    df = pd.read_sql(get_ood_query, con)
    df['count'] = 1
    df = df.groupby(['id']).sum()
    df['social_science'] = df['social_science'] / df['count']
    df['interdisciplinary'] = df['interdisciplinary'] / df['count']
    df['year'] = df.index.map(lambda x: get_article_year(x, cur))
    df = df.drop(columns=['count'])
    matplotlib.style.use(MATPLOT_STYLE)
    df.boxplot(column=['interdisciplinary'], by='year', notch=True, patch_artist=True, boxprops = dict(facecolor = '#235789'))
    plt.title('Percentage of Out-of-Discipline Citations in a Publication by Year')
    plt.suptitle('')
    plt.xlabel('Year')
    plt.ylabel('Average Percentage of Citations')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)


def generate_ss_boxplot_year(con: sqlite3.Connection, filename: str):
    """
    Generate a box plot showing the distribution of social science citations by year.
    This function creates a box plot visualization showing how the percentage of
    social science citations in NLP/CSS publications varies across different years.
    The data is retrieved from a SQLite database containing publication and citation information.
    Args:
        con (sqlite3.Connection): Connection object to the SQLite database
        filename (str): Path where the resulting plot image will be saved
    The plot shows:
        - X-axis: Years
        - Y-axis: Average percentage of social science citations per publication
        - Each box represents the distribution of social science citations
    """
    cur = con.cursor()
    get_ood_query = "SELECT article.id, citations.social_science, citations.interdisciplinary FROM article \
        JOIN article_citations ON article.id = article_citations.article_id \
            JOIN citations ON article_citations.citation_id = citations.id \
                WHERE (article.nlp_css = 1 OR article.nlp_css_prediction = 1) AND citations.field_of_study IS NOT NULL ;"
    
    df = pd.read_sql(get_ood_query, con)
    df['count'] = 1
    df = df.groupby(['id']).sum()
    df['social_science'] = df['social_science'] / df['count']
    df['interdisciplinary'] = df['interdisciplinary'] / df['count']
    df['year'] = df.index.map(lambda x: get_article_year(x, cur))
    df = df.drop(columns=['count'])
    matplotlib.style.use(MATPLOT_STYLE)
    df.boxplot(column=['social_science'], by='year', notch=True, patch_artist=True, boxprops = dict(facecolor = '#235789'))
    plt.title('Percentage of Social Science Citations in a Publication by Year')
    plt.suptitle('')
    plt.xlabel('Year')
    plt.ylabel('Average Percentage of Citations')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)


def ss_ood_boxplot_year(con: sqlite3.Connection, filename: str):
    """
    Generate a box plot showing the distribution of social science citations by year.
    This function creates a box plot visualization showing how the percentage of ood
    citations are social science citations in NLP/CSS publications varies across 
    different years.
    The data is retrieved from a SQLite database containing publication and citation information.
    Args:
        con (sqlite3.Connection): Connection object to the SQLite database
        filename (str): Path where the resulting plot image will be saved
    The plot shows:
        - X-axis: Years
        - Y-axis: Average percentage of ood citations that are ss citations per publication
        - Each box represents the distribution of social science citations
    """

    cur = con.cursor()
    select_query = "SELECT article.id, citations.social_science, citations.interdisciplinary FROM article\
                        JOIN article_citations ON article.id = article_citations.article_id \
                            JOIN citations ON article_citations.citation_id = citations.id \
                                WHERE (article.nlp_css = 1 OR article.nlp_css_prediction = 1)  AND citations.interdisciplinary = 1;"
    df = pd.read_sql(select_query, con)
    df = df.groupby('id').sum()
    df['percent social science'] = df['social_science'] / df['interdisciplinary']
    df['year'] = df.index.map(lambda x: get_article_year(x, cur))
    df = df.drop(columns=['social_science', 'interdisciplinary'])
    matplotlib.style.use(MATPLOT_STYLE)
    df.boxplot(column=['percent social science'], by='year', notch=True, patch_artist=True, boxprops = dict(facecolor = '#235789'))
    plt.title('Percentage of Out-of-Discipline Citations Identified\nas Social Science in a Publication')
    plt.suptitle('')
    plt.xlabel('Year')
    plt.ylabel('Average Percentage of Citations')
    plt.savefig(filename, dpi=300)


def generate_ood_boxplot_venue(con: sqlite3.Connection, filename: str):
    """
    Generate a box plot showing the distribution of out-of-discipline citations by venue.
    This function creates a box plot visualization showing how the percentage of
    interdisciplinary citations in NLP/CSS publications varies across different venues.
    The data is retrieved from a SQLite database containing publication and citation information.
    Args:
        con (sqlite3.Connection): Connection object to the SQLite database
        filename (str): Path where the resulting plot image will be saved
    The plot shows:
        - X-axis: Venues
        - Y-axis: Average percentage of interdisciplinary citations per publication
        - Each box represents the distribution of interdisciplinary citations
    """

    cur = con.cursor()
    get_ood_query = "SELECT article.id, citations.social_science, citations.interdisciplinary FROM article \
        JOIN article_citations ON article.id = article_citations.article_id \
            JOIN citations ON article_citations.citation_id = citations.id \
                WHERE (article.nlp_css = 1 OR article.nlp_css_prediction = 1)  AND citations.field_of_study IS NOT NULL;"
    df = pd.read_sql(get_ood_query, con)
    df['count'] = 1
    df = df.groupby(['id']).sum()
    df['interdisciplinary'] = df['interdisciplinary'] / df['count']
    df['venue'] = df.index.map(lambda x: get_article_venue(x, cur))
    df = df.replace("nlp_css", "NLP+CSS")
    df = df.drop(columns=['count'])
    matplotlib.style.use(MATPLOT_STYLE)
    df.boxplot(column=['interdisciplinary'], by='venue', notch=True, patch_artist=True, boxprops = dict(facecolor = '#235789'), vert=False)
    plt.title('Percentage of Out-of-Discipline Citations in a Publication by Venue')
    plt.suptitle('')
    plt.xlabel('Average Percentage of Citations')
    plt.ylabel('Venue')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)


def generate_venue_interdisc_plot(con: sqlite3.Connection, filename: str):
    """
    Generate a stacked bar plot showing the average percentage of interdisciplinary citations by venue.
    This function queries a database for articles with out-of-discipline citations, calculates the average
    percentage of interdisciplinary citations per publication by venue, and creates a stacked bar visualization.
    Args:
        con: Database connection object (sqlite3.Connection)
        filename (str): Path where the generated plot should be saved
    The plot shows:
        - X-axis: Venues
        - Y-axis: Average percentage of interdisciplinary citations
        - Each bar represents the average percentage of interdisciplinary citations per publication by venue
    """
    cur = con.cursor()
    get_ood_query = "SELECT article.id, citations.social_science, citations.interdisciplinary FROM article \
        JOIN article_citations ON article.id = article_citations.article_id \
            JOIN citations ON article_citations.citation_id = citations.id \
                WHERE (article.nlp_css = 1 OR article.nlp_css_prediction = 1) AND citations.field_of_study IS NOT NULL ;"

    df = pd.read_sql(get_ood_query, con)
    df['count'] = 1
    df = df.groupby(['id']).sum()
    df['interdisciplinary'] = df['interdisciplinary'] - df['social_science']
    df['interdisciplinary'] = df['interdisciplinary'] / df['count']
    df['social_science'] = df['social_science'] / df['count']
    df['venue'] = df.index.map(lambda x: get_article_venue(x, cur))
    df = df.drop(columns=['count'])
    df = df.replace("nlp_css", "NLP+CSS")

    matplotlib.style.use(MATPLOT_STYLE)
    df = df.groupby('venue').mean().plot(kind='bar', stacked=True, color=['#235789', '#D5573B'])
    plt.title('Percentage of Out-of-Discipline Citations in a Publication by Venue')
    plt.xlabel('Venue')
    plt.ylabel('Average Percentage of Citations')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)


def generate_top_10_ss_fos_plot(con: sqlite3.Connection, ss_field_filename: str, filename: str):
    """
    Generate a bar plot showing the top 10 social science fields cited in NLP+CSS publications.
    This function queries a database for articles with social science citations, calculates the count of
    each field of study, and creates a bar plot visualization of the top 10 most cited fields.
    Args:
        con: Database connection object (sqlite3.Connection)
        ss_field_filename (str): Path to the CSV file containing the list of social science fields
        filename (str): Path where the generated plot should be saved
    The plot shows:
        - X-axis: Field of Study
        - Y-axis: Number of Citations
        - Each bar represents the number of citations for the top 10 most cited social science fields
    """
    # get anything ss
    get_ood_query = "SELECT article.id, citations.field_of_study FROM article \
        JOIN article_citations ON article.id = article_citations.article_id \
            JOIN citations ON article_citations.citation_id = citations.id \
                WHERE (article.nlp_css = 1 OR article.nlp_css_prediction = 1) AND citations.field_of_study IS NOT NULL \
                    AND citations.social_science = 1;"
    cur = con.cursor()
    rows = cur.execute(get_ood_query).fetchall()

    # get list of social science fields
    f = open(ss_field_filename, "r")
    df = pd.read_csv(f)
    ss_list = df[df['SS'] == 1]['field_of_study'].tolist()
    ss_list = [s.lower().strip() for s in ss_list]
    f.close()

    # populate dict with fos and count
    fos_dict = {}
    for field in ss_list:
        fos_dict[field] = 0

    for r in rows:
        fos = r[1]
        fos = fos.split("; ")
        fos = [f.strip() for f in fos]
        fos = list(set(fos))
        for f in fos:
            if f in fos_dict:
                fos_dict[f] += 1

    # sort dict by count
    sorted_fos = sorted(fos_dict.items(), key=lambda x: x[1], reverse=True)
    top_ten = sorted_fos[:10]

    field = [x[0] for x in top_ten]
    count = [x[1] for x in top_ten]

    matplotlib.style.use(MATPLOT_STYLE)
    plt.bar(field, count, color='#235789')
    plt.xticks(rotation=90)
    plt.title('Top 10 Social Science Fields Cited in NLP+CSS Publications')
    plt.ylabel('Number of Citations')
    plt.xlabel('Field of Study')
    plt.tight_layout()

    plt.savefig(filename, dpi=300)


def generate_final_sec_arg_id_plot(con: sqlite3.Connection, filename: str):
    """
    Generate a bar plot showing the number of in-discipline citations by section.
    This function queries a database for articles with in-discipline citations, calculates the count of
    each section, and creates a bar plot visualization of the number of citations per section.
    Args:
        con: Database connection object (sqlite3.Connection)
        filename (str): Path where the generated plot should be saved
    The plot shows:
        - X-axis: Section
        - Y-axis: Number of Citations
        - Bar color represents the argument type (Background Claim, Data, Own Claim)
        - Each bar represents the number of in-discipline citations for each section
    """
    get_ood_query = "SELECT article_citations.background_claim, article_citations.data, article_citations.own_claim, article_citations.section FROM article \
            JOIN article_citations ON article.id = article_citations.article_id \
                JOIN citations ON article_citations.citation_id = citations.id\
                    WHERE section IS NOT NULL AND citations.interdisciplinary = 0 AND (article.nlp_css = 1 OR article.nlp_css_prediction = 1);"

    df = pd.read_sql(get_ood_query, con)
    df = df.rename(columns={'background_claim': 'Background Claim', 'data': 'Data', 'own_claim': 'Own Claim'})
    df = df.groupby('section').sum().rename(index={'introduction':"Introduction", 'related work':"Related Work", 'method':"Method", 'experiments': "Experiments", 'conclusion':"Conclusion", 'appendix':"Appendix"})
    df = df.reindex(['Introduction', 'Related Work', 'Method', 'Experiments', 'Conclusion', 'Appendix'])
    matplotlib.style.use(MATPLOT_STYLE)
    df.plot(kind='bar', color=['#235789', '#D5573B', '#FFC15E'])
    plt.title("In-Discipline Citations by Section")
    plt.xlabel('Section')
    plt.ylabel('Number of Citations')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)


def generate_final_sec_arg_ood_plot(con: sqlite3.Connection, filename: str):
    """
    Generate a bar plot showing the number of out-of-discipline citations by section.
    This function queries a database for articles with out-of-discipline citations, calculates the count of
    each section, and creates a bar plot visualization of the number of citations per section.
    Args:
        con: Database connection object (sqlite3.Connection)
        filename (str): Path where the generated plot should be saved
    The plot shows:
        - X-axis: Section
        - Y-axis: Number of Citations
        - Bar color represents the argument type (Background Claim, Data, Own Claim)
        - Each bar represents the number of out-of-discipline citations for each section
    """
    get_ood_query = "SELECT article_citations.background_claim, article_citations.data, article_citations.own_claim, article_citations.section FROM article \
            JOIN article_citations ON article.id = article_citations.article_id \
                JOIN citations ON article_citations.citation_id = citations.id\
                    WHERE section IS NOT NULL AND citations.interdisciplinary = 1 AND (article.nlp_css = 1 OR article.nlp_css_prediction = 1);"

    df = pd.read_sql(get_ood_query, con)
    df = df.rename(columns={'background_claim': 'Background Claim', 'data': 'Data', 'own_claim': 'Own Claim'})
    df = df.groupby('section').sum().rename(index={'introduction':"Introduction", 'related work':"Related Work", 'method':"Method", 'experiments': "Experiments", 'conclusion':"Conclusion", 'appendix':"Appendix"})
    df = df.reindex(['Introduction', 'Related Work', 'Method', 'Experiments', 'Conclusion', 'Appendix'])
    df.plot(kind='bar', color=['#235789', '#D5573B', '#FFC15E'], legend=False)

    plt.title("Out-of-Discipline Citations by Section")
    plt.xlabel('Section')
    plt.ylabel('Number of Citations')
    plt.tight_layout()

    plt.savefig(filename, dpi=300)


def generate_dataset_stats_year(con: sqlite3.Connection, filename: str):
    """
    Generate a bar plot showing the number of labelled and unlabelled publications by year.
    This function queries a database for labelled and unlabelled articles, calculates the count of
    each category by year, and creates a bar plot visualization of the number of publications per year.
    Args:
        con: Database connection object (sqlite3.Connection)
        filename (str): Path where the generated plot should be saved
    The plot shows:
        - X-axis: Years
        - Y-axis: Number of Papers
        - Each bar represents the number of labelled and unlabelled publications for each year
    """

    get_labelled_query = "SELECT id, year, venue, nlp_css FROM article;"
    df = pd.read_sql(get_labelled_query, con)
    df = df.replace("nlp_css", "NLP+CSS")

    df['labelled'] = df.index.map(lambda x: 1 if df.loc[x,'nlp_css'] == 1 or df.loc[x,'nlp_css'] == 0 else 0)
    df['unlabelled'] = df.index.map(lambda x: 1 if df.loc[x,'labelled'] != 1 else 0)

    df['Unlabelled'] = df['unlabelled']
    df['Labelled'] = df['labelled']

    matplotlib.style.use(MATPLOT_STYLE)
    df_y = df.drop(columns=['nlp_css', 'id', 'unlabelled', 'labelled', 'venue'])
    df_y = df_y.groupby('year').sum()
    df_y.plot(kind='bar', stacked=True, color=['#235789', '#D5573B'])
    plt.title('Number of Labelled and Unlabelled Publications in Dataset')
    plt.xlabel('Year')
    plt.ylabel('Number of Papers')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)


def generate_dataset_stats_venue(con: sqlite3.Connection, filename: str):
    """
    Generate a bar plot showing the number of labelled and unlabelled publications by venue.
    This function queries a database for labelled and unlabelled articles, calculates the count of
    each category by venue, and creates a bar plot visualization of the number of publications per venue.
    Args:
        con: Database connection object (sqlite3.Connection)
        filename (str): Path where the generated plot should be saved
    The plot shows:
        - X-axis: Venues
        - Y-axis: Number of Papers
        - Each bar represents the number of labelled and unlabelled publications for each venue
    """
    get_labelled_query = "SELECT id, year, venue, nlp_css FROM article;"
    df = pd.read_sql(get_labelled_query, con)
    df = df.replace("nlp_css", "NLP+CSS")

    df['labelled'] = df.index.map(lambda x: 1 if df.loc[x,'nlp_css'] == 1 or df.loc[x,'nlp_css'] == 0 else 0)
    df['unlabelled'] = df.index.map(lambda x: 1 if df.loc[x,'labelled'] != 1 else 0)

    df['Unlabelled'] = df['unlabelled']
    df['Labelled'] = df['labelled']

    df_v = df.drop(columns=['nlp_css', 'id', 'unlabelled', 'labelled', 'year'])
    df_v = df_v.groupby('venue').sum()
    df_v.plot(kind='bar', stacked=True, color=['#235789', '#D5573B'])
    plt.title('Number of Labelled and Unlabelled Publications in Dataset')
    plt.xlabel('Venue')
    plt.ylabel('Number of Papers')
    plt.tight_layout()

    plt.savefig(filename, dpi=300)


def generate_css_per_year_venue_plot(con: sqlite3.Connection, filename: str):
    """
    Generate a bar plot showing the percentage of NLP publications identified as CSS-related per venue.
    This function queries a SQLite database for articles' CSS classification data, processes it to calculate
    venue-wise percentages, and creates a bar plot visualization.
    Args:
        con (sqlite3.Connection): Connection object to the SQLite database
        filename (str): Path where the resulting plot image will be saved
    The plot shows:
        - X-axis: Venues
        - Y-axis: Percentage of papers classified as CSS-related
        - Each bar represents the combined percentage of manually labeled and predicted CSS papers for that venue
    """

    unlabelled_select_query = "SELECT article.id, article.year, article.venue, article.nlp_css, predictions.nlp_css_prediction FROM article \
        JOIN predictions on article.id = predictions.article_id;"
    df = pd.read_sql(unlabelled_select_query, con)
    df = df.replace("nlp_css", "NLP+CSS")

    df['total_count'] = 1
    df = df.groupby(['year', 'venue']).sum()
    df['percent pos'] = df['nlp_css_prediction'] / df['total_count']
    df = df.drop(columns=['nlp_css', 'nlp_css_prediction', 'total_count', 'id'])
    df['percent pos'] = df['percent pos'].apply(lambda x: x * 100)
    df['percent pos'] = df['percent pos'].apply(lambda x: round(x, 2))

    matplotlib.style.use(MATPLOT_STYLE)
    rcParams.update({'figure.autolayout': True})
    df = df.unstack().plot(kind='bar', y='percent pos', color=['#235789', '#D5573B', '#92B4A7', '#FFC15E', '#26532B', '#A9C7EE'])
    plt.title('Percentage of NLP Publications Identified as NLP+CSS by Venue')
    plt.xlabel('Venue')
    plt.ylabel('Percentage of Papers')
    plt.savefig(filename, dpi=300)


def generate_training_stats_venue_plot(con: sqlite3.Connection, filename: str, processed_only: bool = False):
    """
    Generate a stacked bar plot showing the number of positive and negative training examples by venue.
    This function queries a database for labelled articles, calculates the count of positive and negative examples
    by venue, and creates a stacked bar visualization of the number of training examples per venue.
    Args:
        con: Database connection object (sqlite3.Connection)
        filename (str): Path where the generated plot should be saved
    The plot shows:
        - X-axis: Venues
        - Y-axis: Number of Training Examples
        - Each bar represents the number of positive and negative training examples for each venue
    """
    if processed_only:
        # check that there's an entry for paper in article_citations table
        select_query = "SELECT venue, nlp_css FROM article WHERE nlp_css IS NOT NULL AND id IN (SELECT DISTINCT article_id FROM article_citations);"
        if "processed_only" not in filename:
            filename = filename.replace(".png", "_processed_only.png")
    else:
        select_query = "SELECT venue, nlp_css FROM article WHERE nlp_css IS NOT NULL;"
    df = pd.read_sql_query(select_query, con)
    df = df.replace("nlp_css", "NLP+CSS")
    df['count'] = 1
    df = df.groupby(['venue', 'nlp_css']).sum().unstack().fillna(0)

    df.plot(kind='bar', stacked=True, color=['#235789', '#D5573B'])
    plt.legend(['Negative', 'Positive'])
    plt.title('Number of Positive and Negative Training Examples by Venue')
    plt.xlabel('Conference/Workshop')
    plt.ylabel('Number of Articles')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)


def generate_training_stats_year_plot(con: sqlite3.Connection, filename: str, processed_only: bool = False):
    """
    Generate a stacked bar plot showing the number of positive and negative training examples by year.
    This function queries a database for labelled articles, calculates the count of positive and negative examples
    by year, and creates a stacked bar visualization of the number of training examples per year.
    Args:
        con: Database connection object (sqlite3.Connection)
        filename (str): Path where the generated plot should be saved
    The plot shows:
        - X-axis: Years
        - Y-axis: Number of Training Examples
        - Each bar represents the number of positive and negative training examples for each year       
    """
    if processed_only:
        # check that there's an entry for paper in article_citations table
        select_query = "SELECT year, nlp_css FROM article WHERE nlp_css IS NOT NULL AND abstract IS NOT NULL AND id IN (SELECT DISTINCT article_id FROM article_citations);"
        if "processed_only" not in filename:
            filename = filename.replace(".png", "_processed_only.png")
    else: 
        select_query = "SELECT year, nlp_css FROM article WHERE nlp_css IS NOT NULL;"
    df = pd.read_sql_query(select_query, con)
    df = df.replace("nlp_css", "NLP+CSS")
    df['count'] = 1
    df = df.groupby(['year', 'nlp_css']).sum().unstack().fillna(0)

    df.plot(kind='bar', stacked=True, color=['#235789', '#D5573B'])
    plt.legend(['Negative', 'Positive'])
    plt.title('Number of Positive and Negative Training Examples by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Articles')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
