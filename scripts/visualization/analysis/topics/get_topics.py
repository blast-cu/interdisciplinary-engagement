import tomotopy as tp
import sqlite3
import pandas as pd
from matplotlib import pyplot as plt
from data_utils.utils import DB_FILENAME
import matplotlib

import nltk
from nltk.corpus import stopwords
 
nltk.download('stopwords')

def main():
    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()

    select_query = "SELECT article_id, nlp_css_prediction FROM predictions;"
    cur.execute(select_query)

    for row in cur.fetchall():
            article_id = row[0]
            nlp_css_prediction = row[1]
            if nlp_css_prediction is not None:
                update_query = f"UPDATE article SET nlp_css_prediction = {nlp_css_prediction} WHERE id = {article_id};"
                # print(update_query)
                cur.execute(update_query)

    # declare topic model
    mdl = tp.LDAModel(k=15, seed=42, rm_top=50)

    # connect article_id to doc_id
    article_doc = {}
    select_abstracts_query = f"SELECT id, clean_abstract FROM article WHERE (nlp_css = 1 or nlp_css_prediction = 1);"
    cur.execute(select_abstracts_query)

    for idx, (id, abstract) in enumerate(cur.fetchall()):
         article_doc[id] = idx
         abstract = abstract.strip().split()
        #  abstract = [word for word in abstract if word not in stopwords.words('english')]
         mdl.add_doc(abstract)

    for i in range(0, 100, 10):
        mdl.train(10, workers=1)

    for k in range(mdl.k):
        print('Top 10 words of topic #{}'.format(k))
        words = [w[0] for w in mdl.get_topic_words(k, top_n=10)]
        print(words)
        print()
    
    get_articles_query = "SELECT id, year FROM article WHERE (nlp_css = 1 or nlp_css_prediction = 1);"
    df = pd.read_sql_query(get_articles_query, con)
    df = df.set_index('id')

    df_1 = df.copy()
    df_2 = df.copy()

    # for k in range(0,7):
    #      df_1['topic_{}'.format(k)] = df_1.index.map(lambda x: mdl.docs[article_doc[x]].get_topic_dist()[k])
    # df_1 = df_1.groupby('year').mean()

    # df_1.plot()
    # plt.savefig('topics1.png')
    # plt.show()


    # print(df_2)
    
    # for k in range(7,15):
    #      df_2['topic_{}'.format(k)] = df_2.index.map(lambda x: mdl.docs[article_doc[x]].get_topic_dist()[k])
    # df_2 = df_2.groupby('year').mean()
    # # print(df_2)

    # df_2.plot()
    # plt.savefig('topics2.png')
    # plt.show()

    matplotlib.style.use('seaborn-v0_8-darkgrid')

    for k in range(0,15):
        df['topic_{}'.format(k)] = df.index.map(lambda x: mdl.docs[article_doc[x]].get_topic_dist()[k])
    df = df.groupby('year').mean()


    # map declining topics to titles
    declining_topic_map = {
         'topic_0': 'Discourse',
         'topic_4': 'Twitter Data',
         'topic_7': 'Political Issues',
         'topic_9': 'Structure',
         'topic_11': 'Word Embeddings',
         'topic_12': 'Events'
    }

    df_decline = df.copy()
    drop_columns = ['topic_1','topic_2', 'topic_3', 'topic_5', 'topic_6', 'topic_8', 'topic_10', 'topic_12', 'topic_13', 'topic_14']
    df_decline = df_decline.drop(columns=drop_columns)
    df_decline = df_decline.rename(columns=declining_topic_map)
    df_decline.plot(color=['#235789', '#D5573B', '#92B4A7', '#FFC15E', '#26532B', '#A9C7EE'])
    legend = plt.legend(frameon = 1)
    frame = legend.get_frame()
    frame.set_color('white')
    plt.title('Topics Becoming Less Popular in NLP+CSS Research')
    plt.ylabel('Topic Proportion')
    plt.xlabel('Publication Year')
    plt.savefig('topics/declining_topics.png')
    plt.show()

    # map increasing topics to titles

    increasing_topic_map = {
         'topic_5': 'Mental Health',
         'topic_8': 'Hate Speech',
         'topic_13': 'Machine Learning'
    }
    df_increase = df.copy()
    drop_columns = ['topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_6', 'topic_7', 'topic_9', 'topic_10', 'topic_11', 'topic_12', 'topic_14']
    df_increase = df_increase.drop(columns=drop_columns)
    df_increase = df_increase.rename(columns=increasing_topic_map)
    df_increase.plot(color=['#235789', '#D5573B', '#92B4A7', '#FFC15E', '#26532B', '#A9C7EE'])
    legend = plt.legend(frameon = 1)
    frame = legend.get_frame()
    frame.set_color('white')
    plt.title('Topics Increasing in Popularity in NLP+CSS Research')
    plt.ylabel('Topic Proportion')
    plt.xlabel('Publication Year')
    plt.savefig('topics/increasing_topics.png')
    plt.show()



        

    # for d in mdl.docs:
    #     print(d.get_topic_dist())
        


    

    

    # get_years_query = "SELECT DISTINCT year FROM article;"
    # cur.execute(get_years_query)
    # years = [int(y[0]) for y in cur.fetchall()]
    # years.sort()

    # for year in years:

    #     select_abstracts_query = f"SELECT clean_abstract FROM article WHERE (nlp_css = 1 or nlp_css_prediction = 1) AND year = {year};"
    #     cur.execute(select_abstracts_query)
    #     abstracts = cur.fetchall()
    #     abstracts = [a[0] for a in abstracts]
        

    #     mdl = tp.LDAModel(k=15, seed=42, rm_top=50)
    #     for ab in abstracts:
    #         mdl.add_doc(ab.strip().split())

    #     for i in range(0, 100, 10):
    #         mdl.train(10, workers=1)
    #         # print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

    #     print('--- Year: {} ---'.format(year))
    #     print()
    #     for k in range(mdl.k):
    #         print('Top 10 words of topic #{}'.format(k))
    #         words = [w[0] for w in mdl.get_topic_words(k, top_n=10)]
    #         print(words)
    #         if k == 4:
    #             break
    #         print()
    #     print()

    #     # print(year)
    #     # mdl.summary()
    #     # print()

    

    con.close()

if __name__ == "__main__":
    main()