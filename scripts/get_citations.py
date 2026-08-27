
from data_utils.utils import DB_FILENAME

import sqlite3

def main():
    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()

    # get all citations for articles from 2024 and 2025 where citation_text is not null
    select_query =  """
        SELECT article.id, article.venue, ac.paragraph_context, \
        ac.article_id, ac.citation_id, ac.citation_text, ac.paragraph_context \
        FROM article \
        JOIN article_citations ac \
        ON article.id = ac.article_id
        WHERE article.year IN (2024, 2025)
        AND ac.paragraph_context IS NOT NULL
        AND article.abstract IS NOT NULL;
    """
    cur.execute(select_query)
    rows = cur.fetchall()
    article_ids = set([r[0] for r in rows])
    print(f"Total rows to process: {len(rows)}")

    # print venues
    venues = set([r[1] for r in rows])
    print(f"Venues: {venues}")

    citation_ids = set([r[3] for r in rows])
    print(f"Total unique citations to process: {len(citation_ids)}")

    # select interdisciplinary and social science attributes
    select_query =  """
        SELECT id, social_science, interdisciplinary
        FROM citations
        WHERE id IN ({});
    """.format(','.join([str(id) for id in citation_ids]))
    cur.execute(select_query)
    citation_attrs = cur.fetchall()

    # print stats
    ss_count = 0
    inter_count = 0
    for c in citation_attrs:
        if c[1] == 1:
            ss_count += 1
        if c[2] == 1:
            inter_count += 1
    print(f"Total social science citations: {ss_count}")
    print(f"Total interdisciplinary citations: {inter_count}")

    # for row in rows[:5]:
    #     print("Article ID:", row[0])
    #     print("Article Paragraph Context:", row[1])


    # get conclusion and introduction text for articles in the article_citations table, print a few
    select_query = """
        SELECT article.id, at.introduction, at.conclusion
        FROM article
        JOIN article_text at ON article.id = at.id
        WHERE article.year >= 2024 AND (at.introduction IS NOT NULL AND at.conclusion IS NOT NULL);
    """
    cur.execute(select_query)
    rows = cur.fetchall()
    print(f"Total articles with intro/conclusion: {len(rows)}")
    # for row in rows[:5]:
    #     print("Article ID:", row[0])
    #     print("Introduction:", row[1])
    #     print("Conclusion:", row[2])

    #
    group_by_query =  """
        SELECT article.year, article.venue, COUNT(article.id) \
        FROM article \
        JOIN article_citations ac \
        ON article.id = ac.article_id
        WHERE ac.paragraph_context IS NOT NULL
        AND article.abstract IS NOT NULL
        GROUP BY article.year, article.venue;
    """
    cur.execute(group_by_query)
    rows = cur.fetchall()
    for r in rows:
        print(r)

    con.close()


if __name__ == "__main__":
    main()