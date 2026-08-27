import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
import sqlite3
import os

from src.utils import load_env
import scripts.visualization.plot_generators as pg


def main():
    env_vars = load_env()

    con = sqlite3.connect(env_vars["DB_PATH"])

    fig_output_dir = "data/paper/figures"
    os.makedirs(fig_output_dir, exist_ok=True)

    table_output_dir = "data/paper/tables"
    os.makedirs(table_output_dir, exist_ok=True)

    ### MAIN PAPER ###

    # pos_label_plot
    # caption: Resulting dataset of NLP+CSS publications over the last decade
    pos_label_filename = f"{fig_output_dir}/dataset_stats_pos_labels_year_venue.png"
    pg.generate_pos_label_plot(con, pos_label_filename)

    # css_per_year
    # caption: Percentage of all gathered NLP papers that were predicted or 
    #   labelled as NLP+CSS papers
    css_per_year_filename = f"{fig_output_dir}/dataset_stats_labels_year.png"
    pg.generate_css_per_year_plot(con, css_per_year_filename)

    # ood_boxplot_year
    # caption: Average % of Out-of-Discipline citations in NLP+CSS papers 
    #   per year
    ood_boxplot_year_filename = f"{fig_output_dir}/ood_boxplot_citations_by_year.png"
    pg.generate_ood_boxplot_year(con, ood_boxplot_year_filename)

    # ss_boxplot_year
    # not in paper
    ss_boxplot_year_filename = f"{fig_output_dir}/ss_boxplot_citations_by_year.png"
    pg.generate_ss_boxplot_year(con, ss_boxplot_year_filename)

    # ss_ood_boxplot_year
    # caption: Average percentage of Out-of-Discipline citations within a 
    #   publication that are considered Social Science-specific
    ss_ood_boxplot_year_filename = f"{fig_output_dir}/boxplot_social_science_by_year.png"
    pg.ss_ood_boxplot_year(con, ss_ood_boxplot_year_filename)

    # ood_boxplot_venue
    # caption: % of Out-of-Discipline citations per venue
    ood_boxplot_venue_filename = f"{fig_output_dir}/ood_boxplot_citations_by_venue.png"
    pg.generate_ood_boxplot_venue(con, ood_boxplot_venue_filename)

    # venue_interdisc
    # caption: % Out-of-Discipline per venue
    venue_interdisc_filename = f"{fig_output_dir}/both_percent_interdisciplinary_citations_by_venue.png"
    pg.generate_venue_interdisc_plot(con, venue_interdisc_filename)

    # top10_ss
    # caption: Top 10 most cited social science fields
    # ss_field_filename = "data/hand_tagged/reviewer_labelled_fos.csv"
    # filename = f"{fig_output_dir}/top10_ss.png"
    # pg.generate_top_10_ss_fos_plot(con, ss_field_filename, filename)

    # combined figures 
    # caption: Argument Types Supported by In-discipline vs. Out-of-Discipline citations per Section
    # final_sec_arg_id


    # ERROR HERE
    # final_sec_arg_id_filename = f"{fig_output_dir}/final_sec_arg_id.png"
    # pg.generate_final_sec_arg_id_plot(con, final_sec_arg_id_filename)
    # # final_sec_arg_ood
    # final_sec_arg_ood_filename = f"{fig_output_dir}/final_sec_arg_ood.png"
    # pg.generate_final_sec_arg_ood_plot(con, final_sec_arg_ood_filename)



    ### APPENDIX ###

    # dataset_statistics
    # caption: Full unlabelled dataset by year
    dataset_stats_year_filename = f"{fig_output_dir}/dataset_stats_year.png"
    pg.generate_dataset_stats_year(con, dataset_stats_year_filename)

    # dataset_stats_venue
    # caption: Full unlabelled dataset by venue
    dataset_stats_venue_filename = f"{fig_output_dir}/dataset_stats_venue.png"
    pg.generate_dataset_stats_venue(con, dataset_stats_venue_filename)
    
    # training_stats_venue
    training_stats_venue_filename = f"{fig_output_dir}/training_stats_venue.png"
    pg.generate_training_stats_venue_plot(con, training_stats_venue_filename)

    # training_stats_year
    training_stats_year_filename = f"{fig_output_dir}/training_stats_year.png"
    pg.generate_training_stats_year_plot(con, training_stats_year_filename)


    # css_per_year_venue
    css_per_year_venue_filename = f"{fig_output_dir}/dataset_stats_labels_year_venue.png"
    pg.generate_css_per_year_venue_plot(con, css_per_year_venue_filename)


    con.close()


if __name__ == "__main__":
    main()