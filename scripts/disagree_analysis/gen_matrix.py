import argparse
import sqlite3
# from nltk.metrics.agreement import AnnotationTask
# from nltk.metrics import binary_distance
from collections import Counter

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
import numpy as np
import itertools
import pandas as pd


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Annotator B',
           xlabel='Annotator A')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def get_agreement_matrix(df:pd.DataFrame, output_dir="", excerpts_dict=None):
        
    x_ann = []
    y_ann = []

    # get all unique entries in the 'category' row
    classes = ['analysis', 'basis', 'definition', 'related work', 'substantiation', 'substantiation + basis', 'use']
    # class_combos = itertools.combinations(classes, 2)
    # class_combos = [[c[0], c[1]] for c in class_combos]
    # print(class_combos)
    class_combos = [['substantiation + basis', 'basis'], ['substantiation + basis', 'substantiation'], ['use', 'substantiation + basis'], \
                    ['substantiation + basis', 'definition'], ['substantiation + basis', 'analysis'], ['substantiation + basis', 'related work'], \
                    ['substantiation', 'basis'], ['use', 'basis'], ['definition', 'basis'], ['basis', 'analysis'], ['related work', 'basis',], \
                    ['use', 'substantiation'], ['substantiation', 'definition'], ['substantiation', 'analysis'], ['substantiation', 'related work'], \
                    ['use', 'definition'], ['use', 'analysis'], ['use', 'related work'], ['definition', 'analysis'], ['related work', 'definition'],\
                    ['related work', 'analysis']]

    plot_title = "Agreement Confusion Matrix"

    for idx, row in df.iterrows():
        # get list of annotations
        anns = [row["ann1"], row["ann2"], row["ann3"]]
        anns = [a for a in anns if a is not np.nan]

        # get all combos
        combinations = itertools.combinations(anns, 2)
        for c in combinations:
            combo = [c[0], c[1]]
            if combo not in class_combos: # switch them
                if c[0] != c[1]:
                    print(f"old: {combo}")
                    combo = [c[1], c[0]]
                    print(f"new: {combo}\n")


            x_ann.append(combo[0])
            y_ann.append(combo[1])

    plot_confusion_matrix(x_ann, y_ann, classes, normalize=False, title=plot_title)
    plt.savefig(f"{output_dir}confusion_matrix.png", dpi=500)



def main():
    # read in annotations
    df = pd.read_csv("data/manual_anns/citations.csv")
    get_agreement_matrix(df)


if __name__ == "__main__":
    main()