from __future__ import division
from glob import glob
from os.path import basename
from os.path import splitext
from shutil import copy

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors


SOURCE_DIR = './immidata/modata/93chosun_stop/*.*'
TOPIC_FILE = './immidata/modata/93chosun_stop_t3/t3.txt'
ORI_DIR  = './immidata/endata/93chosun/'
TARGET_DIR = './immidata/modata/93chosun_stop_t3/'

def read_file(filename, readline=False):
    with open(filename, 'r') as f:
        return f.readlines() if readline else f.read()

def init_data():
    get_filename = lambda x: splitext(basename(x))
    filenames = glob(SOURCE_DIR)
    return filenames, map(read_file, filenames), read_file(TOPIC_FILE)

def get_knn_score(data, targetdata, filenames, num=20):
    vectorizer = CountVectorizer()
    tfidfvectorizer = TfidfTransformer()

    counts = vectorizer.fit_transform(data)
    tfidf_data = tfidfvectorizer.fit_transform(counts)

    knn = NearestNeighbors(n_neighbors=num)
    knn.fit(tfidf_data)

    counts = vectorizer.transform(targetdata)
    tfidf_target_data = tfidfvectorizer.transform(counts)

    result = knn.kneighbors(tfidf_target_data)
    score = result[0][0]
    index = result[1][0]

    """
    for i in index.tolist():
        print files[i]
    for i in index.tolist():
    print map(float, score)
    print index.tolist()
    """
    #return index.tolist(), score.tolist()
    for i in index.tolist():
        fname = basename(filenames[i])
        copy(ORI_DIR + fname, TARGET_DIR + fname)

def _main():
    filenames, data, targetdata = init_data()
    files = get_knn_score(data, [targetdata], filenames)


if __name__ == '__main__':
     _main()
