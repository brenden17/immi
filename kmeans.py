from glob import glob
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import decomposition
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt

def file2list(filename):
    with open(filename, 'r') as f:
        rawdata = f.read()
    return rawdata

def init_data():
    target = './immidata/modata/93han/*.txt'
    files = glob(target)
    return map(file2list, files)

def _main(prepro=True):
    data = init_data()
    vectorizer = CountVectorizer(max_df=10, min_df=3)
    counts = vectorizer.fit_transform(data)
    tfidf = TfidfTransformer().fit_transform(counts)

    x = range(0, 18)
    y = []
    if prepro:
        lsa = TruncatedSVD(4)
        tfidf = lsa.fit_transform(tfidf)
    for i in range(2, 20):
        km = KMeans(n_clusters=i, init='k-means++', max_iter=400, n_init=1)
        km.fit(tfidf)
        y.append(km.inertia_)
        print km.inertia_

    plt.plot(x, y)
    plt.show()

if __name__ == '__main__':
    _main()

