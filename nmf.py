from glob import glob

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import decomposition
from sklearn.metrics.pairwise import euclidean_distances

def file2list(filename):
    with open(filename, 'r') as f:
        rawdata = f.read()
    return rawdata

def init_data():
    target = './immidata/modata/93chosun_stop/*.txt'
    files = glob(target)
    return map(file2list, files)

def decompose_by_nmf(debug=True):
    initdata = init_data()

    vectorizer = CountVectorizer(max_df=10, min_df=3)
    counts = vectorizer.fit_transform(initdata)
    tfidf =  TfidfTransformer().fit_transform(counts)

    for i in range(2, 7):
        nmf = decomposition.NMF(n_components=i).fit(tfidf)
        feature_names = vectorizer.get_feature_names()

        print "features %d" % (i, )
        if debug:
            for topic_idx, topic in enumerate(nmf.components_):
                print "Topic #%d:" % topic_idx
                #print " ".join([str(i) for i in topic.argsort()[:-100:-1]])
                print " ".join([feature_names[i] for i in topic.argsort()[:-50:-1]])


if __name__ == '__main__':
    decompose_by_nmf()
