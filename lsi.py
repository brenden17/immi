from glob import glob

from gensim import corpora, models, similarities

import numpy as np

def file2list(filename):
    with open(filename, 'r') as f:
        rawdata = f.read()
        lines = rawdata.split('\n')
        l = []
        for line in lines:
            l.extend(line.split(' '))
    return l

def prepare_data():
    target = './immidata/modata/93chosun_stop/*.txt'
    files = glob(target)
    texts = map(file2list, files)
    dictionary = corpora.Dictionary.from_documents(texts)

    dictionary.save('data/text.dict')
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.save_corpus('data/text.mm', corpus)
    corpora.BleiCorpus.save_corpus('data/text.lda-c', corpus)


#prepare_data()

target = './immidata/modata/93chosun_stop/*.txt'
files = glob(target)
texts = map(file2list, files)
dictionary = corpora.Dictionary.from_documents(texts)
#dictionary.save('data/text.dict')
corpus = [dictionary.doc2bow(text) for text in texts]
#corpora.MmCorpus.save_corpus('data/text.mm', corpus)
corpus = corpora.MmCorpus('data/text.mm')

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
for i in range(2, 8):
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=i) 
    print('Topic #%d' % i)
    for ll in lsi.print_topics(i):
        print('---------')
        print ll
"""
for text in texts:
    vec_bow = dictionary.doc2bow(text)
    vec_lsi = lsi[vec_bow]
    print(vec_lsi)
"""
