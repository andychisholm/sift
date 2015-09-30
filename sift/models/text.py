import ujson as json
from operator import add
from collections import Counter

from sift.models import Model
from sift.util import ngrams

class TermFrequencies(Model):
    """ Get term frequencies over a corpus """
    def build(self, corpus):
        return corpus\
            .flatMap(lambda d: ngrams(d['text']))\
            .map(lambda t: (t, 1))\
            .reduceByKey(add)\
            .filter(lambda (k,v): v > 1)

    def format(self, model):
        return model\
            .map(lambda (term, count): {
                '_id': term,
                'count': count,
            })\
            .map(json.dumps)

    @classmethod
    def add_arguments(cls, p):
        p.set_defaults(modelcls=cls)
        return p

class TermDocumentFrequencies(TermFrequencies):
    """ Get document frequencies for terms in a corpus """
    def build(self, corpus):
        return corpus\
            .flatMap(lambda d: set(ngrams(d['text'])))\
            .map(lambda t: (t, 1))\
            .reduceByKey(add)\
            .filter(lambda (k,v): v > 1)

class TermIndicies(TermDocumentFrequencies):
    """ Generate uniqe indexes for termed based on their document frequency ranking. """
    def build(self, corpus):
        dfs = super(TermIndicies, self).build(corpus)
        return dfs\
            .map(lambda (t, df): (df, t))\
            .sortByKey(False)\
            .zipWithIndex()\
            .map(lambda ((dt, t), idx): (t, idx))

