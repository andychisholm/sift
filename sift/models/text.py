import ujson as json

from . import Model
from operator import add
from collections import Counter

class TermDocumentFrequencies(Model):
    """ Get document frequencies for terms in a corpus """
    def build(self, corpus):
        return corpus\
            .map(lambda d: d['text'].lower().split())\
            .flatMap(lambda tokens: ((t, 1) for t in set(tokens)))\
            .reduceByKey(add)

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

class TermIndicies(TermDocumentFrequencies):
    """ Generate uniqe indexes for termed based on their document frequency ranking. """
    def build(self, corpus):
        dfs = super(TermIndicies, self).build(corpus)
        return dfs\
            .map(lambda (t, df): (df, t))\
            .sortByKey(False)\
            .zipWithIndex()\
            .map(lambda ((dt, t), idx): (t, idx))

