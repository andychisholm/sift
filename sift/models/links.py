import ujson as json

from . import Model
from operator import add
from collections import Counter

class EntityNameCounts(Model):
    """ Entity counts by name """
    @staticmethod
    def iter_link_anchor_target_pairs(doc):
        for link in doc['links']:
            yield doc['text'][link['start']:link['stop']], link['target']
    
    def build(self, corpus):
        return corpus\
            .flatMap(self.iter_link_anchor_target_pairs)\
            .map(lambda (a,t): (a.strip().lower(), t))\
            .filter(lambda (a, t): a)\
            .mapValues(self.trim_subsection_link)\
            .mapValues(self.trim_link_protocol)\
            .groupByKey()\
            .mapValues(Counter)

    def format(self, model):
        return model\
            .map(lambda (anchor, counts): {
                '_id': anchor,
                'counts': [{
                    'target': target,
                    'count': count
                } for target, count in counts.iteritems()]
            })\
            .map(json.dumps)

    @classmethod
    def add_arguments(cls, p):
        p.set_defaults(modelcls=cls)
        return p

class EntityCounts(Model):
    """ Inlink counts """
    def build(self, corpus, threshold=1):
        return corpus\
            .flatMap(lambda d: d['links'])\
            .map(lambda l: l['target'])\
            .map(self.trim_subsection_link)\
            .map(self.trim_link_protocol)\
            .map(lambda l: (l, 1))\
            .reduceByKey(add)\
            .filter(lambda (t, c): c > threshold)

    def format(self, model):
        return model\
            .map(lambda (target, count): {
                '_id': target,
                'count': count
            })\
            .map(json.dumps)

    @classmethod
    def add_arguments(cls, p):
        p.set_defaults(modelcls=cls)
        return p
