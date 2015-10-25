import ujson as json

from operator import add
from collections import Counter

from sift.dataset import Model
from sift.util import trim_link_subsection, trim_link_protocol

import logging
log = logging.getLogger()

class EntityCounts(Model):
    """ Inlink counts """
    def build(self, corpus, threshold=1):
        return corpus\
            .flatMap(lambda d: d['links'])\
            .map(lambda l: l['target'])\
            .map(trim_link_subsection)\
            .map(trim_link_protocol)\
            .map(lambda l: (l, 1))\
            .reduceByKey(add)\
            .filter(lambda (t, c): c > threshold)

    def format_items(self, model):
        return model\
            .map(lambda (target, count): {
                '_id': target,
                'count': count
            })

class EntityNameCounts(Model):
    """ Entity counts by name """
    def __init__(self, **kwargs):
        self.lowercase = kwargs.pop('lowercase')
        super(EntityNameCounts, self).__init__(**kwargs)

    def iter_link_anchor_target_pairs(doc):
        for link in doc['links']:
            anchor = doc['text'][link['start']:link['stop']].strip()
            if self.lowercase:
                anchor = anchor.lower()
            yield anchor, link['target']

    def build(self, corpus):
        return corpus\
            .flatMap(self.iter_link_anchor_target_pairs)\
            .filter(lambda (a, t): a)\
            .mapValues(trim_link_subsection)\
            .mapValues(trim_link_protocol)\
            .groupByKey()\
            .mapValues(Counter)

    def format_items(self, model):
        return model\
            .map(lambda (anchor, counts): {
                '_id': anchor,
                'counts': dict(counts),
                'total': sum(counts.itervalues())
            })

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('--lowercase', dest='lowercase', required=False, default=False, action='store_true')
        return super(EntityNameCounts, cls).add_arguments(p)

class EntityInlinks(Model):
    """ Comention counts """
    def build(self, corpus):
        return corpus\
            .flatMap(lambda d: ((d['_id'], l) for l in set(l['target'] for l in d['links'])))\
            .mapValues(trim_link_subsection)\
            .mapValues(trim_link_protocol)\
            .map(lambda (k, v): (v, k))\
            .groupByKey()\
            .mapValues(list)

    def format_items(self, model):
        return model\
            .map(lambda (target, inlinks): {
                '_id': target,
                'inlinks': inlinks
            })
