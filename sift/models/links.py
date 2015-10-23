import ujson as json

from operator import add
from collections import Counter

from ..models import Model
from ..util import trim_link_subsection, trim_link_protocol

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

    def iter_link_anchor_target_pairs(self, doc):
        for link in doc['links']:
            anchor = doc['text'][link['start']:link['stop']]
            if self.lowercase:
                anchor = anchor.strip().lower()
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
    def add_model_arguments(cls, p):
        p.add_argument('--lowercase', dest='lowercase', required=False, default=False, action='store_true')

    @classmethod
    def add_arguments(cls, p):
        cls.add_model_arguments(p)
        return super(EntityNameCounts, cls).add_arguments(p)

class EntityInlinks(Model):
    """ Comention counts """
    def build(self, corpus):
        return corpus\
            .flatMap(lambda d: ((d['title'], l) for l in set(l['target'] for l in d['links'])))\
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

class EntityComentions(Model):
    """ Comention counts """
    def build(self, corpus):
        def iter_comentions(links):
            links = list(set(trim_link_protocol(trim_link_subsection(l['target'])) for l in links))
            for i in xrange(len(links)):
                yield links[i], Counter(links[:i] + links[i+1:])

        return corpus\
            .flatMap(lambda d: iter_comentions(d['links']))\
            .reduceByKey(add)

    def format_items(self, model):
        return model\
            .map(lambda (target, comentions): {
                '_id': target,
                'comentions': dict(comentions)
            })
