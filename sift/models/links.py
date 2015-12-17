import ujson as json

from operator import add
from collections import Counter
from itertools import chain

from sift.dataset import DocumentModel
from sift.util import trim_link_subsection, trim_link_protocol, ngrams

import logging
log = logging.getLogger()

class EntityCounts(DocumentModel):
    """ Inlink counts """
    def __init__(self, **kwargs):
        self.threshold = kwargs.pop('threshold')
        self.filter_target = kwargs.pop('filter_target')
        super(EntityCounts, self).__init__(**kwargs)

    def build(self, corpus):
        links = corpus\
            .flatMap(lambda d: d['links'])\
            .map(lambda l: l['target'])\
            .map(trim_link_subsection)\
            .map(trim_link_protocol)\

        if self.filter_target:
            links = links.filter(lambda l: l.startswith(self.filter_target))

        return links\
            .map(lambda l: (l, 1))\
            .reduceByKey(add)\
            .filter(lambda (t, c): c > self.threshold)

    def format_items(self, model):
        return model\
            .map(lambda (target, count): {
                '_id': target,
                'count': count
            })

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('--threshold', required=False, default=1, type=int)
        p.add_argument('--filter', dest='filter_target', required=False, default=None)
        return super(EntityCounts, cls).add_arguments(p)

class EntityNameCounts(DocumentModel):
    """ Entity counts by name """
    def __init__(self, **kwargs):
        self.lowercase = kwargs.pop('lowercase')
        self.filter_target = kwargs.pop('filter_target')
        super(EntityNameCounts, self).__init__(**kwargs)

    def iter_anchor_target_pairs(self, doc):
        for link in doc['links']:
            target = link['target']
            target = trim_link_subsection(target)
            target = trim_link_protocol(target)

            anchor = doc['text'][link['start']:link['stop']].strip()

            if self.lowercase:
                anchor = anchor.lower()

            if anchor and target:
                yield anchor, target

    def build(self, corpus):
        m = corpus.flatMap(lambda d: self.iter_anchor_target_pairs(d))

        if self.filter_target:
            m = m.filter(lambda (a, t): t.startswith(self.filter_target))

        return m\
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
        p.add_argument('--filter', dest='filter_target', required=False, default=None)
        p.add_argument('--lowercase', dest='lowercase', required=False, default=False, action='store_true')
        return super(EntityNameCounts, cls).add_arguments(p)

class NamePartCounts(Model):
    """ Occurrence counts for ngrams at different positions within link anchors """
    def __init__(self, **kwargs):
        self.lowercase = kwargs.pop('lowercase')
        self.filter_target = kwargs.pop('filter_target')
        super(NamePartCounts, self).__init__(**kwargs)

    @staticmethod
    def iter_span_count_types(anchor, n):
        parts = list(ngrams(anchor, n, n))
        if parts:
            yield parts[0], 'B'
            yield parts[-1], 'E'
            for i in xrange(1, len(parts)-1):
                yield parts[i], 'I'

    def build(self, corpus):
        n = 2
        part_counts = corpus\
            .flatMap(self.iter_anchor_target_pairs)\
            .map(lambda (a,t): a)\
            .flatMap(lambda a: chain.from_iterable(self.iter_span_count_types(a, i) for i in xrange(1, n+1)))\
            .map(lambda p: (p, 1))\
            .reduceByKey(add)\
            .map(lambda ((term, spantype), count): (term, (spantype, count)))

        part_counts += corpus\
            .flatMap(lambda d: ngrams(d['text'], n))\
            .map(lambda t: (t, 1))\
            .reduceByKey(add)\
            .filter(lambda (t, c): c > 1)\
            .map(lambda (t, c): (t, ('O', c)))

        return part_counts\
            .groupByKey()\
            .mapValues(dict)\
            .filter(lambda (t, cs): 'O' in cs and len(cs) > 1)

    def format_items(self, model):
        return model\
            .map(lambda (term, part_counts): {
                '_id': term,
                'counts': dict(part_counts)
            })

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('--filter', dest='filter_target', required=False, default=None)
        p.add_argument('--lowercase', dest='lowercase', required=False, default=False, action='store_true')
        return super(NamePartCounts, cls).add_arguments(p)

class EntityInlinks(DocumentModel):
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

class EntityVocab(EntityCounts):
    """ Generate unique indexes for entities in a corpus. """
    def __init__(self, **kwargs):
        self.min_rank = kwargs.pop('min_rank')
        self.max_rank = kwargs.pop('max_rank')
        super(EntityVocab, self).__init__(**kwargs)

    def build(self, corpus):
        log.info('Building entity vocab: df rank range=(%i, %i)', self.min_rank, self.max_rank)
        m = super(EntityVocab, self)\
            .build(corpus)\
            .map(lambda (target, count): (count, target))\
            .sortByKey(False)\
            .zipWithIndex()\
            .map(lambda ((df, t), idx): (t, (df, idx)))

        if self.min_rank != None:
            m = m.filter(lambda (t, (df, idx)): idx >= self.min_rank)
        if self.max_rank != None:
            m = m.filter(lambda (t, (df, idx)): idx < self.max_rank)
        return m

    def format_items(self, model):
        return model\
            .map(lambda (term, (f, idx)): {
                '_id': term,
                'count': f,
                'rank': idx
            })

    @staticmethod
    def load(sc, path, fmt=json):
        log.info('Loading entity-index mapping: %s ...', path)
        return sc\
            .textFile(path)\
            .map(fmt.loads)\
            .map(lambda r: (r['_id'], (r['count'], r['rank'])))

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('--min-rank', dest='min_rank', required=False, default=0, type=int, metavar='MIN_RANK')
        p.add_argument('--max-rank', dest='max_rank', required=False, default=int(1e5), type=int, metavar='MAX_RANK')
        return super(EntityVocab, cls).add_arguments(p)
