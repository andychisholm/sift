import ujson as json

from operator import add
from collections import Counter
from itertools import chain

from sift.dataset import ModelBuilder, Documents, Model
from sift.util import trim_link_subsection, trim_link_protocol, ngrams

from sift import logging
log = logging.getLogger()

class EntityCounts(ModelBuilder, Model):
    """ Inlink counts """
    def __init__(self, min_count=1, filter_target=None):
        self.min_count = min_count
        self.filter_target = filter_target

    def build(self, docs):
        links = docs\
            .flatMap(lambda d: d['links'])\
            .map(lambda l: l['target'])\
            .map(trim_link_subsection)\
            .map(trim_link_protocol)

        if self.filter_target:
            links = links.filter(lambda l: l.startswith(self.filter_target))

        return links\
            .map(lambda l: (l, 1))\
            .reduceByKey(add)\
            .filter(lambda (t, c): c > self.min_count)

    @staticmethod
    def format_item((target, count)):
        return {
            '_id': target,
            'count': count
        }

class EntityNameCounts(ModelBuilder, Model):
    """ Entity counts by name """
    def __init__(self, lowercase=False, filter_target=None):
        self.lowercase = lowercase
        self.filter_target = filter_target

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

    def build(self, docs):
        m = docs.flatMap(lambda d: self.iter_anchor_target_pairs(d))

        if self.filter_target:
            m = m.filter(lambda (a, t): t.startswith(self.filter_target))

        return m\
            .groupByKey()\
            .mapValues(Counter)

    @staticmethod
    def format_item((anchor, counts)):
        return {
            '_id': anchor,
            'counts': dict(counts),
            'total': sum(counts.itervalues())
        }

class NamePartCounts(ModelBuilder, Model):
    """
    Occurrence counts for ngrams at different positions within link anchors.
        'B' - beginning of span
        'E' - end of span
        'I' - inside span
        'O' - outside span
    """
    def __init__(self, max_ngram=2, lowercase=False, filter_target=None):
        self.lowercase = lowercase
        self.filter_target = filter_target
        self.max_ngram = max_ngram

    def iter_anchors(self, doc):
        for link in doc['links']:
            anchor = doc['text'][link['start']:link['stop']].strip()
            if self.lowercase:
                anchor = anchor.lower()
            if anchor:
                yield anchor

    @staticmethod
    def iter_span_count_types(anchor, n):
        parts = list(ngrams(anchor, n, n))
        if parts:
            yield parts[0], 'B'
            yield parts[-1], 'E'
            for i in xrange(1, len(parts)-1):
                yield parts[i], 'I'

    def build(self, docs):
        part_counts = docs\
            .flatMap(self.iter_anchors)\
            .flatMap(lambda a: chain.from_iterable(self.iter_span_count_types(a, i) for i in xrange(1, self.max_ngram+1)))\
            .map(lambda p: (p, 1))\
            .reduceByKey(add)\
            .map(lambda ((term, spantype), count): (term, (spantype, count)))

        part_counts += docs\
            .flatMap(lambda d: ngrams(d['text'], self.max_ngram))\
            .map(lambda t: (t, 1))\
            .reduceByKey(add)\
            .filter(lambda (t, c): c > 1)\
            .map(lambda (t, c): (t, ('O', c)))

        return part_counts\
            .groupByKey()\
            .mapValues(dict)\
            .filter(lambda (t, cs): 'O' in cs and len(cs) > 1)

    @staticmethod
    def format_item((term, part_counts)):
        return {
            '_id': term,
            'counts': dict(part_counts)
        }

class EntityInlinks(ModelBuilder, Model):
    """ Inlink sets for each entity """
    def build(self, docs):
        return docs\
            .flatMap(lambda d: ((d['_id'], l) for l in set(l['target'] for l in d['links'])))\
            .mapValues(trim_link_subsection)\
            .mapValues(trim_link_protocol)\
            .map(lambda (k, v): (v, k))\
            .groupByKey()\
            .mapValues(list)

    @staticmethod
    def format_item((target, inlinks)):
        return {
            '_id': target,
            'inlinks': inlinks
        }

class EntityVocab(ModelBuilder, Model):
    """ Generate unique indexes for entities in a corpus. """
    def __init__(self, min_rank=0, max_rank=10000):
        self.min_rank = min_rank
        self.max_rank = max_rank

    def build(self, docs):
        log.info('Building entity vocab: df rank range=(%i, %i)', self.min_rank, self.max_rank)
        m = super(EntityVocab, self)\
            .build(docs)\
            .map(lambda (target, count): (count, target))\
            .sortByKey(False)\
            .zipWithIndex()\
            .map(lambda ((df, t), idx): (t, (df, idx)))

        if self.min_rank != None:
            m = m.filter(lambda (t, (df, idx)): idx >= self.min_rank)
        if self.max_rank != None:
            m = m.filter(lambda (t, (df, idx)): idx < self.max_rank)
        return m

    @staticmethod
    def format_item((term, (f, idx))):
        return {
            '_id': term,
            'count': f,
            'rank': idx
        }

    @staticmethod
    def load(sc, path, fmt=json):
        log.info('Loading entity-index mapping: %s ...', path)
        return sc\
            .textFile(path)\
            .map(fmt.loads)\
            .map(lambda r: (r['_id'], (r['count'], r['rank'])))

class EntityComentions(ModelBuilder, Model):
    """ Entity comentions """
    @staticmethod
    def iter_unique_links(doc):
        links = set()
        for l in doc['links']:
            link = trim_link_subsection(l['target'])
            link = trim_link_protocol(link)
            if link not in links:
                yield link
                links.add(link)

    def build(self, docs):
        return docs\
            .map(lambda d: (d['_id'], list(self.iter_unique_links(d))))\
            .filter(lambda (uri, es): es)

    @staticmethod
    def format_item((uri, es)):
        return {
            '_id': uri,
            'entities': es
        }

class MappedEntityComentions(EntityComentions):
    """ Entity comentions with entities mapped to a numeric index """
    def build(self, docs, entity_vocab):
        ev = sc.broadcast(dict(ev.collect()))
        return super(MappedEntityComentions, self)\
            .build(docs)\
            .map(lambda (uri, es): (uri, [ev.value[e] for e in es if e in ev.value]))\
            .filter(lambda (uri, es): es)