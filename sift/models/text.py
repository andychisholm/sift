import math
import numpy
import ujson as json
from bisect import bisect_left, bisect_right
from operator import add
from collections import Counter

from sift.dataset import Model
from sift.util import ngrams, iter_sent_spans, trim_link_subsection, trim_link_protocol

import logging
log = logging.getLogger()

class TermFrequencies(Model):
    """ Get term frequencies over a corpus """
    def __init__(self, **kwargs):
        self.lowercase = kwargs.pop('lowercase')
        self.max_ngram = kwargs.pop('max_ngram')

    def build(self, corpus):
        max_ngram = self.max_ngram

        m = corpus.map(lambda d: d['text'])
        if self.lowercase:
            m = m.map(unicode.lower)

        return m\
            .flatMap(lambda text: ngrams(text, max_ngram))\
            .map(lambda t: (t, 1))\
            .reduceByKey(add)\
            .filter(lambda (k,v): v > 1)

    def format_items(self, model):
        return model\
            .map(lambda (term, count): {
                '_id': term,
                'count': count,
            })

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('--lowercase', dest='lowercase', required=False, default=False, action='store_true')
        p.add_argument('--max-ngram', dest='max_ngram', required=False, default=1, type=int, metavar='MAX_NGRAM')
        return super(TermFrequencies, cls).add_arguments(p)

class EntityMentions(Model):
    """ Get aggregated sentence context around links in a corpus """
    @staticmethod
    def iter_mentions(doc):
        sent_spans = list(iter_sent_spans(doc['text']))
        sent_offsets = [s.start for s in sent_spans]

        for link in doc['links']:
            # align the link span over sentence spans in the document
            # mention span may cross sentence bounds if sentence tokenisation is dodgy
            # if so, the entire span between bounding sentences will be used as context
            sent_start_idx = bisect_right(sent_offsets, link['start']) - 1
            sent_end_idx = bisect_left(sent_offsets, link['stop']) - 1

            target = trim_link_subsection(link['target'])
            target = trim_link_protocol(target)

            sent_offset = sent_spans[sent_start_idx].start
            span = (link['start'] - sent_offset, link['stop'] - sent_offset)

            yield target, (span, doc['text'][sent_spans[sent_start_idx].start:sent_spans[sent_end_idx].stop])

    def build(self, corpus):
        return corpus\
            .flatMap(self.iter_mentions)\
            .groupByKey()\
            .mapValues(list)

    def format_items(self, model):
        return model\
            .map(lambda (link, mentions): {
                '_id': link,
                'mentions': mentions,
            })

class TermDocumentFrequencies(Model):
    """ Get document frequencies for terms in a corpus """
    def __init__(self, **kwargs):
        self.lowercase = kwargs.pop('lowercase')
        self.max_ngram = kwargs.pop('max_ngram')
        self.min_df = kwargs.pop('min_df')
        super(TermDocumentFrequencies, self).__init__(**kwargs)

    def build(self, corpus):
        log.info('Building df model: max-ngram=%i, min-df=%i', self.max_ngram, self.min_df)

        m = corpus.map(lambda d: d['text'])

        if self.lowercase:
            m = m.map(lambda text: text.lower())

        return m\
            .flatMap(lambda text: set(ngrams(text, self.max_ngram)))\
            .map(lambda t: (t, 1))\
            .reduceByKey(add)\
            .filter(lambda (k,v): v > self.min_df)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('--lowercase', dest='lowercase', required=False, default=False, action='store_true')
        p.add_argument('--max-ngram', dest='max_ngram', required=False, default=1, type=int, metavar='MAX_NGRAM')
        p.add_argument('--min-df', dest='min_df', required=False, default=1, type=int, metavar='MIN_DF')
        return super(TermDocumentFrequencies, cls).add_arguments(p)

class TermIndicies(TermDocumentFrequencies):
    """ Generate uniqe indexes for termed based on their document frequency ranking. """
    def build(self, corpus):
        return super(TermIndicies, self)\
            .build(corpus)\
            .map(lambda (t, df): (df, t))\
            .sortByKey(False)\
            .zipWithIndex()\
            .map(lambda ((df, t), idx): (t, (df, idx)))

class TermIdfs(TermIndicies):
    """ Compute tf-idf weighted token counts over sentence contexts around links in a corpus """
    def __init__(self, **kwargs):
        self.min_rank = kwargs.pop('min_rank')
        self.max_rank = kwargs.pop('max_rank')
        super(TermIdfs, self).__init__(**kwargs)

    def build(self, corpus):
        log.info('Counting documents in corpus...')
        N = float(corpus.count())

        log.info('Building idf model: N=%i, df-range=(%i, %i)', N, self.min_rank, self.max_rank)
        min_rank = self.min_rank
        max_rank = self.max_rank

        return super(TermIdfs, self)\
            .build(corpus)\
            .filter(lambda (token, (df, rank)): rank >= min_rank and rank < max_rank)\
            .map(lambda (token, (df, rank)): (token, df))\
            .mapValues(lambda df: math.log(N/df))

    def format_items(self, model):
        return model\
            .map(lambda (term, idf): {
                '_id': term,
                'idf': idf,
            })

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('--min-rank', dest='min_rank', required=False, default=100, type=int, metavar='MIN_RANK')
        p.add_argument('--max-rank', dest='max_rank', required=False, default=int(5e5), type=int, metavar='MAX_RANK')
        return super(TermIdfs, cls).add_arguments(p)

class EntityMentionTermFrequency(Model):
    """ Compute tf-idf weighted token counts over sentence contexts around links in a corpus """
    def __init__(self, **kwargs):
        self.idf_model = TermIdfs(**kwargs)
        self.normalize = kwargs.pop('normalize')
        self.filter_target = kwargs.pop('filter_target')
        super(EntityMentionTermFrequency, self).__init__(**kwargs)

    def build(self, corpus):
        idfs = self.idf_model.build(corpus)

        m = corpus.flatMap(EntityMentions.iter_mentions)

        if self.filter_target:
            log.info('Filtering mentions targeting: %s', self.filter_target)
            m = m.filter(lambda (target, _): target.startswith(self.filter_target))

        m = m\
            .map(lambda (target, (span, text)): (target, text))\
            .mapValues(lambda v: ngrams(v, self.idf_model.max_ngram))\
            .flatMap(lambda (target, tokens): (((target, t), 1) for t in tokens))\
            .reduceByKey(add)\
            .map(lambda ((target, token), count): (token, (target, count)))\
            .leftOuterJoin(idfs)\
            .filter(lambda (token, ((target, count), idf)): idf != None)\
            .map(lambda (token, ((target, count), idf)): (target, (token, math.sqrt(count)*idf)))\
            .groupByKey()

        return m.mapValues(self.normalize_counts if self.normalize else list)

    def format_items(self, model):
        return model\
            .map(lambda (link, counts): {
                '_id': link,
                'counts': dict(counts),
            })

    @staticmethod
    def normalize_counts(counts):
        norm = numpy.linalg.norm([v for _, v in counts])
        return [(k, v/norm) for k, v in counts]

    @classmethod
    def add_arguments(cls, p):
        TermIdfs.add_arguments(p)
        p.add_argument('--filter', dest='filter_target', required=False, default=None, metavar='FILTER')
        p.add_argument('--skip-norm', dest='normalize', action='store_false', default=True)
        return super(EntityMentionTermFrequency, cls).add_arguments(p)

class TermEntityIndex(Model):
    """ Build an inverted index mapping high idf terms to entities """
    def __init__(self, **kwargs):
        self.num_terms = kwargs.pop('num_terms')
        super(TermEntityIndex, self).__init__(**kwargs)

    def build(self, corpus):
        num_terms = self.num_terms
        return corpus\
            .map(lambda item: (item['_id'], item['counts'].items()))\
            .mapValues(lambda vs: sorted(vs, key=lambda (k,v): v, reverse=True)[:num_terms])\
            .flatMap(lambda (e, cs): ((t, (e, c)) for t, c in cs))\
            .groupByKey()

    def format_items(self, model):
        return model\
            .map(lambda (term, entities): {
                '_id': term,
                'entities': dict(entities),
            })

    @classmethod
    def add_arguments(cls, p):
        TermIdfs.add_arguments(p)
        p.add_argument('--num-terms', dest='num_terms', required=False, default=3, type=int, metavar='NUM_TERMS')
        return super(TermEntityIndex, cls).add_arguments(p)
