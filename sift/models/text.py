import math
import numpy
import ujson as json
from bisect import bisect_left, bisect_right
from operator import add
from collections import Counter

from sift.models import Model
from sift.util import ngrams, iter_sent_spans, trim_link_subsection, trim_link_protocol

import logging
log = logging.getLogger()

class TermFrequencies(Model):
    """ Get term frequencies over a corpus """
    def build(self, corpus, n = 1):
        return corpus\
            .flatMap(lambda d: ngrams(d['text'], n))\
            .map(lambda t: (t, 1))\
            .reduceByKey(add)\
            .filter(lambda (k,v): v > 1)

    def format_items(self, model):
        return model\
            .map(lambda (term, count): {
                '_id': term,
                'count': count,
            })

class TermDocumentFrequencies(TermFrequencies):
    """ Get document frequencies for terms in a corpus """
    def build(self, corpus, n = 1):
        return corpus\
            .flatMap(lambda d: set(ngrams(d['text'], n)))\
            .map(lambda t: (t, 1))\
            .reduceByKey(add)\
            .filter(lambda (k,v): v > 1)

class EntityMentions(Model):
    """ Get aggregated sentence context around links in a corpus """
    @staticmethod
    def iter_mentions(doc):
        sent_spans = list(iter_sent_spans(doc['text']))
        sent_offsets = [s.start for s in sent_spans]

        for link in doc['links']:
            # align the link span over sentence spans in the document
            sent_start_idx = bisect_right(sent_offsets, link['start']) - 1
            sent_end_idx = bisect_left(sent_offsets, link['stop']) - 1

            target = trim_link_subsection(link['target'])
            target = trim_link_protocol(target)

            # mention span may cross sentence bounds if sentence tokenisation is dodgy
            # if so, the entire span between bounding sentences will be used as context
            yield target, doc['text'][sent_spans[sent_start_idx].start:sent_spans[sent_end_idx].stop]

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

class TermIndicies(TermDocumentFrequencies):
    """ Generate uniqe indexes for termed based on their document frequency ranking. """
    def build(self, corpus):
        return super(TermIndicies, self)\
            .build(corpus)\
            .map(lambda (t, df): (df, t))\
            .sortByKey(False)\
            .zipWithIndex()\
            .map(lambda ((df, t), idx): (t, (df, idx)))

class TermIdfs(Model):
    """ Compute tf-idf weighted token counts over sentence contexts around links in a corpus """
    def __init__(self, **kwargs):
        self.max_ngram = kwargs.pop('max_ngram')
        self.min_rank = kwargs.pop('min_rank')
        self.max_rank = kwargs.pop('max_rank')
        super(TermIdfs, self).__init__(**kwargs)

    def build(self, corpus):
        N = float(corpus.count())

        # locals needed to keep spark happy
        max_ngram = self.max_ngram
        min_rank = self.min_rank
        max_rank = self.max_rank

        return TermIndicies()\
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
    def add_model_arguments(cls, p):
        p.add_argument('--max-ngram', dest='max_ngram', required=False, default=2, type=int, metavar='MAX_NGRAM')
        p.add_argument('--min-rank', dest='min_rank', required=False, default=50, type=int, metavar='MIN_RANK')
        p.add_argument('--max-rank', dest='max_rank', required=False, default=int(1e6), type=int, metavar='MAX_RANK')

    @classmethod
    def add_arguments(cls, p):
        cls.add_model_arguments(p)
        return super(TermIdfs, cls).add_arguments(p)

class EntityMentionTermFrequency(Model):
    """ Compute tf-idf weighted token counts over sentence contexts around links in a corpus """
    def __init__(self, **kwargs):
        self.max_ngram = kwargs.pop('max_ngram')
        self.min_rank = kwargs.pop('min_rank')
        self.max_rank = kwargs.pop('max_rank')
        self.normalize = kwargs.pop('normalize')
        super(EntityMentionTermFrequency, self).__init__(**kwargs)

    def build(self, corpus):
        log.info('Building tf-idf model: N=%i, ngrams=%i, df-range=(%i, %i), norm=%s', N, self.max_ngram, self.min_rank, self.max_rank, str(self.normalize))
        idfs = TermIdfs(
            max_ngram=self.max_ngram,
            min_rank=self.min_rank,
            max_rank=self.max_rank)

        m = corpus\
            .flatMap(EntityMentions.iter_mentions)\
            .mapValues(lambda v: ngrams(v, max_ngram))\
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
        TermIdfs.add_model_arguments(p)
        p.add_argument('--skip-norm', dest='normalize', action='store_false')
        p.set_defaults(normalize=True)
        return super(EntityMentionTermFrequency, cls).add_arguments(p)
