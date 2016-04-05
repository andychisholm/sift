import math
import numpy
import ujson as json
from bisect import bisect_left, bisect_right
from operator import add
from collections import Counter

from sift.models.links import EntityVocab
from sift.dataset import ModelBuilder, Documents, Model, Mentions, IndexedMentions, Vocab
from sift.util import ngrams, iter_sent_spans, trim_link_subsection, trim_link_protocol

from sift import logging
log = logging.getLogger()

class TermFrequencies(ModelBuilder, Model):
    """ Get term frequencies over a corpus """
    def __init__(self, lowercase, max_ngram):
        self.lowercase = lowercase
        self.max_ngram = max_ngram

    def build(self, docs):
        m = docs.map(lambda d: d['text'])
        if self.lowercase:
            m = m.map(unicode.lower)

        return m\
            .flatMap(lambda text: ngrams(text, self.max_ngram))\
            .map(lambda t: (t, 1))\
            .reduceByKey(add)\
            .filter(lambda (k,v): v > 1)

    @staticmethod
    def format_item(self, (term, count)):
        return {
            '_id': term,
            'count': count,
        }

class EntityMentions(ModelBuilder, Mentions):
    """ Get aggregated sentence context around links in a corpus """
    def __init__(self, sentence_window = 1, lowercase=False):
        self.sentence_window = sentence_window
        self.lowercase = lowercase

    @staticmethod
    def iter_mentions(doc, window = 1):
        sent_spans = list(iter_sent_spans(doc['text']))
        sent_offsets = [s.start for s in sent_spans]

        for link in doc['links']:
            # align the link span over sentence spans in the document
            # mention span may cross sentence bounds if sentence tokenisation is dodgy
            # if so, the entire span between bounding sentences will be used as context
            sent_start_idx = bisect_right(sent_offsets, link['start']) - 1
            sent_end_idx = bisect_left(sent_offsets, link['stop']) - 1

            lhs_offset = window / 2
            rhs_offset = (window - lhs_offset) - 1
            sent_start_idx = max(0, sent_start_idx - lhs_offset)
            sent_end_idx = min(len(sent_spans)-1, sent_end_idx + rhs_offset)
            sent_offset = sent_spans[sent_start_idx].start

            span = (link['start'] - sent_offset, link['stop'] - sent_offset)
            target = trim_link_subsection(link['target'])
            target = trim_link_protocol(target)
            mention = doc['text'][sent_spans[sent_start_idx].start:sent_spans[sent_end_idx].stop]

            # filter out instances where the mention span is the entire sentence
            if span == (0, len(mention)):
                continue

            # filter out list item sentences
            sm = mention.strip()
            if not sm or sm.startswith('*') or sm[-1] not in '.!?"\'':
                continue

            yield target, doc['_id'], mention, span

    def build(self, docs):
        m = docs.flatMap(lambda d: self.iter_mentions(d, self.sentence_window))
        if self.lowercase:
            m = m.map(lambda (t, src, m, s): (t, src, m.lower(), s))
        return m

class IndexMappedMentions(EntityMentions, IndexedMentions):
    """ Entity mention corpus with terms mapped to numeric indexes """
    def build(self, sc, docs, vocab):
        tv = sc.broadcast(dict(vocab.map(lambda r: (r['_id'], r['rank'])).collect()))
        return super(IndexMappedMentions, self)\
            .build(docs)\
            .map(lambda m: self.transform(m, tv))

    @staticmethod
    def transform((target, source, text, span), vocab):
        vocab = vocab.value

        start, stop = span
        pre = list(ngrams(text[:start], 1))
        ins = list(ngrams(text[start:stop], 1))
        post = list(ngrams(text[stop:], 1))
        indexes = [vocab.get(t, len(vocab)-1) for t in (pre+ins+post)]

        return target, source, indexes, (len(pre), len(pre)+len(ins))

class TermDocumentFrequencies(ModelBuilder):
    """ Get document frequencies for terms in a corpus """
    def __init__(self, lowercase=False, max_ngram=1, min_df=2):
        self.lowercase = lowercase
        self.max_ngram = max_ngram
        self.min_df = min_df

    def build(self, docs):
        m = docs.map(lambda d: d['text'])
        if self.lowercase:
            m = m.map(lambda text: text.lower())

        return m\
            .flatMap(lambda text: set(ngrams(text, self.max_ngram)))\
            .map(lambda t: (t, 1))\
            .reduceByKey(add)\
            .filter(lambda (k,v): v > self.min_df)

class TermVocab(TermDocumentFrequencies, Vocab):
    """ Generate unique indexes for termed based on their document frequency ranking. """
    def __init__(self, max_rank, min_rank=100, *args, **kwargs):
        self.max_rank = max_rank
        self.min_rank = min_rank
        super(TermVocab, self).__init__(*args, **kwargs)

    def build(self, docs):
        m = super(TermVocab, self)\
            .build(docs)\
            .map(lambda (t, df): (df, t))\
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

class TermIdfs(TermDocumentFrequencies, Model):
    """ Compute tf-idf weighted token counts over sentence contexts around links in a corpus """
    def build(self, corpus):
        log.info('Counting documents in corpus...')
        N = float(corpus.count())
        dfs = super(TermIdfs, self).build(corpus)

        log.info('Building idf model: N=%i', N)
        return dfs\
            .map(lambda (term, (df, rank)): (term, df))\
            .mapValues(lambda df: math.log(N/df))

    @staticmethod
    def format_item((term, idf)):
        return {
            '_id': term,
            'idf': idf,
        }

class EntityMentionTermFrequency(ModelBuilder, Model):
    """ Compute tf-idf weighted token counts over sentence contexts around links in a corpus """
    def __init__(self, max_ngram=1, normalize = True):
        self.max_ngram = max_ngram
        self.normalize = normalize

    def build(self, mentions, idfs):
        m = mentions\
            .map(lambda (target, (span, text)): (target, text))\
            .mapValues(lambda v: ngrams(v, self.max_ngram))\
            .flatMap(lambda (target, tokens): (((target, t), 1) for t in tokens))\
            .reduceByKey(add)\
            .map(lambda ((target, token), count): (token, (target, count)))\
            .leftOuterJoin(idfs)\
            .filter(lambda (token, ((target, count), idf)): idf != None)\
            .map(lambda (token, ((target, count), idf)): (target, (token, math.sqrt(count)*idf)))\
            .groupByKey()

        return m.mapValues(self.normalize_counts if self.normalize else list)

    @staticmethod
    def normalize_counts(counts):
        norm = numpy.linalg.norm([v for _, v in counts])
        return [(k, v/norm) for k, v in counts]

    @staticmethod
    def format_item((link, counts)):
        return {
            '_id': link,
            'counts': dict(counts),
        }