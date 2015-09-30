import ujson as json
from bisect import bisect_left, bisect_right
from operator import add
from collections import Counter

from sift.models import Model
from sift.util import ngrams, iter_sent_spans, trim_link_subsection, trim_link_protocol

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
            })

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

class EntityMentions(Model):
    """ Get aggregated sentence context around links in a corpus """
    def build(self, corpus):
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

        return corpus\
            .flatMap(iter_mentions)\
            .groupByKey()\
            .mapValues(list)

    def format(self, model):
        return model\
            .map(lambda (link, mentions): {
                '_id': link,
                'mentions': mentions,
            })

    @classmethod
    def add_arguments(cls, p):
        p.set_defaults(modelcls=cls)
        return p

class EntityMentionTermFrequency(Model):
    """ Get aggregated sentence context around links in a corpus """
    def build(self, corpus):
        return EntityMentions()\
            .build(corpus)\
            .mapValues(lambda mentions: Counter(t for m in mentions for t in ngrams(m)))

    def format(self, model):
        return model\
            .map(lambda (link, counts): {
                '_id': link,
                'counts': dict(counts),
            })

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
