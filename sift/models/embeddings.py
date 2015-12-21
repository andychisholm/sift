from operator import add
from itertools import chain

from sift.models.text import EntityMentions
from sift.util import ngrams
from sift.dataset import DocumentModel

import logging
log = logging.getLogger()

class EntitySkipGramEmbeddings(DocumentModel):
    """ Learn distributed representations for words and entities in a corpus via skip-gram embedding """
    def __init__(self, **kwargs):
        self.dimensions = kwargs.pop('dimensions')
        self.lowercase = kwargs.pop('lowercase')
        self.min_word_count = kwargs.pop('min_word_count')
        self.min_entity_count = kwargs.pop('min_entity_count')
        self.filter_target = kwargs.pop('entity_prefix')
        self.exclude_words = kwargs.pop('exclude_words')
        self.exclude_entities = kwargs.pop('exclude_entities')
        self.workers = kwargs.pop('workers')
        self.coalesce = kwargs.pop('coalesce')
        super(EntitySkipGramEmbeddings, self).__init__(**kwargs)

    def get_trim_rule(self):
        from gensim.utils import RULE_KEEP, RULE_DISCARD
        def trim_rule(word, count, min_count):
            if not word.startswith(self.filter_target):
                return RULE_KEEP if count >= self.min_word_count else RULE_DISCARD
            else:
                return RULE_KEEP if count >= self.min_entity_count else RULE_DISCARD
            return RULE_KEEP
        return trim_rule

    def build(self, corpus):
        from gensim.models.word2vec import Word2Vec
        sentences = corpus\
            .flatMap(EntityMentions.iter_mentions)\
            .filter(lambda (target, (span, text)): target.startswith(self.filter_target))\

        if self.lowercase:
            sentences = sentences.map(lambda (target, (span, text)): (target, (span, text.lower())))

        sentences = sentences\
            .map(lambda (target, ((s,e), text)): list(chain(ngrams(text[:s],1), [target], ngrams(text[e:],1))))

        if self.coalesce:
            sentences = sentences.coalesce(self.coalesce)

        sentences = sentences.cache()

        model = Word2Vec(sample=1e-5, size=self.dimensions, workers=self.workers)

        log.info('Preparing corpus...')
        model.corpus_count = sentences.count()

        log.info('Computing vocab statistics...')
        term_counts = sentences\
            .flatMap(lambda tokens: ((t, 1) for t in tokens))\
            .reduceByKey(add)\
            .filter(lambda (t, count): \
                (t.startswith(self.filter_target) and count >= self.min_entity_count) or \
                (count >= self.min_word_count))

        model.raw_vocab = dict(term_counts.collect())
        model.scale_vocab(trim_rule=self.get_trim_rule())
        model.finalize_vocab()

        log.info('Training local word2vec model...')
        model.train(sentences.toLocalIterator())

        log.info('Normalising embeddings...')
        model.init_sims(replace=True)

        total_entities = sum(1 if t.startswith(self.filter_target) else 0 for t in model.vocab.iterkeys())
        total_words = len(model.vocab) - total_entities

        vocab_sz = 0
        if not self.exclude_entities:
            log.info('Including %i entity embeddings in exported vocab...', total_entities)
            vocab_sz += total_entities
        if not self.exclude_words:
            log.info('Including %i word embeddings in exported vocab...', total_words)
            vocab_sz += total_words

        log.info('Parallelizing %i learned embeddings...', vocab_sz)
        return corpus\
            .context\
            .parallelize(
                (t, model.syn0[vi.index].tolist())
                for t, vi in model.vocab.iteritems()
                    if (not self.exclude_entities and t.startswith(self.filter_target)) or
                       (not self.exclude_words and not t.startswith(self.filter_target)))

    def format_items(self, model):
        return model\
            .map(lambda (entity, embedding): {
                '_id': entity,
                'embedding': embedding
            })

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('--dimensions', required=False, default=100, type=int)
        p.add_argument('--lowercase', dest='lowercase', required=False, default=False, action='store_true')
        p.add_argument('--min-word-count', required=False, default=500, type=int)
        p.add_argument('--min-entity-count', required=False, default=10, type=int)
        p.add_argument('--entity-prefix', dest='entity_prefix', required=False, default='en.wikipedia.org/wiki/')
        p.add_argument('--exclude-words', dest='exclude_words', required=False, default=False, action='store_true')
        p.add_argument('--exclude-entities', dest='exclude_entities', required=False, default=False, action='store_true')
        p.add_argument('--workers', required=False, default=4, type=int)
        p.add_argument('--coalesce', required=False, default=None, type=int)
        return super(EntitySkipGramEmbeddings, cls).add_arguments(p)
