from operator import add
from itertools import chain

from sift.models.text import EntityMentions
from sift.util import ngrams
from sift.dataset import ModelBuilder, Model

from sift import logging
log = logging.getLogger()

class EntitySkipGramEmbeddings(ModelBuilder, Model):
    """ Learn distributed representations for words and entities in a corpus via skip-gram embedding """
    def __init__(
        self,
        dimensions=100,
        min_word_count=500,
        min_entity_count=10,
        entity_prefix='en.wikipedia.org/wiki/',
        exclude_words=False,
        exclude_entities=False,
        workers=4,
        coalesce=None,
        *args, **kwargs):

        self.dimensions = dimensions
        self.min_word_count = min_word_count
        self.min_entity_count = min_entity_count
        self.filter_target = entity_prefix
        self.exclude_words = exclude_words
        self.exclude_entities = exclude_entities
        self.workers = workers
        self.coalesce = coalesce

    def get_trim_rule(self):
        from gensim.utils import RULE_KEEP, RULE_DISCARD
        def trim_rule(word, count, min_count):
            if not word.startswith(self.filter_target):
                return RULE_KEEP if count >= self.min_word_count else RULE_DISCARD
            else:
                return RULE_KEEP if count >= self.min_entity_count else RULE_DISCARD
            return RULE_KEEP
        return trim_rule

    def build(self, mentions):
        from gensim.models.word2vec import Word2Vec
        sentences = mentions\
            .filter(lambda (target, source, text, span): target.startswith(self.filter_target))\

        sentences = sentences\
            .map(lambda (target, source, text, (s,e)): list(chain(ngrams(text[:s],1), [target], ngrams(text[e:],1))))

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

    @staticmethod
    def format_item(self, (entity, embedding)):
        return {
            '_id': entity,
            'embedding': embedding
        }