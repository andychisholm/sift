import ujson as json
from sift.build import DatasetBuilder
from sift.models import links, text, embeddings

class BuildDocModel(DatasetBuilder):
    """ Build a model over a corpus of text documents """
    def __init__(self, *args, **kwargs): 
        self.corpus_path = kwargs.pop('corpus_path')
        super(BuildDocModel, self).__init__(*args, **kwargs)

    def prepare(self, sc):
        return sc\
            .textFile(self.corpus_path)\
            .map(json.loads)

    @classmethod
    def providers(cls):
        return [
            links.EntityCounts,
            links.EntityNameCounts,
            links.EntityInlinks,
            text.TermIndicies,
            text.TermIdfs,
            text.TermFrequencies,
            text.TermDocumentFrequencies,
            text.EntityMentions,
            text.EntityMentionTermFrequency,
            embeddings.EntitySkipGramEmbeddings,
        ]

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('corpus_path', metavar='CORPUS_PATH')
        super(BuildDocModel, cls).add_arguments(p)
        return p
