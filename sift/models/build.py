import ujson as json
from sift.build import DatasetBuilder
from sift.models import links, text, embeddings

class BuildDocModel(DatasetBuilder):
    """ Build a model over a corpus of text documents """
    @classmethod
    def providers(cls):
        return [
            links.EntityCounts,
            links.EntityNameCounts,
            links.EntityInlinks,
            links.EntityVocab,
            links.NamePartCounts,
            text.TermVocab,
            text.TermIdfs,
            text.TermFrequencies,
            text.TermDocumentFrequencies,
            text.EntityMentions,
            text.MappedEntityMentions,
            text.EntityMentionTermFrequency,
            text.TermEntityIndex,
            embeddings.EntitySkipGramEmbeddings,
        ]
