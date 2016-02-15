from sift.build import DatasetBuilder
from sift.corpora import wikipedia, redirects

class BuildCorpus(DatasetBuilder):
    """ Prepare a dataset from some corpus or knowledge base """
    @classmethod
    def providers(cls):
        return [
            wikipedia.WikipediaArticles,
            wikipedia.WikipediaRedirects,
            redirects.MapRedirects,
            redirects.RedirectDocuments,
        ]
