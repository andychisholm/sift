import ujson as json

class Model(object):
    def __init__(self, **kwargs):
        self.corpus_path = kwargs.pop('corpus_path')

    def prepare(self, sc):
        return sc\
            .textFile(self.corpus_path)\
            .map(json.loads)

    def build(self, corpus):
        raise NotImplementedError

    def format_items(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('corpus_path', metavar='CORPUS_PATH')
        p.set_defaults(modelcls=cls)
        return p
