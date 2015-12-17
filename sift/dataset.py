import ujson as json

class Model(object):
    def __init__(self, **kwargs):
        pass

    def prepare(self, sc):
        raise NotImplementedError

    def build(self, *args, **kwargs):
        raise NotImplementedError

    def format_items(self, dataset):
        raise NotImplementedError

    @classmethod
    def add_arguments(cls, p):
        p.set_defaults(modelcls=cls)
        return p

class DocumentModel(Model):
    def __init__(self, **kwargs):
        self.corpus_path = kwargs.pop('corpus_path')

    def prepare(self, sc):
        return {
            'corpus': sc.textFile(self.corpus_path).map(json.loads)
        }

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('corpus_path', metavar='CORPUS_PATH')
        return super(DocumentModel, cls).add_arguments(p)
