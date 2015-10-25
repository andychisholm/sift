class Model(object):
    def __init__(self, *args, **kwargs):
        pass

    def build(self, corpus):
        raise NotImplementedError

    def format_items(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def add_arguments(cls, p):
        p.set_defaults(modelcls=cls)
        return p
