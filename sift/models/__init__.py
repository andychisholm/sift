class Model(object):
    def __init__(self):
        pass

    def build(self, corpus):
        raise NotImplementedError

    def format(self, model):
        raise NotImplementedError
