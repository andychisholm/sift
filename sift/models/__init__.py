class Model(object):
    def __init__(self):
        pass
    
    @staticmethod
    def trim_subsection_link(s):
        idx = s.find('#')
        return s if idx == -1 else s[:idx]

    @staticmethod
    def trim_link_protocol(s):
        idx = s.find('://')
        return s if idx == -1 else s[idx+3:]

    def build(self, corpus):
        raise NotImplementedError

    def format(self, model):
        raise NotImplementedError
