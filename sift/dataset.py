import ujson as json

class ModelBuilder(object):
    def __init__(self, *args, **kwargs): pass

    def __call__(self, *args, **kwargs):
        return self.build(*args, **kwargs).map(self.format_item)

    def build(self, *args, **kwargs):
        raise NotImplementedError

class Model(object):
    @staticmethod
    def format_item(item):
        raise NotImplementedError

    @staticmethod
    def load(sc, path, fmt=json):
        return sc.textFile(path).map(json.loads)

    @staticmethod
    def save(m, path, fmt=json):
        m.map(json.dumps).saveAsTextFile(path, 'org.apache.hadoop.io.compress.GzipCodec')

class Redirects(Model):
    @staticmethod
    def format_item((source, target)):
        return {'_id': source, 'target': target}

class Vocab(Model):
    @staticmethod
    def format_item((term, (count, rank))):
        return {
            '_id': term,
            'count': count,
            'rank': rank
        }

class Mentions(Model):
    @staticmethod
    def format_item((target, source, text, span)):
        return {
            '_id': target,
            'source': source,
            'text': text,
            'span': span
        }

class IndexedMentions(Model):
    @staticmethod
    def format_item((target, source, text, span)):
        return {
            '_id': target,
            'source': source,
            'sequence': text,
            'span': span
        }

class Documents(Model):
    @staticmethod
    def format_item((uri, (text, links))):
        return {
            '_id': uri,
            'text': text,
            'links': [{
                 'target': target,
                 'start': span.start,
                 'stop': span.stop
             } for target, span in links]
        }