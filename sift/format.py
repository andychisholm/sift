import cPickle as pickle
import ujson as json
import msgpack
import base64

class ModelFormat(object):
    def __init__(self):
        pass
    def __call__(self, model):
        raise NotImplemented

    @classmethod
    def iter_options(cls):
        yield JsonFormat
        yield RedisFormat
        yield TsvFormat

class TsvFormat(ModelFormat):
    """ Format model output as tab separated values """
    @staticmethod
    def items_to_tsv(items):
        key_order = None
        for item in items:
            if key_order == None:
                key_order = []
                if '_id' in item:
                    key_order.append('_id')
                key_order += sorted(k for k in item.iterkeys() if k != '_id')

            # todo: proper field serialization and escapes
            yield u'\t'.join(unicode(item[k]) for k in key_order).encode('utf-8')

    def __call__(self, model):
        return model.mapPartitions(self.items_to_tsv)

    @classmethod
    def add_arguments(cls, p):
        p.set_defaults(fmtcls=cls)
        return p

class JsonFormat(ModelFormat):
    """ Format model output as json """
    def __call__(self, model):
        return model.map(json.dumps)

    @classmethod
    def add_arguments(cls, p):
        p.set_defaults(fmtcls=cls)
        return p

class RedisFormat(ModelFormat):
    """ Format model output as redis protocol SET commands """
    def __init__(self, prefix, serializer, field):
        if serializer == 'raw' and not field:
            raise Exception("Target field required for raw serializer")

        self.prefix = prefix
        self.field = field
        self.serializer = {
            'json': json.dumps,
            'msgpack': lambda o: base64.b64encode(msgpack.dumps(o)),
            'pickle': lambda o: base64.b64encode(pickle.dumps(o, -1)),
            'raw': lambda o: o
        }[serializer]

    def to_value(self, item):
        if self.field:
            item = unicode(item[self.field])
        else:
            item.pop('_id', None)
        return self.serializer(item)

    def __call__(self, model):
        cmd = '\r\n'.join(["*3", "$3", "SET", "${}", "{}", "${}", "{}"])+'\r'
        return model\
            .map(lambda i: ((self.prefix+i['_id'].replace('"','\\"')).encode('utf-8'), self.to_value(i)))\
            .map(lambda (t, c): cmd.format(len(t), t, len(c), c))

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('--prefix', required=False, default='', metavar='PREFIX')
        p.add_argument('--serializer', choices=['json', 'pickle', 'msgpack', 'raw'], required=False, default='json', metavar='SERIALIZER')
        p.add_argument('--field', required=False, metavar='FIELD_TO_SERIALIZE')
        p.set_defaults(fmtcls=cls)
        return p
