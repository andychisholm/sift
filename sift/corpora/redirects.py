import urllib
import ujson as json

from sift.dataset import Model
from sift.util import trim_link_protocol, iter_sent_spans, ngrams

import logging
log = logging.getLogger()

class MapRedirects(Model):
    """ Map redirects """
    def __init__(self, *args, **kwargs):
        self.from_path = kwargs.pop('from_path')
        self.to_path = kwargs.pop('to_path')

    def prepare(self, sc):
        return {
            "from_rds": self.load(sc, self.from_path).cache(),
            "to_rds": self.load(sc, self.to_path).cache()
        }

    @staticmethod
    def map_redirects(source, target):
        return source\
            .map(lambda (s, t): (t, s))\
            .leftOuterJoin(target)\
            .map(lambda (t, (s, r)): (s, r or t))\
            .distinct()

    def build(self, from_rds, to_rds):
        # map source of destination kb
        # e.g. (a > b) and (a > c) becomes (b > c)
        mapped_to = to_rds\
            .leftOuterJoin(from_rds)\
            .map(lambda (s, (t, f)): (f or s, t))\

        # map target of origin kb
        # e.g. (a > b) and (b > c) becomes (a > c)
        mapped_from = from_rds\
            .map(lambda (s, t): (t, s))\
            .leftOuterJoin(mapped_to)\
            .map(lambda (t, (s, r)): (s, r))\
            .filter(lambda (s, t): t)

        rds = (mapped_from + mapped_to).distinct()
        rds.cache()

        log.info('Resolving transitive mappings over %i redirects...', rds.count())
        rds = self.map_redirects(rds, rds)

        log.info('Resolved %i redirects...', rds.count())
        return rds

    @staticmethod
    def load(sc, path, fmt=json):
        log.info('Using redirects: %s', path)
        return sc\
            .textFile(path)\
            .map(fmt.loads)\
            .map(lambda r: (r['_id'], r['target']))

    def format_items(self, model):
        return model\
            .map(lambda (source, target): {
                '_id': source,
                'target': target
            })

    @classmethod
    def add_arguments(cls, p):
        super(MapRedirects, cls).add_arguments(p)
        p.add_argument('from_path', metavar='FROM_REDIRECTS_PATH')
        p.add_argument('to_path', metavar='TO_REDIRECTS_PATH')
        return p
