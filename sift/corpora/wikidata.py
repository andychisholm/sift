import ujson as json

from sift.corpora import wikicorpus
from sift.dataset import ModelBuilder, Model, Relations

from sift import logging
log = logging.getLogger()

ENTITY_PREFIX = 'Q'
PREDICATE_PREFIX = 'P'

class WikidataCorpus(ModelBuilder, Model):
    @staticmethod
    def iter_item_for_line(line):
        line = line.strip()
        if line != '[' and line != ']':
            yield json.loads(line.rstrip(',\n'))

    def build(self, sc, path):
        return sc\
            .textFile(path)\
            .flatMap(self.iter_item_for_line)\
            .map(lambda i: (i['id'], i))

    @staticmethod
    def format_item((wid, item)):
        return {
            '_id': wid,
            'data': item
        }

class WikidataRelations(ModelBuilder, Relations):
    """ Prepare a corpus of relations from wikidata """
    @staticmethod
    def iter_relations_for_item(item):
        for pid, statements in item.get('claims', {}).iteritems():
            for statement in statements:
                if statement['mainsnak'].get('snaktype') == 'value':
                    datatype = statement['mainsnak'].get('datatype')
                    if datatype == 'wikibase-item':
                        yield pid, int(statement['mainsnak']['datavalue']['value']['numeric-id'])
                    elif datatype == 'time':
                        yield pid, statement['mainsnak']['datavalue']['value']['time']
                    elif datatype == 'string' or datatype == 'url':
                        yield pid, statement['mainsnak']['datavalue']['value']

    def build(self, corpus):
        entities = corpus\
            .filter(lambda item: item['_id'].startswith(ENTITY_PREFIX))

        entity_labels = entities\
            .map(lambda item: (item['_id'], item['data'].get('labels', {}).get('en', {}).get('value', None)))\
            .filter(lambda (pid, label): label)\
            .map(lambda (pid, label): (int(pid[1:]), label))

        wiki_entities = entities\
            .map(lambda item: (item['data'].get('sitelinks', {}).get('enwiki', {}).get('title', None), item['data']))\
            .filter(lambda (e, _): e)\
            .cache()
       
        predicate_labels = corpus\
            .filter(lambda item: item['_id'].startswith(PREDICATE_PREFIX))\
            .map(lambda item: (item['_id'], item['data'].get('labels', {}).get('en', {}).get('value', None)))\
            .filter(lambda (pid, label): label)\
            .cache()

        relations = wiki_entities\
            .flatMap(lambda (eid, item): ((pid, (value, eid)) for pid, value in self.iter_relations_for_item(item)))\
            .join(predicate_labels)\
            .map(lambda (pid, ((value, eid), label)): (value, (label, eid)))

        return relations\
            .leftOuterJoin(entity_labels)\
            .map(lambda (value, ((label, eid), value_label)): (eid, (label, value_label or value)))\
            .groupByKey()\
            .mapValues(dict)
