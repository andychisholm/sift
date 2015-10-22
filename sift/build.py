import re
import os
import shutil
import textwrap
import argparse
import ujson as json

from pyspark import SparkContext, SparkConf

from .models import links, text

import logging
log = logging.getLogger() 

MODELS = [
    links.EntityCounts,
    links.EntityNameCounts,
    links.EntityComentions,
    links.EntityInlinks,
    text.TermIndicies,
    text.TermFrequencies,
    text.TermDocumentFrequencies,
    text.EntityMentions,
    text.EntityMentionTermFrequency
]

class BuildModel(object):
    """ Wrapper for modules which extract models of entities or text using a corpus of linked documents """
    def __init__(self, **kwargs):
        self.corpus_path = kwargs.pop('corpus_path')
        self.output_path = kwargs.pop('output_path')
        self.sample = kwargs.pop('sample')
        self.sort = False

        modelcls = kwargs.pop('modelcls')
        self.model_name = re.sub('([A-Z])', r' \1', modelcls.__name__).strip()

        log.info("Building %s...", self.model_name)
        self.model = modelcls(**kwargs)

    def __call__(self):
        log.info('Processing corpus: %s ...', self.corpus_path)
        c = SparkConf().setAppName('Build %s' % self.model_name)

        log.info('Using spark master: %s', c.get('spark.master'))
        sc = SparkContext(conf=c)

        corpus = sc.textFile(self.corpus_path).map(json.loads)
        m = self.model.build(corpus)
        m = self.model.format(m)

        if self.sample > 0:
            if self.sort:
                m = m.map(lambda (k,v): (v,k)).sortByKey(False)
            print '\n'.join(str(i) for i in m.take(self.sample))
        elif self.output_path:
            log.info("Saving to: %s", self.output_path)
            if os.path.isdir(self.output_path):
                log.warn('Writing over output path: %s', self.output_path)
                shutil.rmtree(self.output_path)
            m.saveAsTextFile(self.output_path, 'org.apache.hadoop.io.compress.GzipCodec')

        log.info('Done.')

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('corpus_path', metavar='CORPUS_PATH')
        p.add_argument('--save', dest='output_path', required=False, default=None, metavar='OUTPUT_PATH')
        p.add_argument('--sample', dest='sample', required=False, default=0, type=int, metavar='N')
        p.set_defaults(cls=cls)

        sp = p.add_subparsers()
        for modelcls in MODELS:
            name = modelcls.__name__
            help_str = modelcls.__doc__.split('\n')[0]
            desc = textwrap.dedent(modelcls.__doc__.rstrip())
            csp = sp.add_parser(name,
                                help=help_str,
                                description=desc,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
            modelcls.add_arguments(csp)

        return p
