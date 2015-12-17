import re
import os
import shutil
import textwrap
import argparse
import ujson as json

from pyspark import SparkContext, SparkConf
from sift.format import ModelFormat

import logging
log = logging.getLogger() 

class DatasetBuilder(object):
    """ Wrapper for modules which extract models of entities or text from a corpus of linked documents """
    def __init__(self, **kwargs):
        self.output_path = kwargs.pop('output_path')
        self.sample = kwargs.pop('sample')

        fmtcls = kwargs.pop('fmtcls')
        fmt_args = {p:kwargs[p] for p in fmtcls.__init__.__code__.co_varnames if p in kwargs}
        self.formatter = fmtcls(**fmt_args)

        modelcls = kwargs.pop('modelcls')
        self.model_name = re.sub('([A-Z])', r' \1', modelcls.__name__).strip()

        log.info("Building %s...", self.model_name)
        self.model = modelcls(**kwargs)

    def __call__(self):
        c = SparkConf().setAppName('Build %s' % self.model_name)

        log.info('Using spark master: %s', c.get('spark.master'))
        sc = SparkContext(conf=c)

        kwargs = self.model.prepare(sc)
        m = self.model.build(**kwargs)
        m = self.model.format_items(m)
        m = self.formatter(m)

        if self.output_path:
            log.info("Saving to: %s", self.output_path)
            if os.path.isdir(self.output_path):
                log.warn('Writing over output path: %s', self.output_path)
                shutil.rmtree(self.output_path)
            m.saveAsTextFile(self.output_path, 'org.apache.hadoop.io.compress.GzipCodec')
        elif self.sample > 0:
            print '\n'.join(str(i) for i in m.take(self.sample))

        log.info('Done.')

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('--save', dest='output_path', required=False, default=None, metavar='OUTPUT_PATH')
        p.add_argument('--sample', dest='sample', required=False, default=1, type=int, metavar='NUM_SAMPLES')
        p.set_defaults(cls=cls)

        sp = p.add_subparsers()
        for modelcls in cls.providers():
            name = modelcls.__name__
            help_str = modelcls.__doc__.split('\n')[0]
            desc = textwrap.dedent(modelcls.__doc__.rstrip())
            csp = sp.add_parser(name,
                                help=help_str,
                                description=desc,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
            modelcls.add_arguments(csp)
            cls.add_formatter_arguments(csp)

        return p

    @classmethod
    def add_formatter_arguments(cls, p):
        sp = p.add_subparsers()
        for fmtcls in ModelFormat.iter_options():
            name = fmtcls.__name__.lower()
            if name.endswith('format'):
                name = name[:-len('format')]
            help_str = fmtcls.__doc__.split('\n')[0]
            desc = textwrap.dedent(fmtcls.__doc__.rstrip())
            csp = sp.add_parser(name,
                                help=help_str,
                                description=desc,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
            fmtcls.add_arguments(csp)
        return p
