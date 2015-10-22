import textwrap
import argparse
from ..format import ModelFormat, JsonFormat

class Model(object):
    def __init__(self, **kwargs):
        fmtcls = kwargs.pop('fmtcls', JsonFormat)
        fmt_args = {p:kwargs[p] for p in fmtcls.__init__.__code__.co_varnames if p in kwargs}
        self.formatter = fmtcls(**fmt_args)

    def build(self, corpus):
        raise NotImplementedError

    def format_items(self, *args, **kwargs):
        raise NotImplementedError

    def format(self, model):
        return self.formatter(self.format_items(model))

    @classmethod
    def add_arguments(cls, p):
        p.set_defaults(modelcls=cls)

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
