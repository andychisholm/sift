import argparse
import re
import sys
import textwrap

from . import build

import logging
log = logging.getLogger()

logFormat = '%(asctime)s|%(levelname)s|%(module)s|%(message)s'
logging.basicConfig(format=logFormat)
log = logging.getLogger()
log.setLevel(logging.INFO)

APPS = [
    build.BuildModel
]

def main(args=sys.argv[1:]):
    p = argparse.ArgumentParser(description='Extract text models from corpus of documents.')
    sp = p.add_subparsers()
    for cls in APPS:
        name = re.sub('([A-Z])', r'-\1', cls.__name__).lstrip('-').lower()
        help = cls.__doc__.split('\n')[0]
        desc = textwrap.dedent(cls.__doc__.rstrip())
        csp = sp.add_parser(name,
                            help=help,
                            description=desc,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
        cls.add_arguments(csp)
    namespace = vars(p.parse_args(args))
    cls = namespace.pop('cls')
    try:
        obj = cls(**namespace)
    except ValueError as e:
        p.error(str(e))
    obj()
