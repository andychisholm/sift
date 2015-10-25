import re
from pattern import en

# todo: use spacy tokenization
def ngrams(text, n=1):
    for i in xrange(n):
        for n in en.ngrams(text, n=i+1):
            yield ' '.join(n)

SENT_RE = re.compile('((?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|[\?!])\s)|(\s*\n\s*)')
def iter_sent_spans(text):
    last = 0
    for m in SENT_RE.finditer(text):
        yield slice(last, m.start())
        last = m.end()
    if last != len(text):
        yield slice(last, len(text))

def trim_link_subsection(s):
    idx = s.find('#')
    return s if idx == -1 else s[:idx]

def trim_link_protocol(s):
    idx = s.find('://')
    return s if idx == -1 else s[idx+3:]
