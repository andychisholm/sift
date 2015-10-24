import re
from pattern.en import tokenize, parsetree

# todo: use spacy tokenization
def ngrams(text, n=1, lowercase=False):
    for s in tokenize(text):
        if lowercase:
            s = s.lower()
        s = s.split()
        for i in xrange(n):
            for j in xrange(len(s)-i):
                yield ' '.join(s[j:j+i+1])

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
