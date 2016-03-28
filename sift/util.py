import re
from pattern import en

# todo: use spacy tokenization
def ngrams(text, max_n=1, min_n=1):
    for i in xrange(min_n-1,max_n):
        for n in en.ngrams(text, n=i+1):
            yield ' '.join(n)


# sentences can't end with a single lowercase letter
SENT_NO_END_LC = "(?<!(\s[a-z]\.))"

# abbreviation sequences don't delimit
ABBREV = "(?<!\w\.\w.)"

SENT_POS_HEUR = [
    "(?<=\.|[\?!])"     # should end with punctuation
]
SENT_NEG_ABBREVS = '(('+')|('.join([
    "[Ii]nc",
    "[Pp]ty",
    "[Ll]td",
]) + '))'
SENT_NEG_HEUR = [
    "(?<!(\s[a-z]\.))",  # can't end with a single lowercase letter (e.g. "c.")
    "(?<!\w\.\w.)",      # abbreviation sequences don't delimit (e.g. "e.g.")
    "(?<![A-Z][a-z]\.)", # can't end in two character capitalised word (e.g. "Ph.D")
    "(?<!"+SENT_NEG_ABBREVS+"\.)" # can't end with a hardcoded abbreviation (e.g. "Inc.")
]
SENT_HEURISTICS = '('+''.join(SENT_NEG_HEUR+SENT_POS_HEUR)+')'

SENT_RE = re.compile('('+SENT_HEURISTICS+'\s)|(\s*\n\s*)')
def iter_sent_spans(text):
    last = 0
    for m in SENT_RE.finditer(text):
        if last != m.start():
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
