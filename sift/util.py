from pattern.en import tokenize

def ngrams(text, n=2):
    for s in tokenize(text):
        s = s.lower().split()
        for i in xrange(n):
            for j in xrange(len(s)-i):
                yield ' '.join(s[j:j+i+1])
