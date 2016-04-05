import ujson as json

from sift.corpora import wikicorpus
from sift.dataset import Model

from sift import logging
log = logging.getLogger()

class WikipediaCorpus(Model):
    def __init__(self, **kwargs):
        self.dump_path = kwargs.pop('dump_path')
        super(WikipediaCorpus, self).__init__(**kwargs)

    def prepare(self, sc):
        # dodgy yet effective text delimiter to split xml elements for the input rdd
        PAGE_DELIMITER = "\n  </page>\n"
        PAGE_START = '<page>\n'
        PAGE_END = '</page>'

        raw = sc.newAPIHadoopFile(
            self.dump_path,
            "org.apache.hadoop.mapreduce.lib.input.TextInputFormat",
            "org.apache.hadoop.io.LongWritable",
            "org.apache.hadoop.io.Text",
            conf = { "textinputformat.record.delimiter": PAGE_DELIMITER })
        
        return {
            "pages": raw\
                .map(lambda (_, part): (part.find(PAGE_START), part))\
                .filter(lambda (offset, _): offset >= 0)\
                .map(lambda (offset, content): content[offset:]+PAGE_END)\
                .map(wikicorpus.extract_page)
        }

    @classmethod
    def add_arguments(cls, p):
        super(WikipediaCorpus, cls).add_arguments(p)
        p.add_argument('dump_path', metavar='WIKIDUMP_PATH')
        return p

class WikipediaArticles(WikipediaCorpus):
    """ Prepare a corpus of documents from wikipedia """
    def __init__(self, *args, **kwargs):
        self.redirects_path = kwargs.pop('redirects_path')
        super(WikipediaArticles, self).__init__(*args, **kwargs)

    def prepare(self, sc):
        p = super(WikipediaArticles, self).prepare(sc)
        p.update({
            "redirects": WikipediaRedirects.load(sc, self.redirects_path) if self.redirects_path else None
        })
        return p

    def build(self, pages, redirects = None):
        articles = pages\
            .filter(lambda info: info[1] == '0' and info[3] == None and info[4])\
            .map(lambda info: (info[0], info[4]))\
            .mapValues(wikicorpus.remove_markup)\
            .mapValues(wikicorpus.extract_links)

        if redirects:
            articles.cache()

            # redirect set is typically too large to be broadcasted for a map-side join
            articles = articles\
                .flatMap(lambda (pid, (text, links)): ((t, (pid, span)) for t, span in links))\
                .leftOuterJoin(redirects)\
                .map(lambda (t, ((pid, span), r)): (pid, (r if r else t, span)))\
                .groupByKey()\
                .mapValues(list)\
                .join(articles)\
                .map(lambda (pid, (links, (text, _))): (pid, (text, links)))

        return articles

    def format_items(self, corpus):
        return corpus\
            .map(lambda (uri, (text, links)): {
                '_id': uri,
                'text': text,
                'links': [{
                    'target': target,
                    'start': span.start,
                    'stop': span.stop
                } for target, span in links]
            })

    @classmethod
    def add_arguments(cls, p):
        super(WikipediaArticles, cls).add_arguments(p)
        p.add_argument('--redirects', dest='redirects_path', required=False, default=None, metavar='REDIRECTS_PATH')
        return p

class WikipediaRedirects(WikipediaCorpus):
    """ Extract a set of redirects from wikipedia """
    def __init__(self, *args, **kwargs):
        self.resolve_transitive = kwargs.pop('resolve_transitive')
        super(WikipediaRedirects, self).__init__(*args, **kwargs)

    def build(self, pages):
        pfx = wikicorpus.wikilink_prefix
        redirects = pages\
            .filter(lambda info: info[3] != None)\
            .map(lambda info: (info[0], info[3]))\
            .mapValues(wikicorpus.normalise_wikilink)\
            .map(lambda (s, t): (pfx+s, pfx+t))

        if self.resolve_transitive:
            redirects = redirects.cache()

            num_targets = redirects\
                .map(lambda (k,v): v)\
                .distinct()\
                .count()

            redirects = redirects\
                .map(lambda (s, t): (t, s)).leftOuterJoin(redirects)\
                .map(lambda (target, (source, redirect)): (source, redirect or target))\
                .cache()

            final_num_targets = redirects.map(lambda (k,v): v).distinct().count()
            log.info('Resolved %i transitive redirects...', num_targets - final_num_targets)

        return redirects.distinct()

    def format_items(self, corpus):
        return corpus\
            .map(lambda (source, target): {
                '_id': source,
                'target': target
            })

    @staticmethod
    def load(sc, path, fmt=json):
        log.info('Using redirects: %s', path)
        return sc.textFile(path)\
            .map(fmt.loads)\
            .map(lambda r: (r['_id'], r['target']))

    @classmethod
    def add_arguments(cls, p):
        super(WikipediaRedirects, cls).add_arguments(p)
        p.add_argument('--no-transitive', dest='resolve_transitive', required=False, default=True, action='store_false')
        return p
