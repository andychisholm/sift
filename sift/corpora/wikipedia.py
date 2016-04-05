import ujson as json

from sift.corpora import wikicorpus
from sift.dataset import ModelBuilder, Model, Redirects, Documents

from sift import logging
log = logging.getLogger()

class WikipediaCorpus(ModelBuilder, Model):
    def build(self, sc, path):
        PAGE_DELIMITER = "\n  </page>\n"
        PAGE_START = '<page>\n'
        PAGE_END = '</page>'
        return sc\
            .newAPIHadoopFile(
                path,
                "org.apache.hadoop.mapreduce.lib.input.TextInputFormat",
                "org.apache.hadoop.io.LongWritable",
                "org.apache.hadoop.io.Text",
                conf = { "textinputformat.record.delimiter": PAGE_DELIMITER })\
            .map(lambda (_, part): (part.find(PAGE_START), part))\
            .filter(lambda (offset, _): offset >= 0)\
            .map(lambda (offset, content): content[offset:]+PAGE_END)\
            .map(wikicorpus.extract_page)

    @staticmethod
    def format_item((title, ns, pid, redirect, content)):
        return {
            '_id': title,
            'pid': pid,
            'namespace': ns,
            'redirect': redirect,
            'content': content
        }

class WikipediaRedirects(ModelBuilder, Redirects):
    """ Extract a set of redirects from wikipedia """
    def __init__(self, resolve_transitive=False):
        self.resolve_transitive = resolve_transitive

    def build(self, pages, verbose=False):
        pfx = wikicorpus.wikilink_prefix
        redirects = pages\
            .filter(lambda page: page['redirect'] != None)\
            .map(lambda page: (page['_id'], page['redirect']))\
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
                .map(lambda (target, (source, redirect)): (source, redirect or target))

            if verbose:
                redirects = redirects.cache()
                final_num_targets = redirects.map(lambda (k,v): v).distinct().count()
                log.info('Resolved %i transitive redirects...', num_targets - final_num_targets)

        return redirects.distinct()

class WikipediaArticles(ModelBuilder, Documents):
    """ Prepare a corpus of documents from wikipedia """
    def build(self, corpus, redirects=None):
        articles = corpus\
            .filter(lambda page: page['namespace'] == '0' and page['redirect'] == None and page['content'])\
            .map(lambda page: (page['_id'], page['content']))\
            .map(wikicorpus.remove_markup)\
            .mapValues(wikicorpus.extract_links)

        if redirects:
            redirects = redirects.map(lambda r: (r['_id'], r['target']))
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