import re
from cStringIO import StringIO
from warc import WARCFile
from dragnet import content_extractor, BlockifyError
from lxml import etree
from bs4 import BeautifulSoup
from sift.dataset import ModelBuilder, Model, Documents
from sift import logging
import pycld2 as cld
from pycld2 import error as cld_error

LINKS_RE = re.compile(r'<a href="(.+?)">(.+?)</a>')

class WARCCorpus(ModelBuilder, Model):
    def __init__(self, language=None):
        self.language = language

    @staticmethod
    def parse_warc_content(buf):
        try:
            wf = WARCFile(fileobj=StringIO(buf))
            record = wf.read_record()
            payload = record.payload.read()
            top = payload[:15]

            if top.startswith('HTTP/') and top.endswith('200 OK'):
                content_start = payload.find('\r\n\r\n')
                if content_start != -1:
                    yield record.url, payload[content_start+4:]
        except IOError:
            pass

    @staticmethod
    def try_get_lang(content):
        try:
            reliable, _, details = cld.detect(content)
            if reliable:
                return details[0][1]
        except cld_error:
            pass
        return None

    def build(self, sc, path):
        PAGE_DELIMITER = "WARC/1.0\r\n"
        warcs = sc\
            .newAPIHadoopFile(
                path,
                "org.apache.hadoop.mapreduce.lib.input.TextInputFormat",
                "org.apache.hadoop.io.LongWritable",
                "org.apache.hadoop.io.Text",
                conf = { "textinputformat.record.delimiter": PAGE_DELIMITER })\
            .filter(lambda (_, part): part)\
            .map(lambda (_, part): PAGE_DELIMITER+part.encode('utf-8'))\
            .flatMap(self.parse_warc_content)

        if self.language != None:
            warcs = warcs.filter(lambda (url, content): self.try_get_lang(content) == self.language)
        return warcs

    @staticmethod
    def format_item((url, content)):
        return {
            '_id': url,
            'content': content,
        }

class CommonCrawlArticles(ModelBuilder, Documents):
    THRESHOLD_CONTENT_SZ = 250000

    @staticmethod
    def clean_content((url, content)):
        try:
            blocks = content_extractor.analyze(content, blocks=True)
            content = ''.join(etree.tostring(b.features['block_start_element']) for b in blocks)
            if len(content) < CommonCrawlArticles.THRESHOLD_CONTENT_SZ:
                yield url, content
        except (BlockifyError, etree.SerialisationError):
            pass

    @staticmethod
    def parse_article(content):
        soup = BeautifulSoup(content, 'lxml')

        for tag in soup.find_all():
            if tag.name == 'a' and tag.attrs.get('href') and tag.text.strip():
                tag.attrs = {'href': tag.attrs['href']}
            else:
                tag.unwrap()

        return soup.encode_contents().decode('utf-8').strip()

    @staticmethod
    def extract_links(content):
        links = []
        offset = 0
        for match in LINKS_RE.finditer(content):
            target = match.group(1)
            anchor = match.group(2)
            start = match.start() - offset
            offset += len(match.group())-len(anchor)
            links.append((target, slice(start, start+len(anchor))))

        return LINKS_RE.sub(r'\2', content), links

    def build(self, corpus):
        return corpus\
            .map(lambda item: (item['_id'], item['content']))\
            .flatMap(self.clean_content)\
            .mapValues(self.parse_article)\
            .mapValues(self.extract_links)
