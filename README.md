sift - Knowledge extraction from web data
================================================

__sift__ is a toolkit for extracting models of entities and text from a corpus of linked documents.


## What can it do?

__sift__ is written in python, runs on Spark and is completely modular.

Out of the box, you can:

- Convert wikipedia articles into json objects without all the mediawiki cruft
- Extract entity relations from wikidata and align them with wikipedia mentions
- Extract plain-text content html and WARC encoded web page crawls
- Model entity popularity, alternative names and relatedness using inlinks
- Preprocess text documents for machine learning pipelines
- Push output into datastores like MongoDB and Redis

## Quick Start

### Install
```bash
pip install git+http://git@github.com/wikilinks/sift.git
```

## Getting Started

To use sift, you'll need some data.

If you'd like to use Wikipedia data, sift includes a helper script for downloading the latest dumps.

Download the latest paritioned Wikipedia dump into the 'latest' directory.
```bash
download-wikipedia latest
```

Once you've got some data, take a look at the sample notebook: [sift.ipynb](sift.ipynb).

## Spark

__sift__ uses Spark to process corpora in parallel.

If you'd like to make use of an existing Spark cluster, ensure the `SPARK_HOME` environment variable is set.

If not, that's fine. `sift` will prompt you to download and run Spark locally, utilising multiple cores on your system.

## Datasets

[Web KB](github.com/andychisholm/web-kb) datasets built from commoncrawl data are available under a public S3 bucket: [s3.amazonaws.com/webkb](https://s3.amazonaws.com/webkb/)

- `docs-2017` is built from news articles under the [CC-NEWS](http://commoncrawl.org/2016/10/news-dataset-available/) collection from January to June 2017 ([sample](https://s3.amazonaws.com/webkb/docs-2017/part-00000))
- `web-201707` is built from a full web crawl for [July 2017](http://commoncrawl.org/2017/07/july-2017-crawl-archive-now-available/) filted to English language pages ([sample](https://s3.amazonaws.com/webkb/web-201707/part-00000.gz))

The web collection contains plain-text content, entity mentions and endpoint annotations extracted from 1.5 billion documents with over 4 billion web links.
Data is encoded in a simple one-JSON-blob-per-line structure.

For example, the first document in the collection is an article from 2012 describing an [upcoming tour by Nicki Minaj](http://1019ampradio.cbslocal.com/2012/11/06/nicki-minaj-promises-man-bits-on-her-upcoming-tour/):

```json
{
  "_id": "http://1019ampradio.cbslocal.com/2012/11/06/nicki-minaj-promises-man-bits-on-her-upcoming-tour/",
  "text": "Nicki Minaj has had quite the year. Currently in the U.K. on her Reloaded Tour she sat down with London DJ Tim Westwood and her U.K. Barbz for a Q & A session. While Nicki took questions from both Westwood and her fans one answer in particular caused the room to pay attention...",
  "links":[{
      "start": 0,
      "endpoint": 0.6358972797,
      "stop": 11,
      "target": "http://1019ampradio.cbslocal.com/tag/nicki-minaj"
    }, {
      "start": 145,
      "endpoint": 0.2769776554,
      "stop": 160,
      "target": "http://www.youtube.com/watch?v=vnyuhDBcQo0"
  }],
  "mentions":[{
      "start": 0,
      "stop": 11,
      "label": "PERSON"
    }, {
      "start": 53,
      "stop": 57,
      "label": "GPE"
    },
    // truncated
}
```
