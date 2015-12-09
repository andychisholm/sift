sift - Text modelling framework
==================================

__sift__ is a toolkit for extracting models of entities and text from a corpus of linked documents.

## Quick Start

### Install
```bash
virtualenv ve
. ve/bin/activate
pip install git+http://git@github.com/wikilinks/sift.git
```

## Getting Started

To use sift, you'll need some data. Out of the box, sift includes utilities for downloading and extracting text from the latest Wikipedia dumps.

Download the latest paritioned Wikipedia dump into the 'latest' directory.
```bash
download-wikipedia latest
```

Extract a clean json formatted corpus of documents from raw Mediawiki markup, saving results under the 'wikidocs' directory.
```bash
sift build-corpus --save wikidocs WikipediaArticles latest json
```

Build a model over our corpus of processed Wikipedia documents which counts the number inlinks for each entity.
```bash
sift build-doc-model --save counts EntityCounts wikidocs json
```

### Output Formats

In the examples above, we passed the `json` flag for formatting results.

The json record format allows for easy inspection of model output:
```bash
zcat -r counts/*.gz | grep -E '/Apple_Inc."' | python -m json.tool
```

Result:
```javascript
{
    "_id": "en.wikipedia.org/wiki/Apple_Inc.",
    "count": 6379
}
```

This format also allows for easy bulk import of results into MongoDB:
```
zcat -r counts/*.gz | mongoimport --db models --collection counts
```

__sift__ also supports the generation of redis protcol via the `redis` format flag.

This allows for very fast bulk inserts via redis-cli:
```bash
zcat -r counts/*.gz | redis-cli --pipe
```

## Spark

__sift__ uses Spark to process corpora in parallel.

If you'd like to make use of an existing Spark cluster, ensure the `SPARK_HOME` environment variable is set.

If not, `sift` will prompt you to download and run Spark locally in standalone mode.
