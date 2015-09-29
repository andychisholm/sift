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

## Example

### Entity Prominence in Wikipedia

Using the [wikijson](https://github.com/wikilinks/wikijson) framework, prepare a corpus of documents from Wikipedia
```bash
wkdl latest
wkjs process-dump latest wikidocs
```

Extract link counts over this corpus:
```bash
sift build-model wikidocs --save counts EntityCounts
```

Quick inspection of json formatted output:
```bash
zcat -r counts/*.gz | grep -E '/Apple_Inc."' | python -m json.tool
# {
#     "_id": "en.wikipedia.org/wiki/Apple_Inc.",
#     "count": 6379
# }
```

## Spark

__sift__ uses Spark to process corpora in parallel.

If you'd like to make use of an existing Spark cluster, ensure the `SPARK_HOME` environment variable is set.

If not, `sift` will prompt you to download and run Spark locally in standalone mode.
