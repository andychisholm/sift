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

Extract link counts over this corpus, saving records under the 'counts' directory as compressed json.
```bash
sift build-model wikidocs --save counts EntityCounts json
```

### Json Output

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

### Bulk Import
This format also allows for easy bulk import of results into mongo:
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
