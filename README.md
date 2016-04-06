sift - Text modelling framework
==================================

__sift__ is a toolkit for extracting models of entities and text from a corpus of linked documents.


## What can it do?

__sift__ is written in python, runs on Spark and is completely modular.

Out of the box, you can:

- Convert wikipedia articles into json objects without all the mediawiki cruft
- Model entity popularity, alternative names and relatedness
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
