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

Once you've got some data, take a look at the sample notebook: [sift.ipynb](sift.ipynb).

## Spark

__sift__ uses Spark to process corpora in parallel.

If you'd like to make use of an existing Spark cluster, ensure the `SPARK_HOME` environment variable is set.

If not, `sift` will prompt you to download and run Spark locally in standalone mode.
