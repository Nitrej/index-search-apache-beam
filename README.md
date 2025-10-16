# index-search-apache-beam-app

TF-IDF Indexer and Searcher (Apache Beam & NLTK)

A scalable TF-IDF indexing and search application built with Apache Beam, NLTK, and Docker.

This project implements a distributed document indexing and search application using the TF-IDF (Term Frequency–Inverse Document Frequency) algorithm. It utilizes Apache Beam for scalable, pipeline-based indexing, and NLTK (Natural Language Toolkit) for advanced text preprocessing (tokenization, stopword removal, and lemmatization).

## Setup and Installation (Using Docker)

The easiest way to run this application is by building and using the provided Docker image.

Navigate to the project's root directory (where Dockerfile and Python files are located) and run the build command. This process will install Python dependencies and download the necessary NLTK data (wordnet, omw-1.4) inside the image.

```
docker build -t index-search-app .
```
# Usage Guide

The application has two main components: index (via indexer.py) and search (via searcher.py).

## A. Indexing Documents

You need a directory containing all your .txt files. We will map this directory and an output directory into the Docker container.

Create Data Directory: Place your documents in a local folder, e.g., ./data.

Run Indexer: Map your local ./data folder to /app/data inside the container, and map the current directory ($(pwd)) to /app/output to retrieve the generated index file.

Ensure you are running this from the project root.
```
docker run --rm \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd):/app/output" \
    index-search-app \
    python indexer.py index --input_dir /app/data --output /app/output/index_tfidf.json
```


The generated index_tfidf.json file will appear in your local project root.

## B. Searching the Index

Once the index is generated, you can run the search command. We map the generated index_tfidf.json file back into the container for the searcher to read.

Replace "your search phrase" with the query you want to use
```
docker run --rm \
    -v "$(pwd)/index_tfidf.json:/app/index.json" \
    index-search-app \
    python searcher.py search --index /app/index.json --query "your search phrase here" --top_k 5
```


Output Example:

```
Found 5 results (max 5):
0.987654    /app/data/document_a.txt    a3c2f0d9-...
0.543210    /app/data/document_b.txt    1e8b7f2a-...
...
```


## C. Running Unit Tests

To verify that the Apache Beam logic is correct, you can run the built-in tests:
```
docker run --rm tf-idf-app python indexer.py test
```
