import apache_beam as beam
from apache_beam.pvalue import AsSingleton 
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to

import uuid
import re
import json
import math
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.data import find as nltk_find 

try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words('english'))

TOKEN_RE = re.compile(r"[a-zA-Z]+")

def tokenize(text, lemmatizer=None): 
    """Tokenizes text, removes stopwords, and lemmatizes.
    The lemmatizer is passed in Beam, created in searcher.py."""
    text = text.lower()
    tokens = TOKEN_RE.findall(text)
    
    if not lemmatizer:
        try:
            lemmatizer = WordNetLemmatizer()
        except LookupError:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            lemmatizer = WordNetLemmatizer()
            
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens

# === APACHE BEAM CLASSES ===

class ReadFilesDoFn(beam.DoFn):
    """Reads a file and returns a dictionary with text and metadata.
    Accepts a tuple (UUID, path)."""
    def process(self, elem):
        doc_id, path = elem
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
             text = f"Simulated text for {doc_id}" 
        except Exception:
            text = ''
        yield {'doc_id': doc_id, 'path': path, 'text': text}

class DocToTokenCounts(beam.DoFn):
    """Calculates document Term Frequency (TF) and yields (token, doc_id) pairs for DF."""
    def setup(self):
        import nltk
        from nltk.stem import WordNetLemmatizer
        
        required_corpora = ['wordnet', 'omw-1.4']
        
        for resource in required_corpora:
            try:
                nltk_find(f'corpora/{resource}')
            except LookupError:
                print(f"NLTK: Downloading missing resource: {resource} on worker...")
                nltk.download(resource, quiet=True)
        
        self.lemmatizer = WordNetLemmatizer() 

    def process(self, elem):
        doc_id = elem['doc_id']
        path = elem['path']
        text = elem['text']
        
        tokens = tokenize(text, lemmatizer=self.lemmatizer) 
        
        counts = Counter(tokens)
        total = sum(counts.values())
        
        yield beam.pvalue.TaggedOutput('doc_tf', (doc_id, path, dict(counts), total))
        for token in counts.keys():
            yield beam.pvalue.TaggedOutput('df_pair', (token, doc_id))

class ComputeTfIdf(beam.DoFn):
    """Calculates the final TF-IDF score for each document, using DF and N as Side Inputs."""
    
    def process(self, element, df, N):
        doc_id, path, tf_counts, total_tokens = element
        
        tf_normalized = {
            token: count / total_tokens 
            for token, count in tf_counts.items()
        }
        
        idf = {
            token: math.log(1 + (N) / float(df.get(token, 1))) 
            for token in tf_counts.keys()
        } 
        
        tfidf = {
            token: tf_normalized[token] * idf.get(token, 0.0) 
            for token in tf_counts.keys()
        }
        
        yield {'id': doc_id, 'path': path, 'tfidf': tfidf}

# === APACHE BEAM UNIT TESTS ===

TEST_DOC_ID_1 = str(uuid.uuid4())
TEST_DOC_ID_2 = str(uuid.uuid4())

INPUT_FOR_TOKEN_COUNT = [
    {'doc_id': TEST_DOC_ID_1, 'path': 'path1', 'text': 'apple apple banana'},
    {'doc_id': TEST_DOC_ID_2, 'path': 'path2', 'text': 'banana orange'},
]

EXPECTED_DOC_TF = [
    (TEST_DOC_ID_1, 'path1', {'apple': 2, 'banana': 1}, 3), 
    (TEST_DOC_ID_2, 'path2', {'banana': 1, 'orange': 1}, 2), 
]
EXPECTED_DF_PAIR = [
    ('apple', TEST_DOC_ID_1),
    ('banana', TEST_DOC_ID_1),
    ('banana', TEST_DOC_ID_2),
    ('orange', TEST_DOC_ID_2),
]

EXPECTED_TFIDF = [
    {'id': TEST_DOC_ID_1, 'path': 'path1', 'tfidf': {'apple': 0.7324081924454064, 'banana': 0.23104906018664842}},
    {'id': TEST_DOC_ID_2, 'path': 'path2', 'tfidf': {'banana': 0.34657359027997264, 'orange': 0.5493061443340549}},
]

def test_doc_to_token_counts():
    """Tests if DocToTokenCounts correctly computes TF and generates DF pairs."""
    with TestPipeline() as p:
        input_pc = p | beam.Create(INPUT_FOR_TOKEN_COUNT)
        
        results = input_pc | beam.ParDo(DocToTokenCounts()).with_outputs('doc_tf', 'df_pair')
        
        doc_tf_pc = results.doc_tf
        assert_that(doc_tf_pc, equal_to(EXPECTED_DOC_TF), label='TestDocTF')
        
        df_pair_pc = results.df_pair
        assert_that(df_pair_pc, equal_to(EXPECTED_DF_PAIR), label='TestDFPair')

def test_compute_tfidf():
    """Tests if ComputeTfIdf correctly calculates TF-IDF with side inputs."""
    
    df_map = {'apple': 1, 'banana': 2, 'orange': 1}
    N_count = 2 
    input_doc_tf = [
        (TEST_DOC_ID_1, 'path1', {'apple': 2, 'banana': 1}, 3),
        (TEST_DOC_ID_2, 'path2', {'banana': 1, 'orange': 1}, 2),
    ]

    with TestPipeline() as p:
        
        df_pc = p | 'CreateDF' >> beam.Create([df_map])
        N_pc = p | 'CreateN' >> beam.Create([N_count])

        df_view = beam.pvalue.AsSingleton(df_pc)
        N_view = beam.pvalue.AsSingleton(N_pc)

        input_pc = p | 'CreateDocTF' >> beam.Create(input_doc_tf)
        
        output_pc = input_pc | 'ComputeTfIdfTest' >> beam.ParDo(
            ComputeTfIdf(),
            df=df_view,
            N=N_view
        )

        assert_that(output_pc, equal_to(EXPECTED_TFIDF), label='TestTfIdf')