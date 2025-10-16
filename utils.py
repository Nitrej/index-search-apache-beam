import apache_beam as beam
from apache_beam.pvalue import AsSingleton 


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