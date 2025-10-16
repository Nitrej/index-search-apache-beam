import argparse
import json
import math
import os
from collections import Counter

from utils import tokenize 

def load_index(json_path):
    """Loads the TF-IDF index and calculates document vector norms (for cosine similarity)."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Index file not found at: {json_path}")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    
    for d in docs:
        d['norm'] = math.sqrt(sum(v*v for v in d['tfidf'].values()))
    return docs

def query_to_vector(query, docs):
    """Creates a TF-IDF vector for the query based on index statistics."""
    
    df = Counter()
    N = len(docs)
    all_tokens = set()
    for d in docs:
        all_tokens.update(d['tfidf'].keys())
    
    for token in all_tokens:
        df[token] = sum(1 for d in docs if token in d['tfidf'])
        
    q_tokens = tokenize(query)
    q_counts = Counter(q_tokens)

    idf = {
        t: math.log(1 + (N) / float(df.get(t, 1))) 
        for t in all_tokens if df.get(t, 0) > 0 
    }
    
    qvec = {
        t: q_counts[t] * idf.get(t, 0.0) 
        for t in q_tokens if t in idf
    }
            
    qnorm = math.sqrt(sum(v*v for v in qvec.values()))
    return qvec, qnorm

def cosine_similarity(qvec, qnorm, doc):
    """Calculates the cosine similarity between the query vector and the document vector."""
    if qnorm == 0 or doc['norm'] == 0:
        return 0.0
        
    dot = sum(qvec.get(t, 0.0) * doc['tfidf'].get(t, 0.0) for t in qvec)
    
    return dot / (qnorm * doc['norm'])

def search(json_path, query, top_k=10):
    """Searches the index for a given query."""
    try:
        docs = load_index(json_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return []
        
    qvec, qnorm = query_to_vector(query, docs)
    
    if qnorm == 0:
        print("Warning: Query contains only stopwords or unknown terms, resulting in zero query vector norm.")
        return []
        
    results = [(d['id'], d['path'], cosine_similarity(qvec, qnorm, d)) for d in docs]
    results = [r for r in results if r[2] > 0]
    results.sort(key=lambda x: x[2], reverse=True)
    
    return results[:top_k]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search a TF-IDF index built by indexer.py')
    subparsers = parser.add_subparsers(dest='cmd')

    p_search = subparsers.add_parser('search', help='Perform a search query.')
    p_search.add_argument('--index', default='index_tfidf.json', help='TF-IDF JSON index file')
    p_search.add_argument('--query', required=True, help='Search query string')
    p_search.add_argument('--top_k', type=int, default=10, help='Number of results to return')

    args = parser.parse_args()
    
    if args.cmd == 'search':
        results = search(args.index, args.query, top_k=args.top_k)
        if not results:
            print('No results found.')
        else:
            print(f"Found {len(results)} results (max {args.top_k}):")
            for doc_id, path, score in results:
                print(f'{score:.6f}\t{path}\t{doc_id}')
    else:
        parser.print_help()
