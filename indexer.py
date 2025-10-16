import argparse
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import glob
import os
import json
import uuid
import sys

from utils import (
    ReadFilesDoFn, 
    DocToTokenCounts, 
    ComputeTfIdf
)

def build_index(input_dir, output_file, tmp_dir=None):
    """Creates a TF-IDF index for .txt files from the input directory using Apache Beam."""
    
    paths = sorted(glob.glob(os.path.join(input_dir, "**", "*.txt"), recursive=True))
    docs = [(str(uuid.uuid4()), p) for p in paths]
    
    if not docs:
        print('No .txt files found in', input_dir)
        return

    options = PipelineOptions()
    
    with beam.Pipeline(options=options) as p:
        docs_pc = p | 'CreateDocs' >> beam.Create(docs)
        read = docs_pc | 'ReadFiles' >> beam.ParDo(ReadFilesDoFn())
        
        splits = read | 'ToCounts' >> beam.ParDo(DocToTokenCounts()).with_outputs('doc_tf', 'df_pair')

        doc_tf_pc = splits.doc_tf
        df_pairs_pc = splits.df_pair

        df_unique = (df_pairs_pc
                      | 'DF pair to key' >> beam.Map(lambda t: (t[0], t[1]))
                      | 'Distinct token-doc' >> beam.Distinct()
                      | 'Token to 1' >> beam.Map(lambda t: (t[0], 1))
                      | 'Count DF' >> beam.CombinePerKey(sum)
                      | 'DF To Dict' >> beam.combiners.ToDict()) 

        N_value = docs_pc | 'Count N' >> beam.combiners.Count.Globally()
        
        df_view = beam.pvalue.AsSingleton(df_unique)
        N_view = beam.pvalue.AsSingleton(N_value)

        docs_tfidf_pc = doc_tf_pc | 'ComputeTfIdf' >> beam.ParDo(
            ComputeTfIdf(),
            df=df_view,                      
            N=N_view                         
        )

        (docs_tfidf_pc 
         | 'CollectAll' >> beam.combiners.ToList()
         | 'DumpIndex' >> beam.Map(lambda final_list: 
             open(output_file, 'w', encoding='utf-8').write(
                 json.dumps(final_list, ensure_ascii=False, indent=2))))

    print(f'Indexed documents with TF-IDF into {output_file}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='English TF-IDF Indexer using Apache Beam.')
    subparsers = parser.add_subparsers(dest='cmd')

    p_index = subparsers.add_parser('index', help='Build the TF-IDF index.')
    p_index.add_argument('--input_dir', required=True, help='Directory with .txt files')
    p_index.add_argument('--output', default='index_tfidf.json', help='Output JSON file')

    p_search_placeholder = subparsers.add_parser('search', help='Placeholder. Run search operations using "python searcher.py search..."')

    args = parser.parse_args()

    if args.cmd == 'index':
        build_index(args.input_dir, args.output, tmp_dir=args.tmp_dir)
    elif args.cmd == 'search':
        print("Please run search operations using 'python searcher.py search ...'")
    else:
        parser.print_help()