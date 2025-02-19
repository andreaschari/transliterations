import uroman as ur
import ir_datasets
import logging
from tqdm import tqdm
import argparse
from pyterrier_anserini import AnseriniTokenizer
# import anserini tokenizer
tokenizer = AnseriniTokenizer.zh.tokenize

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# setup argparse
parser = argparse.ArgumentParser(description='Create transliterations for a dataset')

parser.add_argument('--dataset', type=str, help='The dataset to create transliterations for (in irds format)', required=True)
parser.add_argument('--lang', type=str, help='Language of queries to load any custom uroman rules (ISO 639-3 standard)', required=True)
# add optinal argument for transliterating the docs
parser.add_argument('--do_docs', action='store_true', help='Transliterate the documents instead of the queries')
args = parser.parse_args()

# Set up the output path and the dataset
output_path = "/root/nfs/CLIR/data/transliterations/"
dataset_name = args.dataset
lang = args.lang
do_docs = args.do_docs

# Load the uroman romanizer
romanizer = ur.Uroman()
# Load mMARCO
dataset = ir_datasets.load(dataset_name)

logging.info(f"Loaded dataset: {dataset_name}")

# change dataset name from / to _
dataset_name = dataset_name.replace("/", "_")

if do_docs:
    # No. of docs in any mmarco dataset: 8841823
    for doc in tqdm(dataset.docs_iter(), desc="Romanizing docs"):
        doc_id = doc.doc_id
        doc_text = doc.text
        romanized_doc = romanizer.romanize_string(doc_text, lang=lang)
        
        # save all docs to the same file
        with open(f"{output_path}/{dataset_name}_uroman_docs.tsv", "a") as f:
            f.write(f"{doc_id}\t{romanized_doc}\n")
else:
    for query in tqdm(dataset.queries_iter(), desc="Romanizing queries"):
        query_id = query.query_id
        if "neuclir" in dataset_name:
            ht_title = query.ht_title
            mt_title = query.mt_title
            ht_description = query.ht_description
  
            romanized_ht_title = romanizer.romanize_string(ht_title, lang=lang)
            romanized_mt_title = romanizer.romanize_string(mt_title, lang=lang)
            romanized_ht_desc = romanizer.romanize_string(ht_description, lang=lang)

            # save all queries to the same file in TSV format
            with open(f"{output_path}/{dataset_name}_uroman.tsv", "a") as f:
                f.write(f"{query_id}\t{romanized_ht_title}\t{romanized_mt_title}\t{romanized_ht_desc}\n")
        else:
            query_text = query.text
            # tokenize the query text
            # tokenized_query = tokenizer(query_text)
            # romanized_query = " ".join(tokenized_query)
            # romanized_query = romanizer.romanize_string(" ".join(tokenized_query), lang=lang)
            romanized_query = romanizer.romanize_string(query_text, lang=lang)
            
            # save all queries to the same file in TSV format
            with open(f"{output_path}/{dataset_name}_uroman.tsv", "a") as f:
                f.write(f"{query_id}\t{romanized_query}\n")
logging.info("Done")