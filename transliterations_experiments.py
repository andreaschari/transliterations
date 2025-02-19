import argparse
import pyterrier as pt
import pandas as pd
import os
import logging
from pyterrier_dr import BGEM3, FlexIndex
from ir_measures import R, MRR, nDCG

# load model paths
from models import model_paths

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

argparser = argparse.ArgumentParser()
argparser.add_argument("--lang", type=str, help="language of queries", required=True)
argparser.add_argument("--index", type=str, help="index path", required=True)
argparser.add_argument("--model", type=str, help="first stage retrieval model", required=True)
argparser.add_argument("--dataset", type=str, help="dataset to use (irds format)", default="mmarco/v2")
argparser.add_argument("--evaluate", action='store_true', help="whether to evaluate the results")
args = argparser.parse_args()

lang = args.lang
index_path = args.index
first_stage_model = args.model
dataset_name = args.dataset
evaluate = args.evaluate

# Check if the index directory exists
if not os.path.isdir(index_path):
    raise FileNotFoundError(f"The index: {index_path} does not exist")

if first_stage_model not in model_paths:
    raise ValueError(f"Model {first_stage_model} not supported")
model_name = model_paths[first_stage_model]

dataset_name_storage = dataset_name.replace("/", "_")

# Load Queries
if "neuclir" in dataset_name:
    dataset = pt.get_dataset(f"irds:{dataset_name}")
    queries_orig = dataset.get_topics(tokenise_query=False)
    # only use ht_title and mt_title
    queries_orig = queries_orig[["qid", "ht_title"]]
    queries_orig = queries_orig.rename(columns={"ht_title": "query"})
    # print no. of queries
    logging.info(f"Loaded {len(queries_orig)} queries for {dataset_name} (HT)")

    # pre-process transliterated queries (assumes they are already tokenized before romanisation)
    queries_translit = pd.read_csv(f"/root/nfs/CLIR/data/transliterations/{dataset_name_storage}_uroman.tsv", sep="\t", header=None, names=["qid", "ht_title", "mt_title", "ht_description"])
    queries_translit["qid"] = queries_translit["qid"].astype(str)
    queries_translit["ht_title"] = queries_translit["ht_title"].astype(str)
    queries_translit["mt_title"] = queries_translit["mt_title"].astype(str)
    queries_translit["ht_description"] = queries_translit["ht_description"].astype(str)
    queries_translit = queries_translit.rename(columns={"ht_title": "query"})
    logging.info(f"Loaded {len(queries_translit)} queries for transliterated {dataset_name} (HT)")
    # Load Metrics
    metrics = [nDCG@20, R@1000]
else:
    dataset = pt.get_dataset(f"irds:{dataset_name}/{lang}/dev/small")
    queries_orig = dataset.get_topics(tokenise_query=False)

    logging.info(f"Loaded {len(queries_orig)} queries for {dataset_name}/{lang}")

    # pre-process transliterated queries
    queries_translit = pd.read_csv(f"/root/nfs/CLIR/data/transliterations/mmarco_v2_{lang}_dev_small_uroman.tsv", sep="\t", header=None, names=["qid", "query"])
    queries_translit["query"] = queries_translit["query"].astype(str)
    queries_translit["qid"] = queries_translit["qid"].astype(str)
    logging.info(f"Loaded {len(queries_translit)} queries for transliterated {dataset_name}/{lang}")
    # Load Metrics
    metrics = [MRR@10, R@1000]

factory = BGEM3(batch_size=32, max_length=1024, model_name=model_name) if model_name else BGEM3(batch_size=32, max_length=1024)
encoder = factory.query_encoder()
# Load Pisa Index
idx = FlexIndex(index_path)
logging.info(f"Loaded index {index_path}")
# Retrieval Pipeline
pipeline = encoder >> idx.np_retriever()

# create a directory for retrieval results if they do not exist
os.makedirs(f"/root/nfs/CLIR/data/retrieval_results/{first_stage_model}", exist_ok=True)

# dataset_name = dataset_name.replace("/", "")
## Run Retrieval for Native and Transliterated Queries
res_orig_path = f"/root/nfs/CLIR/data/retrieval_results/{first_stage_model}/{first_stage_model}_{dataset_name_storage}_{lang}.res.gz"
res_trans_path = f"/root/nfs/CLIR/data/retrieval_results/{first_stage_model}/{first_stage_model}_{dataset_name_storage}_{lang}_trans_uro.res.gz"

if os.path.isfile(res_orig_path):
    logging.info(f"Results for {first_stage_model} retrieval for {lang} already exists")
    res_orig = pt.io.read_results(res_orig_path)
    res_orig = res_orig.merge(queries_orig, on="qid")
    logging.info(f"Loaded results from disk: {res_orig_path}")
else:
    logging.info(f"Running {first_stage_model} retrieval for {lang} queries")
    res_orig = pipeline(queries_orig)
    pt.io.write_results(res_orig, res_orig_path)
    logging.info(f"Saved results to disk: {res_orig_path}")

if os.path.isfile(res_trans_path):
    logging.info(f"Results for {first_stage_model} retrieval for transliterated {lang} already exists")
    res_trans = pt.io.read_results(res_trans_path)
    res_trans = res_trans.merge(queries_translit, on="qid")
    logging.info(f"Loaded results from disk: {res_trans_path}")
else:
    logging.info(f"Running {first_stage_model} retrieval for transliterated {lang} queries")
    res_trans = pipeline(queries_translit)
    pt.io.write_results(res_trans, res_trans_path)
    logging.info(f"Saved results to disk: {res_trans_path}")

if evaluate:
    logging.info(f"Evaluating results between native and romanised script using {first_stage_model}")
    experiment = pt.Experiment(
        [res_orig, res_trans],
        dataset.get_topics(),
        dataset.get_qrels(),
        metrics,
        names=[f"{lang}", f"{lang} transliterated"],
        baseline=0
    )
    print(experiment)
    logging.info("Done")
