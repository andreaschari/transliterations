import argparse
import pyterrier as pt
import pandas as pd
import os
import logging
from pyterrier_t5 import mT5ReRanker
from ir_measures import R, MRR, nDCG
from models import t5_model_paths
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

argparser = argparse.ArgumentParser()
argparser.add_argument("--lang", type=str, help="language of queries (same as index)", required=True)
argparser.add_argument("--first_stage_model", type=str, help="first stage retrieval model", default="bgem3")
argparser.add_argument("--dataset", type=str, help="dataset to use (irds format)", default="mmarco/v2")
argparser.add_argument("--rerank_model", type=str, help="reranking model", default="mt5",)
argparser.add_argument("--evaluate", action='store_true', help="whether to evaluate the results")

args = argparser.parse_args()
lang = args.lang
first_stage_model = args.first_stage_model
dataset_name = args.dataset
rerank_model = args.rerank_model
evaluate = args.evaluate

RETRIEVAL_RESULTS_DIR = "/root/nfs/CLIR/data/retrieval_results" # change this to the directory where the retrieval results are stored

if rerank_model in t5_model_paths:
    model_path = t5_model_paths[rerank_model]
    reranker = mT5ReRanker(model=model_path, verbose=True) if model_path else mT5ReRanker(verbose=True)
else:
    raise ValueError(f"Invalid reranking model {rerank_model}")

print(f"Using {rerank_model} for re-ranking")
dataset_name_storage = dataset_name.replace("/", "_")

# Load Queries
if "neuclir" in dataset_name:
    dataset = pt.get_dataset(f"irds:{dataset_name}")
    queries_orig = dataset.get_topics(tokenise_query=False)
    # only use ht_title and mt_title
    queries_orig = queries_orig[["qid", "ht_title", "mt_title"]]
    # rename ht_title to query
    queries_orig = queries_orig.rename(columns={"ht_title": "query"})
    # print no. of queries
    logging.info(f"Loaded {len(queries_orig)} queries for {dataset_name} (HT)")
    # pre-process transliterated queries
    queries = pd.read_csv(f"/root/nfs/CLIR/data/transliterations/{dataset_name_storage}_uroman.tsv", sep="\t", header=None, names=["qid", "ht_title", "mt_title", "ht_description"])
    queries["qid"] = queries["qid"].astype(str)
    queries["ht_title"] = queries["ht_title"].astype(str)
    # rename ht_title to query
    queries = queries.rename(columns={"ht_title": "query"})
    logging.info(f"Loaded {len(queries)} queries for transliterated {dataset_name} (HT)")

    # Load Metrics
    metrics = [nDCG@20, R@1000]
else:
    dataset = pt.get_dataset(f"irds:{dataset_name}/{lang}/dev/small")
    queries_orig = dataset.get_topics(tokenise_query=False)
    # pre-process transliterated queries
    queries = pd.read_csv(f"/root/nfs/CLIR/data/transliterations/mmarco_v2_{lang}_dev_small_uroman.tsv", sep="\t", header=None, names=["qid", "query"])
    queries["query"] = queries["query"].astype(str)
    queries["qid"] = queries["qid"].astype(str)
    logging.info(f"Loaded {len(queries)} queries for {lang}")
    
    # Load Metrics
    metrics = [MRR@10, R@1000]

# Load first stage retrieval results (assumes they exist)
res_first_orig_path = f"{RETRIEVAL_RESULTS_DIR}/{first_stage_model}/{first_stage_model}_{dataset_name_storage}_{lang}.res.gz"
res_first_trans_path = f"{RETRIEVAL_RESULTS_DIR}/{first_stage_model}/{first_stage_model}_{dataset_name_storage}_{lang}_trans_uro.res.gz"

if os.path.isfile(res_first_orig_path):
    res_first_orig = pt.io.read_results(res_first_orig_path)

    res_first_orig = res_first_orig.merge(queries_orig, on="qid")
    logging.info(f"Loaded results for {first_stage_model} retrieval for original {lang} from {res_first_orig_path}")
else:
    raise FileNotFoundError(f"First stage retrieval results for {first_stage_model} for {lang} do not exist in {res_first_orig_path}")

if os.path.isfile(res_first_trans_path):
    res_first_trans = pt.io.read_results(res_first_trans_path)

    res_first_trans = res_first_trans.merge(queries, on="qid")
    logging.info(f"Loaded results for {first_stage_model} retrieval for transliterated {lang} from {res_first_trans_path}")
else:
    raise FileNotFoundError(f"First stage retrieval results for {first_stage_model} for transliterated {lang} do not exist")


pipeline = pt.text.get_text(dataset, "text") >> reranker
# create results folder if it does not exist
os.makedirs(f"{RETRIEVAL_RESULTS_DIR}/{rerank_model}", exist_ok=True)

res_orig_path = f"{RETRIEVAL_RESULTS_DIR}/{rerank_model}/{rerank_model}_{first_stage_model}_{dataset_name_storage}_{lang}.res.gz"
res_trans_path = f"{RETRIEVAL_RESULTS_DIR}/{rerank_model}/{rerank_model}_{first_stage_model}_{dataset_name_storage}_{lang}_trans_uro.res.gz"

if os.path.isfile(res_orig_path):
    res = pt.io.read_results(res_orig_path)
    # res = res.merge(queries_orig, on="qid")
    logging.info(f"Loaded results for {rerank_model} retrieval for original {lang} from {res_orig_path}")
else:
    logging.info(f"Re-ranking {first_stage_model} results for {lang}")
    res = pipeline(res_first_orig)
    pt.io.write_results(res, res_orig_path)
    logging.info(f"Saved re-ranked results to disk: {res_orig_path}")

if os.path.isfile(res_trans_path):
    res2 = pt.io.read_results(res_trans_path)
    # res2 = res2.merge(queries, on="qid")
    logging.info(f"Loaded results for {rerank_model} retrieval for transliterated {lang} from {res_trans_path}")
else:
    logging.info(f"Re-ranking {first_stage_model} results for transliterated {lang}")
    res2 = pipeline(res_first_trans)
    pt.io.write_results(res2, res_trans_path)
    logging.info(f"Saved re-ranked results to disk: {res_trans_path}")

if evaluate:
    # change res1 and res2 for whatever you want to evaluate
    logging.info("Evaluating results between variaties")
    experiment = pt.Experiment(
        [res, res2],
        dataset.get_topics(),
        dataset.get_qrels(),
        metrics,
        names=[f"native {lang}", f" transliterated {lang}"],
        baseline=0,
        correction="b"
    )
    print(experiment)
    logging.info("Done")