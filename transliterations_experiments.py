import argparse
import pyterrier as pt
import pandas as pd
import os
import logging
from pyterrier_dr import BGEM3, FlexIndex
from ir_measures import R, MRR, nDCG

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

argparser = argparse.ArgumentParser()
argparser.add_argument("--lang", type=str, help="language of queries", required=True)
argparser.add_argument("--index", type=str, help="index path", required=True)
argparser.add_argument("--model", type=str, help="first stage retrieval model", required=True)
argparser.add_argument("--dataset", type=str, help="dataset to use (irds format)", default="mmarco/v2")
argparser.add_argument("--evaluate", action='store_true', help="whether to evaluate the results")
argparser.add_argument("--neuclir_field", type=str, help="field to use for neuclir dataset", default="ht_title")
args = argparser.parse_args()

lang = args.lang
index_path = args.index
first_stage_model = args.model
dataset_name = args.dataset
evaluate = args.evaluate
neuclir_field = args.neuclir_field
# Check if the index directory exists
if not os.path.isdir(index_path):
    raise FileNotFoundError(f"The index: {index_path} does not exist")

# Load Queries
if "neuclir" in dataset_name:
    dataset = pt.get_dataset(f"irds:{dataset_name}")
    queries_orig = dataset.get_topics(tokenise_query=False)
    # only use ht_title and mt_title
    queries_orig = queries_orig[["qid", "ht_title", "mt_title", "ht_description"]]

    queries_orig = queries_orig.rename(columns={neuclir_field: "query"})
    # print no. of queries
    logging.info(f"Loaded {len(queries_orig)} queries for {dataset_name} using {neuclir_field}")
    # pre-process transliterated queries
    dataset_name = dataset_name.replace("/", "_")
    queries = pd.read_csv(f"/root/nfs/CLIR/data/transliterations/{dataset_name}_uroman.tsv", sep="\t", header=None, names=["qid", "ht_title", "mt_title", "ht_description"])
    queries["qid"] = queries["qid"].astype(str)
    queries["ht_title"] = queries["ht_title"].astype(str)
    queries["mt_title"] = queries["mt_title"].astype(str)
    queries["ht_description"] = queries["ht_description"].astype(str)

    queries = queries.rename(columns={neuclir_field: "query"})
    logging.info(f"Loaded {len(queries)} queries for transliterated {dataset_name} using {neuclir_field}")

    # Load Metrics
    metrics = [nDCG@20, R@1000]
else:
    dataset = pt.get_dataset(f"irds:{dataset_name}/{lang}/dev/small")
    queries_orig = dataset.get_topics(tokenise_query=False)

    # pre-process transliterated queries
    queries = pd.read_csv(f"/root/nfs/CLIR/data/transliterations/mmarco_v2_{lang}_dev_small_uroman.txt", sep="\t", header=None, names=["qid", "query"])
    queries["query"] = queries["query"].astype(str)
    queries["qid"] = queries["qid"].astype(str)
    logging.info(f"Loaded {len(queries)} queries for transliterated {lang}")

    # Load Metrics
    metrics = [MRR@10, R@1000]

# Load Dense Model
model_paths = {
    "bgem3": None,
    "bgem3-ja-zh-random": "/root/nfs/CLIR/data/models/bge-m3-JA_ZH_MMARCO",
    "bgem3-ru-zh-50": "/root/nfs/CLIR/data/models/bge-m3-RU_ZH_MMARCO_50",
    "bgem3-ru-zh-50-2M": "/root/nfs/CLIR/data/models/bge-m3-RU_ZH_MMARCO_50_2M",
    "bgem3-zh-50": "/root/nfs/CLIR/data/models/bge-m3-ZH_MMARCO_50",
    "bgem3_ru-50": "/root/nfs/CLIR/data/models/bge-m3-RU_MMARCO_50",
    "bgem3-ru-zh-50-2M-new": "/root/nfs/CLIR/data/models/bge-m3-RU-ZH_MMARCO_50_2M_NEW",
    "bgem3-ru-zh-50-2M-ALL_TRANS": "/root/nfs/CLIR/data/models/bge-m3-RU-ZH_MMARCO_50_2M_ALL_TRANS",
    "bgem3_tzh": "/root/nfs/CLIR/data/models/bge-m3-tZH_MMARCO",
    "bgem3_zh_native": "/root/nfs/CLIR/data/models/bge-m3-ZH_MMARCO_NATIVE",
    "bgem3_ru_native": "/root/nfs/CLIR/data/models/bge-m3-RU_MMARCO_NATIVE",
    "bgem3_ru_translit": "/root/nfs/CLIR/data/models/bge-m3-RU_MMARCO_TRANSLIT"
}

if first_stage_model not in model_paths:
    raise ValueError(f"Model {first_stage_model} not supported")

model_name = model_paths[first_stage_model]

if model_name is None:
    factory = BGEM3(batch_size=32, max_length=1024)
else:
    factory = BGEM3(batch_size=32, max_length=1024, model_name=model_name)

encoder = factory.query_encoder()

# Load Pisa Index
idx = FlexIndex(index_path)
logging.info(f"Loaded index {index_path}")

# Retrieval Pipeline
pipeline = encoder >> idx.np_retriever()

# create a directory for retrieval results if they do not exist
os.makedirs(f"/root/nfs/CLIR/data/retrieval_results/{first_stage_model}", exist_ok=True)

dataset_name = dataset_name.replace("/", "")
## Run Retrieval for Native and Transliterated Queries
if "neuclir" in dataset_name:
    res_orig_path = f"/root/nfs/CLIR/data/retrieval_results/{first_stage_model}/{first_stage_model}_{dataset_name}_{lang}_{neuclir_field}.res.gz"
    res_trans_path = f"/root/nfs/CLIR/data/retrieval_results/{first_stage_model}/{first_stage_model}_{dataset_name}_{lang}_{neuclir_field}_trans_uro.res.gz"
else:
    res_orig_path = f"/root/nfs/CLIR/data/retrieval_results/{first_stage_model}/{first_stage_model}_{dataset_name}_{lang}.res.gz"
    res_trans_path = f"/root/nfs/CLIR/data/retrieval_results/{first_stage_model}/{first_stage_model}_{dataset_name}_{lang}_trans_uro.res.gz"

if os.path.isfile(res_orig_path):
    logging.info(f"Results for {first_stage_model} retrieval for {lang} already exists")
    res_orig = pt.io.read_results(res_orig_path)
    res_orig = res_orig.merge(queries_orig, on="qid")
    logging.info("Loaded results from disk")
else:
    logging.info(f"Running {first_stage_model} retrieval for {lang} queries")
    res_orig = pipeline(queries_orig)
    pt.io.write_results(res_orig, res_orig_path)
    logging.info("Saved results to disk")

if os.path.isfile(res_trans_path):
    logging.info(f"Results for {first_stage_model} retrieval for transliterated {lang} already exists")
    res_trans = pt.io.read_results(res_trans_path)
    res_trans = res_trans.merge(queries, on="qid")
    logging.info("Loaded results from disk")
else:
    logging.info(f"Running {first_stage_model} retrieval for transliterated {lang} queries")
    res_trans = pipeline(queries)
    pt.io.write_results(res_trans, res_trans_path)
    logging.info("Saved results to disk")

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