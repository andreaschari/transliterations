import pyterrier as pt
import pandas as pd
import os
import logging
from ir_measures import R, MRR, nDCG

# Load both trec 2022 and trec 2023 queries sets for neuclir and evaluate as one
dataset_2022 = pt.get_dataset("irds:neuclir/1/ru/trec-2022")
dataset_2023 = pt.get_dataset("irds:neuclir/1/ru/trec-2023")

res_orig_path_2022 = "/root/nfs/CLIR/data/retrieval_results/mt5/mt5_bgem3_neuclir_1_ru_trec-2022_ru_trans_uro.res.gz"
res_orig_path_2023 = "/root/nfs/CLIR/data/retrieval_results/mt5/mt5_bgem3_neuclir_1_ru_trec-2023_ru_trans_uro.res.gz"

res_trans_path_2022 = "/root/nfs/CLIR/data/retrieval_results/mt5_ru_50/mt5_ru_50_bgem3_ru-50_neuclir_1_ru_trec-2022_ru_trans_uro.res.gz"
res_trans_path_2023 = "/root/nfs/CLIR/data/retrieval_results/mt5_ru_50/mt5_ru_50_bgem3_ru-50_neuclir_1_ru_trec-2023_ru_trans_uro.res.gz"

# Load Queries
queries_orig_2022 = dataset_2022.get_topics(tokenise_query=False)
queries_orig_2023 = dataset_2023.get_topics(tokenise_query=False)

# combine queries
queries_orig = pd.concat([queries_orig_2022, queries_orig_2023])

# only use ht_title and mt_title
queries_orig_2022 = queries_orig_2022[["qid", "ht_title"]]
queries_orig_2022 = queries_orig_2022.rename(columns={"ht_title": "query"})

queries_orig_2023 = queries_orig_2023[["qid", "ht_title"]]
queries_orig_2023 = queries_orig_2023.rename(columns={"ht_title": "query"})

# Load Metrics
metrics = [nDCG@20, R@1000]

res_orig_2022 = pt.io.read_results(res_orig_path_2022)
res_orig_2023 = pt.io.read_results(res_orig_path_2023)

res_trans_2022 = pt.io.read_results(res_trans_path_2022)
res_trans_2023 = pt.io.read_results(res_trans_path_2023)
logging.info("Loaded results")

# Combine results
res_orig = pd.concat([res_orig_2022, res_orig_2023])
res_trans = pd.concat([res_trans_2022, res_trans_2023])

# Load and combine qrels

qrels_2022 = dataset_2022.get_qrels()
qrels_2023 = dataset_2023.get_qrels()

qrels = pd.concat([qrels_2022, qrels_2023])

# Evaluate
logging.info("Evaluating")
experiment = pt.Experiment(
    [res_orig, res_trans],
    queries_orig,
    qrels,
    metrics,
    names=["baseline", "transliterated"],
    baseline=0,
    correction='b'
)
print(experiment)
logging.info("Done")