# Lost in Transliteration: Bridging the Script Gap in Neural IR

Code for the ECIR 2025 IR4Good Paper

## Abstract

Most human languages use scripts other than the Latin alphabet. Search users in these languages often formulate their information needs in a transliterated --usually Latinized-- form for ease of typing. For example, Greek speakers might use Greeklish, and Arabic speakers might use Arabizi. This paper shows that current search systems, including those that use multilingual dense embeddings such as BGE-M3, do not generalise to this setting, and their performance rapidly deteriorates when exposed to transliterated queries. This creates a "script gap" between the performance of the same queries when written in their native or transliterated form. We explore whether adapting the popular "translate-train" paradigm to transliterations can enhance the robustness of multilingual Information Retrieval (IR) methods and bridge the gap between native and transliterated scripts. By exploring various combinations of non-Latin and Latinized query text for training, we investigate whether we can enhance the capacity of existing neural retrieval techniques and enable them to apply to this important setting. We show that by further fine-tuning IR models on an even mixture of native and Latinized text, they can perform this cross-script matching at nearly the same performance as when the query was formulated in the native script. Out-of-domain evaluation and further qualitative analysis show that transliterations can also cause queries to lose some of their nuances, motivating further research.

## Released Artifacts

We release the model checkpoints and transliterated queries in the Hugging Face Collection [here](https://huggingface.co/collections/andreaschari/sigir2025-lost-in-transliteration-680a15e761a763a3d7e04775). The collection contains:

- The BGE-M3 and mT5 models used in the experiments.
- The transliterated queries for the datasets used in the experiments.

## Reproducing the results

### Setup

The instructions here are focused on setting up a conda environment.

This code was developed and tested with Python 3.10.

First, create a virtual environment:

```bash
conda create -n lt
conda activate lt
```

To run the retrieval and re-ranking experiments for BGE-M3 and mT5, you will need to install the following dependencies to run `variations_experiments.py`:

- [pyterrier](https://pyterrier.readthedocs.io/)
- [pyterrier_dr[bgem3]](https://github.com/terrierteam/pyterrier_dr) (Check BGE-M3 Encoder section of `pyterrier_dr` for installation instructions.)
- [pyterrier_pisa](https://github.com/terrierteam/pyterrier_pisa)
- [pyterrier_t5](https://github.com/terrierteam/pyterrier_t5)

### Transliterating the Queries

The transliterations depend on the [uroman](https://github.com/isi-nlp/uroman) library. You can install it using pip:

```bash
python3 -m pip install uroman
```

You can then use the `transliterate.py` script to transliterate the queries. e.g.

```bash
python transliterate.py --lang <language of queries in (ISO 639-3 standard)> --dataset <dataset in IRDS format> --do_docs <if you want to transliterate the documents of the dataset instead>
```

Note: you need to set the `output_path` variable in the `transliterate.py` script to point to where you want to save the transliterated queries. This should be the same path as the `TRANSLITERATIONS_DIR` variable in the `transliterations_experiments.py` script.

We provide the transliterated queries in the Hugging Face Collection [here](https://huggingface.co/collections/andreaschari/sigir2025-lost-in-transliteration-680a15e761a763a3d7e04775). You can download the transliterated queries from there and place them in the `TRANSLITERATIONS_DIR` path.

### Fine-tuning the Models

#### BGE-M3

You can use the `finetune_bgem3.sh` script to fine-tune the BGE-M3 model on the mMARCO dataset. The only requirement is setting the `output_dir` variable to the desired output directory and `train_data` to the path of the training JSONL file. (We provide our JSONL files in the Hugging Face Collection but if you want to create your own you can follow the steps in our other repo [here](https://github.com/andreaschari/linguistic-transfer).)

#### mT5

You can use the `finetune_mt5.py` script to fine-tune the mT5 model on the mMARCO dataset.

### Retrieval

The list of available BGE-M3 and mT5 models can be found in the `models.py` file. The models are available in this [Hugging Face Collection](https://huggingface.co/collections/andreaschari/sigir2025-lost-in-transliteration-680a15e761a763a3d7e04775). You can use the `--model` argument to specify which model you want to use.

#### BGE-M3 Retrieval

First you need to use the `indexing.py` to index the collection. e.g.

```bash
python indexing.py --dataset <dataset in IRDS format> --model <BGE-M3 model>
```

The `transliterations_experiments.py` script is used to run the BGE-M3 retrieval experiments. e.g.

```bash
python transliterations_experiments.py --lang <language of queries> --index <index path> --model <BGE-M3 model>  --dataset  <dataset in IRDS format> --evaluate 
```

Note: there are two paths you need to manual set in the `transliterations_experiments.py` script:

`TRANSLITERATIONS_DIR` which should be the path point to where you have the transliterated queries and `RETRIEVAL_DIR` which should be the path point to where you want to save the retrieval results.

The `--evaluate` flag will run the evaluation in addition to retrieval.

#### mT5 Re-Ranking

The mT5 re-ranking experiments are run using the `transliterations_rerank.py` script. e.g.

```bash
python transliterations_rerank.py --lang <language of queries> --index <index path> --first_stage_model <BGE-M3 model>  --rerank_model >mT5 model> --dataset  <dataset in IRDS format> --evaluate 
```

Note: Similar to the BGE-M3 retrieval experiments, you need to set the `TRANSLITERATIONS_DIR` and `RETRIEVAL_DIR` variables in the `transliterations_rerank.py` script to point to where the transliterated queries are saved and where to save the retrieval results respectively.

#### Qualitative Analysis and Significance Testing

The `qualitative_analysis.ipynb` and the `significance_testing.ipynb` notebooks are used to run the qualitative analysis and significance testing respectively. You can run them using Jupyter Notebook.

## Citation

If you use this code in your research, please cite the following paper:

```
@inproceedings{10.1007/978-3-031-88717-8_22,
author = {Chari, Andreas and MacAvaney, Sean and Ounis, Iadh},
title = {Improving Low-Resource Retrieval Effectiveness Using Zero-Shot Linguistic Similarity Transfer},
year = {2025},
isbn = {978-3-031-88716-1},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/978-3-031-88717-8_22},
doi = {10.1007/978-3-031-88717-8_22},
abstract = {Globalisation and colonisation have led the vast majority of the world to use only a fraction of languages, such as English and French, to communicate, excluding many others. This has severely affected the survivability of many now-deemed vulnerable or endangered languages, such as Occitan and Sicilian. These languages often share some characteristics, such as elements of their grammar and lexicon, with other high-resource languages, e.g. French or Italian. They can be clustered into groups of language varieties with various degrees of mutual intelligibility. Current search systems are not usually trained on many of these low-resource varieties, leading search users to express their needs in a high-resource language instead. This problem is further complicated when most information content is expressed in a high-resource language, inhibiting even more retrieval in low-resource languages. We show that current search systems are not robust across language varieties, severely affecting retrieval effectiveness. Therefore, it would be desirable for these systems to leverage the capabilities of neural models to bridge the differences between these varieties. This can allow users to express their needs in their low-resource variety and retrieve the most relevant documents in a high-resource one. To address this, we propose fine-tuning neural rankers on pairs of language varieties, thereby exposing them to their linguistic similarities (). We find that this approach improves the performance of the varieties upon which the models were directly trained, thereby regularising these models to generalise and perform better even on unseen language variety pairs. We also explore whether this approach can transfer across language families and observe mixed results that open doors for future research.},
booktitle = {Advances in Information Retrieval: 47th European Conference on Information Retrieval, ECIR 2025, Lucca, Italy, April 6–10, 2025, Proceedings, Part IV},
pages = {290–306},
numpages = {17},
keywords = {low resource information retrieval, zero-shot transfer},
location = {Lucca, Italy}
}
```
