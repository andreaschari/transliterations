"""use pyterrier_dr to build Flex indices using BGE-M3 encodings"""
import argparse
import pyterrier as pt
from pyterrier_dr import BGEM3, FlexIndex

# Create an argument parser
parser = argparse.ArgumentParser(description='Indexing script')

# Add an argument for the language and the dataset
parser.add_argument('--dataset', help='Dataset to index (ir-datasets name format)')
parser.add_argument('--batch_size', help='Batch size for encoding', default=64)
parser.add_argument('--max_length', help='Max length for encoding', default=1024)
parser.add_argument('--model', help='BGE-M3 model to use for encoding')

# Parse the command line arguments
args = parser.parse_args()
dataset = args.dataset
model = args.model

batch_size = int(args.batch_size)
max_length = int(args.max_length)

# create a BGEM3 encoder
model_paths = {
    "bgem3": None,
    "bgem3-ru-zh-random-50": "/root/nfs/CLIR/data/models/bge-m3-RU_ZH_MMARCO_50",
    "bgem3-ru-zh-random-50-2M": "/root/nfs/CLIR/data/models/bge-m3-RU_ZH_MMARCO_50_2M",
    "bgem3-zh-50": "/root/nfs/CLIR/data/models/bge-m3-ZH_MMARCO_50",
    "bgem3-ru-zh-50-2M-NEW": "/root/nfs/CLIR/data/models/bge-m3-RU-ZH_MMARCO_50_2M_NEW",
    "bge-m3-ZH_MMARCO_NATIVE": "/root/nfs/CLIR/data/models/bge-m3-ZH_MMARCO_NATIVE",
    "bge-m3-RU_MMARCO_NATIVE": "/root/nfs/CLIR/data/models/bge-m3-RU_MMARCO_NATIVE",
    "bge-m3-RU_MMARCO_50": "/root/nfs/CLIR/data/models/bge-m3-RU_MMARCO_50",
    "bge-m3-RU_MMARCO_TRANSLIT": "/root/nfs/CLIR/data/models/bge-m3-RU_MMARCO_TRANSLIT",
    "bge-m3-ZH_MMARCO_TRANSLIT": "/root/nfs/CLIR/data/models/bge-m3-tZH_MMARCO"
}

if model not in model_paths:
    raise ValueError(f"Model {model} not supported")

model_path = model_paths[model]
factory = BGEM3(batch_size=batch_size, max_length=max_length, model_name=model_path) if model_path else BGEM3(batch_size=batch_size, max_length=max_length)
index = FlexIndex(f"/root/nfs/CLIR/data/indices/{dataset}_{model}", verbose=True)
encoder = factory.doc_encoder()

print(f"Building Flex index for {dataset} dataset...")

if "neuclir" in dataset:
    indexing_pipeline = pt.apply.text(lambda x: '{title}\n{text}'.format(**x)) >> encoder >> index
else:
    indexing_pipeline = encoder >> index

indexing_pipeline.index(pt.get_dataset(f"irds:{dataset}").get_corpus_iter())
