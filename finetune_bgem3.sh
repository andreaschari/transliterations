output_dir=$1
train_data=$2

torchrun --nproc_per_node 1 \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir $1 \
--model_name_or_path BAAI/bge-m3 \
--train_data $2 \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 1 \
--per_device_train_batch_size 8 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 64 \
--passage_max_len 256 \
--train_group_size 2 \
--negatives_cross_device \
--logging_steps 10 \
--save_steps 50000 \
--query_instruction_for_retrieval ""