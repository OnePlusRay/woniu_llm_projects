CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 \
-m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
--model_name_or_path bge-large-zh-v1.5 \
--input_file ./embedding_finetune/data_process/finetune_data.jsonl \
--output_file ./embedding_finetune/data_process/finetune_data_minedHN_bge.jsonl \
--range_for_sampling 2-200 \
--negative_number 5 \
# --use_gpu_for_searching 
