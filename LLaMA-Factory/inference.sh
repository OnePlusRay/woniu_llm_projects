CUDA_VISIBLE_DEVICES=1 llamafactory-cli webchat \  
    --model_name_or_path /data/disk4/home/chenrui/Yi-1.5-34B-Chat-bnb-4bit \  
    --adapter_name_or_path saves/yi-1.5-34b/lora/sft \  
    --template yi \  
    --finetuning_type lora \