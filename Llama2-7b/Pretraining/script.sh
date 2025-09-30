# Based on https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/training/run_pt.sh


# Change the values accordingly.

python3 "/raid/home/prabhasreddy/nlp_fine/fine_tune_lama/TeluguLLM/Pretrain/run_pretrain.py" \
    --model_name_or_path "/raid/home/prabhasreddy/nlp_fine/fine_tune_lama/TeluguLLM/Pretrain/llama_7b_hf" \
    --tokenizer_name_or_path "/raid/home/prabhasreddy/nlp_fine/fine_tune_lama/TeluguLLM/Tokenizer/merged_tokenizer_huggingface" \
    --dataset_dir "/raid/home/prabhasreddy/nlp_fine/fine_tune_lama/TeluguLLM/data/te/" \
    --data_cache_dir "/raid/home/prabhasreddy/nlp_fine/fine_tune_lama/TeluguLLM/Pretrain/cache_dir" \
    --validation_split_percentage 0.1 \
    --per_device_train_batch_size 16 \
    --do_train \
    --seed 128 \
    --num_train_epochs 0.02 \
    --lr_scheduler_type cosine \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 1 \
    --save_steps 100 \
    --gradient_accumulation_steps 2 \
    --preprocessing_num_workers 50 \
    --block_size 512 \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank 64 \
    --lora_alpha 128 \
    --trainable "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
    --lora_dropout 0.05 \
    --modules_to_save "embed_tokens,lm_head" \
    --torch_dtype float16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    --output_dir "output_dir" \
    --overwrite_output_dir \
    #--resume_from_checkpoint "output_dir/checkpoint-2400" \   #To resume from a checkpoint uncomment this line
    #--fp16 \
    #--deepspeed "ds_zero2_no_offload.json" \
    #--flash_attn True \
    # --load_in_kbits 16 \
    
