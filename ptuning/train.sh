PRE_SEQ_LEN=4096
LR=2e-2

CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
    --do_train \
    --train_file dichandata_demo/train.json \
    --validation_file dichandata_demo/dev.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm-6b-int4 \
    --output_dir output/demo-chatglm-6b-int4-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 2048 \
    --max_target_length 2048 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 100 \
    --logging_steps 5 \
    --save_steps 10 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

