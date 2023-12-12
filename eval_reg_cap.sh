MODEL_PATH=$1

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --num_processes 1 \
        --main_process_port 23787 \
        mllm/pipeline/finetune.py \
        config/nextchat_eval_reg_cap.py \
        --cfg-options model_args.model_name_or_path=$MODEL_PATH model_args.mm_projector_depth=2 \
        --output_dir "$MODEL_PATH"/reg_cap/  --per_device_eval_batch_size 8