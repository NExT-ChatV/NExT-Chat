MODEL_PATH=$1

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --num_processes 8 \
        --main_process_port 23781 \
        mllm/pipeline/finetune.py \
        config/nextchat_eval_multi_res.py \
        --cfg-options model_args.model_name_or_path=$MODEL_PATH model_args.mm_projector_depth=2 \
        --output_dir "$MODEL_PATH"/res/  --per_device_eval_batch_size 1