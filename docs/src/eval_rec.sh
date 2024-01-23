MODEL_PATH=$1

CUDA_VISIBLE_DEVICES="5,6" accelerate launch --num_processes 2 \
        --main_process_port 23781 \
        mllm/pipeline/finetune.py \
        config/nextchat_eval_multi_rec.py \
        --cfg-options model_args.model_name_or_path=$MODEL_PATH model_args.mm_projector_depth=2 \
        --output_dir "$MODEL_PATH"/rec/