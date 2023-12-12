accelerate launch --num_processes 8 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/nextchat_stage3.py \
        --cfg-options model_args.model_name_or_path=$1 \
        model_args.mm_projector_depth=2  \
        --num_train_epochs 3 --save_steps 5000 \
        --output_dir ./output/stage3