accelerate launch --num_processes 8 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/nextchat_stage1.py \
        --cfg-options model_args.model_name_or_path=/data/public/multimodal/multimodal_model_ckpts/vicuna-7b-v1.5 \
        model_args.mm_projector_depth=2 \
        --num_train_epochs 2 \
        --output_dir ./output/stage1