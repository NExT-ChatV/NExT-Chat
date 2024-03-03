_base_ = ['_base_/dataset/nextchat_stage2.py', '_base_/model/nextchat.py', '_base_/train/nextchat_fsdp.py']

training_args = dict(
    num_train_epochs=2,
    output_dir='./exp/{{fileBasenameNoExtension}}',
)

model_args = dict(
    conv_args=dict(
        tokenize_kwargs=dict(truncation_size=4096),
    ),
    model_name_or_path=None,
)
