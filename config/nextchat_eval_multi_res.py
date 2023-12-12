_base_ = ['_base_/dataset/DEFAULT_TEST_DATASET.py', '_base_/model/nextchat.py', '_base_/train/eval.py']

training_args = dict(
    output_dir='./exp/{{fileBasenameNoExtension}}',

    do_train=False,
    do_eval=False,
    do_predict=False,
    do_multi_predict=True,

    fp16=False,
    fp16_full_eval=False,
    bf16=True,
    bf16_full_eval=True,
    per_device_eval_batch_size=1,
)

model_args = dict(
    type='nextchat_seg',
    model_name_or_path=None,
    conv_args=dict(
        conv_template='vicuna_v1.1',
        transforms=dict(type='Expand2square'),
        tokenize_kwargs=dict(truncation_size=2048),
    ),
)

data_args = dict(
    train=None,
    validation=None,
    test=None,
    multitest={k: {'cfg': v, 'compute_metric': dict(type='RESComputeMetrics')} for k, v in _base_.DEFAULT_TEST_RES_VARIANT.items()},

    compute_metric=None,

    # padding collator kwargs
    collator_kwargs=dict(
        padding=True,
        max_length=1024,
    ),
    dataset_wrapper='conv_seg',

    # generate config
    gen_kwargs=dict(
        max_new_tokens=20,
        num_beams=1,
    ),
)
