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
    per_device_eval_batch_size=8,
)

model_args = dict(
    type='nextchat_seg',
    model_name_or_path=None,
)

data_args = dict(
    train=None,
    validation=None,
    test=None,
    multitest={k: {'cfg': v, 'compute_metric': dict(type='REGCapComputeMetrics')} for k, v in _base_.DEFAULT_TEST_REFCOCOG_VARIANT.items()},

    compute_metric=None,

    # padding collator kwargs
    collator_kwargs=dict(
        padding=True,
        max_length=1024,
    ),

    # generate config
    gen_kwargs=dict(
        max_new_tokens=500,
        num_beams=1,
    ),
)