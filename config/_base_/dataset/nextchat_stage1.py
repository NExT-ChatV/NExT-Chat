_base_ = ['DEFAULT_TRAIN_DATASET.py']

data_args = dict(
    #
    train=dict(
        type='ConcatDataset',
        cfgs=[
            dict(
                type='SubSet',
                portion=1/20,
                do_shuffle=True,
                seed=42,
                cfg={{_base_.DEFAULT_TRAIN_DATASET.gc}},
            ),
            dict(
                type='SubSet',
                portion=1/10,
                do_shuffle=True,
                seed=43,
                cfg={{_base_.DEFAULT_TRAIN_DATASET.recvg}},
            ),

            {{_base_.DEFAULT_TRAIN_DATASET.VQAv2_train}},
            # {{_base_.DEFAULT_TRAIN_DATASET.VQAE_train}},
            # {{_base_.DEFAULT_TRAIN_DATASET.VQAX_train}},

            {{_base_.DEFAULT_TRAIN_DATASET.caption}},

            {{_base_.DEFAULT_TRAIN_DATASET.rec}},
            {{_base_.DEFAULT_TRAIN_DATASET.reg}},
            {{_base_.DEFAULT_TRAIN_DATASET.flickr}},

            {{_base_.DEFAULT_TRAIN_DATASET.VCR_q_ra}},
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_V7W_b}},

        ],
    ),
    validation=None,
    test=None,

    # compute_metric
    compute_metric=None,

    # padding collator kwargs
    collator_kwargs=dict(
        padding=True,
        max_length=1024,
    ),

    # generate config
    gen_kwargs=dict(
        max_new_tokens=1024,
        num_beams=1,
    ),
)
