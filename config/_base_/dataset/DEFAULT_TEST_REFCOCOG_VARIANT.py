REG_CAP_TEST_COMMON_CFG = dict(
    type='REGDataset',
    template_file=r'{{fileDirname}}/template/REG.json',
    image_folder=r'/data/public/multimodal/multimodal_data/coco_imgs/train2014/',
    max_dynamic_size=None,
)

DEFAULT_TEST_REFCOCOG_VARIANT = dict(
    REG_REFCOCOG_GOOGLE_TEST=dict(
        **REG_CAP_TEST_COMMON_CFG,
        filename=r'{{fileDirname}}/../../../data/REC_refcocog_google_val.jsonl',
    ),
)
