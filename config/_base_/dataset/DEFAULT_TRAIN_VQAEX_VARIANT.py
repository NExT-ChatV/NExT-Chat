VQAEX_TRAIN_COMMON_CFG = dict(
    type='VQAEXDataset',
    image_folder=r'/data/public/multimodal/multimodal_data/coco_imgs/',
    template_file=r"{{fileDirname}}/template/VQA_CoT.json",
)

DEFAULT_TRAIN_VQAEX_VARIANT = dict(
    VQAE_train=dict(
        **VQAEX_TRAIN_COMMON_CFG,
        is_e_dataset=True,
        filename=r'{{fileDirname}}/../../../data/vqa_E_train.jsonl',
    ),
    VQAX_train=dict(
        **VQAEX_TRAIN_COMMON_CFG,
        is_e_dataset=False,
        filename=r'{{fileDirname}}/../../../data/vqa_X_train.jsonl',
    ),
)
