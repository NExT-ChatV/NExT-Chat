VQAv2_TRAIN_COMMON_CFG = dict(
    type='VQAv2Dataset',
    filename=r'{{fileDirname}}/../../../data/v2_OpenEnded_mscoco_train2014_questions.jsonl',
    image_folder=r'/data/public/multimodal/multimodal_data/coco_imgs',
    template_file=r"{{fileDirname}}/template/VQA.json",
)

DEFAULT_TRAIN_VQAv2_VARIANT = dict(
    VQAv2_train=dict(**VQAv2_TRAIN_COMMON_CFG),
)