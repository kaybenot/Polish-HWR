auto_scale_lr = dict(base_batch_size=512)
cute80_textrecog_data_root = 'data/cute80'
cute80_textrecog_test = dict(
    ann_file='textrecog_test.json',
    data_root='data/rec',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw_gt=False,
        draw_pred=False,
        enable=False,
        interval=1,
        show=False,
        type='VisualizationHook'))
default_scope = 'mmocr'
dictionary = dict(
    dict_file=
    '/home/kaybenot/MMOCR/mmocr/configs/textrecog/satrn/../../../dicts/english_digits_symbols.txt',
    same_start_end=True,
    type='Dictionary',
    with_end=True,
    with_padding=True,
    with_start=True,
    with_unknown=True)
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
gpu_ids = range(0, 1)
# icdar2013_857_textrecog_test = dict(
#     ann_file='textrecog_test_857.json',
#     data_root='data/icdar2013',
#     pipeline=None,
#     test_mode=True,
#     type='OCRDataset')
# icdar2013_textrecog_data_root = 'data/icdar2013'
# icdar2013_textrecog_test = dict(
#     ann_file='textrecog_test.json',
#     data_root='data/icdar2013',
#     pipeline=None,
#     test_mode=True,
#     type='OCRDataset')
# icdar2013_textrecog_train = dict(
#     ann_file='textrecog_train.json',
#     data_root='data/icdar2013',
#     pipeline=None,
#     type='OCRDataset')
# icdar2015_1811_textrecog_test = dict(
#     ann_file='textrecog_test_1811.json',
#     data_root='data/icdar2015',
#     pipeline=None,
#     test_mode=True,
#     type='OCRDataset')
# icdar2015_textrecog_data_root = 'data/icdar2015'
# icdar2015_textrecog_test = dict(
#     ann_file='textrecog_test.json',
#     data_root='data/icdar2015',
#     pipeline=None,
#     test_mode=True,
#     type='OCRDataset')
# icdar2015_textrecog_train = dict(
#     ann_file='textrecog_train.json',
#     data_root='data/icdar2015',
#     pipeline=None,
#     type='OCRDataset')
# iiit5k_textrecog_data_root = 'data/iiit5k'
# iiit5k_textrecog_test = dict(
#     ann_file='textrecog_test.json',
#     data_root='data/iiit5k',
#     pipeline=None,
#     test_mode=True,
#     type='OCRDataset')
# iiit5k_textrecog_train = dict(
#     ann_file='textrecog_train.json',
#     data_root='data/iiit5k',
#     pipeline=None,
#     type='OCRDataset')
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=10)
# mjsynth_sub_textrecog_train = dict(
#     ann_file='subset_textrecog_train.json',
#     data_root='data/mjsynth',
#     pipeline=None,
#     type='OCRDataset')
# mjsynth_textrecog_data_root = 'data/mjsynth'
# mjsynth_textrecog_train = dict(
#     ann_file='textrecog_train.json',
#     data_root='data/mjsynth',
#     pipeline=None,
#     type='OCRDataset')
model = dict(
    backbone=dict(hidden_dim=512, input_channels=3, type='ShallowCNN'),
    data_preprocessor=dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='TextRecogDataPreprocessor'),
    decoder=dict(
        d_embedding=512,
        d_inner=2048,
        d_k=64,
        d_model=512,
        d_v=64,
        dictionary=dict(
            dict_file=
            '/home/kaybenot/MMOCR/mmocr/configs/textrecog/satrn/../../../dicts/english_digits_symbols.txt',
            same_start_end=True,
            type='Dictionary',
            with_end=True,
            with_padding=True,
            with_start=True,
            with_unknown=True),
        max_seq_len=25,
        module_loss=dict(
            flatten=True, ignore_first_char=True, type='CEModuleLoss'),
        n_head=8,
        n_layers=6,
        postprocessor=dict(type='AttentionPostprocessor'),
        type='NRTRDecoder'),
    encoder=dict(
        d_inner=2048,
        d_k=64,
        d_model=512,
        d_v=64,
        dropout=0.1,
        n_head=8,
        n_layers=12,
        n_position=100,
        type='SATRNEncoder'),
    type='SATRN')
optim_wrapper = dict(
    optimizer=dict(lr=0.0003, type='Adam'), type='OptimWrapper')
param_scheduler = [
    dict(end=5, milestones=[
        3,
        4,
    ], type='MultiStepLR'),
]
randomness = dict(seed=None)
resume = False
seed = 0
# svt_textrecog_data_root = 'data/svt'
# svt_textrecog_test = dict(
#     ann_file='textrecog_test.json',
#     data_root='data/svt',
#     pipeline=None,
#     test_mode=True,
#     type='OCRDataset')
# svt_textrecog_train = dict(
#     ann_file='textrecog_train.json',
#     data_root='data/svt',
#     pipeline=None,
#     type='OCRDataset')
# svtp_textrecog_data_root = 'data/svtp'
# svtp_textrecog_test = dict(
#     ann_file='textrecog_test.json',
#     data_root='data/svtp',
#     pipeline=None,
#     test_mode=True,
#     type='OCRDataset')
# svtp_textrecog_train = dict(
#     ann_file='textrecog_train.json',
#     data_root='data/svtp',
#     pipeline=None,
#     type='OCRDataset')
# synthtext_an_textrecog_train = dict(
#     ann_file='alphanumeric_textrecog_train.json',
#     data_root='data/synthtext',
#     pipeline=None,
#     type='OCRDataset')
# synthtext_sub_textrecog_train = dict(
#     ann_file='subset_textrecog_train.json',
#     data_root='data/synthtext',
#     pipeline=None,
#     type='OCRDataset')
# synthtext_textrecog_data_root = 'data/synthtext'
# synthtext_textrecog_train = dict(
#     ann_file='textrecog_train.json',
#     data_root='data/synthtext',
#     pipeline=None,
#     type='OCRDataset')
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        datasets=[
            dict(
                ann_file='textrecog_test.json',
                data_root='data/rec',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
            # dict(
            #     ann_file='textrecog_test.json',
            #     data_root='data/iiit5k',
            #     pipeline=None,
            #     test_mode=True,
            #     type='OCRDataset'),
            # dict(
            #     ann_file='textrecog_test.json',
            #     data_root='data/svt',
            #     pipeline=None,
            #     test_mode=True,
            #     type='OCRDataset'),
            # dict(
            #     ann_file='textrecog_test.json',
            #     data_root='data/svtp',
            #     pipeline=None,
            #     test_mode=True,
            #     type='OCRDataset'),
            # dict(
            #     ann_file='textrecog_test.json',
            #     data_root='data/icdar2013',
            #     pipeline=None,
            #     test_mode=True,
            #     type='OCRDataset'),
            # dict(
            #     ann_file='textrecog_test.json',
            #     data_root='data/icdar2015',
            #     pipeline=None,
            #     test_mode=True,
            #     type='OCRDataset'),
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                100,
                32,
            ), type='Resize'),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'valid_ratio',
                ),
                type='PackTextRecogInputs'),
        ],
        type='ConcatDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_dataset = dict(
    datasets=[
        dict(
            ann_file='textrecog_test.json',
            data_root='data/rec',
            pipeline=None,
            test_mode=True,
            type='OCRDataset'),
        # dict(
        #     ann_file='textrecog_test.json',
        #     data_root='data/iiit5k',
        #     pipeline=None,
        #     test_mode=True,
        #     type='OCRDataset'),
        # dict(
        #     ann_file='textrecog_test.json',
        #     data_root='data/svt',
        #     pipeline=None,
        #     test_mode=True,
        #     type='OCRDataset'),
        # dict(
        #     ann_file='textrecog_test.json',
        #     data_root='data/svtp',
        #     pipeline=None,
        #     test_mode=True,
        #     type='OCRDataset'),
        # dict(
        #     ann_file='textrecog_test.json',
        #     data_root='data/icdar2013',
        #     pipeline=None,
        #     test_mode=True,
        #     type='OCRDataset'),
        # dict(
        #     ann_file='textrecog_test.json',
        #     data_root='data/icdar2015',
        #     pipeline=None,
        #     test_mode=True,
        #     type='OCRDataset'),
    ],
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(keep_ratio=False, scale=(
            100,
            32,
        ), type='Resize'),
        dict(type='LoadOCRAnnotations', with_text=True),
        dict(
            meta_keys=(
                'img_path',
                'ori_shape',
                'img_shape',
                'valid_ratio',
            ),
            type='PackTextRecogInputs'),
    ],
    type='ConcatDataset')
test_evaluator = dict(
    dataset_prefixes=[
        'CUTE80',
        'IIIT5K',
        'SVT',
        'SVTP',
        'IC13',
        'IC15',
    ],
    metrics=[
        dict(
            mode=[
                'exact',
                'ignore_case',
                'ignore_case_symbol',
            ],
            type='WordMetric'),
        dict(type='CharMetric'),
    ],
    type='MultiDatasetsEvaluator')
test_list = [
    dict(
        ann_file='textrecog_test.json',
        data_root='data/rec',
        pipeline=None,
        test_mode=True,
        type='OCRDataset'),
    # dict(
    #     ann_file='textrecog_test.json',
    #     data_root='data/iiit5k',
    #     pipeline=None,
    #     test_mode=True,
    #     type='OCRDataset'),
    # dict(
    #     ann_file='textrecog_test.json',
    #     data_root='data/svt',
    #     pipeline=None,
    #     test_mode=True,
    #     type='OCRDataset'),
    # dict(
    #     ann_file='textrecog_test.json',
    #     data_root='data/svtp',
    #     pipeline=None,
    #     test_mode=True,
    #     type='OCRDataset'),
    # dict(
    #     ann_file='textrecog_test.json',
    #     data_root='data/icdar2013',
    #     pipeline=None,
    #     test_mode=True,
    #     type='OCRDataset'),
    # dict(
    #     ann_file='textrecog_test.json',
    #     data_root='data/icdar2015',
    #     pipeline=None,
    #     test_mode=True,
    #     type='OCRDataset'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        100,
        32,
    ), type='Resize'),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'valid_ratio',
        ),
        type='PackTextRecogInputs'),
]
train_cfg = dict(max_epochs=20, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=128,
    dataset=dict(
        datasets=[
            dict(
                ann_file='textrecog_train.json',
                data_root='data/rec',
                pipeline=None,
                type='OCRDataset'),
            dict(
                ann_file='textrecog_train.json',
                data_root='data/rec',
                pipeline=None,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(ignore_empty=True, min_size=2, type='LoadImageFromFile'),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(keep_ratio=False, scale=(
                100,
                32,
            ), type='Resize'),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'valid_ratio',
                ),
                type='PackTextRecogInputs'),
        ],
        type='ConcatDataset'),
    num_workers=24,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_dataset = dict(
    datasets=[
        dict(
            ann_file='textrecog_train.json',
            data_root='data/rec',
            pipeline=None,
            type='OCRDataset'),
        dict(
            ann_file='textrecog_train.json',
            data_root='data/rec',
            pipeline=None,
            type='OCRDataset'),
    ],
    pipeline=[
        dict(ignore_empty=True, min_size=2, type='LoadImageFromFile'),
        dict(type='LoadOCRAnnotations', with_text=True),
        dict(keep_ratio=False, scale=(
            100,
            32,
        ), type='Resize'),
        dict(
            meta_keys=(
                'img_path',
                'ori_shape',
                'img_shape',
                'valid_ratio',
            ),
            type='PackTextRecogInputs'),
    ],
    type='ConcatDataset')
train_list = [
    dict(
        ann_file='textrecog_train.json',
        data_root='data/rec',
        pipeline=None,
        type='OCRDataset'),
    dict(
        ann_file='textrecog_train.json',
        data_root='data/rec',
        pipeline=None,
        type='OCRDataset'),
]
train_pipeline = [
    dict(ignore_empty=True, min_size=2, type='LoadImageFromFile'),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(keep_ratio=False, scale=(
        100,
        32,
    ), type='Resize'),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'valid_ratio',
        ),
        type='PackTextRecogInputs'),
]
tta_model = dict(type='EncoderDecoderRecognizerTTAModel')
tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                    true_transforms=[
                        dict(
                            args=[
                                dict(cls='Rot90', k=0, keep_size=False),
                            ],
                            type='ImgAugWrapper'),
                    ],
                    type='ConditionApply'),
                dict(
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                    true_transforms=[
                        dict(
                            args=[
                                dict(cls='Rot90', k=1, keep_size=False),
                            ],
                            type='ImgAugWrapper'),
                    ],
                    type='ConditionApply'),
                dict(
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                    true_transforms=[
                        dict(
                            args=[
                                dict(cls='Rot90', k=3, keep_size=False),
                            ],
                            type='ImgAugWrapper'),
                    ],
                    type='ConditionApply'),
            ],
            [
                dict(keep_ratio=False, scale=(
                    100,
                    32,
                ), type='Resize'),
            ],
            [
                dict(type='LoadOCRAnnotations', with_text=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'valid_ratio',
                    ),
                    type='PackTextRecogInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        datasets=[
            dict(
                ann_file='textrecog_test.json',
                data_root='data/rec',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
            # dict(
            #     ann_file='textrecog_test.json',
            #     data_root='data/iiit5k',
            #     pipeline=None,
            #     test_mode=True,
            #     type='OCRDataset'),
            # dict(
            #     ann_file='textrecog_test.json',
            #     data_root='data/svt',
            #     pipeline=None,
            #     test_mode=True,
            #     type='OCRDataset'),
            # dict(
            #     ann_file='textrecog_test.json',
            #     data_root='data/svtp',
            #     pipeline=None,
            #     test_mode=True,
            #     type='OCRDataset'),
            # dict(
            #     ann_file='textrecog_test.json',
            #     data_root='data/icdar2013',
            #     pipeline=None,
            #     test_mode=True,
            #     type='OCRDataset'),
            # dict(
            #     ann_file='textrecog_test.json',
            #     data_root='data/icdar2015',
            #     pipeline=None,
            #     test_mode=True,
            #     type='OCRDataset'),
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                100,
                32,
            ), type='Resize'),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'valid_ratio',
                ),
                type='PackTextRecogInputs'),
        ],
        type='ConcatDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    dataset_prefixes=[
        'CUTE80',
        'IIIT5K',
        'SVT',
        'SVTP',
        'IC13',
        'IC15',
    ],
    metrics=[
        dict(
            mode=[
                'exact',
                'ignore_case',
                'ignore_case_symbol',
            ],
            type='WordMetric'),
        dict(type='CharMetric'),
    ],
    type='MultiDatasetsEvaluator')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='TextRecogLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/satrn_shallow_5e_st_mj/'