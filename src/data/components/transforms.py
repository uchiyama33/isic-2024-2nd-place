import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import v2
import torch


def get_transforms(version, img_size, finetuning=True):
    if finetuning:
        normalize = A.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0
        )
    else:
        normalize = A.Normalize(
            mean=[0.702, 0.522, 0.416], std=[0.139, 0.131, 0.123], max_pixel_value=255.0, p=1.0
        )

    if version == 1:
        transforms_train = A.Compose(
            [
                A.Resize(img_size, img_size),
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.Downscale(p=0.25),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=60, p=0.5),
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                normalize,
                ToTensorV2(),
            ],
            p=1.0,
        )
        transforms_test = A.Compose(
            [
                A.Resize(img_size, img_size),
                normalize,
                ToTensorV2(),
            ],
            p=1.0,
        )
        transforms_type = "albumentations"
    elif version == 2:
        transforms_train = A.Compose(
            [
                A.Transpose(p=0.5),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=5),
                        A.MedianBlur(blur_limit=5),
                        A.GaussianBlur(blur_limit=5),
                        A.GaussNoise(var_limit=(5.0, 30.0)),
                    ],
                    p=0.7,
                ),
                A.OneOf(
                    [
                        A.OpticalDistortion(distort_limit=1.0),
                        A.GridDistortion(num_steps=5, distort_limit=1.0),
                        A.ElasticTransform(alpha=3),
                    ],
                    p=0.7,
                ),
                A.CLAHE(clip_limit=4.0, p=0.7),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
                A.Resize(img_size, img_size),
                A.CoarseDropout(
                    max_height=int(img_size * 0.375), max_width=int(img_size * 0.375), max_holes=1, p=0.7
                ),
                normalize,
                ToTensorV2(),
            ]
        )
        transforms_test = A.Compose(
            [
                A.Resize(img_size, img_size),
                normalize,
                ToTensorV2(),
            ],
        )
        transforms_type = "albumentations"
    elif version == 3:
        transforms_train = v2.Compose(
            [
                v2.ToImage(),
                v2.RandomApply([v2.JPEG([30, 80])]),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.RandAugment(6, 12),
                v2.RandomErasing(scale=(0.02, 0.05)),
                v2.Resize(img_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        transforms_test = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(img_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        transforms_type = "torchvision"

    elif version == 4:
        transforms_train = A.Compose(
            [
                A.Transpose(p=0.5),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=5),
                        A.MedianBlur(blur_limit=5),
                        A.GaussianBlur(blur_limit=5),
                        A.GaussNoise(var_limit=(5.0, 30.0)),
                    ],
                    p=0.7,
                ),
                A.OneOf(
                    [
                        A.OpticalDistortion(distort_limit=0.1),
                        A.GridDistortion(num_steps=5, distort_limit=0.03),
                        A.ElasticTransform(alpha=1),
                    ],
                    p=0.7,
                ),
                A.CLAHE(clip_limit=4.0, p=0.7),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
                A.OneOf(
                    [
                        A.ImageCompression(quality_lower=50, quality_upper=80),
                        A.Solarize((168, 224), p=0.5),  # 128がデフォルトでrandaugの中間の強さ
                        A.Posterize((1, 3), p=0.5),  # 4がデフォルトでrandaugの中間の強さ
                    ]
                ),
                A.Resize(img_size, img_size),
                A.CoarseDropout(
                    max_height=int(img_size * 0.2),
                    max_width=int(img_size * 0.2),
                    max_holes=3,
                    min_holes=1,
                    p=0.7,
                ),
                normalize,
                ToTensorV2(),
            ]
        )
        transforms_test = A.Compose(
            [
                A.Resize(img_size, img_size),
                normalize,
                ToTensorV2(),
            ],
        )
        transforms_type = "albumentations"

    elif version == 5:
        transforms_train = A.Compose(
            [
                A.OneOf(
                    [
                        A.Posterize(num_bits=1),
                        A.ImageCompression(quality_lower=50, quality_upper=80),
                        A.Downscale(scale_min=0.5, scale_max=0.75),
                    ],
                    p=0.7,
                ),
                A.Transpose(p=0.5),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=5),
                        A.MedianBlur(blur_limit=5),
                        A.GaussianBlur(blur_limit=5),
                        A.GaussNoise(var_limit=(5.0, 30.0)),
                    ],
                    p=0.7,
                ),
                A.OneOf(
                    [
                        A.OpticalDistortion(distort_limit=1.0),
                        A.GridDistortion(num_steps=5, distort_limit=1.0),
                        A.ElasticTransform(alpha=3),
                    ],
                    p=0.7,
                ),
                A.CLAHE(clip_limit=4.0, p=0.7),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
                A.Resize(img_size, img_size),
                A.CoarseDropout(
                    max_height=int(img_size * 0.375), max_width=int(img_size * 0.375), max_holes=1, p=0.7
                ),
                normalize,
                ToTensorV2(),
            ]
        )
        transforms_test = A.Compose(
            [
                A.Resize(img_size, img_size),
                normalize,
                ToTensorV2(),
            ],
        )
        transforms_type = "albumentations"

    elif version == 6:
        # meta_target向けにv2を弱める
        transforms_train = A.Compose(
            [
                A.Transpose(p=0.5),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=5),
                        A.MedianBlur(blur_limit=5),
                        A.GaussianBlur(blur_limit=5),
                        A.GaussNoise(var_limit=(5.0, 30.0)),
                    ],
                    p=0.7,
                ),
                A.OneOf(
                    [
                        A.OpticalDistortion(distort_limit=0.05),
                        A.GridDistortion(num_steps=5, distort_limit=0.3),
                        A.ElasticTransform(alpha=1),
                    ],
                    p=0.7,
                ),
                A.CLAHE(clip_limit=4.0, p=0.7),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
                A.Resize(img_size, img_size),
                A.CoarseDropout(
                    max_height=int(img_size * 0.1), max_width=int(img_size * 0.1), max_holes=1, p=0.7
                ),
                normalize,
                ToTensorV2(),
            ]
        )
        transforms_test = A.Compose(
            [
                A.Resize(img_size, img_size),
                normalize,
                ToTensorV2(),
            ],
        )
        transforms_type = "albumentations"

    elif version == 7:
        # v6からblur,noise,ShiftScaleRotateを少し強めた
        transforms_train = A.Compose(
            [
                A.Transpose(p=0.5),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=7),
                        A.MedianBlur(blur_limit=7),
                        A.GaussianBlur(blur_limit=7),
                        A.GaussNoise(var_limit=(10, 50)),
                    ],
                    p=0.7,
                ),
                A.OneOf(
                    [
                        A.OpticalDistortion(distort_limit=0.05),
                        A.GridDistortion(num_steps=5, distort_limit=0.3),
                        A.ElasticTransform(alpha=1),
                    ],
                    p=0.7,
                ),
                A.CLAHE(clip_limit=4.0, p=0.7),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.15, scale_limit=0.15, rotate_limit=20, border_mode=0, p=0.85
                ),
                A.Resize(img_size, img_size),
                A.CoarseDropout(
                    max_height=int(img_size * 0.1), max_width=int(img_size * 0.1), max_holes=1, p=0.7
                ),
                normalize,
                ToTensorV2(),
            ]
        )
        transforms_test = A.Compose(
            [
                A.Resize(img_size, img_size),
                normalize,
                ToTensorV2(),
            ],
        )
        transforms_type = "albumentations"

    elif version == 8:
        # TIP pretrain向け, cutoutなし非剛体変形弱め、劣化あり回転強め
        transforms_train = A.Compose(
            [
                A.OneOf(
                    [
                        A.Posterize(num_bits=1),
                        A.ImageCompression(quality_lower=50, quality_upper=80),
                        A.Downscale(scale_min=0.5, scale_max=0.75),
                    ],
                    p=0.7,
                ),
                A.Transpose(p=0.5),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=5),
                        A.MedianBlur(blur_limit=5),
                        A.GaussianBlur(blur_limit=5),
                        A.GaussNoise(var_limit=(5.0, 30.0)),
                    ],
                    p=0.7,
                ),
                A.OneOf(
                    [
                        A.OpticalDistortion(distort_limit=0.05),
                        A.GridDistortion(num_steps=5, distort_limit=0.3),
                        A.ElasticTransform(alpha=1),
                    ],
                    p=0.7,
                ),
                A.CLAHE(clip_limit=4.0, p=0.7),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, border_mode=0, p=0.85),
                A.Resize(img_size, img_size),
                normalize,
                ToTensorV2(),
            ]
        )
        transforms_test = A.Compose(
            [
                A.Resize(img_size, img_size),
                normalize,
                ToTensorV2(),
            ],
        )
        transforms_type = "albumentations"
    return transforms_train, transforms_test, transforms_type
