from typing import Tuple

import albumentations as albu
import numpy as np
from albumentations.pytorch import ToTensorV2
from numpy.typing import NDArray
from torch import Tensor


def get_train_transforms(img_width: int, img_height: int) -> albu.Compose:
    return albu.Compose(
        [
            albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            albu.GaussianBlur(),
            albu.RandomResizedCrop(height=img_height, width=img_width, always_apply=True),
        ],
    )


def get_valid_transforms(img_width: int, img_height: int) -> albu.Compose:
    return albu.Compose(
        [albu.Resize(height=img_height, width=img_width)],
    )


def cv_image_to_tensor(img: NDArray[float], normalize: bool = True) -> Tensor:
    ops = [
        ToTensorV2(),
    ]
    if normalize:
        ops.insert(0, albu.Normalize())
    to_tensor = albu.Compose(ops)
    return to_tensor(image=img)['image']


def denormalize(
    img: NDArray[int],
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    max_value: int = 255,
):
    denorm = albu.Normalize(
        mean=[-me / st for me, st in zip(mean, std)],  # noqa: WPS221
        std=[1.0 / st for st in std],
        always_apply=True,
        max_pixel_value=1.0,
    )
    denorm_img = denorm(image=img)['image'] * max_value
    return denorm_img.astype(np.uint8)


def tensor_to_cv_image(tensor: Tensor) -> NDArray[float]:
    return tensor.permute(1, 2, 0).cpu().numpy()
