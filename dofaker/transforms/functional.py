import numbers
import numpy as np

import cv2


def center_crop(image: np.ndarray, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])

    image_height, image_width, c = image.shape
    crop_height, crop_width = output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) //
            2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) //
            2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) //
            2 if crop_height > image_height else 0,
        ]
        image = cv2.copyMakeBorder(image,
                                   padding_ltrb[1],
                                   padding_ltrb[3],
                                   padding_ltrb[0],
                                   padding_ltrb[2],
                                   cv2.BORDER_CONSTANT,
                                   value=(0, 0, 0))
        image_height, image_width, c = image.shape
        if crop_width == image_width and crop_height == image_height:
            return image

    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return image[crop_top:crop_top + crop_height,
                 crop_left:crop_left + crop_width]


def pad(image,
        left,
        top,
        right,
        bottom,
        fill: int = 0,
        padding_mode: str = "constant"):
    if padding_mode == 'constant':
        return cv2.copyMakeBorder(image,
                                  top,
                                  bottom,
                                  left,
                                  right,
                                  cv2.BORDER_CONSTANT,
                                  value=(fill, fill, fill))
    else:
        raise UserWarning('padding mode {} not supported.'.format(padding_mode))
