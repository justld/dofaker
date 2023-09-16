import numpy as np


def _pad_image(image, stride=1, padvalue=0):
    assert len(image.shape) == 2 or len(image.shape) == 3
    h, w = image.shape[:2]
    pads = [None] * 4
    pads[0] = 0  # left
    pads[1] = 0  # top
    pads[2] = 0 if (w % stride == 0) else stride - (w % stride)  # right
    pads[3] = 0 if (h % stride == 0) else stride - (h % stride)  # bottom
    num_channels = 1 if len(image.shape) == 2 else image.shape[2]
    image_padded = np.ones(
        (h + pads[3], w + pads[2], num_channels), dtype=np.uint8) * padvalue
    image_padded = np.squeeze(image_padded)
    image_padded[:h, :w] = image
    return image_padded, pads


def _get_keypoints(candidates, subsets):
    k = subsets.shape[0]
    keypoints = np.zeros((k, 18, 3), dtype=np.int32)
    for i in range(k):
        for j in range(18):
            index = np.int32(subsets[i][j])
            if index != -1:
                x, y = np.int32(candidates[index][:2])
                keypoints[i][j] = (x, y, 1)
    return keypoints
