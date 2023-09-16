import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from .pose_utils import _get_keypoints, _pad_image
from insightface import model_zoo
from dofaker.utils import download_file, get_model_url
from dofaker.transforms import center_crop, pad


class PoseTransfer:

    def __init__(self,
                 name='pose_transfer',
                 root='weights/models',
                 pose_estimator=None):
        assert pose_estimator is not None, "The pose_estimator of PoseTransfer shouldn't be None"
        self.pose_estimator = pose_estimator
        _, model_file = download_file(get_model_url(name),
                                      save_dir=root,
                                      overwrite=False)
        providers = model_zoo.model_zoo.get_default_providers()
        self.session = model_zoo.model_zoo.PickableInferenceSession(
            model_file, providers=providers)

        self.input_mean = 127.5
        self.input_std = 127.5
        inputs = self.session.get_inputs()
        self.input_names = []
        for inp in inputs:
            self.input_names.append(inp.name)
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.output_names = output_names
        assert len(
            self.output_names
        ) == 1, "The output number of PoseTransfer model should be 1, but got {}, please check your model.".format(
            len(self.output_names))
        output_shape = outputs[0].shape
        input_cfg = inputs[0]
        input_shape = input_cfg.shape
        self.input_shape = input_shape
        print('pose transfer shape:', self.input_shape)

    def forward(self, source_image, target_image, image_format='rgb'):
        h, w, c = source_image.shape
        if image_format == 'rgb':
            pass
        elif image_format == 'bgr':
            source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
            target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            image_format = 'rgb'
        else:
            raise UserWarning(
                "PoseTransfer not support image format {}".format(image_format))
        imgA = self._resize_and_pad_image(source_image)
        kptA = self._estimate_keypoints(imgA, image_format=image_format)
        mapA = self._keypoints2heatmaps(kptA)

        imgB = self._resize_and_pad_image(target_image)
        kptB = self._estimate_keypoints(imgB)
        mapB = self._keypoints2heatmaps(kptB)

        imgA_t = (imgA.astype('float32') - self.input_mean) / self.input_std
        imgA_t = imgA_t.transpose([2, 0, 1])[None, ...]
        mapA_t = mapA.transpose([2, 0, 1])[None, ...]
        mapB_t = mapB.transpose([2, 0, 1])[None, ...]
        mapAB_t = np.concatenate((mapA_t, mapB_t), axis=1)
        pred = self.session.run(self.output_names, {
            self.input_names[0]: imgA_t,
            self.input_names[1]: mapAB_t
        })[0]
        target_image = pred.transpose((0, 2, 3, 1))[0]
        bgr_target_image = np.clip(
            self.input_std * target_image + self.input_mean, 0,
            255).astype(np.uint8)[:, :, ::-1]
        crop_size = (256,
                     min((256 * target_image.shape[1] // target_image.shape[0]),
                         176))
        bgr_image = center_crop(bgr_target_image, crop_size)
        bgr_image = cv2.resize(bgr_image, (w, h), interpolation=cv2.INTER_CUBIC)
        return bgr_image

    def get(self, source_image, target_image, image_format='rgb'):
        return self.forward(source_image, target_image, image_format)

    def _resize_and_pad_image(self, image: np.ndarray, size=256):
        w = size * image.shape[1] // image.shape[0]
        w_box = min(w, size * 11 // 16)
        image = cv2.resize(image, (w, size), interpolation=cv2.INTER_CUBIC)
        image = center_crop(image, (size, w_box))
        image = pad(image,
                    size - w_box,
                    size - w_box,
                    size - w_box,
                    size - w_box,
                    fill=255)
        image = center_crop(image, (size, size))
        return image

    def _estimate_keypoints(self, image: np.ndarray, image_format='rgb'):
        keypoints = self.pose_estimator.get(image, image_format)
        keypoints = keypoints[0] if len(keypoints) > 0 else np.zeros(
            (18, 3), dtype=np.int32)
        keypoints[np.where(keypoints[:, 2] == 0), :2] = -1
        keypoints = keypoints[:, :2]
        return keypoints

    def _keypoints2heatmaps(self, keypoints, size=256):
        heatmaps = np.zeros((size, size, keypoints.shape[0]), dtype=np.float32)
        for k in range(keypoints.shape[0]):
            x, y = keypoints[k]
            if x == -1 or y == -1:
                continue
            heatmaps[y, x, k] = 1.0
        return heatmaps
