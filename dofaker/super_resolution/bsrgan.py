import numpy as np

import cv2

from insightface import model_zoo
from dofaker.utils import download_file, get_model_url


class BSRGAN:

    def __init__(self, name='bsrgan', root='weights/models', scale=1) -> None:
        _, model_file = download_file(get_model_url(name),
                                      save_dir=root,
                                      overwrite=False)
        self.scale = scale
        providers = model_zoo.model_zoo.get_default_providers()
        self.session = model_zoo.model_zoo.PickableInferenceSession(
            model_file, providers=providers)

        self.input_mean = 0.0
        self.input_std = 255.0
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
        ) == 1, "The output number of BSRGAN model should be 1, but got {}, please check your model.".format(
            len(self.output_names))
        output_shape = outputs[0].shape
        input_cfg = inputs[0]
        input_shape = input_cfg.shape
        self.input_shape = input_shape
        print('image super resolution shape:', self.input_shape)

    def forward(self, image, image_format='bgr'):
        if isinstance(image, str):
            image = cv2.imread(image, 1)
            image_format = 'bgr'
        elif isinstance(image, np.ndarray):
            if image_format == 'bgr':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image_format == 'rgb':
                pass
            else:
                raise UserWarning(
                    "BSRGAN not support image format {}".format(image_format))
        else:
            raise UserWarning(
                "BSRGAN input must be str or np.ndarray, but got {}.".format(
                    type(image)))
        img = (image - self.input_mean) / self.input_std
        pred = self.session.run(self.output_names,
                                {self.input_names[0]: img})[0]
        return pred

    def get(self, img, image_format='bgr'):
        if image_format.lower() == 'bgr':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif image_format.lower() == 'rgb':
            pass
        else:
            raise UserWarning(
                "gfpgan not support image format {}".format(image_format))
        h, w, c = img.shape
        blob = cv2.dnn.blobFromImage(
            img,
            1.0 / self.input_std, (w, h),
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=False)
        pred = self.session.run(self.output_names,
                                {self.input_names[0]: blob})[0]
        image_aug = pred.transpose((0, 2, 3, 1))[0]
        rgb_aug = np.clip(self.input_std * image_aug + self.input_mean, 0,
                          255).astype(np.uint8)
        rgb_aug = cv2.resize(rgb_aug,
                             (int(w * self.scale), int(h * self.scale)))
        bgr_aug = rgb_aug[:, :, ::-1]
        return bgr_aug
