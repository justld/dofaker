WEIGHT_URLS = {
    'buffalo_l':
    'https://github.com/justld/dofaker/releases/download/v0.1/buffalo_l.zip',
    'buffalo_s':
    'https://github.com/justld/dofaker/releases/download/v0.1/buffalo_s.zip',
    'buffalo_sc':
    'https://github.com/justld/dofaker/releases/download/v0.1/buffalo_sc.zip',
    'inswapper':
    'https://github.com/justld/dofaker/releases/download/v0.1/inswapper_128.onnx',
    'gfpgan':
    'https://github.com/justld/dofaker/releases/download/v0.1/GFPGANv1.3.onnx',
    'bsrgan':
    'https://github.com/justld/dofaker/releases/download/v0.1/bsrgan_4.onnx',
    'openpose_body':
    'https://github.com/justld/dofaker/releases/download/v0.1/openpose_body.onnx',
    'pose_transfer':
    'https://github.com/justld/dofaker/releases/download/v0.1/pose_transfer.onnx',
}


def get_model_url(model_name):
    return WEIGHT_URLS[model_name]
