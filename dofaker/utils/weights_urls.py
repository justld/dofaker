
WEIGHT_URLS = {
    'buffalo_l': 'https://github.com/justld/dofaker/releases/download/v0.0/buffalo_l.zip',
    'buffalo_s': 'https://github.com/justld/dofaker/releases/download/v0.0/buffalo_s.zip',
    'buffalo_sc': 'https://github.com/justld/dofaker/releases/download/v0.0/buffalo_sc.zip',
    'inswapper': 'https://github.com/justld/dofaker/releases/download/v0.0/inswapper_128.onnx',
}

def get_model_url(model_name):
    return WEIGHT_URLS[model_name]
