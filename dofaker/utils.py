from .face_swap import InSwapper


def get_swapper_model(name='', onnx_path=None):
    if name.lower() == 'inswapper':
        return InSwapper(model_file=onnx_path)
    else:
        raise UserWarning('The swapper model {} not support.'.format(name))
