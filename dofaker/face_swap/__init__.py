from .inswapper import InSwapper


def get_swapper_model(name='', root=None, **kwargs):
    if name.lower() == 'inswapper':
        return InSwapper(name=name, root=root, **kwargs)
    else:
        raise UserWarning('The swapper model {} not support.'.format(name))
