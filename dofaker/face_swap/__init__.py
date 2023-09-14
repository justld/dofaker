from .inswapper import InSwapper


def get_swapper_model(name='', root=None):
    if name.lower() == 'inswapper':
        return InSwapper(name=name, root=root)
    else:
        raise UserWarning('The swapper model {} not support.'.format(name))
