__all__ = ['log']


def log(msg='', end='\n'):
    print(msg, flush=True, end=end)
