# edit also setup.py!
__version__ = '3.0.8'
__version_info__ = tuple([int(i) for i in __version__.split('.')])

try:
    from pygco import *  # noqa: F401 F403
except ImportError:
    from gco.pygco import *  # noqa: F401 F403
