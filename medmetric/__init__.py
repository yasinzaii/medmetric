try:
    from ._version import __version__
except (ImportError, ModuleNotFoundError):
    from importlib.metadata import version as _v
    __version__ = _v("medmetric")
