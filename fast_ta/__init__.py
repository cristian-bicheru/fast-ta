try:
    from fast_ta import core, momentum, volume, volatility
except ImportError:
    raise ImportError("Run setup.py to build library before importing.")

__all__ = ["fast_ta"]
