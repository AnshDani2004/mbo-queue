"""mbo-queue: exact FIFO queue reconstruction from CME MBO data and a benchmark of MBP queue estimators."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mbo-queue")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0+unknown"
