"""Shared logging and progress helpers.

Logs and progress bars go to stderr so stdout stays machine-readable JSON.
Never log advertisement text: only counts, ids, durations and error types.
"""
import logging
import os
import sys

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # tqdm is optional; progress degrades to plain iteration
    _tqdm = None


def setup_logging(level=None):
    """Configure the root logger once; level from argument or AVITO_LOG_LEVEL."""
    level = (level or os.environ.get('AVITO_LOG_LEVEL') or 'INFO').upper()
    logging.basicConfig(level=level, stream=sys.stderr, force=True,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        datefmt='%H:%M:%S')


def progress(iterable=None, **kwargs):
    """tqdm on stderr, or the plain iterable when tqdm is not installed."""
    if _tqdm is None:
        return iterable
    kwargs.setdefault('file', sys.stderr)
    return _tqdm(iterable, **kwargs)
