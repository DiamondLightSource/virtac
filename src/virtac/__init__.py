"""virtac: a python based virtual accelerator using ATIP.

.. data:: __version__
    :type: str

    Version number as calculated by https://github.com/pypa/setuptools_scm
"""

from . import create_csv, masks, mirror_objects, virtac_server
from ._version import __version__

__all__ = [
    "__version__",
    "virtac_server",
    "create_csv",
    "masks",
    "mirror_objects",
]
