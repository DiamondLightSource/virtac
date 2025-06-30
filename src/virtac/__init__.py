"""virtac: a python based virtual accelerator using ATIP.

.. data:: __version__
    :type: str

    Version number as calculated by https://github.com/pypa/setuptools_scm
"""

from . import atip_server, create_csv, masks, mirror_objects
from ._version import __version__

__all__ = [
    "__version__",
    "atip_server",
    "create_csv",
    "masks",
    "mirror_objects",
]
