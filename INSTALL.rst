=================
VIRTAC Installation
=================

This guide is for Linux and is based on the current structures of AT and Pytac,
if you find a mistake anywhere in VIRTAC please raise an issue on VIRTAC's GitHub
page, `here. <https://github.com/DiamondLightSource/virtac>`_

Initial Setup and Installation
------------------------------

**Option 1: Install VIRTAC using pip**::

    $ pip install virtac

**Option 2: Install VIRTAC from GitHub**:

1. Clone VIRTAC::

    $ cd <source-directory>
    $ git clone https://github.com/DiamondLightSource/virtac.git

2. From within a python virtual environment, install the dependencies::

    $ cd virtac
    $ pip install -e ./

3. Run the tests to ensure everything is working correctly::

    $ python -m pytest

Troubleshooting
---------------

Please note that for VIRTAC to function with Python 3.7 or later, you must
use Cothread>=2.16.
