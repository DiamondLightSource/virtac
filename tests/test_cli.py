import subprocess
import sys

from virtac import __version__


def test_cli_version():
    cmd = [sys.executable, "-m", "virtac", "--version"]
    assert __version__ in subprocess.check_output(cmd).decode().strip()
