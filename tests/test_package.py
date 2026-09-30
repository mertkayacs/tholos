from importlib.metadata import version

import tholos


def test_version():
    assert tholos.__version__ == version("tholos") == "0.1.0"
