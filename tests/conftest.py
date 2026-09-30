import pytest

from tholos.db import clock, connect, init


@pytest.fixture
def db(tmp_path):
    connection = connect(str(tmp_path / "tholos.db"))
    init(connection)
    yield connection
    connection.close()


@pytest.fixture
def pin_clock():
    yield clock.pin
    clock.unpin()
