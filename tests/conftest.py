import pytest

from tholos.db import connect, init


@pytest.fixture
def db(tmp_path):
    connection = connect(str(tmp_path / "tholos.db"))
    init(connection)
    yield connection
    connection.close()
