"""Global test configuration: the zero-network guarantee.

Every test runs with sockets disabled unless it explicitly opts in with the
`llm` marker — CI runs `pytest -m "not llm"` with no secrets configured, so a
test that tries to reach the network fails loudly instead of silently
depending on it.
"""

import pytest
from pytest_socket import disable_socket, enable_socket


def pytest_runtest_setup(item: pytest.Item) -> None:
    if "llm" in item.keywords:
        enable_socket()
    else:
        disable_socket()
