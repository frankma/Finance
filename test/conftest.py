import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def close_figures():
    # plt.show() does not close figures on a non-interactive backend, so a 3D axes left by one test
    # would otherwise receive the next test's 2D plot calls
    yield
    plt.close('all')
