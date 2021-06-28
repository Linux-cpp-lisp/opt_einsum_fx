import pytest

import torch

FLOAT_TOLERANCE = {
    t: torch.as_tensor(v, dtype=t)
    for t, v in {torch.float32: 1e-5, torch.float64: 1e-10}.items()
}


@pytest.fixture(scope="session", autouse=True, params=["float32", "float64"])
def float_tolerance(request):
    """Run all tests with various PyTorch default dtypes.

    This is a session-wide, autouse fixture — you only need to request it explicitly if a test needs to know the tolerance for the current default dtype.

    Returns
    --------
        A precision threshold to use for closeness tests.
    """
    old_dtype = torch.get_default_dtype()
    dtype = {"float32": torch.float32, "float64": torch.float64}[request.param]
    torch.set_default_dtype(dtype)
    yield FLOAT_TOLERANCE[dtype]
    torch.set_default_dtype(old_dtype)


@pytest.fixture(scope="session")
def allclose(float_tolerance):
    return lambda x, y: torch.allclose(x, y, atol=float_tolerance)
