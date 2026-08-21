import pytest
import numpy as np


def generate_configs():
    """Generates randomized parameter configurations."""
    configs = []

    np.random.seed(10)
    for i in range(10):
        configs.append(
            {
                "id": f"Randomized_{i+1}",
                "E": float(np.random.uniform(0.0, 2.0)),
                "U": float(np.random.uniform(0.0, 5.0)),
                "V": float(np.random.uniform(0.0, 5.0)),
                "J": float(np.random.uniform(0.0, 5.0)),
                "Jp": float(np.random.uniform(0.0, 5.0)),
                "B": np.random.uniform(-5.0, 5.0, 3).tolist(),
                "T": float(np.random.uniform(0.01, 2.0)),
            }
        )

    # Compute the expected macro-state energy for N=2 for the Direct mapping tests
    for r in configs:
        r["N2_E"] = 2 * r["E"] + r["U"]

    return configs


@pytest.fixture(params=generate_configs(), ids=lambda c: c["id"])
def param_config(request):
    """
    Globally shared fixture to run tests over randomized parameter configs.
    """
    return request.param