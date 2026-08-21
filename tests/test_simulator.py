import pytest
import numpy as np
from qpyd.parameters import DirectParameters, DerivedParameters
from qpyd.simulator import Simulator


@pytest.fixture(params=["direct", "derived"])
def solved_simulator(request):
    """Provides a solved simulator, parameterizing over both model paradigms."""
    
    shared_kwargs = {
        "T": 0.1,
        "slopes": [1.0, 1.0],
        "gammas": [1.0, 1.0],
        "setup": "symmetric",
        "Vb_range": (-2.0, 2.0, 11),
        "Vg_range": (-2.0, 2.0, 11)
    }

    if request.param == "direct":
        params = DirectParameters(
            charge_numbers=[0, 1, 2],
            energies=[[(0.0, 'CS')], [(1.0, 'D')], [(3.0, 'CS')]],
            gamma_rel=0.0,
            spin_labels=False,
            **shared_kwargs
        )
    else:
        params = DerivedParameters(
            charge_numbers=[0, 1, 2],
            orbital_energies=[1.0],
            U=1.0, V=0.0, J=0.0, Jp=0.0,
            B=[0.0, 0.0, 0.0],
            M_rel=0.0,
            **shared_kwargs
        )

    sim = Simulator(params)
    sim.solve()
    
    return sim


def test_probability_normalization(solved_simulator):
    """Steady-state probabilities must conserve particle probability (sum to 1.0)."""
    results = solved_simulator.get_results()
    
    probabilities = results['P']
    total_prob = np.sum(probabilities, axis=-1)
    
    np.testing.assert_allclose(
        total_prob, 1.0, 
        atol=1e-5, 
        err_msg=f"Probabilities do not sum to 1.0 in {solved_simulator.p.__class__.__name__}."
    )


def test_zero_bias_equilibrium(solved_simulator):
    """At exactly zero bias (Vb = 0), the net DC current must be absolutely zero."""
    results = solved_simulator.get_results()
    current = results['I']
    Vb_array = solved_simulator.p.Vbs_host
    
    # Locate the exact index for 0.0V 
    zero_bias_idx = np.argmin(np.abs(Vb_array))
    zero_bias_current = current[:, zero_bias_idx]
    
    np.testing.assert_allclose(
        zero_bias_current, 0.0, 
        atol=1e-8, 
        err_msg=f"Net current is non-zero at Vb=0 in {solved_simulator.p.__class__.__name__}."
    )


def test_occupation_bounds(solved_simulator):
    """Average occupation <N> must strictly obey fermionic limits (0 <= <N> <= N_max)."""
    results = solved_simulator.get_results()
    occupation = results['N_avg']
    max_electrons = max(solved_simulator.p.charge_numbers)
    
    assert np.all(occupation >= -1e-5), \
        f"Occupation dropped below 0 in {solved_simulator.p.__class__.__name__}."
        
    assert np.all(occupation <= max_electrons + 1e-5), (
        f"Occupation exceeded max ({max_electrons}) in {solved_simulator.p.__class__.__name__}."
    )