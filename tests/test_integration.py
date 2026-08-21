import numpy as np
from qpyd.parameters import DerivedParameters, DirectParameters
from qpyd.simulator import Simulator


def test_direct_vs_derived_parity(param_config):
    """
    Both models must produce identically equal physical observables when 
    evaluating equivalent quantum systems across all distinct regimes.
    """
    shared_kwargs = {
        "T": param_config["T"],
        "slopes": [1.0, 1.0],
        "gammas": [1.0, 1.0],
        "setup": "symmetric",
        "Vb_range": (-2.0, 2.0, 15),
        "Vg_range": (-2.0, 10.0, 15), 
    }

    params_derived = DerivedParameters(
        charge_numbers=[0, 1, 2],
        orbital_energies=[param_config["E"]],
        U=param_config["U"],
        V=0.0, J=0.0, Jp=0.0,
        B=[0.0, 0.0, 0.0],
        M_rel=0.0,
        **shared_kwargs
    )
    sim_derived = Simulator(params_derived)
    sim_derived.solve()
    results_derived = sim_derived.get_results()
    
    params_direct = DirectParameters(
        charge_numbers=[0, 1, 2],
        energies=[
            [(0.0, 'CS')],
            [(param_config["E"], 'D')],
            [(param_config["N2_E"], 'CS')]
        ],
        spin_labels=True,  
        gamma_rel=0.0,
        **shared_kwargs
    )
    sim_direct = Simulator(params_direct)
    sim_direct.solve()
    results_direct = sim_direct.get_results()

    np.testing.assert_allclose(
        results_derived['N_avg'], 
        results_direct['N_avg'], 
        rtol=1e-3, 
        atol=1e-3, 
        err_msg=f"Occupation <N> mismatch in regime: {param_config['id']}"
    )

    np.testing.assert_allclose(
        results_derived['I'], 
        results_direct['I'], 
        rtol=1e-3,
        atol=1e-3,
        err_msg=f"Current I mismatch in regime: {param_config['id']}"
    )

    np.testing.assert_allclose(
        results_derived['G'], 
        results_direct['G'], 
        rtol=1e-3,
        atol=1e-3,
        err_msg=f"Differential conductance G mismatch in regime: {param_config['id']}"
    )