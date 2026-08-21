import pytest
import numpy as np
from qpyd.parameters import DerivedParameters


@pytest.fixture
def derived_params():
    """Provides a standard 2-orbital system with symmetric couplings."""
    return DerivedParameters(
        charge_numbers=[1, 2],
        orbital_energies=[0.0, 1.0],
        U=2.0, V=0.2, J=0.2, Jp=0.1,
        T=0.1, slopes=[1.0, 1.0],
        B=[1.0, 0.5, 0.8], 
        setup="symmetric"
    )


def test_fermionic_commutation(derived_params):
    """Creation/annihilation operators must strictly follow {c_i, c_j} = 0."""
    state_a, sign_a = derived_params._apply_ops((), [('c', 0), ('c', 2)])
    state_b, sign_b = derived_params._apply_ops((), [('c', 2), ('c', 0)])
    
    assert state_a == state_b == (0, 2)
    assert sign_a == -sign_b


def test_hamiltonian_hermiticity(param_config):
    """The fully constructed many-body Hamiltonian must be strictly Hermitian."""
    params = DerivedParameters(
        charge_numbers=[1, 2],
        orbital_energies=[0.0, 1.0],
        U=param_config["U"], 
        V=param_config["V"], 
        J=param_config["J"], 
        Jp=param_config["Jp"],
        T=param_config["T"], 
        slopes=[1.0, 1.0],
        B=param_config["B"], 
        setup="symmetric"
    )
    
    H, _, _ = params.construct_hamiltonian(N=2)
    H_dagger = H.conj().T
    
    np.testing.assert_allclose(
        H, H_dagger, 
        atol=1e-10, 
        err_msg=f"Hermiticity failed in regime: {param_config['id']}"
    )


def test_spin_eigenvalues():
    """Simultaneous diagonalization must properly resolve Singlets (S=0) and Triplets (S=1)."""
    params = DerivedParameters(
        charge_numbers=[1, 2],
        orbital_energies=[0.0, 1.0],
        U=2.0, V=1.0, J=0.2, Jp=0.1,
        T=0.1, slopes=[1.0, 1.0],
        B=[0.0, 0.0, 0.0],  # B=0 required to maintain pure spin eigenstates
        setup="symmetric"
    )
    
    n2_states = params.states_dict[2]
    spins = [state['S'] for state in n2_states]
    
    assert 1.0 in spins, "No Triplet state (S=1.0) found in N=2 sector."
    assert 0.0 in spins, "No Singlet state (S=0.0) found in N=2 sector."
    
    triplet_sz = sorted([state['Sz'] for state in n2_states if state['S'] == 1.0])
    np.testing.assert_allclose(
        triplet_sz, [-1.0, 0.0, 1.0], 
        err_msg="Triplet Sz projections do not correctly span -1, 0, 1."
    )