import time
import numpy as np
import jax
import jax.numpy as jnp

from .parameters import BaseParameters, DirectParameters, DerivedParameters
from .solver import solve_chunk, bose, fermi


class SimulatorBase:
    """
    Foundational base class for Quantum Dot Master Equation simulations.

    This class provides the main computational framework for both the direct and 
    derived simulator subclasses. It manages the JAX-accelerated steady-state 
    solver, which computes stability diagrams over a 2D voltage grid (gate and 
    bias voltages) in memory-efficient batches.

    It handles all universal physics operations (Fermi-Dirac and Bose-Einstein 
    distributions), computes macroscopic observables (DC current, average charge 
    occupation, and differential conductance), and provides safe utilities for 
    manual rate modification (`set_rate`) and CPU-side data extraction 
    (`get_results`).

    Subclasses inherit this engine and are only required to implement the 
    `_build_energy_vectors` and `_build_transition_matrices` methods to define 
    how the specific state space and transition rates are assembled from their 
    respective parameter objects.

    Attributes
    ----------
    p : BaseParameters
        The validated system configuration and grid parameters.
    P_flat : jax.Array or None
        Flattened 2D array of steady-state probabilities for all grid points.
    I_flat : jax.Array or None
        Flattened 1D array of the calculated DC current for all grid points.
    NN_flat : jax.Array or None
        Flattened 1D array of the average charge occupation for all grid points.
    GG : jax.Array or None
        2D array representing the differential conductance (dI/dVb).
    """
    
    def __init__(self, sys_params: BaseParameters):
        """
        Initializes the simulator base.

        Parameters
        ----------
        sys_params : BaseParameters
            System configuration and grid parameters.
        """
        self.p = sys_params
        
        self.P_flat = None
        self.I_flat = None
        self.NN_flat = None
        self.GG = None
        
        self._build_energy_vectors()
        self._build_transition_matrices()

    def solve(self, batch_size: int = 25000) -> None:
        """
        Main execution method for the solution of the steady-state rate equation.

        It solves the stationary Pauli master equation (dP/dt = 0) over the entire 
        2D voltage grid (gate and bias voltages) to find the steady-state probability 
        distribution of the quantum dot's many-body states. Using these probabilities, 
        it computes the resulting DC current, average particle occupation, and 
        differential conductance across the device. 

        To safely manage memory and maximize JAX hardware acceleration efficiency 
        (especially on GPUs), the flattened grid is evaluated in chunks.

        Parameters
        ----------
        batch_size : int, optional
            Number of grid points to process simultaneously per batch. Lower this 
            if you experience Out-Of-Memory (OOM) errors on your hardware. 
            Default is 25000.

        Returns
        -------
        None
            The method does not return values directly. Instead, it populates the 
            instance attributes `self.P_flat`, `self.I_flat`, `self.NN_flat`, 
            and `self.GG`. Use `get_results()` to retrieve the reshaped arrays.
        """
        total_parameters = self.p.M_points * self.N_states
        print(
            f"Starting solver (Total points: {self.p.M_points:.2e}, "
            f"Total parameters: {total_parameters:.2e})..."
        )
        t0 = time.time()
        
        # Ensure flat input arrays
        Vext_flat = jnp.asarray(self.p.Vext_flat)
        muL_flat = jnp.asarray(self.p.muL_flat)
        muR_flat = jnp.asarray(self.p.muR_flat)

        P_out = []
        I_out = []
        sector_bounds_tuple = tuple(self.sector_bounds)

        # Process grid in batches
        for i in range(0, self.p.M_points, batch_size):
            end = min(i + batch_size, self.p.M_points)
            
            Vext_chunk = Vext_flat[i:end]
            muL_chunk = muL_flat[i:end]
            muR_chunk = muR_flat[i:end]

            # Execute compiled solver
            P_chunk, I_chunk = solve_chunk(
                Vext_chunk, muL_chunk, muR_chunk,
                self.dE, self.dNe, self.rate_L_seq, self.rate_R_seq, self.rate_rel,
                self.p.kB_T, sector_bounds_tuple
            )
            
            P_out.append(P_chunk)
            I_out.append(I_chunk)

        # Combine batches
        self.P_flat = jnp.vstack(P_out)
        self.I_flat = jnp.concatenate(I_out)
            
        print(f"Solved full grid in {time.time() - t0:.2f}s.")

        self._calculate_occupation()
        self._calculate_conductance()

    def set_rate(self, 
                 state_i: dict[int, float] | dict[int, list[float]], 
                 state_f: dict[int, float] | dict[int, list[float]], 
                 rate_matrix: jax.Array, 
                 value: float
                 ) -> jax.Array:
        """
        Manually overrides a specific transition rate matrix element by state definition.

        This method allows for direct control over individual state-to-state transition rates. 
        It accepts dictionaries defining the physical quantum numbers of the states. The 
        provided value replaces the target matrix element as given.

        Parameters
        ----------
        state_i : dict[int, float] | dict[int, list[float]]
            The initial state of the transition. Formatted as `{N: energy}` for 
            direct parameters, or `{N: [energy, S, Sz]}` for derived parameters.
        state_f : dict[int, float] | dict[int, list[float]]
            The final state of the transition, formatted identically to `state_i`.
        rate_matrix : jax.Array
            The target transition rate matrix to modify (e.g., `self.rate_L_seq`).
        value : float
            The transition rate value to input.

        Returns
        -------
        jax.Array
            A new JAX array representing the updated transition rate matrix.
            
        Raises
        ------
        ValueError
            If the defined state cannot be found or if the definition is ambiguous 
            (e.g., matching multiple degenerate states without sufficient quantum numbers).

        Notes
        -----
        - If DirectSimulator is used, the state inputs must have the form {N_i: E_i}.
        - If DerivedSimulator is used, the state inputs must have the form {N_i: [E_i, S_i, Sz_i]}.
        - N_i: charge number, E_i: eigenenergy, S_i: total spin eigenvalue, Sz_i: spin z-projection eigenvalue
        """
        if len(state_i) != 1 or len(state_f) != 1:
            raise ValueError("Each state dictionary must contain exactly one charge number key.")

        # Extract the values (energy or [energy, S, Sz]) from the dictionaries
        val_i = list(state_i.values())[0]
        val_f = list(state_f.values())[0]

        # Enforce constraints for DirectParameters
        if self.p.__class__.__name__ == 'DirectParameters':
            if not isinstance(val_i, (float, int)) or not isinstance(val_f, (float, int)):
                raise ValueError("If DirectSimulator is used, the state inputs must have the form {N_i: E_i}.")
                
        # Enforce constraints for DerivedParameters
        elif self.p.__class__.__name__ == 'DerivedParameters':
            if not isinstance(val_i, (list, tuple, np.ndarray)) or len(val_i) != 3:
                raise ValueError("If DerivedSimulator is used, the initial state inputs must have the form {N_i: [E_i, S_i, Sz_i]}.")
            if not isinstance(val_f, (list, tuple, np.ndarray)) or len(val_f) != 3:
                raise ValueError("If DerivedSimulator is used, the final state inputs must have the form {N_f: [E_f, S_f, Sz_f]}.")

        def _get_state_idx(state: dict[int, float] | dict[int, list[float]]) -> int:
            # Extract charge number N (key) and quantum numbers (value)
            N_target = list(state.keys())[0]
            val = state[N_target]
            
            if isinstance(val, (list, tuple, np.ndarray)):
                E_target = val[0]
                S_target = val[1] if len(val) > 1 else None
                Sz_target = val[2] if len(val) > 2 else None
            else:
                E_target = val
                S_target = None
                Sz_target = None

            Ne_arr = np.asarray(self.Ne)
            E0_arr = np.asarray(self.E0)
            
            # Base masking: match charge number and energy
            mask = (Ne_arr == N_target) & np.isclose(E0_arr, E_target, atol=1e-6)
            
            # Optional masking: match spin values if provided and if they exist in the simulator
            if S_target is not None and hasattr(self, 'S'):
                S_arr = np.asarray(self.S)
                mask &= np.isclose(S_arr, S_target, atol=1e-6)
                
            if Sz_target is not None and hasattr(self, 'Sz'):
                Sz_arr = np.asarray(self.Sz)
                mask &= np.isclose(Sz_arr, Sz_target, atol=1e-6)
                
            indices = np.where(mask)[0]
            
            if len(indices) == 0:
                raise ValueError(f"State {state} not found in the eigensystem.")
            if len(indices) > 1:
                raise ValueError(f"Ambiguous state definition. {len(indices)} states match {state}. "
                                 "Check for degeneracies or provide extra quantum numbers (S, Sz).")
                
            return int(indices[0])

        # Retrieve 0-based internal matrix indices
        idx_i = _get_state_idx(state_i)
        idx_f = _get_state_idx(state_f)
        
        # Execute array update
        return rate_matrix.at[idx_f, idx_i].set(value)

    def _calculate_occupation(self) -> None:
        """Calculates average occupation number N_avg."""
        self.NN_flat = self.P_flat @ self.Ne

    def _calculate_conductance(self) -> None:
        """Calculates differential conductance."""
        I_grid = self.I_flat.reshape((self.p.NVg, self.p.NVb))
        Vbs_t = jnp.asarray(self.p.Vbs_host)
        dVb = jnp.diff(Vbs_t)
        self.GG = jnp.diff(I_grid, axis=1) / dVb[None, :]

    def get_results(self) -> dict[str, np.ndarray]:
        """
        Returns simulation results as CPU NumPy arrays.

        Returns
        -------
        dict
            Dictionary containing 'P', 'N_avg', 'I', 'G', 'Vgs', 'Vbs'.
        """
        P = np.asarray(self.P_flat).reshape((self.p.NVg, self.p.NVb, self.N_states))
        NN = np.asarray(self.NN_flat).reshape((self.p.NVg, self.p.NVb))
        I = np.asarray(self.I_flat).reshape((self.p.NVg, self.p.NVb))
        GG = np.asarray(self.GG)
        
        return {'P': P, 'N_avg': NN, 'I': I, 'G': GG, 'Vgs': self.p.Vgs_host, 'Vbs': self.p.Vbs_host}


class DirectSimulator(SimulatorBase):
    """
    Simulator for phenomenological quantum dot models using direct energy inputs.

    This subclass implements the Master Equation matrix builder for systems where 
    the many-body energy levels, charge states, and tunneling rates are explicitly 
    provided by the user (via `DirectParameters`). This approach bypasses 
    microscopic Exact Diagonalization, making it ideal for fitting experimental 
    data or modeling macroscopic effective energy levels.

    It specifically handles flattening the user-defined state dictionaries into 
    the system's transition rate matrices. If `spin_labels` is enabled in the 
    parameters, it automatically scales the sequential tunneling matrix elements 
    using the predefined spin-dependent degeneracy weights (e.g., scaling 
    transitions between Singlet, Doublet, and Triplet states).

    See Also
    --------
    DirectParameters : The parameter configuration class required for this simulator.
    Simulator : The factory function recommended for instantiating this class.
    """
    
    def _build_energy_vectors(self) -> None:
        """
        Flattens the nested energy dictionaries into parallel arrays.

        Iterates sequentially through the charge states and their corresponding 
        energy levels defined in `states_dict`. Constructs the flattened 1D tensors 
        (energies, particle numbers, and spin labels) and uses them to build the 
        primary antisymmetric difference matrices (`dE`, `dNe`) required by the solver. 
        Also defines the boundary indices separating different charge sectors.

        Returns
        -------
        None
        """
        charge_sectors = sorted(self.p.states_dict.keys())
        E0_list, Ne_list, is_excited_list, labels_list = [], [], [], []
        self.sector_bounds = [0] 

        for n in charge_sectors:
            levels = self.p.states_dict[n]
            
            for i, state_data in enumerate(levels):
                E0_list.append(state_data['energy'])
                labels_list.append(state_data['label'])
                Ne_list.append(n)
                is_excited_list.append(i > 0) 
                
            self.sector_bounds.append(self.sector_bounds[-1] + len(levels))

        self.N_states = len(E0_list)
        self.E0 = jnp.array(E0_list, dtype=float)
        self.Ne = jnp.array(Ne_list, dtype=float)
        self.labels = labels_list 
        
        self.E_mat = jnp.tile(self.E0[:, None], (1, self.N_states))
        
        # Antisymmetric difference matrices
        self.dE = self.E_mat - self.E_mat.T
        self.dNe = self.Ne[:, None] - self.Ne[None, :]

    def _build_transition_matrices(self) -> None:
        """
        Constructs the sequential tunneling and relaxation rate matrices.

        Iterates over adjacent charge sectors to compute the transition rate 
        amplitudes from state `i` to state `j` based on Fermi's golden rule. 
        Integrates explicit transition matrix elements derived via spin-dependent 
        weighting rules if `spin_labels` is enabled or sets all allowed transition 
        matrix elements equal to 1 if `spin_labels` is disabled. Each matrix element
        is multiplied by the associated tunnel couplings to the leads.
        
        Finally, it integrates the pre-parsed block-diagonal relaxation matrix 
        and applies the thermodynamic Bose-Einstein distribution for intra-sector transitions.

        Returns
        -------
        None
        """
        N = self.N_states
        
        rate_L_seq = np.zeros((N, N), dtype=float)
        rate_R_seq = np.zeros((N, N), dtype=float)

        bounds = self.sector_bounds
        for s in range(len(bounds) - 2): 
            n_start, n_end = bounds[s], bounds[s+1]       
            np1_start, np1_end = bounds[s+1], bounds[s+2] 
            
            for i in range(n_start, n_end):        
                for j in range(np1_start, np1_end): 
                    if self.p.spin_labels:
                        i_label, j_label = self.labels[i], self.labels[j]
                        trans_key_fwd = f"{i_label}_to_{j_label}"
                        trans_key_rev = f"{j_label}_to_{i_label}"
                        
                        g_in = self.p.g_deg.get(trans_key_fwd, 1.0)
                        g_out = self.p.g_deg.get(trans_key_rev, 1.0)
                    else:
                        g_in, g_out = 1.0, 1.0
                        
                    # Extract the pre-mapped exact transition element
                    gamma_L = self.p.base_gL[j, i]
                    gamma_R = self.p.base_gR[j, i]

                    rate_L_seq[j, i] = g_in * gamma_L
                    rate_R_seq[j, i] = g_in * gamma_R
                    rate_L_seq[i, j] = g_out * gamma_L
                    rate_R_seq[i, j] = g_out * gamma_R

        # Relaxation rate calculation
        if isinstance(self.p.gamma_rel, np.ndarray):
            # Uses the zero-padded N x N array pre-built in DirectParameters
            base_rate_rel = self.p.gamma_rel
        else:
            # If a global scalar is provided, manually build the intra-sector block mask
            base_rate_rel = np.zeros((N, N), dtype=float)
            for s in range(len(bounds) - 1):
                start, end = bounds[s], bounds[s+1]
                base_rate_rel[start:end, start:end] = float(self.p.gamma_rel)
            
        # Convert to JAX arrays
        base_rate_rel = jnp.array(base_rate_rel)
        dE_abs = jnp.abs(self.dE)
        
        # Calculate Bose distribution using imported function
        n_b = bose(dE_abs, self.p.kB_T)
        
        # Apply thermodynamic weighting (Bose distribution) and zero the diagonal
        rate_rel = base_rate_rel.at[jnp.diag_indices_from(base_rate_rel)].set(0.0)
        self.rate_rel = jnp.tril(rate_rel, -1) * n_b + jnp.triu(rate_rel, 1) * (1.0 + n_b)

        self.rate_L_seq = jnp.array(rate_L_seq)
        self.rate_R_seq = jnp.array(rate_R_seq)


class DerivedSimulator(SimulatorBase):
    """
    Simulator for microscopic quantum dot models utilizing Exact Diagonalization (ED).

    This subclass constructs the Master Equation matrices from first-principles 
    microscopic calculations. It takes the many-body energy levels, charge 
    sectors, and eigenstate overlaps provided by the user (via 
    `DerivedParameters`) to assemble the system's transition rates.

    Unlike the phenomenological direct model, this simulator does not rely on 
    manual degeneracy weights. Instead, it calculates sequential tunneling rates 
    by directly evaluating the transition matrix elements (`M_2` matrices) between 
    the eigenvectors of adjacent charge states.

    See Also
    --------
    DerivedParameters : The parameter configuration class required for this simulator.
    Simulator : The factory function recommended for instantiating this class.
    """
    
    def _build_energy_vectors(self) -> None:
        """
        Flattens the Exact Diagonalization (ED) state dictionaries into JAX arrays.

        Iterates sequentially through the charge sectors defined in `states_dict` 
        to extract the many-body eigenenergies, charge numbers, and total spin (S) 
        eigenvalues. Constructs the flattened 1D tensors and uses them to build 
        the primary antisymmetric difference matrices (`dE`, `dNe`, and `dS`) 
        required by the solver and the relaxation transition rules. Finally, it 
        defines the boundary indices separating different charge sectors.

        Returns
        -------
        None
        """
        charge_sectors = sorted(self.p.states_dict.keys())
        
        E0_list, Ne_list, S_list = [], [], []
        self.sector_bounds = [0] 

        for n_val in charge_sectors:
            sector_data = self.p.states_dict[n_val]
            
            E_sector = [item['energy'] for item in sector_data]
            S_sector = [item['S'] for item in sector_data]
            
            E0_list.extend(E_sector)
            Ne_list.extend([n_val] * len(E_sector))
            S_list.extend(S_sector)
            self.sector_bounds.append(self.sector_bounds[-1] + len(E_sector))

        self.N_states = len(E0_list)
        self.E0 = jnp.array(E0_list, dtype=float)
        self.Ne = jnp.array(Ne_list, dtype=float)
        self.S = jnp.array(S_list, dtype=float)
        
        self.E_mat = jnp.tile(self.E0[:, None], (1, self.N_states))
        self.S_mat = jnp.tile(self.S[:, None], (1, self.N_states))
        
        self.dE = self.E_mat - self.E_mat.T
        self.dNe = self.Ne[:, None] - self.Ne[None, :]
        self.dS = jnp.abs(self.S_mat - self.S_mat.T)
        
    def _build_transition_matrices(self) -> None:
        """
        Constructs the sequential tunneling and relaxation rate matrices using ED overlaps.

        Iterates over adjacent charge sectors to compute transition rates based on 
        Fermi's golden rule. It first flattens the tunneling couplings (`gammas`), 
        accommodating either global or orbital-specific rates. For sequential tunneling, 
        it calculates the absolute squared modulus of the transition matrix elements by 
        evaluating the overlap between the many-body eigenvectors using creation operators. 

        Additionally, it builds the thermally-weighted intra-sector relaxation matrix. 
        The relaxation operator constructs the many-body overlaps and sums them coherently 
        using the provided `M_rel` amplitudes. Because this summation is coherent, 
        `M_rel` functions as the electron-phonon matrix elements. The absolute square is taken, 
        and the Bose-Einstein distribution is applied to evaluate the final relaxation rates.

        Returns
        -------
        None
        """
        num_levels = self.p.num_levels
        N = self.N_states
        rate_L_seq = np.zeros((N, N), dtype=float)
        rate_R_seq = np.zeros((N, N), dtype=float)
        rate_rel = np.zeros((N, N), dtype=float)

        bounds = self.sector_bounds
        charge_sectors = sorted(self.p.states_dict.keys())
        
        # Handle global vs. orbital-specific gammas
        if isinstance(self.p.gammas[0], (list, tuple, np.ndarray)):
            gL_array = [g[0] for g in self.p.gammas]
            gR_array = [g[1] for g in self.p.gammas]
            gamma_L = np.repeat(gL_array, 2)
            gamma_R = np.repeat(gR_array, 2)
        else:
            gamma_L = np.repeat(self.p.gammas[0], 2 * num_levels)
            gamma_R = np.repeat(self.p.gammas[1], 2 * num_levels)

        if isinstance(self.p.M_rel, np.ndarray):
            M_rel = self.p.M_rel
        else:
            M_rel = np.full((2 * num_levels, 2 * num_levels), self.p.M_rel)
            np.fill_diagonal(M_rel, 0.0)
        
        # 1. Sequential Tunneling (N -> N+1)
        for s in range(len(bounds) - 2): 
            low_start, low_end = bounds[s], bounds[s+1]
            high_start, high_end = bounds[s+1], bounds[s+2]
            
            n_val = charge_sectors[s]
            np1_val = charge_sectors[s+1]
            
            if np1_val == n_val + 1:
                basis_N = self.p.subspace_data[n_val]['basis']
                evecs_N = self.p.subspace_data[n_val]['evecs']
                basis_Np1 = self.p.subspace_data[np1_val]['basis']
                evecs_Np1 = self.p.subspace_data[np1_val]['evecs']
                
                M_2 = np.zeros((2 * num_levels, len(basis_Np1), len(basis_N)), dtype=float)
                
                for idx in range(2 * num_levels):
                    C_fock = self.p._op_to_matrix(basis_N, basis_Np1, ('c', idx))
                    T_eigen = evecs_Np1.conj().T @ C_fock @ evecs_N
                    M_2[idx] = np.abs(T_eigen)**2
                    
                rate_L_block = np.tensordot(gamma_L, M_2, axes=([0], [0]))
                rate_R_block = np.tensordot(gamma_R, M_2, axes=([0], [0]))
                
                rate_L_seq[high_start:high_end, low_start:low_end] = rate_L_block
                rate_R_seq[high_start:high_end, low_start:low_end] = rate_R_block
                rate_L_seq[low_start:low_end, high_start:high_end] = rate_L_block.T
                rate_R_seq[low_start:low_end, high_start:high_end] = rate_R_block.T

        # 2. Intra-sector Relaxation
        for s in range(len(bounds) - 1):
            start, end = bounds[s], bounds[s+1]
            n_val = charge_sectors[s]
            
            basis = self.p.subspace_data[n_val]['basis']
            evecs = self.p.subspace_data[n_val]['evecs']
            dim = len(basis)
            state_to_idx = {state: i for i, state in enumerate(basis)}
            
            T_rel_eigen_total = np.zeros((dim, dim), dtype=complex)
            
            for a in range(2 * num_levels):
                for b in range(2 * num_levels):
                    M_val = M_rel[a, b] 
                    
                    if M_val != 0.0:
                        scatter_fock = np.zeros((dim, dim), dtype=float)
                        
                        # Apply c^\dagger_a c_b (destroys b, then creates a)
                        ops = [('d', b), ('c', a)]
                        for i, state in enumerate(basis):
                            new_state, sign = self.p._apply_ops(state, ops)
                            if new_state:
                                scatter_fock[state_to_idx[new_state], i] = sign
                                
                        scatter_eigen = evecs.conj().T @ scatter_fock @ evecs                             
                        T_rel_eigen_total += M_val * scatter_eigen
                        
            rate_rel_block = np.abs(T_rel_eigen_total)**2
            rate_rel[start:end, start:end] = rate_rel_block

        # Convert to JAX arrays
        rate_rel = jnp.array(rate_rel)
        dE_abs = jnp.abs(self.dE)
        
        # Calculate Bose distribution using imported function
        n_b = bose(dE_abs, self.p.kB_T)
        
        # Apply thermodynamic weighting (Bose distribution) and zero the diagonal
        rate_rel = rate_rel.at[jnp.diag_indices_from(rate_rel)].set(0.0)
        self.rate_rel = jnp.tril(rate_rel, -1) * n_b + jnp.triu(rate_rel, 1) * (1.0 + n_b)

        self.rate_L_seq = jnp.array(rate_L_seq)
        self.rate_R_seq = jnp.array(rate_R_seq)


def Simulator(sys_params: BaseParameters) -> SimulatorBase:
    """
    Factory function assigning the required simulator subclass.

    Parameters
    ----------
    sys_params : BaseParameters
        System parameter object.

    Returns
    -------
    SimulatorBase
        Instance of DirectSimulator or DerivedSimulator.

    Raises
    ------
    TypeError
        If `sys_params` is not of a supported type.
    """
    if isinstance(sys_params, DirectParameters):
        return DirectSimulator(sys_params)
    elif isinstance(sys_params, DerivedParameters):
        return DerivedSimulator(sys_params)
    else:
        raise TypeError("sys_params must be an instance of DirectParameters or DerivedParameters")