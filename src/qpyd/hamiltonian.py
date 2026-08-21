import itertools
import numpy as np
import sympy as sp
from .datatypes import StateProperties


class ExactDiagonalization:
    """
    Core physics engine for constructing and diagonalizing many-body quantum dot Hamiltonians.
    """
    def __init__(self, num_levels, orbital_energies, U, V, J, Jp, B, g_factor, charge_numbers):
        self.num_levels = num_levels
        self.orbital_energies = orbital_energies
        self.U = U
        self.V = V
        self.J = J
        self.Jp = Jp
        self.B = B
        self.g_factor = g_factor
        self.charge_numbers = charge_numbers
        self.muB = 0.05788  # meV/T

    def _generate_fock_basis(self, N: int) -> list[tuple[int, ...]]:
        """
        Generates the sorted basis configurations for N particles.
        
        Even indices represent spin-up, odd indices represent spin-down.

        Parameters
        ----------
        N : int
            Number of particles in the subspace.

        Returns
        -------
        list of tuple of int
            List of tuples, where each tuple contains the occupied spin-orbital indices.
        """
        return list(itertools.combinations(range(2 * self.num_levels), N))
    
    def _generate_fock_kets(self, basis: list[tuple[int, ...]]) -> dict[sp.Symbol, np.ndarray]:
        """
        Maps a basis of states to their occupation vector representations.

        Parameters
        ----------
        basis : list of tuple of int
            List of basis states (tuples of occupied spin-orbital indices).

        Returns
        -------
        dict of sp.Symbol to np.ndarray
            Dictionary mapping a SymPy Dirac ket symbol to its physical 
            occupation NumPy array (e.g., [1, 0, 1, 0]).
        """
        basis_dict = {}
        total_slots = 2 * self.num_levels

        for state in basis:
            # 1. Initialize empty orbital array
            basis_ket = np.zeros(total_slots, dtype=int)
            
            # 2. Directly set occupied indices to 1 (vectorized)
            if state:  # Prevents error on N=0 (empty tuple)
                basis_ket[list(state)] = 1
                
            # 3. Convert array directly to string for the SymPy symbol
            state_str = ", ".join(basis_ket.astype(str))
            basis_ket_sym = sp.Symbol(f"|{state_str})", commutative=False)
            
            basis_dict[basis_ket_sym] = basis_ket
                                     
        return basis_dict

    def _op_to_matrix(
        self, 
        basis_initial: list[tuple[int, ...]], 
        basis_final: list[tuple[int, ...]], 
        operator: tuple[str, int]
    ) -> np.ndarray:
        """
        Constructs the matrix representation of a single creation or annihilation operator.

        Because these operators change the particle number, this matrix maps between 
        two different charge subspaces (e.g., from N to N-1).

        Parameters
        ----------
        basis_initial : list of tuple of int
            The basis states of the initial subspace (columns).
        basis_final : list of tuple of int
            The basis states of the final subspace (rows).
        operator : tuple of (str, int)
            Tuple of ('c', idx) for creation, or ('d', idx) for destruction/annihilation.

        Returns
        -------
        np.ndarray
            A rectangular array of shape (len(basis_final), len(basis_initial)).
        """
        dim_i = len(basis_initial)
        dim_f = len(basis_final)
        op_matrix = np.zeros((dim_f, dim_i), dtype=float)
        
        state_to_idx_f = {state: i for i, state in enumerate(basis_final)}
        
        for i, state in enumerate(basis_initial):
            new_state, sign = self._apply_ops(state, [operator])
            
            if new_state is not None and new_state in state_to_idx_f:
                f_idx = state_to_idx_f[new_state]
                op_matrix[f_idx, i] = sign
                
        return op_matrix
    
    def _apply_ops(self, state: tuple[int, ...], operators: list[tuple[str, int]]) -> tuple[tuple[int, ...] | None, int]:
        """
        Applies a sequence of creation/annihilation operators and tracks fermionic signs.

        Parameters
        ----------
        state : tuple of int
            Tuple representing the current occupied spin-orbitals.
        operators : list of tuple of (str, int)
            List of tuples `(action, index)`, where action is 'c' (create) or 'd' (destroy).
            Must be provided in the order they are applied (right to left mathematically).

        Returns
        -------
        tuple of (tuple of int or None, int)
            The resulting state tuple (or None if annihilated), and the 
            accumulated fermionic sign (+1 or -1).
        """
        s = list(state)
        sign = 1
        for action, idx in operators:
            count = sum(1 for x in s if x < idx)
            sign *= (-1)**count
            
            if action == 'd':
                if idx not in s: return None, 0
                s.remove(idx)
            elif action == 'c':
                if idx in s: return None, 0
                s.append(idx)
                s.sort()
                
        return tuple(s), sign
    
    def _sym_op(self, op_type: str, orb: int, spin: str) -> sp.Symbol:
        """
        Helper to generate non-commutative SymPy operator symbols.
        
        Parameters
        ----------
        op_type : str
            The operator type: 'n' (density), 'cdag' (creation), or 'c' (annihilation).
        orb : int
            The spatial orbital index.
        spin : str
            The spin projection: 'up' or 'down'.
            
        Returns
        -------
        sp.Symbol
            A non-commutative SymPy symbol.
        """
        if spin == 'up':
            tex_spin = r"\uparrow"
        elif spin == 'down':
            tex_spin = r"\downarrow"
        else:
            tex_spin = spin
            
        if op_type == 'cdag':
            name = f"c^\\dagger_{{{orb}, {tex_spin}}}"
        else:
            name = f"{op_type}_{{{orb}, {tex_spin}}}"
            
        return sp.Symbol(name, commutative=False)

    def _single_particle_hamiltonian(self, basis: list[tuple[int, ...]]) -> tuple[np.ndarray, sp.Expr]:
        """
        Calculates the tight-binding (orbital energy) block.

        Parameters
        ----------
        basis : list of tuple of int
            The many-body basis for the subspace.

        Returns
        -------
        tuple of (np.ndarray, sp.Expr)
            The numerical matrix block and the symbolic operator expression.
        """
        dim = len(basis)
        H = np.zeros((dim, dim), dtype=float)
        sym_H = 0
        
        for i, state in enumerate(basis):
            for orb_spin in state:
                orb = orb_spin // 2
                H[i, i] += self.orbital_energies[orb]
                
        for a in range(self.num_levels):
            n_up = self._sym_op('n', a, 'up')
            n_down = self._sym_op('n', a, 'down')
            sym_H += self.orbital_energies[a] * (n_up + n_down)
            
        return H, sym_H

    def _direct_int_hamiltonian(self, basis: list[tuple[int, ...]]) -> tuple[np.ndarray, sp.Expr]:
        """
        Calculates Macroscopic Charging, Intra-orbital (U), and Inter-orbital (V).

        Parameters
        ----------
        basis : list of tuple of int
            The many-body basis for the subspace.

        Returns
        -------
        tuple of (np.ndarray, sp.Expr)
            The numerical matrix block and the symbolic operator expression.
        """
        dim = len(basis)
        H = np.zeros((dim, dim), dtype=float)
        sym_H = 0
 
        for i, state in enumerate(basis):
            for orb in range(self.num_levels):
                if (2*orb) in state and (2*orb + 1) in state:
                    H[i, i] += self.U[orb]

            for a in range(self.num_levels):
                for b in range(a + 1, self.num_levels):
                    n_a = (1 if 2*a in state else 0) + (1 if 2*a+1 in state else 0)
                    n_b = (1 if 2*b in state else 0) + (1 if 2*b+1 in state else 0)
                    H[i, i] += self.V[a, b] * (n_a * n_b)

        for a in range(self.num_levels):
            n_aup = self._sym_op('n', a, 'up')
            n_adown = self._sym_op('n', a, 'down')
            sym_H += self.U[a] * n_aup * n_adown
            
            for b in range(a + 1, self.num_levels):
                n_bup = self._sym_op('n', b, 'up')
                n_bdown = self._sym_op('n', b, 'down')
                sym_H += self.V[a, b] * (n_aup + n_adown) * (n_bup + n_bdown)
                
        return H, sym_H

    def _exchange_int_hamiltonian(self, basis: list[tuple[int, ...]], state_to_idx: dict) -> tuple[np.ndarray, sp.Expr]:
        """
        Calculates the Heisenberg Exchange interactions (Ising + Spin-flip).

        Parameters
        ----------
        basis : list of tuple of int
            The many-body basis for the subspace.
        state_to_idx : dict
            Mapping of state tuples to their matrix index.

        Returns
        -------
        tuple of (np.ndarray, sp.Expr)
            The numerical matrix block and the symbolic operator expression.
        """
        dim = len(basis)
        H = np.zeros((dim, dim), dtype=float)
        sym_H = 0
        
        for i, state in enumerate(basis):
            for a in range(self.num_levels):
                for b in range(a + 1, self.num_levels):
                    Sz_a = 0.5 * ((1 if 2*a in state else 0) - (1 if 2*a+1 in state else 0))
                    Sz_b = 0.5 * ((1 if 2*b in state else 0) - (1 if 2*b+1 in state else 0))
                    H[i, i] -= 2 * self.J[a, b] * Sz_a * Sz_b

                    ops_1 = [('d', 2*b), ('c', 2*b+1), ('d', 2*a+1), ('c', 2*a)]
                    new_state, sign = self._apply_ops(state, ops_1)
                    if new_state: H[state_to_idx[new_state], i] -= self.J[a, b] * sign 

                    ops_2 = [('d', 2*b+1), ('c', 2*b), ('d', 2*a), ('c', 2*a+1)]
                    new_state, sign = self._apply_ops(state, ops_2)
                    if new_state: H[state_to_idx[new_state], i] -= self.J[a, b] * sign

        for a in range(self.num_levels):
            for b in range(a + 1, self.num_levels):
                n_aup, n_adown = self._sym_op('n', a, 'up'), self._sym_op('n', a, 'down')
                n_bup, n_bdown = self._sym_op('n', b, 'up'), self._sym_op('n', b, 'down')
                Sz_a, Sz_b = (n_aup - n_adown) / 2, (n_bup - n_bdown) / 2
                
                cdag_aup, c_aup = self._sym_op('cdag', a, 'up'), self._sym_op('c', a, 'up')
                cdag_adown, c_adown = self._sym_op('cdag', a, 'down'), self._sym_op('c', a, 'down')
                cdag_bup, c_bup = self._sym_op('cdag', b, 'up'), self._sym_op('c', b, 'up')
                cdag_bdown, c_bdown = self._sym_op('cdag', b, 'down'), self._sym_op('c', b, 'down')
                
                sym_H -= 2 * self.J[a, b] * Sz_a * Sz_b
                sym_H -= self.J[a, b] * (cdag_aup * c_adown * cdag_bdown * c_bup)
                sym_H -= self.J[a, b] * (cdag_adown * c_aup * cdag_bup * c_bdown)
                
        return H, sym_H

    def _pair_hopping_hamiltonian(self, basis: list[tuple[int, ...]], state_to_idx: dict) -> tuple[np.ndarray, sp.Expr]:
        """
        Calculates the pair hopping interactions.

        Parameters
        ----------
        basis : list of tuple of int
            The many-body basis for the subspace.
        state_to_idx : dict
            Mapping of state tuples to their matrix index.

        Returns
        -------
        tuple of (np.ndarray, sp.Expr)
            The numerical matrix block and the symbolic operator expression.
        """
        dim = len(basis)
        H = np.zeros((dim, dim), dtype=float)
        sym_H = 0
        
        for i, state in enumerate(basis):
            for a in range(self.num_levels):
                for b in range(a + 1, self.num_levels):
                    ops_1 = [('d', 2*b), ('d', 2*b+1), ('c', 2*a+1), ('c', 2*a)]
                    new_state, sign = self._apply_ops(state, ops_1)
                    if new_state: H[state_to_idx[new_state], i] += self.Jp[a, b] * sign

                    ops_2 = [('d', 2*a), ('d', 2*a+1), ('c', 2*b+1), ('c', 2*b)]
                    new_state, sign = self._apply_ops(state, ops_2)
                    if new_state: H[state_to_idx[new_state], i] += self.Jp[a, b] * sign

        for a in range(self.num_levels):
            for b in range(a + 1, self.num_levels):
                cdag_aup, c_aup = self._sym_op('cdag', a, 'up'), self._sym_op('c', a, 'up')
                cdag_adown, c_adown = self._sym_op('cdag', a, 'down'), self._sym_op('c', a, 'down')
                cdag_bup, c_bup = self._sym_op('cdag', b, 'up'), self._sym_op('c', b, 'up')
                cdag_bdown, c_bdown = self._sym_op('cdag', b, 'down'), self._sym_op('c', b, 'down')
                
                sym_H += self.Jp[a, b] * (cdag_aup * cdag_adown * c_bdown * c_bup)
                sym_H += self.Jp[a, b] * (cdag_bup * cdag_bdown * c_adown * c_aup)
                
        return H, sym_H

    def _zeeman_hamiltonian(self, basis: list[tuple[int, ...]], state_to_idx: dict) -> tuple[np.ndarray, sp.Expr]:
        """
        Calculates the Zeeman splitting block using creation and annihilation operators,
        supporting arbitrary 3D magnetic field orientations.

        Parameters
        ----------
        basis : list of tuple of int
            The many-body basis for the subspace.
        state_to_idx : dict
            Mapping of state tuples to their matrix index.

        Returns
        -------
        tuple of (np.ndarray, sp.Expr)
            The numerical matrix block and the symbolic operator expression.
        """
        dim = len(basis)
        Bx, By, Bz = self.B[0], self.B[1], self.B[2]

        # If By is non-zero, the Pauli-Y matrix introduces imaginary components
        is_complex = abs(By) > 1e-12
        H = np.zeros((dim, dim), dtype=complex if is_complex else float)
        sym_H = 0
        
        prefactor = 0.5 * self.g_factor * self.muB
        
        for i, state in enumerate(basis):
            for a in range(self.num_levels):
                
                # Z-component: B_z * sigma_z (diagonal spin polarization)
                if Bz != 0.0:
                    ops_up = [('d', 2 * a), ('c', 2 * a)]
                    new_state_up, sign_up = self._apply_ops(state, ops_up)
                    if new_state_up: 
                        H[state_to_idx[new_state_up], i] += prefactor * Bz * sign_up
                        
                    ops_down = [('d', 2 * a + 1), ('c', 2 * a + 1)]
                    new_state_down, sign_down = self._apply_ops(state, ops_down)
                    if new_state_down: 
                        H[state_to_idx[new_state_down], i] -= prefactor * Bz * sign_down

                # X and Y components: Spin-flip terms via Pauli-X and Pauli-Y
                if Bx != 0.0 or By != 0.0:
                    # c^\dagger_{up} c_{down}  -> destroys down (2a+1), creates up (2a)
                    ops_flip_up = [('d', 2 * a + 1), ('c', 2 * a)]
                    new_state_flip_up, sign_flip_up = self._apply_ops(state, ops_flip_up)
                    if new_state_flip_up: 
                        H[state_to_idx[new_state_flip_up], i] += prefactor * (Bx - 1j * By) * sign_flip_up

                    # c^\dagger_{down} c_{up} -> destroys up (2a), creates down (2a+1)
                    ops_flip_down = [('d', 2 * a), ('c', 2 * a + 1)]
                    new_state_flip_down, sign_flip_down = self._apply_ops(state, ops_flip_down)
                    if new_state_flip_down:
                        H[state_to_idx[new_state_flip_down], i] += prefactor * (Bx + 1j * By) * sign_flip_down

        # Construct the symbolic representation
        for a in range(self.num_levels):
            n_aup = self._sym_op('n', a, 'up')
            n_adown = self._sym_op('n', a, 'down')
            cdag_aup = self._sym_op('cdag', a, 'up')
            c_aup = self._sym_op('c', a, 'up')
            cdag_adown = self._sym_op('cdag', a, 'down')
            c_adown = self._sym_op('c', a, 'down')
            
            # Z-axis (sigma_z)
            sym_H += sp.Rational(1, 2) * self.g_factor * self.muB * self.B[2] * (n_aup - n_adown)
            # X-axis (sigma_x)
            sym_H += sp.Rational(1, 2) * self.g_factor * self.muB * self.B[0] * (cdag_aup * c_adown + cdag_adown * c_aup)
            # Y-axis (sigma_y)
            sym_H += sp.Rational(1, 2) * self.g_factor * self.muB * self.B[1] * (-sp.I * cdag_aup * c_adown + sp.I * cdag_adown * c_aup)
                    
        return H, sym_H

    def construct_hamiltonian(self, N: int) -> tuple[np.ndarray, list[tuple[int, ...]], sp.Expr]:
        """
        Builds the full Hamiltonian and its symbolic representation.

        Parameters
        ----------
        N : int
            Number of particles in the subspace.

        Returns
        -------
        tuple of (np.ndarray, list of tuple, sp.Expr)
            The dense numerical Hamiltonian matrix, the list of basis states, 
            and the fully constructed symbolic operator expression.
        """
        basis = self._generate_fock_basis(N)
        state_to_idx = {state: i for i, state in enumerate(basis)}
        
        H_tb, sym_tb = self._single_particle_hamiltonian(basis)
        H_dir, sym_dir = self._direct_int_hamiltonian(basis)
        H_exch, sym_exch = self._exchange_int_hamiltonian(basis, state_to_idx)
        H_pair, sym_pair = self._pair_hopping_hamiltonian(basis, state_to_idx)
        H_z, sym_z = self._zeeman_hamiltonian(basis, state_to_idx)

        H = H_tb + H_dir + H_exch + H_pair + H_z
        sym_H = sym_tb + sym_dir + sym_exch + sym_pair + sym_z
             
        return H, basis, sym_H
    
    def _S2_matrix(self, basis: list[tuple[int, ...]], state_to_idx: dict) -> np.ndarray:
        """
        Constructs the total spin squared matrix S^2 for the given basis.

        Parameters
        ----------
        basis : list of tuple of int
            The many-body basis for the subspace.
        state_to_idx : dict
            Mapping of state tuples to their matrix index.

        Returns
        -------
        np.ndarray
            The S^2 operator matrix in the given basis.
        """
        dim = len(basis)
        S2 = np.zeros((dim, dim), dtype=float)
        
        for i, state in enumerate(basis):
            m_i = sum(0.5 if x % 2 == 0 else -0.5 for x in state)
            S2[i, i] += m_i**2 - m_i
            
            for a in range(self.num_levels):
                for b in range(self.num_levels):
                    ops = [('d', 2*b), ('c', 2*b+1), ('d', 2*a+1), ('c', 2*a)]
                    new_state, sign = self._apply_ops(state, ops)
                    if new_state:
                        S2[state_to_idx[new_state], i] += 1.0 * sign
                        
        return S2   

    def _diagonalize_sector(self, n: int) -> tuple[list[tuple[int, ...]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[sp.Expr]] | None:
        """
        Constructs operators and diagonalizes the Hamiltonian for a single charge sector.

        Parameters
        ----------
        n : int
            Number of particles in the charge subspace.

        Returns
        -------
        tuple or None
            A tuple containing `(evals, evecs, Sz_basis, S2_mat, basis_symbols, basis_vecs)`. 
            Returns `None` if the basis for the given charge sector is empty.
            - evals : np.ndarray (1D array of eigenvalues)
            - evecs : np.ndarray (2D array of eigenvectors)
            - Sz_basis : np.ndarray (1D array of diagonal Sz values)
            - S2_mat : np.ndarray (2D array of the S^2 operator)
            - basis_symbols : list of sp.Expr (SymPy quantum kets)
            - basis_vecs : list of np.ndarray (NumPy arrays of orbital occupations)
        """
        H, basis, sym_H = self.construct_hamiltonian(n)
        
        # Override the global property with the latest valid symbolic Hamiltonian
        self.symbolic_hamiltonian = sym_H 
        
        if len(basis) == 0:
            return None

        state_to_idx = {state: i for i, state in enumerate(basis)}
        S2_mat = self._S2_matrix(basis, state_to_idx)
        basis_symbols = list(self._generate_fock_kets(basis).keys())
        basis_vecs = list(self._generate_fock_kets(basis).values())
        
        # 1. Calculate diagonal Sz values for the Fock basis
        Sz_basis = np.array([sum(0.5 if x % 2 == 0 else -0.5 for x in state) for state in basis])
        
        # 2. Build the full Sz matrix (strictly diagonal in the Fock basis)
        Sz_mat = np.diag(Sz_basis)
        
        # 3. Call the custom simultaneous diagonalization solver
        evals, evecs = self._resolve_degeneracies(H, Sz_mat)
        
        # Store raw matrices for external use (e.g., transition matrix builders)
        self.subspace_data[n] = {
            'basis': basis,
            'evecs': evecs
        }
        
        return evals, evecs, Sz_basis, S2_mat, basis_symbols, basis_vecs

    def _extract_state_properties(
        self, 
        n: int, 
        energy: float, 
        evec: np.ndarray, 
        basis_symbols: list[sp.Expr], 
        basis_vecs: list[np.ndarray],
        Sz_basis: np.ndarray, 
        S2_mat: np.ndarray
    ) -> StateProperties:
        """
        Calculates physical observables and formats the properties of a single eigenstate.

        Parameters
        ----------
        n : int
            Number of particles in the charge subspace.
        energy : float
            The energy eigenvalue of the state.
        evec : np.ndarray
            The eigenvector array (probability amplitudes) for this state.
        basis_symbols : list of sp.Expr
            List of SymPy quantum kets representing the physical Fock basis.
        basis_vecs : list of np.ndarray
            List of NumPy arrays representing the orbital occupations.
        Sz_basis : np.ndarray
            Diagonal total spin projection (S_z) values for the Fock basis.
        S2_mat : np.ndarray
            The total spin squared (S^2) operator matrix in the given basis.

        Returns
        -------
        StateProperties
            Dictionary containing the physical properties of the eigenstate:
            - "eigenstate": A rigidly ordered string representation of the state.
            - "single_particle_kets": A list of tuples `(amplitude, ket_string, ket_array)` 
              representing the constituent many-body states expressed in the single-particle basis.
            - "energy": The eigenenergy of the state, rounded.
            - "S": The total spin quantum number.
            - "Sz": The total spin projection quantum number.
        """
        probs = np.abs(evec)**2

        # Calculate expectation values
        sz = np.sum(probs * Sz_basis)
        s2_exp = np.real(evec.conj().T @ S2_mat @ evec)
        total_s = 0.5 * (-1 + np.sqrt(max(0.0, 1.0 + 4.0 * s2_exp)))

        # Build strings and lists manually to preserve exact basis order
        rhs_terms = []
        eigenstate_vec_fock = []
        
        for i in range(len(evec)):
            coeff = np.round(evec[i], 3)
            
            if abs(coeff) > 1e-3:
                # Simplify floats to integers if they are perfectly whole numbers
                if np.isreal(coeff) and float(np.real(coeff)).is_integer():
                    coeff = int(np.real(coeff))
                
                ket_str = str(basis_symbols[i])
                
                # Bundle the coefficient, readable string, and raw numpy array
                eigenstate_vec_fock.append((coeff, ket_str, basis_vecs[i]))
                
                # Format the mathematical string cleanly
                if coeff == 1:
                    rhs_terms.append(f"{ket_str}")
                elif coeff == -1:
                    rhs_terms.append(f"-{ket_str}")
                elif np.iscomplex(coeff):
                    rhs_terms.append(f"({coeff})*{ket_str}")
                else:
                    rhs_terms.append(f"{coeff}*{ket_str}")

        # Join the terms and clean up "+ -" into just "- "
        sym_eigenstate_fock = " + ".join(rhs_terms).replace(" + -", " - ")
        sym_eigenstate_lhs = f"|{n}, {round(total_s, 1)}, {round(sz, 1)}>"

        return {
            "eigenstate": f"{sym_eigenstate_lhs} = {sym_eigenstate_fock}",
            "single_particle_kets": eigenstate_vec_fock,
            "energy": round(energy, 5), 
            "S": round(total_s, 2),
            "Sz": round(sz, 2)
        }

    def compute_eigenspectrum(self) -> tuple[dict[int, list[StateProperties]], dict, sp.Expr | None]:
        """
        Uses Exact Diagonalization and state extraction across all charge sectors.

        This method acts as the primary physics engine during initialization. It iterates 
        over all valid charge numbers and delegates the construction of the many-body 
        Hamiltonian (tight-binding, Coulomb, exchange, pair-hopping, and Zeeman terms). 
        It then diagonalizes the Hamiltonian to resolve the eigenstates, calculates 
        expectation values for S^2 and Sz, and maps the resulting eigenvectors back 
        to their symbolic Fock basis representations.

        Returns
        -------
        tuple
            A tuple containing `(results, subspace_data, symbolic_hamiltonian)`.
            `results` is a dictionary keyed by the charge number `n`, containing a list of dictionaries 
            for every physical eigenstate. Each state dictionary contains its string 
            representation, exact energy, total spin (S), spin projection (Sz), and 
            raw single-particle Fock vector amplitudes.
        """
        results = {}
        total_slots = 2 * self.num_levels  
        self.symbolic_hamiltonian = None 
        self.subspace_data = {}

        for n in self.charge_numbers:
            if n > total_slots or n < 0:
                results[n] = []
                continue

            # Step 1: Solve the physics for this charge sector
            sector_data = self._diagonalize_sector(n)
            
            if sector_data is None:
                results[n] = []
                continue
                
            evals, evecs, Sz_basis, S2_mat, basis_symbols, basis_vecs = sector_data

            # Step 2: Extract observables and format each eigenstate
            configs_for_n = []
            for k in range(len(evals)):
                state_dict = self._extract_state_properties(
                    n=n,
                    energy=evals[k],
                    evec=evecs[:, k],
                    basis_symbols=basis_symbols,
                    basis_vecs=basis_vecs,
                    Sz_basis=Sz_basis,
                    S2_mat=S2_mat
                )
                configs_for_n.append(state_dict)

            # Step 3: Sort by energy (ground state first) and store
            configs_for_n.sort(key=lambda x: x['energy'])
            results[n] = configs_for_n

        return results, self.subspace_data, self.symbolic_hamiltonian
    
    def _resolve_degeneracies(self, H: np.ndarray, Sz_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Simultaneously diagonalizes the Hamiltonian and the Sz spin operator.

        Detects degenerate energy subspaces and mathematically rotates their 
        arbitrary eigenvectors to align perfectly with the physical spin axes.

        Parameters
        ----------
        H : np.ndarray
            The numerical Hamiltonian matrix for the subspace.
        Sz_mat : np.ndarray
            The total spin projection matrix (S_z) in the Fock basis.

        Returns
        -------
        tuple of np.ndarray
            A tuple containing `(evals, evecs_clean)`, where `evals` are 
            the sorted energy eigenvalues and `evecs_clean` are the 
            physically oriented eigenvectors.
        """
        evals, evecs = np.linalg.eigh(H)
        evecs_clean = np.zeros_like(evecs)
        
        tolerance = 1e-8
        i = 0
        
        while i < len(evals):
            j = i + 1
            # Scan forward to find the bounds of the degenerate subspace
            while j < len(evals) and abs(evals[j] - evals[i]) < tolerance:
                j += 1
                
            if j - i == 1:
                # Non-degenerate: No rotation needed
                evecs_clean[:, i] = evecs[:, i]
            else:
                # Degenerate: Isolate subspace and rotate using Sz
                V_sub = evecs[:, i:j]
                
                # Project Sz into the arbitrary subspace
                Sz_sub = V_sub.conj().T @ Sz_mat @ V_sub
                
                # Diagonalize the subspace Sz to find the clean rotation weights
                _, W = np.linalg.eigh(Sz_sub)
                
                # Apply the rotation to map back to the full physical basis
                evecs_clean[:, i:j] = V_sub @ W
                
            i = j
            
        return evals, evecs_clean