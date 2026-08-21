import os
import itertools
import numpy as np
import sympy as sp
import jax
import jax.numpy as jnp
from .datatypes import StateProperties
from .hamiltonian import ExactDiagonalization


class BaseParameters:
    """
    Base container class for shared system parameters, grids, and device settings.
    
    This class handles basic constants, grid initializations, and lever arm 
    calculations. Do not instantiate this directly. Use `DirectParameters` 
    or `DerivedParameters`.
    """
    
    def __init__(
        self, 
        T: float, 
        slopes: list[float] | None = None, 
        lever_arms: list[float] | None = None,
        gammas: list[float] | list[list[float | list[list[float]] | np.ndarray]] = [1.0, 1.0], 
        gamma_rel: float | list[float | list[list[float]] | np.ndarray] = 0.0,
        Vb_range: tuple[float, float, int] | None = None, 
        Vg_range: tuple[float, float, int] | None = None, 
        V_ground: float = 0.0,
        setup: str = "symmetric", 
        datapoints: int = 200, 
        dtype: np.dtype = jnp.float32
    ):
        """
        Initializes the base shared parameters.

        Parameters
        ----------
        T : float
            Temperature of the leads.
        slopes : list of float or None, optional
            Coulomb diamond slopes [sL, sR] for the source (left) and drain (right).
            Must provide either `slopes` or `lever_arms`.
        lever_arms : list of float or None, optional
            Direct electrostatic couplings [a_L, a_R, a_G]. If provided, bypasses 
            the calculation from slopes. Must provide either `slopes` or `lever_arms`.
        gammas : list of float or list of list, optional
            Tunneling couplings to leads. Handled explicitly by the subclasses.
        gamma_rel : float or list, optional
            Relaxation rate for same-charge states. Default is 0.0.
        Vb_range : tuple of (float, float, int) or None, optional
            Bias voltage range as (min, max, points). Default is None.
        Vg_range : tuple of (float, float, int) or None, optional
            Gate voltage range as (min, max, points). Default is None.
        V_ground : float, optional
            Ground potential offset. Default is 0.0.
        setup : str, optional
            Voltage setup configuration ('symmetric', 'ground_L', 'ground_R'). 
            Default is "symmetric".
        datapoints : int, optional
            Number of data points across each grid dimension if auto-calculating. 
            Default is 200.
        dtype : np.dtype, optional
            Data type for tensors. Default is jnp.float32.

        Raises
        ------
        ValueError
            If neither `slopes` nor `lever_arms` is provided.
        """
        if slopes is None and lever_arms is None:
            raise ValueError("You must provide either 'slopes' or 'lever_arms'.")

        # Fundamental physical constants
        self.e: float = 1.602e-10       # nC (NanoCoulombs)
        self.hbar: float = 6.582e-13    # meV·s
        self.muB: float = 0.05788       # meV/T
        self.kB: float = 0.08617        # meV/K
        
        # User inputs
        self.T: float = T
        
        # Pre-calculated State Variables
        self.kB_T: float = self.kB * self.T

        # Shared System Config
        self.slopes = slopes
        self.lever_arms = lever_arms
        self.gammas = gammas
        self.gamma_rel = gamma_rel
        self.setup = setup
        self.dtype = dtype
        
        # Grab the default accelerator.
        self.device = jax.devices()[0]
        self.V_ground = V_ground
        self.datapoints = datapoints

        # Placeholders
        self.aG = self.aL = self.aR = None
        self.Vbmin = self.Vbmax = self.NVb = None
        self.Vgmin = self.Vgmax = self.NVg = None
        self.charge_numbers: list[int] = []
        self.states_dict: dict = {}
        self.U: list[float | int] | np.ndarray = None

    def _finalize_setup(
        self, 
        Vb_range: tuple[float, float, int] | None, 
        Vg_range: tuple[float, float, int] | None, 
        setup: str
    ) -> None:
        """
        Calculates lever arms, auto ranges, and builds the vectorized grids.

        Parameters
        ----------
        Vb_range : tuple of (float, float, int) or None
            Provided Bias voltage range (min, max, points) or None.
        Vg_range : tuple of (float, float, int) or None
            Provided Gate voltage range (min, max, points) or None.
        setup : str
            The voltage setup configuration.

        Raises
        ------
        ValueError
            If an unknown voltage setup configuration or malformed list is provided.
        """
        if self.lever_arms is not None:
            if len(self.lever_arms) != 3:
                raise ValueError("lever_arms must contain exactly three elements: [a_L, a_R, a_G]")
            self.aL, self.aR, self.aG = self.lever_arms
            
        elif self.slopes is not None:
            if len(self.slopes) != 2:
                raise ValueError("slopes must contain exactly two elements: [s_L, s_R]")
                
            s_pos = abs(self.slopes[0])
            s_neg = abs(self.slopes[1])
            self.aG = 1.0 / (1.0/s_pos + 1.0/s_neg)
            
            if round(1/s_pos + 1/s_neg - 1/self.aG, 2) != 0.0:
                print("Warning: The expression 1/s1+1/s2=1/aG does not hold. Adjust parameters.")
                
            if self.setup == "symmetric":
                A = np.array([[-1, 1], [1, 1]])
                b = np.array([(2 * self.aG / s_pos) - 1, 1.0 - self.aG])
            elif self.setup == "ground_R":
                A = np.array([[1, 0], [1, 1]])
                b = np.array([1.0 - (self.aG / s_pos), 1.0 - self.aG])
            elif self.setup == "ground_L":
                A = np.array([[0, 1], [1, 1]])
                b = np.array([1.0 - (self.aG / s_pos), 1.0 - self.aG])
            else:
                raise ValueError(f"Unknown setup '{setup}'. Choose 'symmetric', 'ground_R', or 'ground_L'.")
            
            try:
                aLR = np.linalg.solve(A, b)
                self.aL, self.aR = aLR[0], aLR[1]
            except np.linalg.LinAlgError:
                print("Error: Singular matrix in lever arm calculation.")
                self.aL, self.aR = 0.33, 0.33 

        auto_Vb, auto_Vg = self._calculate_auto_ranges()
        
        if Vb_range is not None:
            self.Vbmin, self.Vbmax, self.NVb = Vb_range
        else:
            self.Vbmin, self.Vbmax, self.NVb = auto_Vb

        if Vg_range is not None:
            self.Vgmin, self.Vgmax, self.NVg = Vg_range
        else:
            self.Vgmin, self.Vgmax, self.NVg = auto_Vg

        self._build_grids(setup)

    def _build_grids(self, setup: str) -> None:
        """
        Generates JAX-native meshgrids for calculation.

        Parameters
        ----------
        setup : str
            The voltage setup configuration determining the local potentials.
        """
        self.Vbs_host = np.linspace(self.Vbmin, self.Vbmax, self.NVb)
        self.Vgs_host = np.linspace(self.Vgmin, self.Vgmax, self.NVg)
        
        Vg_grid, Vb_grid = np.meshgrid(self.Vgs_host, self.Vbs_host, indexing='xy')
        
        Vg_flat_np = Vg_grid.T.ravel()
        Vb_flat_np = Vb_grid.T.ravel()
        self.M_points = Vg_flat_np.size 

        # Map grids to JAX device arrays
        self.Vg_flat = jnp.array(Vg_flat_np, dtype=self.dtype)
        self.Vb_flat = jnp.array(Vb_flat_np, dtype=self.dtype)
        V_ground_tensor = jnp.full_like(self.Vb_flat, self.V_ground)

        if self.setup == "symmetric":
            self.VL_flat = V_ground_tensor + 0.5 * self.Vb_flat
            self.VR_flat = V_ground_tensor - 0.5 * self.Vb_flat
        elif self.setup == "ground_L":
            self.VL_flat = V_ground_tensor
            self.VR_flat = V_ground_tensor - self.Vb_flat
        elif self.setup == "ground_R":
            self.VL_flat = V_ground_tensor + self.Vb_flat
            self.VR_flat = V_ground_tensor

        self.muL_flat = -self.VL_flat
        self.muR_flat = -self.VR_flat
        self.Vext_flat = (self.aG * self.Vg_flat + self.aL * self.VL_flat + self.aR * self.VR_flat)

    def report_memory(self) -> None:
        """Prints current JAX device memory usage if profiling stats are available."""
        try:
            stats = self.device.memory_stats()
            allocated = stats.get('bytes_in_use', 0) / 1024**2
            print(f"JAX Device: {self.device.device_kind}")
            print(f"Allocated: {allocated:.2f} MB")
        except Exception:
            print(f"JAX Device: {self.device}")
            print("Detailed memory stats not available for this backend.")

    def _calculate_auto_ranges(self) -> tuple[tuple[float, float, int], tuple[float, float, int]]:
        """
        Automatically determines the bias (Vb) and gate (Vg) voltage grid ranges.

        Returns
        -------
        tuple of tuple of (float, float, int)
            A tuple containing the auto-scaled ranges: 
            `((Vb_min, Vb_max, Vb_points), (Vg_min, Vg_max, Vg_points))`.
        """
        
        gs_energies = {}
        excited_spreads = []
        
        # 1. Extract ground-state energies and maximum excitation bounds per sector
        for n in self.charge_numbers:
            if hasattr(self, 'states_dict') and self.states_dict.get(n):
                configs = self.states_dict[n]
                gs_E = configs[0]['energy']
                max_E = configs[-1]['energy']
                gs_energies[n] = gs_E
                excited_spreads.append(max_E - gs_E)
            else:
                gs_energies[n] = 0.0
                excited_spreads.append(0.0)

        # 2. Map V_g values for ground-state transitions (evaluated at V_b = 0)
        transitions = []
        for i in range(len(self.charge_numbers) - 1):
            n_curr = self.charge_numbers[i]
            n_next = self.charge_numbers[i+1]
            if n_next == n_curr + 1:
                mu = gs_energies[n_next] - gs_energies[n_curr]
                transitions.append(mu / self.aG)

        # 3. Establish core energy scales
        max_excitation = max(excited_spreads) if excited_spreads else 0.0
        thermal_floor = 15.0 * self.kB_T  # Minimum window to resolve Fermi-Dirac broadening
        
        # Estimate Addition Energy
        if len(transitions) >= 2:
            # Derived naturally from the spacing of successive ground state transitions
            E_add_est = np.mean(np.diff(transitions)) * self.aG
        else:
            # Fallback heuristics for single-transition simulations (e.g. N=1 to N=2)
            if getattr(self, 'U', None) is not None and len(self.U) > 0:
                E_add_est = float(self.U[0])
            else:
                E_add_est = max(1.0, 2.0 * max_excitation)

        # 4. Determine Vb Range (Bias Voltage)
        vb_limit = max(1.25 * max_excitation, 0.4 * E_add_est, thermal_floor)

        # 5. Determine Vg Range (Gate Voltage)
        if transitions:
            center_Vg = 0.5 * (min(transitions) + max(transitions))
            transition_span = max(transitions) - min(transitions)
        else:
            center_Vg = 0.0
            transition_span = 0.0

        # Base gate margin to capture the full width of the zero-bias transition
        base_Vg_margin = max(0.6 * (E_add_est / self.aG), transition_span / 2.0)
        
        # This term prevents the top edges of the diamonds from being cropped off.
        lever_max = max(self.aL, self.aR) if (self.aL is not None and self.aR is not None) else 0.5
        bias_expansion_Vg = vb_limit * (lever_max / self.aG)

        total_Vg_halfwidth = base_Vg_margin + bias_expansion_Vg
        
        # Prevent zero-width grid collapses by enforcing a minimum safety width
        min_gate_halfwidth = max(thermal_floor / self.aG, 0.1)
        total_Vg_halfwidth = max(total_Vg_halfwidth, min_gate_halfwidth)

        vg_min = center_Vg - total_Vg_halfwidth
        vg_max = center_Vg + total_Vg_halfwidth

        return (-vb_limit, vb_limit, self.datapoints), (vg_min, vg_max, self.datapoints)


class DirectParameters(BaseParameters):
    """
    Parameter configuration for phenomenological macroscopic quantum dot models.

    This class handles systems where the many-body energy levels, charge states, 
    and tunneling rates are explicitly known or provided by the user. It bypasses 
    microscopic Exact Diagonalization entirely, making it the ideal choice for 
    fitting experimental stability diagrams or modeling effective macroscopic 
    states.

    When `spin_labels` is enabled, this class will automatically map user-defined 
    macro-states (e.g., 'CS' for Closed-shell Singlet, 'D' for Doublet) to 
    pre-calculated degeneracy weights mapped to transition matrix elements. These weights 
    are later used by the `DirectSimulator` to correctly scale the sequential tunneling rates.
    
    Notes
    -----
    - Available spin configurations are: Closed-Shell Singlet (two electrons in the same orbital),
      Open-Shell Singlet (two electrons in different orbitals), Doublet, Triplet, and Quadruplet.
    - The accepted spin labels are: 'CS', 'OS', 'D', 'T', 'Q', and can be programmatically 
      accessed via the `spin_config` attribute.

    See Also
    --------
    DirectSimulator : The simulator subclass designed to ingest these parameters.
    DerivedParameters : Alternative parameter class for microscopic, first-principles models.
    """
    
    def __init__(
        self, 
        charge_numbers: list[int],
        energies: list[list[float | tuple[float, str]]], 
        T: float, 
        slopes: list[float] | None = None, 
        lever_arms: list[float] | None = None,
        gammas: list[float] | list[list[float | list[list[float]] | np.ndarray]] = [1.0, 1.0], 
        gamma_rel: float | list[float | list[list[float]] | np.ndarray] = 0.0,
        spin_labels: bool = False,
        Vb_range: tuple[float, float, int] | None = None, 
        Vg_range: tuple[float, float, int] | None = None, 
        V_ground: float = 0.0, 
        setup: str = "symmetric",
        datapoints: int = 100, 
        dtype: np.dtype = jnp.float32
    ):
        """
        Initializes the Direct System Parameters.

        Parameters
        ----------
        charge_numbers : list of int
            List containing all sequential charge state numbers.
        energies : list of list of float or tuple
            List containing lists of total energy levels corresponding sequentially 
            to each charge state. Can explicitly map energies to macro-state spin 
            labels using tuples (e.g., `(energy, 'T')`).
        T : float
            Temperature of the leads.
        slopes : list of float or None, optional
            Coulomb diamond slopes [sL, sR]. Must provide `slopes` or `lever_arms`.
        lever_arms : list of float or None, optional
            Direct electrostatic couplings [a_L, a_R, a_G].
        gammas : list of float or list of list 
            Effective tunneling couplings to leads [Γ_L, Γ_R]. Can be a state-independent 
            global scalar list [gL, gR], or a list of transition blocks paralleling charge sector transitions 
            (length = len(charge_numbers) - 1). Each entry is [gL_val, gR_val], where values 
            can be scalars or explicit M_{N+1} x M_{N} arrays mapping specific transitions.
        gamma_rel : float or list of (float or list of list of float or np.ndarray), optional
            Relaxation rate for same-charge states. Can be a global scalar or a list 
            matching the length of `charge_numbers`, where entries are either sector-specific 
            scalars or M x M arrays mapping explicit state-to-state relaxation. Default is 0.0.
        spin_labels : bool, optional
            If set to True, applies spin-dependent Clebsch-Gordan weights. Default is False.
        Vb_range : tuple of (float, float, int) or None, optional
            Bias voltage range (min, max, points). Default is None.
        Vg_range : tuple of (float, float, int) or None, optional
            Gate voltage range (min, max, points). Default is None.
        V_ground : float, optional
            Ground potential offset. Default is 0.0.
        setup : str, optional
            Setup configuration ('symmetric', 'ground_R', 'ground_L'). Default is "symmetric".
        datapoints : int, optional
            Default points for auto calculation. Default is 100.
        dtype : np.dtype, optional
            Tensor type. Default is jnp.float32.
        """
        super().__init__(T=T, slopes=slopes, lever_arms=lever_arms, gammas=gammas, 
                         gamma_rel=gamma_rel, Vb_range=Vb_range, Vg_range=Vg_range, 
                         V_ground=V_ground, setup=setup, datapoints=datapoints, dtype=dtype)
        
        self.spin_labels = spin_labels
        self.charge_numbers = charge_numbers
        
        # Construct physical states
        self.states_dict = self._parse_energies(charge_numbers, energies)
        
        # Parse tunneling and relaxation rates into global transition matrices
        self.base_gL, self.base_gR = self._parse_gammas(gammas)
        self.gamma_rel = self._parse_gamma_rel(gamma_rel)

        # Pre-saved sequence of supported macroscopic spin configurations
        self.spin_config = ['CS', 'OS', 'D', 'T', 'Q']

        # Pre-calculated degeneracy weights derived from Clebsch-Gordan squared sums
        self.g_deg = {
            'CS_to_D': 2.0,       # Closed-shell Singlet -> Doublet
            'D_to_CS': 1.0,       # Doublet -> Closed-shell Singlet
            'D_to_T': 1.5,        # Doublet -> Triplet
            'T_to_D': 1.0,        # Triplet -> Doublet
            'D_to_OS': 0.5,       # Doublet -> Open-shell Singlet
            'OS_to_D': 1.0,       # Open-shell Singlet -> Doublet
            'CS_to_Q': 0.0,       # Closed-shell Singlet -> Quadruplet
            'Q_to_CS': 0.0,       # Quadruplet -> Closed-shell Singlet
            'T_to_Q': 4.0 / 3.0,  # Triplet -> Quadruplet
            'Q_to_T': 1.0,        # Quadruplet -> Triplet
            'OS_to_Q': 0.0,       # Open-shell Singlet -> Quadruplet
            'Q_to_OS': 0.0        # Quadruplet -> Open-shell Singlet    
        }

        self._validate_inputs()
        self._finalize_setup(Vb_range, Vg_range, setup)

        self.params_config = {
            "Occupations & Energies": self.states_dict,
            "Lead Tunnel Couplings [Γ_L, Γ_R]": gammas,
            "Relaxation Rate(s) Γ_rel": self.gamma_rel,
            "Spin Degeneracy Rules": "Active" if self.spin_labels else "Inactive",
            "Coulomb Diamond Slopes [s1, s2]": self.slopes if self.slopes else "N/A (Using lever arms)",
            "Lever Arms [a_L, a_R, a_G]": [float(x) for x in np.round([self.aL, self.aR, self.aG], 3)], 
            "Temperature": self.T,
            "V_g range": [float(x) for x in np.round([self.Vgmin, self.Vgmax], 3)],
            "V_b range": [float(x) for x in np.round([self.Vbmin, self.Vbmax], 3)],
            "Bias-voltage Setup": self.setup
        }

    def _validate_inputs(self) -> None:
        """
        Shields against input inconsistencies between energies and spin labels.

        Raises
        ------
        ValueError
            If the 'energies' dictionary is empty.
            If charge states in 'energies' are not strictly sequential.
            If a charge state contains an empty list of levels.
            If spin tracking is active but labels are missing or unsupported.
        """
        # 1. Energies: Structural Integrity
        charge_states = sorted(self.states_dict.keys())
        if not charge_states:
            raise ValueError("The 'energies' dictionary cannot be empty.")
            
        for i in range(len(charge_states) - 1):
            if charge_states[i+1] != charge_states[i] + 1:
                raise ValueError(f"Charge states in 'energies' must be strictly sequential. "
                                 f"Found a gap between N={charge_states[i]} and N={charge_states[i+1]}.")

        # 2. Energies: Label Extraction & Validation 
        provided_labels = set()
        missing_labels = False
        
        for n, levels in self.states_dict.items():
            if not levels:
                raise ValueError(f"Charge state N={n} in 'energies' has an empty list.")

            for state in levels:
                if state['label'] == 'Unknown':
                    missing_labels = True
                else:
                    provided_labels.add(state['label'])

        if not self.spin_labels:
            return  # If spin rules are off, ignore spin labels

        # If spin rules are on, every level must have a label
        if missing_labels:
            raise ValueError("Spin tracking is active, but some energies are missing labels. "
                             "You must provide energies as tuples, e.g., (0.5, 'CS').")

        unsupported = provided_labels - set(self.spin_config)
        if unsupported:
            raise ValueError(f"Unsupported spin labels found in energies: {unsupported}. "
                             f"Supported labels are: {self.spin_config}")
                                    
    def _parse_energies(
        self, 
        charge_numbers: list[int], 
        energies: list[list[float | tuple[float, str]]]
    ) -> dict[int, list[dict]]: 
        """
        Parses the charge numbers and energies lists to ensure lengths match 
        and all entries are formatted as tuples of (energy, label).

        Parameters
        ----------
        charge_numbers : list of int
            Input list of charge states.
        energies : list of list of float or tuple
            Input list of energy levels corresponding sequentially to the charge states.

        Returns
        -------
        dict of int to list of tuple of (float, str)
            Parsed dictionary of explicitly labeled energy tuples.
        """
        if len(charge_numbers) != len(energies):
            raise ValueError("The lengths of 'charge_numbers' and 'energies' must match exactly.")
            
        parsed = {}
        for n, levels in zip(charge_numbers, energies):
            std_levels = []
            for lvl in levels:
                if isinstance(lvl, tuple):
                    std_levels.append({"energy": float(lvl[0]), "label": lvl[1]})
                else:
                    std_levels.append({"energy": float(lvl), "label": 'Unknown'})
            parsed[n] = std_levels
        return parsed

    def _parse_gammas(
        self, 
        gammas: list[float] | list[list[float | list[list[float]] | np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Parses sequential tunneling rates into global N_tot x N_tot transition matrices.

        Parameters
        ----------
        gammas : list
            Global [gL, gR] list, or a transition-specific list of length 
            `len(charge_numbers) - 1`. Each entry defines [gL_val, gR_val] for a given 
            charge sector jump. Values can be scalars or strict M_{N+1} x M_{N} matrices.

        Returns
        -------
        tuple of np.ndarray
            Two N_tot x N_tot matrices representing base left (gL) and right (gR) 
            sequential tunneling transitions. Off-diagonal non-sequential blocks are zeroed.
        """
        N_total = sum(len(levels) for levels in self.states_dict.values())
        base_gL = np.zeros((N_total, N_total), dtype=float)
        base_gR = np.zeros((N_total, N_total), dtype=float)
        
        num_transitions = max(0, len(self.charge_numbers) - 1)
        
        # Scenario 1: A global [gamma_L, gamma_R] list
        if isinstance(gammas, list) and len(gammas) == 2 and all(isinstance(g, (int, float)) for g in gammas):
            gL_global, gR_global = float(gammas[0]), float(gammas[1])
            idx_low = 0
            for s in range(num_transitions):
                n_curr = self.charge_numbers[s]
                n_next = self.charge_numbers[s+1]
                m_curr = len(self.states_dict[n_curr])
                m_next = len(self.states_dict[n_next])
                idx_high = idx_low + m_curr
                
                base_gL[idx_high:idx_high+m_next, idx_low:idx_low+m_curr] = gL_global
                base_gR[idx_high:idx_high+m_next, idx_low:idx_low+m_curr] = gR_global
                
                idx_low += m_curr
                
        # Scenario 2: Parallel nested lists mapping transition sectors
        elif isinstance(gammas, list):
            if len(gammas) != num_transitions:
                raise ValueError(f"List of transition-specific gammas must have length {num_transitions} "
                                 f"(one less than charge_numbers).")
                
            idx_low = 0
            for s, g_pair in enumerate(gammas):
                if not isinstance(g_pair, (list, tuple)) or len(g_pair) != 2:
                    raise ValueError("Each transition entry in gammas must contain exactly two elements [gL, gR].")
                
                n_curr = self.charge_numbers[s]
                n_next = self.charge_numbers[s+1]
                m_curr = len(self.states_dict[n_curr])
                m_next = len(self.states_dict[n_next])
                idx_high = idx_low + m_curr
                
                gL_val, gR_val = g_pair
                
                # Format Left Lead Matrix
                if isinstance(gL_val, (int, float)):
                    block_L = np.full((m_next, m_curr), float(gL_val))
                else:
                    block_L = np.array(gL_val, dtype=float)
                    if block_L.shape != (m_next, m_curr):
                        raise ValueError(f"gamma_L matrix for transition N={n_curr} -> N={n_next} "
                                         f"must be of shape ({m_next}, {m_curr}).")
                
                # Format Right Lead Matrix
                if isinstance(gR_val, (int, float)):
                    block_R = np.full((m_next, m_curr), float(gR_val))
                else:
                    block_R = np.array(gR_val, dtype=float)
                    if block_R.shape != (m_next, m_curr):
                        raise ValueError(f"gamma_R matrix for transition N={n_curr} -> N={n_next} "
                                         f"must be of shape ({m_next}, {m_curr}).")
                                         
                base_gL[idx_high:idx_high+m_next, idx_low:idx_low+m_curr] = block_L
                base_gR[idx_high:idx_high+m_next, idx_low:idx_low+m_curr] = block_R
                
                idx_low += m_curr
        else:
            raise ValueError("gammas must be a global list of two floats, or a nested list mapping specific charge transitions.")
            
        return base_gL, base_gR

    def _parse_gamma_rel(self, gamma_rel: float | list[float | list[list[float]] | np.ndarray]) -> float | np.ndarray:
        """
        Parses relaxation rates into a global N x N block-diagonal matrix.

        Parameters
        ----------
        gamma_rel : float or list of (float or list of list of float or np.ndarray)
            Global scalar, or a list perfectly paralleling `charge_numbers`. 
            List entries can be sector-specific scalars or M x M arrays 
            (where M is the number of states in that sector).

        Returns
        -------
        float or np.ndarray
            A float if applied globally, or an N x N array blocking physically 
            forbidden inter-sector transitions.
        """
        # Scenario 1: A global scalar
        if isinstance(gamma_rel, (int, float)):
            return float(gamma_rel)
            
        # Scenario 2: Parallel lists mapped to charge sectors
        if not isinstance(gamma_rel, list):
            raise ValueError("gamma_rel must be a global float or a list matching the length of charge_numbers.")
            
        if len(gamma_rel) != len(self.charge_numbers):
            raise ValueError(f"List of specific relaxation rates must have exactly {len(self.charge_numbers)} entries to match charge_numbers.")

        # Calculate total N to initialize the global matrix
        N_total = sum(len(levels) for levels in self.states_dict.values())
        rel_matrix = np.zeros((N_total, N_total), dtype=float)
        
        idx = 0
        for n, provided_rel in zip(self.charge_numbers, gamma_rel):
            num_states = len(self.states_dict[n])
            
            if isinstance(provided_rel, (int, float)):
                # Uniform relaxation scalar for this specific sector
                block = np.full((num_states, num_states), float(provided_rel))
            else:
                # State-specific relaxation matrix for this sector
                block = np.array(provided_rel, dtype=float)
                if block.shape != (num_states, num_states):
                    raise ValueError(
                        f"gamma_rel block for charge state N={n} must be a square "
                        f"matrix of shape ({num_states}, {num_states})."
                    )
            
            # Insert the block into the global matrix
            rel_matrix[idx:idx+num_states, idx:idx+num_states] = block
            idx += num_states
            
        return rel_matrix


class DerivedParameters(BaseParameters):
    """
    Parameter configuration for microscopic quantum dot models utilizing Exact Diagonalization (ED).

    This class takes first-principles microscopic inputs—such as single-particle 
    orbital energies, intra-orbital Coulomb repulsion (U), inter-orbital repulsion (V), 
    exchange interactions (J), and Zeeman splitting—to construct a generalized 
    many-body Kanamori-type Hamiltonian.

    Upon initialization, it automatically delegates the generation of the requisite 
    fermionic Fock basis and Exact Diagonalization to the `ExactDiagonalization` engine.
    It yields a clean physical basis of many-body eigenstates, their energies, 
    and their exact quantum numbers (S, Sz). These properties are then passed to 
    the `DerivedSimulator` to calculate first-principles transition rates.

    See Also
    --------
    ExactDiagonalization : The underlying engine used to compute the eigenspectrum.
    DerivedSimulator : The simulator subclass designed to ingest these parameters.
    DirectParameters : Alternative parameter class for phenomenological, user-defined models.
    """
    
    def __init__(
        self, 
        charge_numbers: list[int], 
        orbital_energies: list[float], 
        U: float | list[float], 
        V: float | list[list[float]], 
        J: float | list[list[float]], 
        Jp: float | list[list[float]], 
        T: float, 
        slopes: list[float] | None = None,
        lever_arms: list[float] | None = None,
        B: list[float] = [0.0, 0.0, 0.0], 
        g_factor: float = 2.0, 
        gammas: list[float] | list[list[float]] = [1.0, 1.0], 
        M_rel: float | list[list[float]] | np.ndarray = 0.0,
        Vb_range: tuple[float, float, int] | None = None, 
        Vg_range: tuple[float, float, int] | None = None, 
        V_ground: float = 0.0, 
        setup: str = 'symmetric',
        datapoints: int = 100, 
        dtype: np.dtype = jnp.float32
    ):
        """
        Initializes the Derived System Parameters with Exact Diagonalization.

        Parameters
        ----------
        charge_numbers : list of int
            List containing all charge state numbers.
        orbital_energies : list of float
            List containing all single-particle energies.
        U : float or list of float
            Intra-orbital Coulomb repulsion (V_a).
        V : float or list of list of float
            Inter-orbital density-density interaction (V_ab).
        J : float or list of list of float
            Exchange interaction (J_ab).
        Jp : float or list of list of float
            Pair-hopping interaction (J_ab').
        T : float
            Temperature of the leads.
        slopes : list of float or None, optional
            Coulomb diamond slopes [sL, sR]. Must provide `slopes` or `lever_arms`.
        lever_arms : list of float or None, optional
            Direct electrostatic couplings [a_L, a_R, a_G].
        B : list of float, optional
            External magnetic field. Default is [0.0, 0.0, 0.0].
        g_factor : float, optional
            g-factor for the system. Default is 2.0.
        gammas : list of float or list of list of float, optional
            Tunneling couplings of orbitals to the leads [Γ_L, Γ_R]. Can be global scalars 
            or specific orbital arrays. Default is [1.0, 1.0].
        M_rel : float or list of list of float or np.ndarray, optional
            Electron-phonon relaxation matrix elements (transition amplitudes). 
            Can be a global scalar applied to all transitions, or an exact M x M, or 2M x 2M 
            matrix, where M is the number of orbitals. The factor of 2 incorporates 
            the spin degree of freedom. Default is 0.0.
        Vb_range : tuple of (float, float, int) or None, optional
            Bias voltage range as (min, max, points).
        Vg_range : tuple of (float, float, int) or None, optional
            Gate voltage range as (min, max, points).
        V_ground : float, optional
            Ground potential offset. Default is 0.0.
        setup : str, optional
            Setup config. Default is "symmetric".
        datapoints : int, optional
            Default points for auto calculation. Default is 100.
        dtype : np.dtype, optional
            Tensor type. Default is jnp.float32.
            
        Raises
        ------
        ValueError
            If matrix dimensions for interactions (U, V, J) are misaligned with 
            the number of orbital energies.
        """
        super().__init__(T=T, slopes=slopes, lever_arms=lever_arms, gammas=gammas, 
                         gamma_rel=0.0, Vb_range=Vb_range, Vg_range=Vg_range, 
                         V_ground=V_ground, setup=setup, datapoints=datapoints, dtype=dtype)
        
        self.charge_numbers = charge_numbers
        self.orbital_energies = orbital_energies
        self.num_levels = len(orbital_energies)
        
        # 1. Initialize Symbolic Property
        self.symbolic_hamiltonian = None

        # 2. Broadcast and Validate U
        if isinstance(U, (int, float)):
            self.U = np.full(self.num_levels, float(U)) 
        else:
            self.U = np.array(U, dtype=float)
            if len(self.U) != self.num_levels:
                raise ValueError(f"U must be scalar or 1D array of length {self.num_levels}")

        # 3. Broadcast and Validate V (Zeroing the diagonal for scalars)
        if isinstance(V, (int, float)):
            self.V = np.full((self.num_levels, self.num_levels), float(V))
            np.fill_diagonal(self.V, 0.0) 
        else:
            self.V = np.array(V, dtype=float)
            if self.V.shape != (self.num_levels, self.num_levels):
                raise ValueError(f"V must be scalar or 2D array of shape ({self.num_levels}, {self.num_levels})")

        # 4. Broadcast and Validate J (Zeroing the diagonal for scalars)
        if isinstance(J, (int, float)):
            self.J = np.full((self.num_levels, self.num_levels), float(J))
            np.fill_diagonal(self.J, 0.0)
        else:
            self.J = np.array(J, dtype=float)
            if self.J.shape != (self.num_levels, self.num_levels):
                raise ValueError(f"J must be scalar or 2D array of shape ({self.num_levels}, {self.num_levels})")

        # 5. Broadcast and Validate Jp (Zeroing the diagonal for scalars)
        if isinstance(Jp, (int, float)):
            self.Jp = np.full((self.num_levels, self.num_levels), float(Jp))
            np.fill_diagonal(self.Jp, 0.0)
        else:
            self.Jp = np.array(Jp, dtype=float)
            if self.Jp.shape != (self.num_levels, self.num_levels):
                raise ValueError(f"Jp must be scalar or 2D array of shape ({self.num_levels}, {self.num_levels})")
            
        # 6. Broadcast and Validate B
        self.B = [float(x) for x in B]
        if len(self.B) != 3:
            raise ValueError("Magnetic field B must be a 3-element list: [B_x, B_y, B_z]")

        # 7. Broadcast and Validate M_rel
        if isinstance(M_rel, (int, float)):
            # populate all transitions (including spin-flips)
            self.M_rel = np.full((2 * self.num_levels, 2 * self.num_levels), float(M_rel))
            np.fill_diagonal(self.M_rel, 0.0)
            
        else:
            M_rel_arr = np.array(M_rel, dtype=float)
            
            if M_rel_arr.shape == (self.num_levels, self.num_levels):
                # User provides M x M (Spatial only). Promote to 2M x 2M spin-conserving matrix.
                self.M_rel = np.zeros((2 * self.num_levels, 2 * self.num_levels), dtype=float)
                for i in range(self.num_levels):
                    for j in range(self.num_levels):
                        if i != j:
                            self.M_rel[2*i, 2*j] = M_rel_arr[i, j]         # Up -> Up
                            self.M_rel[2*i+1, 2*j+1] = M_rel_arr[i, j]     # Down -> Down
                            
            elif M_rel_arr.shape == (2 * self.num_levels, 2 * self.num_levels):
                # User provided full 2M x 2M. Use directly.
                self.M_rel = M_rel_arr
                np.fill_diagonal(self.M_rel, 0.0)
                
            else:
                raise ValueError(
                    f"M_rel must be a scalar, an M x M spatial array {self.num_levels, self.num_levels}, "
                    f"or a full 2M x 2M array {2 * self.num_levels, 2 * self.num_levels}"
                )
        
        self.g_factor = g_factor

        self._validate_inputs()

        # Initialize the ED Engine and compute spectrum
        self.ed_engine = ExactDiagonalization(
            num_levels=self.num_levels, 
            orbital_energies=self.orbital_energies, 
            U=self.U, V=self.V, J=self.J, Jp=self.Jp, 
            B=self.B, g_factor=self.g_factor, 
            charge_numbers=self.charge_numbers
        )
        
        self.states_dict, self.subspace_data, self.symbolic_hamiltonian = self.ed_engine.compute_eigenspectrum()
        
        self._finalize_setup(Vb_range, Vg_range, setup)
        
        # 7. Build Parameter Configuration Dictionary
        self.params_config = {
            "Occupations & Level Spacing": {N: [E for E in orbital_energies] for N in self.charge_numbers},
            "Direct Intra-orbital Coulomb U": self.U,
            "Direct Inter-orbital Coulomb V": self.V,
            "Exchange J": self.J,
            "Pair-hopping Jp": self.Jp,
            "Magnetic Field B [Bx, By, Bz]": self.B,
            "Lead Tunnel Couplings [Γ_L, Γ_R]": self.gammas,
            "Relaxation Matrix Elements M_rel": self.M_rel, 
            "Coulomb Diamond Slopes [s1, s2]": self.slopes if self.slopes else "N/A (Using lever arms)",
            "Lever Arms [a_L, a_R, a_G]": [float(x) for x in np.round([self.aL, self.aR, self.aG], 3)], 
            "g factor": self.g_factor,
            "V_g range": [float(x) for x in np.round([self.Vgmin, self.Vgmax], 3)], 
            "V_b range": [float(x) for x in np.round([self.Vbmin, self.Vbmax], 3)], 
            "Bias-voltage Setup": self.setup
        }

    def _validate_inputs(self) -> None:
        """
        Validates the structural consistency of inputs against the system size.

        Specifically ensures that 'charge_numbers' are contiguous and that the 
        'gammas' list shape correctly corresponds to the number of defined orbitals.

        Raises
        ------
        ValueError
            If 'charge_numbers' are empty or not strictly sequential.
            If 'gammas' is not a global 2-element list or a nested list 
            perfectly matching the number of orbital energies.
        """
        # 1. Validate charge numbers for continuity
        if not self.charge_numbers:
            raise ValueError("The 'charge_numbers' list cannot be empty.")
            
        sorted_charges = sorted(self.charge_numbers)
        for i in range(len(sorted_charges) - 1):
            if sorted_charges[i+1] != sorted_charges[i] + 1:
                raise ValueError(f"Charge states must be strictly sequential. "
                                 f"Found a gap between N={sorted_charges[i]} and N={sorted_charges[i+1]}.")

        # 2. Validate gammas against orbital structure
        if not isinstance(self.gammas, list):
            raise ValueError("gammas must be a list.")
            
        # Case A: Global gammas [gL, gR]
        if len(self.gammas) == 2 and all(isinstance(g, (int, float)) for g in self.gammas):
            pass # Valid globally applied rates
            
        # Case B: Orbital-specific gammas [[gL1, gR1], [gL2, gR2], ...]
        elif len(self.gammas) == self.num_levels:
            for i, g in enumerate(self.gammas):
                if not isinstance(g, (list, tuple)) or len(g) != 2:
                    raise ValueError(
                        f"Orbital-specific gammas must contain exactly two elements [gL, gR]. "
                        f"Invalid entry for orbital {i}: {g}"
                    )
                if not all(isinstance(rate, (int, float)) for rate in g):
                    raise ValueError(
                        f"Gamma rates must be numeric values. Invalid entry for orbital {i}: {g}"
                    )
        else:
            raise ValueError(
                f"gammas must either be a global list of two floats [gL, gR], or a nested "
                f"list of length {self.num_levels} matching the number of orbital energies."
            )

    # Pass-through methods for backward-compatibility with tests & simulator
    def construct_hamiltonian(self, N: int):
        """Pass-through to the ED Engine to rebuild a Hamiltonian block manually."""
        return self.ed_engine.construct_hamiltonian(N)
        
    def _apply_ops(self, state, operators):
        """Pass-through to the ED Engine to apply creation/annihilation operators manually."""
        return self.ed_engine._apply_ops(state, operators)
        
    def _op_to_matrix(self, basis_initial, basis_final, operator):
        """Pass-through to the ED Engine to construct transition matrices."""
        return self.ed_engine._op_to_matrix(basis_initial, basis_final, operator)