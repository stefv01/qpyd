# Getting Started

## Installation

It is recommended to install this package in "editable" mode so that the `qpyd` module can be imported from anywhere on your system.

```bash
git clone https://github.com/stefv01/qpyd.git
cd qpyd
pip install -e .
```

## Physical Units & Constants

To ensure consistent transport calculations, the simulator standardizes its internal variables around the **milli-electron volt (meV)**. When initializing parameters, all inputs should adhere to the following unit conventions:

* **Energy ($E, U, V, J, J^\prime$):** milli-electron volts (**meV**).
* **Voltages ($V_b, V_g$):** milli-Volts (**mV**). Because the electron charge is factored into the energy scaling, an applied voltage of $1\text{ mV}$ corresponds exactly to a $1\text{ meV}$ shift in the chemical potentials.
* **Temperature ($T$):** Kelvin (**K**). The simulator converts this to thermal energy internally using the macroscopic Boltzmann constant ($k_B \approx 0.08617\text{ meV/K}$).
* **Magnetic Field ($B$):** Tesla (**T**). The Zeeman energy splitting is calculated internally using the Bohr magneton ($\mu_B \approx 0.05788\text{ meV/T}$).
* **Couplings & Amplitudes ($\Gamma, \Gamma_{rel}, M_{rel}$):** $\Gamma, \Gamma_{rel}$ are expressed in energy units ($\text{meV}$) and $M_{rel}$ in $(\text{meV})^{1/2}$. The solver automatically scales these by the reduced Planck constant ($\hbar \approx 6.582 \times 10^{-13}\text{ meV}\cdot\text{s}$) to compute physical transition rates in $\text{s}^{-1}$.
* **Current ($I$):** By combining the intrinsic electron charge ($e \approx 1.602 \times 10^{-10}\text{ nC}$) and $\hbar$, the macroscopic current natively evaluates to nano-Amperes (**nA**), and differential conductance to **nA/mV** (equivalent to $\mu\text{S}$).

## Quick Start Examples

The package uses a factory pattern. You simply create your parameter object (`DirectParameters` or `DerivedParameters`), pass it to `Simulator()`, and extract the results.

### Example 1: The Exact Diagonalization Model

The `DerivedParameters` class is designed for first-principles modeling using a Hamiltonian description of the leads/QD system. You provide the raw single-particle orbital energies and Coulomb interactions, and the engine automatically generates the fermionic Fock basis, constructs the Hamiltonian, and finds the exact many-body eigenstates.

```python
import numpy as np
from qpyd import DerivedParameters, Simulator, Plotter

# 1. Define the Microscopic Parameters
params = DerivedParameters(
    charge_numbers=[2, 3],               # Simulate transitions between N=2 and N=3 electrons
    orbital_energies=[1.0, 2.0],         # Bare single-particle energies for two orbitals
    T=0.4,                               # Temperature (0.4 K)
    slopes=[1.0, 1.0],                   # Coulomb diamond slopes (auto-calculates lever arms)
    U=0.5 * np.ones(2),                  # Intra-orbital Coulomb repulsion (0.5 meV)
    V=np.array([[0.0, 0.2],              # Inter-orbital Coulomb repulsion (0.2 meV)
                [0.2, 0.0]]),        
    J=0.1, Jp=0.1, B=[0.0, 0.0, 0.0],    # Exchange, Pair-hopping, and Magnetic field vector
    gammas=[[1.0, 1.0], [1.0, 1.0]],     # Tunnel couplings [Γ_L, Γ_R] for each orbital
    setup="ground_R"                     # Voltage bias setup (Right lead is grounded)
)

# 2. Execute the Master Equation Solver
sim = Simulator(params)
sim.solve()

# 3. Extract and Plot the Results
results = sim.get_results()
plotter = Plotter()
extent = [params.Vgmin, params.Vgmax, params.Vbmin, params.Vbmax]
plotter.plot_heatmap(results['N_avg'], extent, title="Average Occupation")
```

**What is happening here?**
1. **Initialization:** When `DerivedParameters` is called, it automatically determines a safe 2D grid for the Gate and Bias voltages based on the energy scales provided. It then performs Exact Diagonalization to find the many-body states.
2. **Solving:** `sim.solve()` compiles the Master Equation using JAX and evaluates the steady-state probabilities across the entire voltage grid in memory-efficient chunks.
3. **Results:** `get_results()` pulls the final data out of JAX memory and returns a standard dictionary of NumPy arrays. It contains the steady-state probabilities (`'P'`), the average charge occupation (`'N_avg'`), the calculated DC current (`'I'`), and the differential conductance (`'G'`).

---

### Example 2: The Direct Phenomenological Model

If you are trying to quickly test hypotheses to fit experimental data or model a macroscopic effective system, building a Hamiltonian description is often overkill. The `DirectParameters` class allows you to bypass Exact Diagonalization entirely and explicitly define the many-body states yourself.

```python
import numpy as np
from qpyd import DirectParameters, Simulator

# 1. Explicitly define the Many-Body States
# Format: (Energy in meV, Spin Label)
energies = [
    [(0.0, 'CS')],                  # N=0 sector: A single Closed-Shell Singlet state
    [(1.5, 'D'), (2.0, 'D')],       # N=1 sector: A ground-state Doublet and an excited Doublet
    [(4.0, 'T')]                    # N=2 sector: A Triplet state
]

# 2. Initialize the Phenomenological Parameters
params = DirectParameters(
    charge_numbers=[0, 1, 2],       # The sequential charge sectors
    energies=energies, 
    T=0.1, slopes=[1.0, 1.0], 
    gammas=[1.0, 1.0],              # Global tunnel couplings
    gamma_rel=0.01,                 # Explicit relaxation rate between states of the same charge
    spin_labels=True,               # Activate Clebsch-Gordan transition weights
    setup="symmetric"
)

# 3. Execute the Solver
sim = Simulator(params) 
sim.solve()
results = sim.get_results()
```

**Why use this model?**
By setting `spin_labels=True`, the simulator automatically scales the sequential tunneling rates between the charge sectors using predefined degeneracy weights (e.g., recognizing that transitioning from a Doublet to a Triplet has a different statistical weight than transitioning to a Singlet). This allows you to quickly simulate spin-dependent dynamics in a QD without performing the full analytical calculations. For more information, you can navigate to the [Theoretical Description](theoretical_description.md).