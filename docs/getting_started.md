# Getting Started

## Installation

The easiest way to install **QPyD** is directly from PyPI using pip:

```bash
pip install qpyd
```

### Development Installation
If you want to modify the source code, run the examples, or contribute to the project, you can clone the repository and install it in editable mode:

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
    charge_numbers=[2, 3],      
    orbital_energies=[1.0, 2.0],         # Single-particle energies (meV)
    T=0.4,                               # Temperature (K)
    slopes=[1.0, 1.0],                   # Imposed Coulomb diamond slopes (auto-calculates lever arms)
    U=0.5,                               # Intra-orbital Coulomb (meV)
    V=0.2,                               # Inter-orbital Coulomb (meV)     
    J=0.1, Jp=0.1,                       # Exchange, Pair-hopping (meV)
    B=[0.0, 0.0, 0.0],                   # Magnetic field (T)
    gammas=[[1.0, 1.0], [1.0, 1.0]],     # Orbital-Lead tunnel couplings (meV)
    setup="ground_R",                    # Voltage bias setup (Right lead is grounded)
    datapoints=200                       # Number of data points per axis (bias/gate voltage)
)

# 2. Execute the Master Equation Solver
sim = Simulator(params)
sim.solve()

# 3. Extract and Plot the Results
results = sim.get_results()
plotter = Plotter()
extent = [params.Vgmin, params.Vgmax, params.Vbmin, params.Vbmax]
plotter.plot_map(
    results=results, 
    quantity='N_avg', 
    extent=extent, 
    title=r"Average Occupation $\langle N \rangle$",
    axes_units=['mV', 'mV']
)
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
from qpyd import DirectParameters, Simulator, Plotter

# 1. Explicitly define the Many-Body States
# Format: (Energy in meV, Spin Label)
energies = [
    [(0.0, 'CS')],                  # N=0 sector: A single Closed-Shell Singlet state
    [(1.5, 'D'), (2.0, 'D')],       # N=1 sector: A ground-state Doublet and an excited Doublet
    [(4.0, 'T')]                    # N=2 sector: A Triplet state
]

# 2. Initialize the Phenomenological Parameters
params = DirectParameters(
    charge_numbers=[0, 1, 2], 
    energies=energies, 
    T=0.1, slopes=[1.0, 1.0], 
    gammas=[1.0, 1.0],              # Set global tunnel couplings for all transitions
    gamma_rel=0.01,                 # Explicit relaxation rate between states of the same charge
    spin_labels=True,               # Activate degeneracy transition weights
    setup="symmetric",              # Voltage bias setup (Distributed symmetrically between the leads)
    datapoints=200
)

# 3. Execute the Solver
sim = Simulator(params) 
sim.solve()
results = sim.get_results()

# 4. Visualize the results (Differential Conductance)
plotter = Plotter()
extent = [params.Vgmin, params.Vgmax, params.Vbmin, params.Vbmax]
plotter.plot_map(
    results=results, 
    quantity='G', 
    extent=extent, 
    title=r"Differential Conductance",
    axes_units=['mV', 'mV']
)
```

**Why use this model?**
By setting `spin_labels=True`, the simulator automatically scales the sequential tunneling rates between the charge sectors using predefined degeneracy weights (e.g., recognizing that transitioning from a Doublet to a Triplet has a different statistical weight than transitioning to a Singlet). This allows you to quickly simulate spin-dependent dynamics in a QD without performing the full analytical calculations. For more information, you can navigate to the [Theoretical Description](theoretical_description.md).