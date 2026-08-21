# API Reference & Parameter Mapping

When setting up the simulator, the following keywords map directly to the physical parameters of the Quantum Dot.

## General & Grid Settings
* `T`: The temperature of the system.
* `slopes`: The desired slopes of the Coulomb diamonds. Automatically calculates the lever arms ($\alpha_i$) that couple the gate and leads to the QD.
* `lever_arms`: Directly inputs the lever arms values (`[a_L, a_R, a_G]`). Mutually exclusive with `slopes`.
* `setup`: Determines how the applied bias voltage shifts the chemical potentials (e.g., `"symmetric"` splits the bias window evenly, `"ground_R"` grounds the right lead).
* `Vb_range` / `Vg_range`: Explicitly bounds the voltage mesh (e.g., `(min, max, datapoints)`). If omitted, the grid is automatically derived from the energy scales.

## Couplings & Transitions
* `gammas`: Tunneling couplings to the left and right leads ($\Gamma_\mathrm{L}, \Gamma_\mathrm{R}$).
    * **In `DirectParameters`:** Accepts a global 2-element list `[Γ_L, Γ_R]`, or a list of transition blocks paralleling sequential charge jumps.
    * **In `DerivedParameters`:** Accepts a global 2-element list `[Γ_L, Γ_R]`, or an orbital-specific nested list `[[Γ_L0, Γ_R0], ...]`.
* `gamma_rel` (**Direct Model only**): The macroscopic relaxation rate ($\Gamma_\mathrm{rel}$). Can be a global scalar, or a list matching the length of `charge_numbers` containing sector-specific scalars or $M \times M$ transition arrays.
* `M_rel` (**Derived Model only**): The electron-phonon transition matrix element. This is summed coherently across spin states to evaluate spatial quantum interference.
* `spin_labels`: When `True`, automatically calculates generalized spin-selection rules and Clebsch-Gordan transition weights.

## Derived Model Specifics
* `U`, `V`: Intra-orbital and Inter-orbital Direct Coulomb interactions.
* `J`, `Jp`: Exchange and pair-hopping interactions.
* `B`, `g_factor`: External magnetic field and Landé g-factor for Zeeman splitting.

## Internal Architecture
* **`parameters.py`**: Handles all pre-computation, input parsing, and grid setup. For `DerivedParameters`, it acts as a configuration manager that passes inputs to the physics engine.
* **`hamiltonian.py`**: The Exact Diagonalization (ED) physics engine. Generates fermionic Fock bases, constructs the many-body Hamiltonian matrices, and returns the Hamiltonian eigenstates.
* **`simulator.py`**: The computational wrapper. Calculates the transition rate matrices of the system and prepares data to be processed in memory-safe chunks.
* **`solver.py`**: The mathematical backend. Contains the `@jax.jit` compiled Master equation solvers and thermal distributions (Fermi-Dirac/Bose-Einstein) used in the calculation of transition rates. Uses `jax.vmap` to invert matrices across the voltage grid.
* **`plotter.py`**: A wrapper for Matplotlib to quickly generate stability diagrams using the `Plotter` class.