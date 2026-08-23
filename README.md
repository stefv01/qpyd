# QPyD — Sequential Electron Transport Simulator for Quantum Dots

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![JAX](https://img.shields.io/badge/JAX-Accelerated-FF6F00.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**QPyD** (imported as `qpyd`) is an open-source Python framework for simulating sequential electron transport in quantum dot devices.

`qpyd` uses a steady-state rate equation (Pauli Master equation) approach to model sequential electron tunneling. Built on **JAX**, it utilizes XLA compilation and hardware-accelerated vectorization to process high-resolution Coulomb diamond stability diagrams instantly.

## 🚀 Features

* **Microscopic ED Engine (`DerivedParameters`):** Automatically constructs many-body Hamiltonians using Exact Diagonalization. Fully supports single-particle orbitals, intra/inter-orbital Coulomb interactions, exchange/pair-hopping, and Zeeman splitting.
* **Macroscopic Phenomenological Engine (`DirectParameters`):** Bypass microscopic Hamiltonian setup to manually enforce arbitrary many-body state energies, spin labels, and transition spin degeneracies.
* **JAX Backend:** The core Master equation solver uses `@jax.jit` and `jax.vmap` to solve steady-state probabilities across 2D voltage meshes.
* **Memory-Safe Execution:** A hybrid chunking algorithm streams data directly to the CPU host, bypassing GPU VRAM bottlenecks on high-resolution grids.

## ⚙️ Installation

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

## ⚡ Quick Start

The package uses a factory pattern. Define your system parameters, pass them to Simulator(), and let the JAX backend handle the rest.

```python
import numpy as np
from qpyd import DerivedParameters, Simulator, Plotter

# 1. Configure a 2-orbital quantum dot system
params = DerivedParameters(
    charge_numbers=[2, 3],      
    orbital_energies=[1.0, 2.0],         # Single-particle energies (meV)
    T=0.4,                               # Temperature (K)
    slopes=[1.0, 1.0],      
    U=0.5 * np.ones(2),                  # Intra-orbital Coulomb (meV)
    V=np.array([[0.0, 0.2],              # Inter-orbital Coulomb (meV)
                [0.2, 0.0]]),        
    J=0.1, Jp=0.1, B=[0.0, 0.0, 0.0],    # Exchange, Pair-hopping, Mag. field (meV, T)
    gammas=[[1.0, 1.0], [1.0, 1.0]],     # Orbital-Lead tunnel couplings (meV)
    setup="ground_R",                    # Voltage bias setup (Right lead is grounded)
    datapoints=200
)

# 2. Solve the Master Equation
sim = Simulator(params)
sim.solve()
results = sim.get_results()

# 3. Visualize the results (Coulomb Diamonds)
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

## 📚 Documentation & Tutorials

Interested in learning more about how QPyD works? Please check the documentation:

* [Getting Started](https://github.com/stefv01/qpyd/blob/main/docs/getting_started.md)
* [Theoretical Description](https://github.com/stefv01/qpyd/blob/main/docs/theoretical_description.md)
* [API Reference & Parameter Mapping](https://github.com/stefv01/qpyd/blob/main/docs/api_reference.md)
* [Interactive Tutorials (Jupyter Notebooks)](https://github.com/stefv01/qpyd/tree/main/tutorials)

## 🎓 Acknowledgements

This project is an extension of my MSc thesis project *"Electronic Transport Analysis of Atomically Precise Armchair Graphene Nanoribbons"* at TU Delft, under the supervision of Herre S.J. van der Zant and Yongqing Yang. Special thanks to Jaime Ferrer (Universidad de Oviedo). His analytical calculations for Armchair Graphene Nanoribbons guided and inspired the development of the generalized Quantum Dot Hamiltonian implemented in this simulator.

*Note: Users interested in alternate perturbative transport methods are highly encouraged to refer to the open-source QmeQ package (Kiršanskas et al., 2017).*

## 📄 License

This project is licensed under the MIT License.