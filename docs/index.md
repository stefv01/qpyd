# QPyD — Quantum Dot Transport Simulator

A high-performance Python simulation framework for calculating charge transport and stability diagrams in single quantum dot (QD) systems. 

**QPyD** uses a steady-state rate equation (Pauli Master equation) approach to model sequential electron tunneling. Built on **JAX**, it utilizes XLA compilation and hardware-accelerated vectorization to process high-resolution Coulomb diamond stability diagrams instantly.

## 🚀 Features

* **Microscopic ED Engine (`DerivedParameters`):** Automatically constructs many-body Hamiltonians using Exact Diagonalization. Fully supports single-particle orbitals, intra/inter-orbital Coulomb interactions, exchange/pair-hopping, and Zeeman splitting.
* **Macroscopic Phenomenological Engine (`DirectParameters`):** Bypass microscopic Hamiltonian setup to manually enforce arbitrary many-body state energies, spin labels, and transition spin degeneracies.
* **JAX Backend:** The core Master equation solver uses `@jax.jit` and `jax.vmap` to solve steady-state probabilities across 2D voltage meshes.
* **Memory-Safe Execution:** A hybrid chunking algorithm streams data directly to the CPU host, bypassing GPU VRAM bottlenecks on high-resolution grids.