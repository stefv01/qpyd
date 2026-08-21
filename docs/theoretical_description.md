# Theoretical Description

## General Framework

We model the dynamics of transport in single QD systems, comprised of two metallic leads (assumed to be electron reservoirs) weakly connected to a QD, which is electrostatically connected to a single gate, controlling the electrostatic energy of the QD and thus, its energy levels. An applied bias voltage controls the energy (chemical potential) of the electrons in the leads via the relation:

$$\mu _\alpha = \mu _\alpha^0 - e V$$

where $\alpha$ is the lead, $\mu _\alpha^0$ is the chemical potential of the lead at zero bias voltage, $V$, and $e$ is the electron charge. The components coupled to the QD shift its chemical potential linearly via the relation:

$$\mu _\mathrm{QD} = \mu _\mathrm{QD}^0 - e V _\mathrm{ext}$$

where $\mu _\mathrm{QD}^0$ is the intrinsic chemical potential and $V _\mathrm{ext} = \alpha _\mathrm{L} V _\mathrm{L} + \alpha _\mathrm{R} V _\mathrm{R} + \alpha _\mathrm{G} V _\mathrm{G}$. Here $\alpha _i$ are the lever arms which parameterize the capacitive couplings of the leads and the gate to the dot and $V _\alpha = - \mu _\alpha / e$, where $\alpha = \mathrm{L}, \mathrm{R}$ denote the left and right lead indices, respectively.

Electron transport is modeled via transitions between different charge states of the QD (e.g. $N \rightarrow N+1$ for an electron entering the QD, where $N$ is the number of electrons in the QD). To leading order, transport is allowed when the chemical potential of a transition is in the bias window defined by the chemical potential of the leads, i.e.:

$$\mu _\mathrm{L (R)} \leq \mu _\mathrm{QD} \leq \mu _\mathrm{R (L)}$$

In any other condition, transport is blocked to first order (Coulomb blockade). This describes the sequential electron transport (SET) regime. More details can be found in [[1]](#ref1).

## Hamiltonian Description

The nature of the transitions and base characteristics, such as the rate at which the transitions are happening, depend on the Hamiltonian description of the system. For the joint system (QD-leads), we assume the general Hamiltonian form:

$$\hat{H} = \hat{H} _\mathrm{leads} + \hat{H} _\mathrm{QD} + \hat{H} _\mathrm{int}$$

In second-quantization forms, these are:

**The Leads:**

$$\hat{H} _\mathrm{leads} = \sum _{\alpha \mathbf{k} \sigma} (\epsilon _{\alpha \mathbf{k}} - e V _\alpha) \ \hat{c}^\dagger _{\alpha \mathbf{k} \sigma} \hat{c} _{\alpha \mathbf{k} \sigma}$$

Here, $\hat{H} _\mathrm{leads}$ describes the non-interacting electrons in the metallic leads. The index $\alpha \in \{L, R\}$ denotes the left or right lead, $\mathbf{k}$ is the electron momentum (or wavevector), and $\sigma$ is the electron spin. The term $\epsilon _{\alpha \mathbf{k}}$ represents the energy of the continuous electron states in the leads, and $V _\alpha$ is the electrostatic potential applied to the lead (which shifts the chemical potential). The operators $\hat{c}^\dagger _{\alpha \mathbf{k} \sigma}$ and $\hat{c} _{\alpha \mathbf{k} \sigma}$ are the standard fermionic creation and annihilation operators for an electron in the leads. In a general approximation, the electrodes (leads) are assumed to be electron reservoirs at a constant temperature $T$ and chemical potential $\mu _\alpha$ and their electron energy eigenstates are distributed according to the Fermi-Dirac distribution:

$$f _\alpha (\epsilon) = [1 + \exp{(\beta (\epsilon - \mu _\alpha)}]^{-1} ,$$

where $\beta$ is the inverse temperature.

**The Quantum Dot:**

$$\hat{H} _\mathrm{QD} = \sum _{m} E _m | m \rangle \ \langle m |$$

The term $\hat{H} _\mathrm{QD}$ represents the isolated Quantum Dot. Here, we specify the notation for the many-body states as $\vert{}m \rangle \equiv \vert{}N _m, m \rangle$, where $N _m$ is the number of electrons. Instead of single-particle orbitals, this is written in the eigenbasis of $\hat{H} _\mathrm{QD}$, where $m$ indexes the many-body state and $E _m$ is its corresponding total eigenenergy. The external electrostatic environment is explicitly included here by shifting the total energy of the state proportionally to its electron number $N _m$, such that $E _m = E _m^0 - N _m e V _\mathrm{ext}$.

These states are also eigenstates of a set of operators that commute with the Hamiltonian. In the weak-coupling regime, where the QD is sufficiently isolated from the leads so the wavefunctions from each part do not significantly overlap and cause broadened states, the many-body states have almost conserved and well-defined quantities, such as charge number $N$, total spin $S$, and $S _z$. The set of eigenvalues of these operator sets represents the many-body index, i.e., $m \equiv \lbrace N, S, S _z, \ldots \rbrace$.

When dealing with microscopic descriptions of QD systems, the fundamental energy scales arise from quantum confinement effects and various internal interactions, such as electron-lattice and electron-electron interactions. Quantum confinement gives rise to quantized momenta. Depending on the confinement geometry and topological properties of the system, these define a set of quantization indices, represented collectively by an index $\nu$. These give rise to specific QD orbitals characterized by wavefunctions $\phi _\nu (\mathbf{r})$ and single-particle energies $\epsilon _\nu$. We refer to these as single-particle orbitals and energies because they represent the available states an individual electron can occupy, prior to considering electronic correlations between different orbitals. The electronic configurations of an $N$-electron QD can be described in second-quantization form by the Fock states $\vert{} n _{\nu _1}, n _{\nu _2}, \ldots \rangle$, where $\sum _\nu n _\nu = N$, establishing the basis of the single-particle picture.

However, single-particle states do not necessarily coincide with the resulting true many-body states of the QD. Electron-electron interactions introduce strong electronic correlations, meaning true many-body states are often represented by superpositions of entangled single-particle states. We can always transform between the two pictures by expanding the many-body states in the single-particle basis.

A single-particle description is highly dependent on the material under consideration, including its topology and the strength of interactions, and can vary considerably. In nanoscopic solid-state systems, electron-electron interactions are often significant and shape the transport properties accordingly. A general Hamiltonian describing a wide variety of systems in the single-particle picture, explicitly including electron-electron Coulomb interactions, takes the form [[2]](#ref2):

$$\hat{H} _\mathrm{QD} = \sum _{\nu \sigma} \epsilon _\nu \hat{d}^\dagger _{\nu \sigma} \hat{d} _{\nu \sigma} + \frac{1}{2} \sum _{\nu _1 \nu _2 \nu _3 \nu _4} \sum _{\sigma \sigma^\prime} V _{\nu _1 \nu _2 \nu _3 \nu _4} \ \hat{d}^\dagger _{\nu _1 \sigma} \hat{d}^\dagger _{\nu _2 \sigma^\prime} \hat{d} _{\nu _3 \sigma^\prime} \hat{d} _{\nu _4 \sigma}$$

The first term represents the non-interacting single-particle energies. The external potentials act directly on these single-particle levels by shifting them as $\epsilon _\nu = \epsilon _\nu^0 - e V _\mathrm{ext}$. The second term introduces the two-body Coulomb interactions, where the Coulomb matrix elements (integrals) are formally defined as:

$$V _{\nu _1 \nu _2 \nu _3 \nu _4} = \int d^3 \mathbf{r} _1 \int d^3 \mathbf{r} _2 \ \phi^* _{\nu _1} (\mathbf{r} _1) \phi^* _{\nu _2} (\mathbf{r} _2) \frac{e^2}{4 \pi \epsilon \vert{}\mathbf{r} _1 - \mathbf{r} _2\vert{}} \phi _{\nu _3} (\mathbf{r} _2) \phi _{\nu _4} (\mathbf{r} _1)$$

where $\epsilon$ is the effective dielectric constant of the material. In transport models, evaluating every possible scattering permutation is often physically unnecessary. Instead, we typically keep only the most dominant interaction terms:

1. **Direct Coulomb (Hartree) terms:** These occur when $\nu _1 = \nu _4$ and $\nu _2 = \nu _3$. This reduces the matrix element to $V _{\nu _1 \nu _2 \nu _2 \nu _1} \equiv U _{\nu _1 \nu _2}$, which represents the classical electrostatic repulsion between the charge densities of electrons residing in orbitals $\nu _1$ and $\nu _2$. The corresponding Hamiltonian term is:

$$\hat{H} _\mathrm{direct} = \frac{1}{2} \sum _{\nu _1 \nu _2} \sum _{\sigma \sigma^\prime} U _{\nu _1 \nu _2} \ \hat{d}^\dagger _{\nu _1 \sigma} \hat{d}^\dagger _{\nu _2 \sigma^\prime} \hat{d} _{\nu _2 \sigma^\prime} \hat{d} _{\nu _1 \sigma} = \frac{1}{2} \sum _{\nu _1 \neq \nu _2} U _{\nu _1 \nu _2} \ \hat{n} _{\nu _1} \hat{n} _{\nu _2} + \sum _\nu U _{\nu \nu} \ \hat{n} _{\nu \uparrow} \hat{n} _{\nu \downarrow}$$

Where $\hat{n} _\nu = \hat{n} _{\nu \uparrow} + \hat{n} _{\nu \downarrow}$ is the total number operator for the single-particle orbital $\nu$.

2. **Exchange (Fock) terms:** These occur when $\nu _1 = \nu _3$ and $\nu _2 = \nu _4$. This reduces the matrix element to $V _{\nu _1 \nu _2 \nu _1 \nu _2} \equiv J _{\nu _1 \nu _2}$, which represents the purely quantum-mechanical exchange energy that lowers the total energy due to the Pauli exclusion principle for indistinguishable fermions. The corresponding Hamiltonian term is:

$$\hat{H} _\mathrm{exchange} = \frac{1}{2} \sum _{\nu _1 \neq \nu _2} \sum _{\sigma \sigma^\prime} J _{\nu _1 \nu _2} \ \hat{d}^\dagger _{\nu _1 \sigma} \hat{d}^\dagger _{\nu _2 \sigma^\prime} \hat{d} _{\nu _1 \sigma^\prime} \hat{d} _{\nu _2 \sigma} = - \sum _{\nu _1 \neq \nu _2} J _{\nu _1 \nu _2} \left( \hat{\mathbf{S}} _{\nu _1} \cdot \hat{\mathbf{S}} _{\nu _2} + \frac{1}{4} \hat{n} _{\nu _1} \hat{n} _{\nu _2} \right)$$

Where $\hat{\mathbf{S}} _\nu$ denotes the vector spin operator for orbital $\nu$.

3. **Pair-hopping terms:** These occur when $\nu _1 = \nu _2$ and $\nu _3 = \nu _4$ (with $\nu _1 \neq \nu _3$, and requiring opposite spins $\sigma \neq \sigma^\prime$). This reduces the matrix element to $V _{\nu _1 \nu _1 \nu _3 \nu _3} \equiv J^\prime _{\nu _1 \nu _3}$, which describes the simultaneous tunneling (hopping) of a pair of electrons with opposite spins from orbital $\nu _3$ into orbital $\nu _1$. The corresponding Hamiltonian term is:

$$\hat{H} _\mathrm{pair-hopping} = \frac{1}{2} \sum _{\nu _1 \neq \nu _3} \sum _{\sigma \neq \sigma^\prime} J^\prime _{\nu _1 \nu _3} \ \hat{d}^\dagger _{\nu _1 \sigma} \hat{d}^\dagger _{\nu _1 \sigma^\prime} \hat{d} _{\nu _3 \sigma^\prime} \hat{d} _{\nu _3 \sigma} = \sum _{\nu _1 \neq \nu _3} J^\prime _{\nu _1 \nu _3} \ \hat{d}^\dagger _{\nu _1 \uparrow} \hat{d}^\dagger _{\nu _1 \downarrow} \hat{d} _{\nu _3 \downarrow} \hat{d} _{\nu _3 \uparrow}$$

Other permutations are possible as well, but these are usually highly suppressed and can be neglected in our calculations.

If the system is subjected to an external magnetic field $\mathbf{B}$, the spin degeneracy of the energy levels is lifted. This is introduced via an additional Zeeman Hamiltonian term that couples the magnetic field to the electron spin:

$$\hat{H} _\mathrm{Zeeman} = g \mu _B \mathbf{B} \cdot \sum _\nu \hat{\mathbf{S}} _\nu = \frac{1}{2} g \mu _B \sum _{\nu} \sum _{\sigma \sigma^\prime} (\mathbf{B} \cdot \vec{\sigma} _{\sigma \sigma^\prime}) \ \hat{d}^\dagger _{\nu \sigma} \hat{d} _{\nu \sigma^\prime}$$

where $g$ is the effective Landé g-factor of the material, $\mu _B$ is the Bohr magneton, and $\hat{\mathbf{S}} _\nu$ is the vector spin operator for orbital $\nu$. The expanded second-quantized form makes use of the vector of Pauli matrices $\vec{\sigma} = (\sigma _x, \sigma _y, \sigma _z)$.

**The Leads-QD Interaction**

$$\hat{H} _\mathrm{int} = \sum _{\alpha \mathbf{k} \nu \sigma} t _{\alpha \mathbf{k} \nu} \hat{c}^{\dagger} _{\alpha \mathbf{k} \sigma} \hat{d} _{\nu \sigma} + \mathrm{H.c.}$$

Finally, $\hat{H} _\mathrm{int}$ is the interaction (or tunneling) Hamiltonian connecting the leads to the QD. The parameter $t _{\alpha \mathbf{k} \nu}$ is the tunneling amplitude coupling the continuous lead states to the discrete QD orbitals. The operator $\hat{d} _{\nu \sigma}$ annihilates an electron in the QD, while $\hat{c}^{\dagger} _{\alpha \mathbf{k} \sigma}$ creates it in the lead. The Hermitian conjugate ($\mathrm{H.c.}$) describes the reverse process: an electron tunneling from the lead into the QD.

## Transition Rates

### SET Processes

The SET transition rates between two many-body states of the QD are given via Fermi's golden rule. For an electron entering the dot through lead $\alpha$ ($N _n = N _m + 1$), the transition rate is:

$$\gamma _{\alpha, m \rightarrow n} = \frac{1}{\hbar} \sum _{\nu} \gamma _{\alpha, \nu} \ g _{m n} (\nu) \ f _\alpha (\mu _{m n})$$

Conversely, for an electron leaving the dot ($N _n = N _m - 1$), the transition rate is:

$$\gamma _{\alpha, m \rightarrow n} = \frac{1}{\hbar} \sum _{\nu} \gamma _{\alpha, \nu} \ g _{n m} (\nu) \ ( 1 - f _\alpha (\mu _{n m}) )$$

As tunneling through both leads contributes to transport, the total rates are simply given by $\gamma _{m \rightarrow n} = \sum _\alpha \gamma _{\alpha, m \rightarrow n}$.

In both cases, the coupling of the orbital level $\nu$ to the (assumed non-magnetic) lead $\alpha$ is evaluated in the wide-band limit:

$$\gamma _{\alpha, \nu} = 2 \pi \rho _\alpha (\epsilon _F) \vert{}t _{\alpha, k _F, \nu}\vert{}^2$$

where $\rho _\alpha (\epsilon _F)$ is the density of states of the lead $\alpha$ evaluated at its Fermi energy.

The term $g _{m n} (\nu)$ involves the squared transition matrix elements representing the overlap probability of the initial and final states when an electron with spin $\sigma$ is added to or removed from the orbital $\nu$:

$$g _{m n} (\nu) = \sum _\sigma \left\vert{} \langle n \vert{} \hat{d} _{\nu \sigma}^\dagger \vert{} m \rangle \right\vert{}^2$$

When a degenerate state is in the bias window, the total transition rate is the summation of the rates for each individual state. The *degeneracy weights* $g _{m \lbrace n \rbrace}$ encode all the information about the degeneracy of the final state. The following table summarizes these weights for standard spin multiplets:

| Initial Multiplet (Spin $S _m$) | Final Multiplet (Spin $S _n$) | Degeneracy Weight $g _{m \lbrace n \rbrace}$ |
| --- | --- | --- |
| Singlet #1 ($S _m = 0$) | Doublet ($S _n = 1/2$) | $2$ |
| Doublet ($S _m = 1/2$) | Singlet #1 ($S _n = 0$) | $1$ |
| Singlet #2 ($S _m = 0$) | Doublet ($S _n = 1/2$) | $1/2$ |
| Doublet ($S _m = 1/2$) | Singlet #2 ($S _n = 0$) | $1$ |
| Doublet ($S _m = 1/2$) | Triplet ($S _n = 1$) | $3/2$ |
| Triplet ($S _m = 1$) | Doublet ($S _n = 1/2$) | $1$ |
| Triplet ($S _m = 1$) | Quartet ($S _n = 3/2$) | $4/3$ |
| Quartet ($S _m = 3/2$) | Triplet ($S _n = 1$) | $1$ |

### Relaxation/Excitation Processes

When the QD is coupled to external degrees of freedom (e.g. phonons on the substrate), this can induce transitions preserving the electron number ($N _m = N _n$). 

**Hamiltonian Model (`DerivedParameters`):**
In the exact diagonalization framework, the electron-phonon transition amplitude between an initial many-body state $|m\rangle$ and final state $|n\rangle$ is evaluated by coherently summing single-particle scattering events across all composite spin-orbital indices $a$ and $b$ [[3]](#ref3):

$$T_{mn} = \sum_{a, b} M_{ab} \langle n | \hat{c}^\dagger_a \hat{c}_b | m \rangle$$

where $M_{ab}$ represents the electron-phonon transition matrix element (`M_rel`). Because phonons couple to the overall charge density, the transition amplitudes of indistinguishable pathways interfere. The bare transition probability is given by:

$$\gamma_{mn} = |T_{mn}|^2 = \left| \sum_{a, b} M_{ab} \langle n | \hat{c}^\dagger_a \hat{c}_b | m \rangle \right|^2$$

This coherent summation allows the model to phenomenologically capture spin-relaxation when $M_{ab}$ is populated with off-diagonal spin-flip terms.

**Phenomenological Model (`DirectParameters`):**
In the direct macroscopic model, the evaluation of overlapping many-body wavefunctions is bypassed entirely. Instead, the bare transition probability $\gamma_{mn}$ is provided directly by the user as a macroscopic input rate (`gamma_rel`) governing the transitions between explicit states.

**Thermal Weighting:**
Regardless of whether $\gamma_{mn}$ is derived from exact diagonalization calculations or provided as a phenomenological parameter, the final transition rates must satisfy detailed balance with the thermodynamic environment. The rates to lower-energy or higher-energy states via the emission or absorption of a phonon, respectively, are given by:

* **Emission** ($E _m > E _n$): $\gamma _{m \rightarrow n}^\mathrm{rel} = \gamma _{m n} \left( n _B(\vert{}E _m - E _n\vert{}) + 1 \right)$
* **Absorption** ($E _m < E _n$): $\gamma _{m \rightarrow n}^\mathrm{rel} = \gamma _{m n} \ n _B(\vert{}E _m - E _n\vert{})$

where $n _B(E) = \left[ \exp(\beta E) - 1 \right]^{-1}$ is the Bose-Einstein distribution characterizing the thermal phonon bath at temperature $T$.

## Time Evolution and the Rate Equation

An analytical approach for modeling the dynamics of the system requires the evolution of the full density matrix $\hat{\rho}$. For sufficiently weak coupling between the QD and the metallic leads, the coherences of the density matrix are assumed to decay rapidly [[4]](#ref4).

Under this approximation, it is sufficient to model the evolution of the diagonal elements $P _m = \langle m \vert{} \hat{\rho}^\prime \vert{} m \rangle$. The reduced density matrix of the QD is given by:

$$\hat{\rho}^\prime = \mathrm{Tr} _\mathrm{leads} \hat{\rho}$$

The diagonal terms $P _m$ represent the occupation probabilities of these states. SET transport processes are described by the Pauli master equation:

$$\frac{d}{dt} \ P _m = \sum _{n} \left(P _n \gamma _{n \rightarrow m} - P _m \gamma _{m \rightarrow n}\right)$$

For a system in equilibrium, the occupation probabilities are derived from a steady-state solution ($dP _m / dt = 0$). To obtain a unique solution, one must also impose the normalization condition $\sum _m P _m = 1$. Given a steady-state solution $\lbrace P _m \rbrace$, one can then calculate the following macroscopic quantities:

* **Average Occupation:** $\langle \hat{N} \rangle = \sum _m N _m P _m$
* **Net Current through lead $\alpha$:**  $I _\alpha / e = - \sum _{m \neq n} \sum _n (-1)^{N _m - N _n} \delta _{N _m, N _n \pm 1} \ P _m \ \gamma _{\alpha, m \rightarrow n}$
* **Differential Conductance:** $G = \partial I / \partial V$

While the Pauli master equation accurately captures leading-order transport effects like sequential tunneling, it neglects higher-order processes. Phenomena such as cotunneling, which arise from stronger coupling to the leads, necessitate retaining the coherences of the reduced density matrix or employing higher-order approximations of the von Neumann equation. A comprehensive summary of these methods can be found in [[5]](#ref5).

## References

1. <a id="ref1"></a>**Thijssen, J. M., & Van der Zant, H. S. J. (2008).** *Charge transport and single-electron effects in nanoscale systems.* Physica Status Solidi (b), 245(8), 1455-1470. [DOI: 10.1002/pssb.200743470](https://doi.org/10.1002/pssb.200743470)
2. <a id="ref2"></a>**Bruus, H., & Flensberg, K. (2004).** *Many-Body Quantum Theory in Condensed Matter Physics: An Introduction.* Oxford University Press. [DOI: 10.1093/oso/9780198566335.001.0001](https://doi.org/10.1093/oso/9780198566335.001.0001)
3. <a id="ref3"></a>**Florescu, M., & Hawrylak, P. (2006).** *Spin relaxation in lateral quantum dots: Effects of spin-orbit interaction.* Physical Review B—Condensed Matter and Materials Physics, 73(4), 045304. [DOI: 10.1103/PhysRevB.73.045304](https://doi.org/10.1103/PhysRevB.73.045304)
4. <a id="ref4"></a>**Timm, C. (2008).** *Tunneling through molecules and quantum dots: Master-equation approaches.* Physical Review B, 77(19), 195416. [DOI: 10.1103/PhysRevB.77.195416](https://doi.org/10.1103/PhysRevB.77.195416)
5. <a id="ref5"></a>**Kiršanskas, G., Pedersen, J. N., Karlström, O., & Wacker, A. (2017).** *QmeQ 1.0: An open-source Python package for calculations of transport through quantum dot devices.* Computer Physics Communications, 221, 317-342. [DOI: 10.1016/j.cpc.2017.07.024](https://doi.org/10.1016/j.cpc.2017.07.024)