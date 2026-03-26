# Minimal Quantum Plus Gravity Model

This note captures the clearest version of the question:

> What is the smallest system where a computer model can reproduce measurable quantum behavior and measurable gravitational effects using today's physics, even though we still do not have one complete experimentally confirmed theory of quantum gravity?

## Important Clarification

If "perfect" is taken literally, then there is no example that matches the strongest version of the question. A perfect computer simulation is already a mathematical description. So the accurate statement is not:

- there is no math at all that describes both quantum and gravitational effects

Instead, it is:

- there is no final, experimentally confirmed, fundamental unified theory of quantum gravity

We do have mathematical frameworks that describe measurable outputs very well in regimes where quantum mechanics and gravity both matter.

## Short Answer

There are two useful answers, depending on what "simplest" means.

### 1. Simplest overall practical model

A single neutron or atom in Earth's gravitational field.

Examples:

- a neutron quantum bouncer
- an atom interferometer

In these systems:

- quantum mechanics describes the wavefunction, superposition, interference, and measurement probabilities
- gravity enters as a classical gravitational potential or proper-time shift

This is real physics and experimentally meaningful, but it is not a deep quantum-gravity case because gravity is treated as a classical background.

### 2. Simplest genuine quantum-plus-self-gravity model

Two small masses, each in a spatial quantum superposition, interacting only through gravity.

This is the cleanest minimal model if the goal is to see both:

- genuinely quantum behavior
- gravity produced by the masses themselves affecting the measurable outcome

This is the smallest clean setup where gravity can generate a quantum phase difference and, in principle, entanglement.

## Minimal Genuine Model

Take two masses, `m1` and `m2`. Each mass has two possible positions:

- `|L>`
- `|R>`

The joint basis states are:

- `|LL>`
- `|LR>`
- `|RL>`
- `|RR>`

### Initial state

Each mass starts in an equal superposition:

```text
|psi(0)> = (|L> + |R>)/sqrt(2)  ⊗  (|L> + |R>)/sqrt(2)
```

### Gravitational interaction

For each branch `ab`, where `a,b ∈ {L,R}`, the gravitational potential energy is:

```text
U_ab = -G m1 m2 / r_ab
```

where `r_ab` is the distance between the two masses in that branch.

### Hamiltonian

During the interaction, the Hamiltonian is diagonal in the branch basis:

```text
H = Σ_ab U_ab |ab><ab|
```

### Time evolution

Each branch picks up a different phase:

```text
|psi(t)> = 1/2 Σ_ab exp(-i U_ab t / hbar) |ab>
```

This means gravity changes the relative phases between branches.

After the interaction, recombine the branches and measure:

- interference probabilities
- correlation functions
- entanglement witnesses

## What Each Theory Contributes

### Quantum mechanics gives

- superposition
- phase evolution
- interference
- measurement probabilities
- entanglement structure

### Gravity or weak-field relativity gives

- the branch-dependent energies `U_ab`
- equivalently, branch-dependent proper-time shifts

The same phase can also be written in relativistic language as:

```text
phi_ab = m c^2 tau_ab / hbar
```

So in this weak-field regime:

- Newtonian gravity gives the potential-energy picture
- weak-field general relativity gives the proper-time picture
- both describe the same measurable phase shift

## Why This Is the Minimal Interesting Case

A one-particle model is simpler, but then gravity is just an external background field.

Two particles are the minimal clean case where:

- the masses themselves source the gravitational effect
- the gravitational effect changes a quantum observable

That makes the two-mass superposition model the simplest genuine example matching the spirit of the question.

## Even Simpler Equation If You Only Want Gravity as Background

For a neutron or atom above a mirror in Earth's gravity, a standard model is:

```text
i hbar ∂psi/∂t = [-(hbar^2/2m) ∂^2/∂z^2 + mgz] psi
```

with a boundary condition at the mirror.

This produces real measurable outputs, but it is not a full self-gravitating quantum system.

## Best Summary

- Simplest overall: one particle in a classical gravitational field
- Simplest genuine quantum-plus-self-gravity model: two masses in spatial superposition interacting gravitationally

## Final Takeaway

The strongest literal version of the original question is too strong. It is not that there is "no math" describing both aspects at all. Rather:

- we can model measurable outputs in regimes where quantum mechanics and gravity both matter
- but we do not yet have one complete, experimentally confirmed, fundamental theory of quantum gravity

That is why the two-mass superposition model is the cleanest minimal example.
