# Quantum Harmonic Oscillator Simulation

This Python code simulates the time evolution of a quantum harmonic oscillator using the Suzuki-Trotter decomposition method. It's designed for educational and research purposes in quantum mechanics and quantum computing.

# Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib

# Installation

No special installation is required beyond the Python environment. Ensure you have the required libraries (NumPy, SciPy, and Matplotlib) installed in your Python environment.

# Usage

Define the number of spins (Nspins) and the frequency (omega) for the harmonic oscillator system.
Set the time step (dt) and the number of steps (steps) for the simulation.
The main functions are:

- `create_adag(Nspins)`: Creation operator.
- `create_aa(Nspins)`: Annihilation operator.
- `create_hamiltonian(Nspins, omega)`: Hamiltonian of the system.
- `suzuki_trotter_evolution_operator(H, dt)`: Time evolution operator using Suzuki-Trotter decomposition.
- `time_evolve_state(state, U)`: Evolves a given quantum state.
- `calculate_overlap(state1, state2)`: Calculates the overlap between two states.
- `calculate_mean_occupation_number(state, Nspins)`: Calculates the mean occupation number.
- `track_evolution(initial_state, H, Nspins, steps, delta_t)`: Tracks the evolution of the initial state over time.
  The script ends with plotting the time evolution of the overlap with the initial state and the mean occupation number.
  Example
  The provided example initializes a quantum harmonic oscillator with 10 spins and tracks its time evolution using a Suzuki-Trotter decomposition with a first-order approximation.

Notes
The implementation assumes a first-order Suzuki-Trotter decomposition. Higher-order decompositions are not implemented.
The code is optimized for clarity and educational purposes, not for computational efficiency or scalability.
