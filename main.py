import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, expm

def create_adag(Nspins):
    """ Create creation operator for Nspins quantum harmonic oscillators. """
    return np.diag([np.sqrt(i) for i in range(1, Nspins)], -1)

def create_aa(Nspins):
    """ Create annihilation operator for Nspins quantum harmonic oscillators. """
    return np.diag([np.sqrt(i) for i in range(1, Nspins)], 1)

def create_hamiltonian(Nspins, omega):
    """ Create the Hamiltonian for a quantum harmonic oscillator. """
    adag = create_adag(2**Nspins)
    aa = create_aa(2**Nspins)
    constant_term = 0.5 * np.eye(2**Nspins)
    return omega * (np.dot(adag, aa) + constant_term)

def suzuki_trotter_evolution_operator(H, dt, order=1):
    """
    Compute the Suzuki-Trotter approximation of the time evolution operator.
    For simplicity, this implementation assumes a first-order decomposition.

    Parameters:
    H (numpy.ndarray): Hamiltonian of the system.
    dt (float): Time step for the evolution.
    order (int): Order of the Suzuki-Trotter decomposition.

    Returns:
    numpy.ndarray: Approximated time evolution operator.
    """
    if order == 1:
        # First-order Suzuki-Trotter decomposition
        return expm(-1j * H * dt)
    else:
        # Higher-order decompositions can be implemented here
        raise NotImplementedError("Higher-order Suzuki-Trotter decompositions are not yet implemented.")


def time_evolve_state(state, U):
    """ Apply the time evolution operator U to the state. """
    return np.dot(U, state)

def calculate_overlap(state1, state2):
    """ Calculate the overlap between two quantum states. """
    return np.dot(np.conj(state1.T), state2)

def calculate_mean_occupation_number(state, Nspins):
    """
    Calculate the mean occupation number for a quantum state.

    Parameters:
    state (numpy.ndarray): The quantum state for which to calculate the occupation number.
    Nspins (int): Number of spins (oscillators) in the system.

    Returns:
    float: The mean occupation number for the given state.
    """
    adag = create_adag(2**Nspins)
    aa = create_aa(2**Nspins)
    number_operator = np.dot(adag, aa)
    
    # Calculate the expectation value of the number operator in the given state
    mean_occupation = np.dot(state.conj().T, np.dot(number_operator, state)).real
    return mean_occupation

def track_evolution(initial_state, H, Nspins, steps, delta_t):
    """ Track the evolution of the initial state over time, including the mean occupation number. """
    results = []
    current_state = initial_state
    U = suzuki_trotter_evolution_operator(H, delta_t)  # Compute the evolution operator once

    for i in range(steps + 1):
        if i > 0:
            current_state = time_evolve_state(current_state, U)  # Evolve the state by delta_t each step
        overlap = calculate_overlap(initial_state, current_state)
        mean_occupation = calculate_mean_occupation_number(current_state, Nspins)  # Calculate mean occupation number
        results.append((i * delta_t, np.real(overlap), mean_occupation))  # Store time, overlap, and occupation number
    return results

# Define system parameters
Nspins = 10
omega = 1.0
dt = 0.1  # time step
steps = 100

# Create Hamiltonian
H = create_hamiltonian(Nspins, omega)

# Find eigenvalues and eigenvectors
eigenvalues, eigenvectors = eigh(H)

# Initialize the ground state
initial_state = eigenvectors[:, 1]

# Track the time evolution of the initial state
evolution_data = track_evolution(initial_state, H,Nspins, steps, dt)

# Extract times and overlap values
times, overlaps, mean_occupations = zip(*evolution_data)

# Plot the overlaps
plt.plot(times, overlaps, 'ro-', label='Time evolution of overlap')
plt.xlabel('Time')
plt.ylabel('Overlap with initial state')
plt.title('Time Evolution of the Quantum Harmonic Oscillator')
plt.legend()
plt.savefig('time_evolution_overlap.jpg')
plt.close()

plt.plot(times, mean_occupations, 'ro-', label='Time evolution of occupancy')
plt.xlabel('Time')
plt.ylabel('N')
plt.title('Time Evolution of the Quantum Harmonic Oscillator')
plt.legend()
plt.savefig('time_evolution_occupancy.jpg')
plt.close()