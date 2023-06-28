from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator

def counts_plot(qc, backend, shots, multiplier):
    sim_backend = AerSimulator.from_backend(backend)
    

    # Define transpiler options
    transpile_options = {
        'optimization_level': 0,  # Disable all optimization passes
        'basis_gates': ['u1', 'u2', 'u3', 'cx']  # Specify the basis gates to be used
    }

    # Transpile the circuit with the specified options
    tcirc = transpile(qc, sim_backend, **transpile_options)
    # Execute noisy simulation and get counts
    result_noise = sim_backend.run(tcirc, shots=shots).result()
    counts_noise = result_noise.get_counts(0)
    plot_histogram(counts_noise,
                title=f"Counts for 3-qubit GHZ state with device noise model for repetitions of the CX of {multiplier} times")
    

def get_GHZ_fidelity(qc, backend, shots):

    sim_backend = AerSimulator.from_backend(backend)   

    # Define transpiler options
    transpile_options = {
        'optimization_level': 0,  # Disable all optimization passes
        'basis_gates': ['u1', 'u2', 'u3', 'cx']  # Specify the basis gates to be used
    }

    # Transpile the circuit with the specified options
    tcirc = transpile(qc, sim_backend, **transpile_options)
    # Execute noisy simulation and get counts
    result_noise = sim_backend.run(tcirc, shots=shots).result()
    counts_noise = result_noise.get_counts(0)

    N = qc.num_qubits  # Replace this with your number of qubits
    ideal_counts = {"0" * N: 500, "1" * N: 500}  # Replace 500 with your expected counts

    # Get the total number of shots from your noisy simulation
    total_shots = sum(counts_noise.values())

    # Normalize the ideal_counts to the same total number of shots
    normalized_ideal_counts = {state: count * total_shots / sum(ideal_counts.values()) for state, count in ideal_counts.items()}

    # Calculate the fidelity
    fidelity = sum(min(counts_noise.get(state, 0), count) for state, count in normalized_ideal_counts.items()) / total_shots

    return fidelity