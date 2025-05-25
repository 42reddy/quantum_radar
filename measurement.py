import numpy as np
from radar_sweeping import radar_sweep
import matplotlib.pyplot as plt
from main import simulation_box
from main import ray_tracing
from quantum_entanglement import quantum_state


dt = 0.01

simulation_instance = simulation_box()
positions, velocities = simulation_instance.generate(10, 1000, 10)
radii = simulation_instance.radius(10)
state = quantum_state()

angles = np.linspace(180, 0, 1000)

ray_tracing_instance = ray_tracing(20,0.005, [5,-5])
radar_sweep_instance = radar_sweep(10, 0, [5,-5])


def measurement(rho):
    """
    Measures a two-qubit state in the Bell basis.

    rho: density matrix of the two-qubit system

    Returns:
    A list of probabilities for each of the four Bell states.
    """

    H = np.array([1,0])
    V = np.array([0,1])

    phi_1 = (np.kron(H,H) + np.kron(V, V)) / np.sqrt(2)
    phi_2 = (np.kron(H, H) - np.kron(V,V)) / np.sqrt(2)
    psi_1 = (np.kron(H,V) + np.kron(V,H)) / np.sqrt(2)
    psi_2 = (np.kron(H,V) - np.kron(V,H)) / np.sqrt(2)

    bell_states = [phi_1, phi_2, psi_1, psi_2]

    prob = [np.real(np.dot(psi.conj().T, np.dot(rho, psi))) for psi in bell_states]
    prob = np.maximum(prob, 0)
    prob /= np.sum(prob)

    return prob


all_detections = []

# Loop over each radar sweep angle
for i in angles:
    # Sample 20 angles around 'i' with small Gaussian noise
    sampled_angles = np.random.normal(i, 0.5, 20)
    sampled_angles = np.deg2rad(sampled_angles)  # convert to radians

    # Sweep radar for each sampled angle
    for beta in sampled_angles:
        results = radar_sweep_instance.sweep(beta, positions[0], radii, 20, 20)

        # Store the reflected state, arrival time, and sweep angle
        all_detections.extend([(res[0], res[1], beta) for res in results])

probs = []

# Measure each detected state in the Bell basis
for state in all_detections:
    density = state[0]  # joint density matrix
    range = state[1] / 2  # convert round-trip time to one way distance
    angle = state[2]  # angle of detection

    # Store Bell state probabilities, range, and angle
    probs.append([measurement(density), range, angle])

plot = []

# Extract probability of the first Bell state (phi_1)
for i in probs:
    plot.append(i[0][0])

detected_objects = []

# Select only those detections where the first Bell state probability > 0.8
for prob in probs:
    if prob[0][0] > 0.8:
        detected_objects.append([prob[1], prob[2]])  # store range and angle


coordinates = np.array([
    [5 + r * np.cos(theta), -5 + r * np.sin(theta)]
    for r, theta in detected_objects
])

coordinates = np.array(coordinates)
plt.scatter(coordinates[:,0], coordinates[:,1])
plt.scatter(positions[0,:,0], positions[0,:,1])
plt.show()




