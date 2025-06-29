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

ray_tracing_instance = ray_tracing(20,0.025, [5,-5])
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







plt.figure(figsize=(8, 6))
plt.scatter(coordinates[:, 0], coordinates[:, 1], color='red', s=100, label='Detected Objects',
            alpha=0.8, edgecolors='white', linewidth=1)
plt.scatter(positions[0,:, 0], positions[0,:, 1], color='blue', s=80, marker='x',
            label='True Object Locations', alpha=0.6)
plt.scatter(5, -5, color='green', s=200, marker='^',
            label='Radar Origin', zorder=5)

plt.title('Detected Objects vs. True Locations (Cartesian)')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid(True, linestyle=':', alpha=0.6)
plt.axis('equal') # Important for true spatial representation
plt.legend()
plt.tight_layout()
plt.show()




plt.figure(figsize=(7, 5))
plt.hist(plot, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(0.8, color='red', linestyle='--', label='Detection Threshold (0.8)')
plt.title('Distribution of P(Φ1) for All Detections')
plt.xlabel('Probability of first Bell State Φ1')
plt.ylabel('Frequency')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



plt.figure(figsize=(8,8)) # Increased size for better readability with more elements
ax_polar = plt.subplot(111, polar=True)

# Plot detected objects
if detected_objects:
    detected_ranges = [obj[0] for obj in detected_objects]
    detected_angles = [obj[1] for obj in detected_objects] # Angles are already in radians

    ax_polar.scatter(detected_angles, detected_ranges, color='red', s=100,
                     label='Detected Objects', alpha=0.8, edgecolors='white', linewidth=1, zorder=3)

# Plot true initial object positions
true_polar_ranges = []
true_polar_angles = []

# Assuming 'positions' are absolute Cartesian coordinates and radar_origin is also Cartesian.
# Convert true object Cartesian coordinates to polar coordinates relative to the radar_origin.
for p in positions[0, :]:
    dx = p[0] - 5
    dy = p[1] + 5
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx) # Returns angle in radians

    true_polar_ranges.append(r)
    true_polar_angles.append(theta)

ax_polar.scatter(true_polar_angles, true_polar_ranges, color='blue', s=80, marker='x',
                 label='True Object Locations', alpha=0.6, zorder=2)


ax_polar.set_theta_zero_location('N') # Set 0 degrees to North (top)
ax_polar.set_theta_direction(-1)     # Clockwise direction
ax_polar.set_title('Quantum Radar Detections vs. True Locations (Polar View)', va='bottom')
# Dynamically adjust radial limits to fit all data
max_overall_range = 0
if detected_objects:
    max_overall_range = max(max(detected_ranges), max(true_polar_ranges))
else:
    max_overall_range = max(true_polar_ranges)
ax_polar.set_ylim(0, max_overall_range * 1.2 if max_overall_range > 0 else 10)

ax_polar.set_rlabel_position(-22.5)
ax_polar.legend(loc='lower left', bbox_to_anchor=(1.05, 0)) # Place legend outside
plt.tight_layout()
plt.show()




