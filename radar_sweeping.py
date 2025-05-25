import numpy as np
import matplotlib.pyplot as plt
from main import simulation_box
from main import ray_tracing
from quantum_entanglement import quantum_state

dt = 0.01

ray_propagation = ray_tracing(20, 0.005, np.array([5,-5]))
angles = np.linspace(180, 0, 100)


class radar_sweep():

    def __init__(self, detector_max_x, detector_min_x, r_0):

        self.r_0 = r_0
        self.detector_max_x = detector_max_x
        self.detector_min_x = detector_min_x
        self.state = quantum_state()

    def sweep(self, beta, positions, radii, N_pairs=20, N_noise_per_angle=40):
        """
        Simulates a radar sweep at a given angle.

        beta: angle of the radar sweep in radians
        positions: list of object positions
        radii: list of object radii
        N_pairs: number of entangled photon pairs to simulate
        N_noise_per_angle: number of noise events to simulate

        Returns:
        A list of detected joint states with their arrival times.
        """
        detected_joint_states = []  # list of [rho_joint, arrival_time]
        idler_marginals = []  # list of [rho_idler, arrival_time]

        direction = np.array([np.cos(beta), np.sin(beta)])   # normalized direction of the ray

        for _ in range(N_pairs):     # for N photon pairs along each sampled ray
            collision_point, obj = ray_propagation.collision(direction, positions, radii)      # calculate collision point if it exists

            if collision_point is None:    # if it doesn't, pass
                continue

            normal = (positions[obj] - collision_point)      # normal to the objects surface, (all objects are spheres)
            normal /= np.linalg.norm(normal)                 # normalized vector
            incidence_angle = np.dot(direction, normal)      # angle of incidence

            distance = np.linalg.norm(collision_point - self.r_0)     # distance b/w radar and the collided object
            arrival_time = 2 * distance                      # range

            U = self.state.unitary_operator(incidence_angle, n1=1, n2=1.05)   # Unitary operator based on fresnel equations
            U_noise = self.state.noise_operator()            # random noise operator to simulate realistic reflection
            U_total = U @ U_noise

            psi = self.state.bell_state(np.array([1, 0]), np.array([0, 1]))     # initial entangled bell state
            rho = np.outer(psi, psi.conj())                  # initial density matrix

            U_full = np.kron(U_total, np.eye(2))
            rho_reflected = U_full @ rho @ U_full.conj().T   # new density matrix

            d_reflected = direction - 2 * np.dot(direction, normal) * normal    # reflected ray direction
            n = -collision_point[1] / d_reflected[1]
            hit_x = collision_point[0] + n * d_reflected[0]

            if self.detector_min_x <= hit_x <= self.detector_max_x:     # if detected by the detector
                detected_joint_states.append([rho_reflected, arrival_time])

                # Properly store only the idler matrix
                rho_tensor = rho_reflected.reshape(2, 2, 2, 2)
                rho_idler = np.trace(rho_tensor, axis1=0, axis2=2)
                idler_marginals.append(rho_idler)

        # Thermal noise simulation (outside of angle loop)
        if len(idler_marginals) > 0:
            for _ in range(N_noise_per_angle):
                phi = self.random_pure_qubit()    # random single photon state
                rho_noise = np.outer(phi, phi.conj())    # its density matrix

                rho_idler = idler_marginals[np.random.randint(len(idler_marginals))]  # randomly chosen idler photon
                rho_noise_joint = np.kron(rho_noise, rho_idler)       # denisty martix for the noisy photon pair

                arrival_time = np.random.uniform(0, 2 * np.sqrt(2) * 10)
                detected_joint_states.append([rho_noise_joint, arrival_time])

        else:
            for _ in range(N_noise_per_angle):
                phi = self.random_pure_qubit()
                rho_noise = np.outer(phi, phi.conj())

                psi_idler = self.random_pure_qubit()
                rho_idler = np.outer(psi_idler, psi_idler.conj())

                rho_noise_joint = np.kron(rho_noise, rho_idler)

                arrival_time = np.random.uniform(0, 2 * np.sqrt(2) * 10)
                detected_joint_states.append([rho_noise_joint, arrival_time])

        return detected_joint_states

    def random_pure_qubit(self):
        v = np.random.randn(2) + 1j * np.random.randn(2)
        return v / np.linalg.norm(v)









































