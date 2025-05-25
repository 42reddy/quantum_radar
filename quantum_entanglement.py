import numpy as np

class quantum_state():

    def __int__(self):

        pass


    def bell_state(self, H, V):

        return (np.kron(H,H) + np.kron(V,V)) / np.sqrt(2)

    def unitary_operator(self, theta_i, n1=1.0, n2=1.05):
        """
        Returns a 2x2 unitary matrix representing the effect of Fresnel reflection
        on H (horizontal) and V (vertical) polarizations.

        Parameters:
        - theta_i: angle of incidence (in radians)
        - n1: refractive index of medium 1 (incident side)
        - n2: refractive index of medium 2 (transmission side)
        """
        sin_t = (n1 / n2) * np.sin(theta_i)     # refracted direction

        # in case of total internal reflection,
        if np.abs(sin_t) > 1:
            # Approximate phase shift for total internal reflection
            return np.array([
                [np.exp(1j * np.pi / 2), 0],
                [0, np.exp(1j * np.pi / 4)]
            ])

        # Snell's law
        theta_t = np.arcsin(sin_t)

        # Fresnel reflection coefficients
        rs_num = n1 * np.cos(theta_i) - n2 * np.cos(theta_t)
        rs_den = n1 * np.cos(theta_i) + n2 * np.cos(theta_t)
        r_s = rs_num / rs_den

        rp_num = n2 * np.cos(theta_i) - n1 * np.cos(theta_t)
        rp_den = n2 * np.cos(theta_i) + n1 * np.cos(theta_t)
        r_p = rp_num / rp_den

        # Normalize to ensure unitarity, diagonal matrix with unit norm
        norm_rs = np.sqrt(np.abs(r_s) ** 2)
        norm_rp = np.sqrt(np.abs(r_p) ** 2)

        r_s /= norm_rs if norm_rs != 0 else 1
        r_p /= norm_rp if norm_rp != 0 else 1

        # Build 2x2 diagonal matrix
        U = np.array([
            [r_p, 0],
            [0, r_s]
        ])

        return U

    def noise_operator(self):
        """
        Generates a diagonal unitary operator representing phase noise
        in horizontal (H) and vertical (V) polarization components.

        Returns:
        noise_op : array of shape (2, 2)
            Unitary matrix with independent phase noise on each polarization.
        """
        phi_H = np.random.normal(0, 0.01)  # phase noise for H
        phi_V = np.random.normal(0, 0.01)  # phase noise for V

        return np.array([
            [np.exp(1j * phi_H), 0],
            [0, np.exp(1j * phi_V)]
        ])

    def fresnel_reflection(self, U, psi):

        psi_new = np.kron(U, np.identity(2)) @ psi    # updated combined wavefunction

        return psi_new









