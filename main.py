import numpy as np

dt = 0.01

class simulation_box():

    def __int__(self):

        self.shape = 'sphere'

    def generate(self,n_objects, N, L):
        """
        Simulates the motion of n_objects in a 2D periodic box.
        Parameters:
        n_objects : Number of moving objects.
        N : Number of time steps in the simulation.
        L : Size of the 2D periodic box (assumed square of side L).

        Returns:
        r : Array of shape (N, n_objects, 2) of object positions at each time step.
        v : Array of shape (n_objects, 2) Constant velocity vector for each object.
        """

        r = np.zeros((N, n_objects, 2))
        v = np.random.rand(n_objects,2) * L
        r[0] = np.random.rand(n_objects, 2) * L

        for i in range(N-1):

            r[i+1] = r[i] + dt * v

            r[i+1] %= L

        return r, v

    def radius(self, n_objects):
        """
        Generates random positive radii for objects using a normal distribution.
        Parameters:
        n_objects : Number of objects to generate radii for.

        Returns:
        radius : Absolute values of normally distributed random radii (mean=0, std=0.25).
        """
        radius = np.abs(np.random.normal(0, 0.25, n_objects))

        return radius




class ray_tracing:
    def __init__(self, n_rays, sigma, r_0):
        """
        Initializes the ray source.
        Parameters:
        n_rays : Number of rays to simulate.
        sigma : Standard deviation of angular spread in degrees.
        r_0 : Origin of the rays/radar position.
        """

        self.n_rays = n_rays  # number of rays in the cone
        self.sigma = sigma  # standard deviation in degrees
        self.r_0 = np.array(r_0)   # radar position

    def directions(self, theta_deg):
        sampled_angles = np.random.normal(theta_deg, self.sigma, size=self.n_rays)  # sample a cone of rays sampled around the input ray
        return np.deg2rad(sampled_angles)

    def collision(self, ray_direction, positions, r_objects):
        """
        Calculates the first collision point of a ray with objects.

        Parameters:
        ray_direction : Unit direction vector of the ray.
        positions : Array of shape (n_objects, 2), positions of objects.
        r_objects : Radii of the circular objects.

        Returns: (collision_point, object_index) if a collision occurs,
            otherwise (None, None).
        """
        closest_t = np.inf
        collision_point = None
        collided_object = None

        for i in range(len(positions)):
            o = self.r_0 - positions[i]                        # the ray direction vector
            d = ray_direction / np.linalg.norm(ray_direction)  # normalize to unit direction
            r = r_objects[i]      # radius of the object i

            a = 1
            b = 2 * np.dot(o, d)
            c = np.dot(o, o) - r ** 2

            det = b ** 2 - 4 * a * c     # calulate the determinant

            if det >= 0:                 # if a collision occurs/ solution exists even if it is complex
                sqrt_det = np.sqrt(det)
                t1 = (-b - sqrt_det) / 2
                t2 = (-b + sqrt_det) / 2

                for t in [t1, t2]:       # find the first point of impact and if the solution is real
                    if t > 0 and t < closest_t:
                        closest_t = t
                        collision_point = self.r_0 + t * d
                        collided_object = i

        if collision_point is not None:
            return collision_point, collided_object
        else:
            return None, None












































