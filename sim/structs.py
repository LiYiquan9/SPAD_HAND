import numpy as np


def rotate_vec2vec(v_in, v_out):
    """ 
    Calculate the rotation matrix for rotating one vector into another 
    using Rodrigues' formula.

    The rotation axis is the cross product of the two vectors. The rotation 
    angle is the arccosine of the dot product of the two vectors.

    Given the rotation axis w = [x, y, z] (in unit vector) and the rotation 
    angle a, the rotation matrix R is given by

      | 1+(1-cosa)*(x**2-1)   -z*sina+(1-cosa)*x*y  y*sina+(1-cosa)*x*z  |
    R = | z*sina+(1-cosa)x*y    1+(1-cosa)*(y**2-1)   -x*sina+(1-cosa)*y*z |
      | -y*sina+(1-cosa)*x*z  x*sina+(1-cosa)*y*z   1+(1-cosa)*(z**2-1)  |

    Args:
    v_in (float array, (3,)): unit vector before rotation.
    v_out (float array, (3,)): unit vector after rotation.

    Returns:
    R (float array, (3, 3)): rotation matrix.
    """
    # normalize the vectors
    v_in /= np.linalg.norm(v_in)
    v_out /= np.linalg.norm(v_out)
    if (v_in == v_out).all():
        return np.eye(3)

    w = np.cross(v_in, v_out)
    w /= np.linalg.norm(w)               # normalized rotation axis
    x, y, z = w

    a = np.arccos(np.dot(v_in, v_out))   # rotation angle
    cosa, sina = np.cos(a), np.sin(a)

    # calculate rotation matrix elements
    R = np.empty((3, 3))
    R[0, 0] = 1. + (1. - cosa) * (x ** 2 - 1.)
    R[0, 1] = -z * sina + (1. - cosa) * x * y
    R[0, 2] = y * sina + (1. - cosa) * x * z
    R[1, 0] = z * sina + (1. - cosa) * x * y
    R[1, 1] = 1. + (1. - cosa) * (y ** 2 - 1.)
    R[1, 2] = -x * sina + (1. - cosa) * y * z
    R[2, 0] = -y * sina + (1. - cosa) * x * z
    R[2, 1] = x * sina + (1. - cosa) * y * z
    R[2, 2] = 1. + (1. - cosa) * (z ** 2 - 1.)
    return R


class Anchor:

    def __init__(self, loc, normal, fov, is_area=False):
        """
        Args:
            loc (float array, (3, )): anchor location in world frame.
            normal (float array, (3, )): normal vector.
            fov (float): field of view (unit: degree).
            is_area (bool): whether anchor has infinitisimal surface area.
        """

        assert fov >= 0 and fov <= 180, f"invalid field of view: {fov:f}"

        loc = np.array(loc, dtype=np.float32)
        normal = np.array(normal, dtype=np.float32)

        self.fov = fov * np.pi / 180  # degree >> radian
        self.solid_angle = 2 * np.pi * (1 - np.cos(self.fov / 2))
        self.loc = loc
        self.normal = normal / np.linalg.norm(normal)
        self.is_area = is_area

        p_range = (0.0, 2 * np.pi)                  # phi
        cost_range = (np.cos(self.fov / 2), 1.0)    # cos(theta)

        # anchor-to-world transformation
        R = rotate_vec2vec(np.array([0.0, 0.0, 1.0]), self.normal)
        self.a2w = np.hstack((R, self.loc[:, None]))

    def sample_fibonacci(self, num_samples):
        """ Uniform sampling on Fibonacci spiral. """

        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        idx = np.linspace(0, num_samples - 1, num_samples) + 0.5
        p = golden_angle * idx
        cosa = np.cos(self.fov / 2)
        cost = 1.0 - (1.0 - cosa) * idx / num_samples

        sinp, cosp, sint = np.sin(p), np.cos(p), np.sqrt(1.0 - cost ** 2)
        d = np.vstack((sint * cosp, sint * sinp, cost)).T
        return d

    def sample_spherical(self, num_samples):
        """ Uniform sampling of spherical coordinates. """

        phi = 2 * np.pi * np.random.random(num_samples)
        sinp, cosp = np.sin(phi), np.cos(phi)
        cost = np.random.uniform(np.cos(self.fov / 2), 1.0, num_samples)
        sint = np.sqrt(1.0 - cost ** 2)
        d = np.vstack((sint * cosp, sint * sinp, cost)).T
        return d
    
    def sample_circle(self, num_samples):
        """ Uniform sampling of circle coordinates. (for hand reconstruction)"""
        
        phi = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        sinp, cosp = np.sin(phi), np.cos(phi)
        z = np.zeros_like(sinp)
        d = np.vstack((cosp, sinp, z)).T
        return d


class Light(Anchor):
    """ Light source object. """

    def __init__(self, loc, normal, fov, power=1, is_area=False):

        super().__init__(loc, normal, fov, is_area)

        self.power = power

    @staticmethod
    def pulse_kernel(fwhm, peak_power=1):
        """ Pulse shape modeled as a Gaussian function. """
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        def kernel(t):
            return peak_power * np.exp(-(t / sigma) ** 2)
        return kernel


class Camera(Anchor):
    """ Camera object. """

    def __init__(self, loc, normal, fov, num_bins, bin_size, is_area=False):
        """
        Args:
            loc (float array, (3, )): camera location in world frame.
            normal (float array, (3, )): normal vector.
            fov (float): field of view (unit: degree).
            num_bins (int): number of histogram bins.
            bin_size (float): distance covered by a bin (unit: meter).
            is_area (bool): whether anchor has infinitisimal surface area.
        """

        super().__init__(loc, normal, fov, is_area)
        
        self.num_bins = num_bins
        self.bin_size = bin_size