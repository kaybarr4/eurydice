import numpy as np
from abc import ABC, abstractmethod


##### KERNEL HELPER FUNCTION ####
def build_covariance_matrix(kernel, p1, p2):
    """
    Computes the covariance matrix between two sets of points given a kernel object

    Args:
        p1 (np.array): array of first set of points, length N
        p2 (np.array): array of second set of points, length M
        kernel (object): describes the covariance between two points

    Return:
        cov_matrix (np.array): an array of size NxM filled with values computed by kernel
    """
    cov_matrix = np.zeros((len(p1), len(p2)))

    for i in range(len(p1)):
        for j in range(len(p2)):
            cov_matrix[i][j] = kernel.calc_value(p1[i], p2[j])

    return cov_matrix


##### ABSTRACT KERNEL CLASS #####
class Kernel(ABC):
    """
    An abstract base class to set up any kernel functions to use in CrossValidation object.
    """

    def calc_covmatrix(self, p1, p2):
        """Evaluate the covariance matrix with the kernel function at two given sets of independent coordinates."""
        return build_covariance_matrix(self, p1, p2)

    @abstractmethod
    def calc_value(self, x1, x2):
        """Evaluate the kernel at a pair of input coordinates"""
        pass


##### DEFAULT KERNEL: SQ EXP #####
class defaultKernel(Kernel):
    """
    A default kernel that CrossValidation will use if no other kernel function is given.
    Has an amplitude of 1 and a characteristic length-scale of 5 (days)
    """

    def __init__(self):
        self.amplitude = 1
        self.lengthscale = 5

    def calc_value(self, x1, x2):
        return self.amplitude**2 * np.exp(
            -((x1 - x2) ** 2) / (2 * (self.lengthscale) ** 2)
        )
