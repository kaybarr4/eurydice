import numpy as np
import cv_skeleton.kepler as kepler
import cv_skeleton.plot as plot


##### GPR HELPER FUNCTION ####
def build_covariance_matrix(p1, p2, kernel_function):
    """
    Computes the covariance matrix between two sets of points given a kernel function

    Args:
        p1 (np.array): array of first set of points, length N
        p2 (np.array): array of second set of points, length M
        kernel_function (func): describes the covariance between two points

    Return:
        cov_matrix (np.array): an array of size NxM filled with values computed by kernel
    """
    cov_matrix = np.zeros((len(p1), len(p2)))

    for i in range(len(p1)):
        for j in range(len(p2)):
            k = kernel_function(p1[i], p2[j])
            cov_matrix[i][j] = k

    return cov_matrix


##### DEFAULT KERNEL: SQ EXP #####
def default_sqexp_kernel(ti, tj):
    """
    A default kernel that CrossValidation will use if no other kernel function is given.
    Has an amplitude of 1 and a characteristic length-scale of 5 (days)

    Computes values for squared exponential covariance function.

    Args:
        ti, tj: any two times recorded within the data set

    Return:
        float
    """
    return np.exp(-((ti - tj) ** 2) / (2 * (5) ** 2))


##### CROSSVALIDATION OBJECT ######
class CrossValidation:
    """
    Args:
        times (np.array): array of timestamps from data
        rv_measurements (np.array): array of measured radial velocity values
        rv_errors (np.array): array of radial velocity errors
        orbit_params (tuple or list of tuples): (transit time, period, eccentricity, omega, keplerian semi-amplitude) per planet
        kernel_function (func or object): describes the covariance between two points. If none given, uses the default squared exponential kernel function
        gamma (float, optional): Defaults to 0.
        jitter (float, optional): Defaults to 0.

    """

    def __init__(
        self,
        times,
        rv_measurements,
        rv_errors,
        orbit_params,
        kernel_function=default_sqexp_kernel,
        gamma=0,
        jitter=0,
    ):
        self.times = times
        self.rv_measurements = rv_measurements
        self.rv_errors = rv_errors
        self.kernel_function = kernel_function
        self.orbit_params = orbit_params
        self.gamma = gamma
        self.jitter = jitter

    def split(self, train_split):
        """
        Splits data into a training set and test set to perform cross validation

        Args:
            train_split (float): number from 0 to 1 (inclusive), the fraction of data that will be split into the training set

        Returns:
            None. Saves the training mask and test mask within the object
        """

        if 0 <= train_split <= 1:
            pass
        else:
            raise ValueError(
                f"The train_split value must be between 0 and 1 (inclusive). Your input train_split value was {train_split}"
            )

        #### selecting training & test set at random
        n_data = len(self.times)
        training_mask = np.random.choice(
            n_data, size=int(train_split * n_data), replace=False
        )
        test_mask = np.setdiff1d(np.arange(n_data), training_mask)

        self.training_mask = training_mask
        self.test_mask = test_mask

    def calc_total_planetary_signal(self, times):
        """
        Computes the combined radial velocity signals of all transiting exoplanets in the star system

        Args:
            times (np.array): set of times to calculate the planetary signal for

        Returns:
            np.array: total radial velociy signal of the system at given times
        """

        ## check for multiple planets
        if isinstance(self.orbit_params, list):

            total_rv_signal = np.zeros(len(times))

            for planet_params in self.orbit_params:
                if len(planet_params) != 5:
                    raise ValueError(
                        f"Your orbital parameters should have 5 values per tuple: (transit time, period, eccentricity, omega, keplerian semi-amplitude). One of your input tuples only has {len(planet_params)} parameters given."
                    )
                total_rv_signal += kepler.calc_keplerian_signal(times, *planet_params)

            return total_rv_signal

        ## for singlular planets
        elif isinstance(self.orbit_params, tuple):
            if len(self.orbit_params) != 5:
                raise ValueError(
                    f"Your orbital parameters tuple should have 5 values: (transit time, period, eccentricity, omega, keplerian semi-amplitude). Your input orbit parameter tuple only has {len(self.orbit_params)} parameters given."
                )
            return kepler.calc_keplerian_signal(times, *self.orbit_params)

    def GP_predict(self, times_predict, include_keplerian=True):
        """
        Performs Gaussian Process regression conditioned on the training set of data to predict values at new points

        Args:
            times_predict (np.array): set of times for the GPR to predict values for
            include_keplerian (bool): whether or not to include planetary signal as mean function. Defaults to True.

        Returns:
            predictive_means, predictive_variances (tuple): mean and variance values for times you want the GP to predict
        """

        if include_keplerian:
            mean_function = self.calc_total_planetary_signal(self.times) + self.gamma
            mean_function_predict = (
                self.calc_total_planetary_signal(times_predict) + self.gamma
            )
        else:
            mean_function = np.zeros_like(self.times) + self.gamma
            mean_function_predict = np.zeros_like(times_predict) + self.gamma

        tot_train_error = np.sqrt(
            (self.rv_errors[self.training_mask]) ** 2 + self.jitter**2
        )

        K = build_covariance_matrix(
            self.times[self.training_mask],
            self.times[self.training_mask],
            self.kernel_function,
        ) + (tot_train_error**2) * np.identity(len(self.times[self.training_mask]))

        K_star = build_covariance_matrix(
            self.times[self.training_mask], times_predict, self.kernel_function
        )

        K_doublestar = build_covariance_matrix(
            times_predict, times_predict, self.kernel_function
        )

        predictive_means = mean_function_predict + np.linalg.multi_dot(
            [
                np.transpose(K_star),
                np.linalg.inv(K),
                (
                    self.rv_measurements[self.training_mask]
                    - mean_function[self.training_mask]
                ),
            ]
        )

        predictive_covariances = np.subtract(
            K_doublestar,
            np.linalg.multi_dot([np.transpose(K_star), np.linalg.inv(K), K_star]),
        )

        predictive_variances = np.diag(predictive_covariances)

        return predictive_means, predictive_variances

    def run_CV(self, include_keplerian=True):
        """
        Args:
            include_keplerian (bool): whether or not to include planetary signal as mean function. Defaults to True.

        Returns:
            CV_means, CV_variances (tuple): mean and variance values of GP ran on data
        """

        CV_means, CV_variances = self.GP_predict(self.times, include_keplerian)

        return CV_means, CV_variances

    def plot(self, times_predict, include_keplerian=True):
        """
        Wrapper for plot.Plot

        Args:
            times_predict (np.array): set of times for the GPR to predict values for
            include_keplerian (bool): whether or not to include planetary signal as mean function. Defaults to True.

        Returns:
            matplotlib.pyplot.Figure
        """
        return plot.plot_cv(self, times_predict, include_keplerian)

    def histogram(self, include_Gaussian=False):
        """
        Wrapper for plot.Histogram

        Args:
            include_Gaussian (bool): whether or not to fit Gaussians to residuals. Defaults to False.

        Returns:
            matplotlib.pyplot.Figure
        """
        return plot.plot_histogram(self, include_Gaussian)
