import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


##### KEPLERIAN SIGNAL HELPER FUNCTIONS #####

def kepler_solver(M_anom, ecc, tolerance = 1e-7):
    '''
    Solves Kepler's equation using the Newton Raphson method 

    Args:
        M_anom (np.array): array of mean anomalies
        ecc (np.array): array of eccentricities
        tolerance (float, optional): absolute tolerance of iterative computatiom ('exit condition'). Defaults to 1e-7.

    Return:
        E_anom (np.array): array of eccentric anomalies
    '''

    def f(E_anom, ecc, M_enom):
        return E_anom - ecc * np.sin(E_anom) - M_enom

    def df(E_anom, ecc):
        return 1 - ecc * np.cos(E_anom)

    ### intialize guess at E_0 = M ##
    E_anom = np.copy(M_anom)

    ### starting conditions ### 
    E_anom -= f(E_anom, ecc, M_anom) / df(E_anom, ecc)
    diff = f(E_anom, ecc, M_anom) / df(E_anom, ecc)
    error = np.abs(diff)
    indx = np.where(error > tolerance)
    n = 0

    ### run algorithm until a value is found within tolerance
    while (indx[0].size > 0) and (n <= 1000000):
        E_anom[indx] -= diff[indx]
        diff[indx] = f(E_anom[indx], ecc[indx], M_anom[indx]) / df(E_anom[indx], ecc[indx])
        error[indx] = np.abs(diff[indx])
        indx = np.where(error > tolerance)
        n += 1

    if n > 1000000:
        raise ValueError('Sorry, this is taking too long.')
    
    return E_anom

def transit_to_periastron(t_transit, period, ecc, omega):
    """
    Convert Time of Transit to Time of Periastron Passage

    Args:
        t_transit (float): time of transit
        period (float): period of orbit [days]
        ecc (float): eccentricity
        omega (float): longitude of periastron (radians)

    Returns:
        t_periastron (float): time of periastron passage

    """
    f = np.pi/2 - omega
    E_anom = 2 * np.arctan(np.sqrt((1-ecc)/(1+ecc)) * np.tan(f/2))
    t_periastron = t_transit - period / (2*np.pi) * (E_anom - ecc * np.sin(E_anom))

    return t_periastron

def calc_mean_anomaly(times, period, t_periastron):
    """
    Compute the mean anomaly at given times during an orbit

    Args:
        times (np.array): times of measurement
        period (float): period of orbit [days]
        t_periastron (float): time of periastron passage
        
    Returns:
        (np.array): array of mean anomalies

    """
    return 2* np.pi / period * (times - t_periastron)

def calc_true_anomaly(E_anom, ecc):
    """
    Compute the true anomaly at certain points during an orbit

    Args:
        E_anom (np.array): array of eccentric anomalies
        ecc (float): eccentricity
        
    Returns:
        (np.array): array of true anomalies

    """
    return 2 * np.arctan (np.sqrt((1+ecc)/(1-ecc)) * np.tan(E_anom/2))

def calc_keplerian_signal(times, t_transit, period, ecc, omega, K):
    """
    Compute the radial velocity signal of a transiting exoplanet

    Args:
        times (np.array): time measurements
        t_transit (float): time of transit
        period (float): period of orbit [days]
        ecc (float): eccentricity
        omega (float): longitude of periastron (radians)
        K (float): Keplarian semi_amplitude
        
    Returns:
        (np.array): array of radial velocity signals

    """
    t_periastron = transit_to_periastron(t_transit, period, ecc, omega)

    M_anom = calc_mean_anomaly(times, period, t_periastron)

    ecc_array = ecc * np.ones_like(times)

    E_anom = kepler_solver(M_anom, ecc_array)

    nu = calc_true_anomaly(E_anom, ecc)

    rv = K * ( np.cos(omega + nu) + ecc * np.cos(omega) )

    return rv


##### GPR HELPER FUNCTION ####
def build_covariance_matrix(p1, p2, kernel_function):
    '''
    Computes the covariance matrix between two sets of points given a kernel function
    
    Args:
        p1 (np.array): array of first set of points, length N
        p2 (np.array): array of second set of points, length M
        kernel_function (func): describes the covariance between two points

    Return:
        cov_matrix (np.array): an array of size NxM filled with values computed by kernel
    '''
    cov_matrix = np.zeros((len(p1),len(p2)))

    for i in range(len(p1)):
        for j in range(len(p2)):
            k = kernel_function(p1[i], p2[j]) 
            cov_matrix[i][j] = k

    return cov_matrix


class CrossValidation():
    '''
    Args:
        times (np.array): array of timestamps from data
        rv_measurements (np.array): array of measured radial velocity values
        rv_errors (np.array): array of radial velocity errors
        kernel_function (func): describes the covariance between two points
        orbit_params (array or list): [transit time, period, eccentricity, omega, and keplerian semi-amplitude] 
        train_split (float): number from 0 to 1 that will divide data into a training and testing set
        gamma (float, optional): Defaults to 0.
        jitter (float, optional): Defaults to 0.

    '''
    def __init__(self, times, rv_measurements, rv_errors, kernel_function, orbit_params, gamma = 0, jitter = 0):
        self.times = times
        self.rv_measurements = rv_measurements
        self.rv_errors = rv_errors
        self.kernel_function = kernel_function
        self.orbit_params = orbit_params
        self.gamma = gamma
        self.jitter = jitter


    def Split(self, train_split):
        ######## selecting training & test set at random ########
        n_data = len(self.times)
        training_mask = np.random.choice(n_data, size=int(train_split * n_data), replace=False)
        test_mask = np.setdiff1d(np.arange(n_data), training_mask)
        
        self.training_mask = training_mask
        self.test_mask = test_mask

    def GP_Predict(self, times_predict, include_keplerian = True):

        if include_keplerian:
            mean_function = calc_keplerian_signal(self.times, *self.orbit_params) + self.gamma
            mean_function_predict = calc_keplerian_signal(times_predict, *self.orbit_params) 
        else:
            mean_function = np.zeros_like(self.times) + self.gamma 
            mean_function_predict = np.zeros_like(times_predict) 

        tot_train_error = np.sqrt(self.rv_errors[self.training_mask]**2 + self.jitter**2)

        K = build_covariance_matrix(self.times[self.training_mask],self.times[self.training_mask],self.kernel_function) + \
                (tot_train_error**2) * np.identity(len(self.times[self.training_mask]))
        
        K_star_predict = build_covariance_matrix(self.times[self.training_mask],times_predict, self.kernel_function)

        K_doublestar_predict = build_covariance_matrix(times_predict,times_predict, self.kernel_function)


        predictive_means = mean_function_predict + \
            np.linalg.multi_dot([np.transpose(K_star_predict), np.linalg.inv(K),
                                 (self.rv_measurements[self.training_mask] - mean_function[self.training_mask])])
        
        predictive_covariances = np.subtract(K_doublestar_predict, 
                                            np.linalg.multi_dot([np.transpose(K_star_predict),np.linalg.inv(K),K_star_predict]))

        predictive_variances = np.diag(predictive_covariances)

        return predictive_means, predictive_variances

    def Run_CV(self):
        tot_train_error = np.sqrt(self.rv_errors[self.training_mask]**2 + self.jitter**2)

        K = build_covariance_matrix(self.times[self.training_mask],self.times[self.training_mask],self.kernel_function) + \
                (tot_train_error**2) * np.identity(len(self.times[self.training_mask]))
        
        K_star_CV = build_covariance_matrix(self.times[self.training_mask],self.times, self.kernel_function)
        K_doublestar_CV = build_covariance_matrix(self.times, self.times, self.kernel_function)

        CV_means = np.linalg.multi_dot([np.transpose(K_star_CV),np.linalg.inv(K),
                                        (self.rv_measurements[self.training_mask])])
        CV_covariances = np.subtract(K_doublestar_CV, 
                                     np.linalg.multi_dot([np.transpose(K_star_CV),np.linalg.inv(K),K_star_CV]))
        CV_variances = np.diag(CV_covariances)

        return CV_means, CV_variances

    def Plot(self, times_predict, include_keplerian = True):

        predictive_means, predictive_variance = self.GP_Predict(times_predict, include_keplerian)
        std_dev = np.sqrt(predictive_variance)

        CV_means, _ = self.Run_CV()

        fig,  (ax1, ax2) = plt.subplots(2, sharex=True, height_ratios = [0.7, 0.3])
        fig.subplots_adjust(hspace=0)

        ax1.plot(times_predict, predictive_means, label = "Mean prediction", color = 'k')

        ax1.fill_between(times_predict, predictive_means -std_dev, predictive_means+std_dev, label = '1 std. dev', color = '#808080', alpha = 0.3)
        ax1.fill_between(times_predict, predictive_means-2*std_dev, predictive_means +2*std_dev, label = '2 std. dev', color = '#A9A9A9', alpha = 0.5)


        ax1.errorbar(self.times[self.training_mask], self.rv_measurements[self.training_mask], 
                     self.rv_errors[self.training_mask], ls="", marker="o", color="r", label="training set")
        ax1.errorbar(self.times[self.test_mask], self.rv_measurements[self.test_mask], 
                     self.rv_errors[self.test_mask], ls="", marker="o", color="b",label="test set")


        ax1.legend(fontsize = 8)

        ax2.axhline(color="g", ls="--")


        ax2.errorbar(self.times[self.training_mask], self.rv_measurements[self.training_mask] - CV_means[self.training_mask], 
                     self.rv_errors[self.training_mask], ls="", marker="o", color="r", label="training set")
        ax2.errorbar(self.times[self.test_mask], self.rv_measurements[self.test_mask] - CV_means[self.test_mask], 
                     self.rv_errors[self.test_mask], ls="", marker="o", color="b",label="test set")

        ax2.legend(fontsize = 8)


        ax2.set_xlabel("Time")
        ax2.set_ylabel("RV residual")
        ax1.set_ylabel("RV")

        plt.show()
        plt.close()

    def Histogram(self):
        CV_means, _ = self.Run_CV()

        plt.figure()

        test_set_gp_resids = []
        for i in self.test_mask:
            a = (self.rv_measurements[i] - CV_means[i]) + self.rv_errors[i]
            total_residual = np.sign(a) * (np.abs(a))**0.5
            test_set_gp_resids.append(total_residual)

        training_set_gp_resids = []
        for i in self.training_mask:
            a = (self.rv_measurements[i] - CV_means[i]) + self.rv_errors[i]
            total_residual = np.sign(a) * (np.abs(a))**0.5
            training_set_gp_resids.append(total_residual)

    
        plt.hist(test_set_gp_resids, color="b", histtype="step", label="test set", range=(-5, 5), bins=50, density=True)
        plt.hist( training_set_gp_resids, color="r", alpha=0.5, range=(-5, 5), bins=50, density=True, label="training set",)
        plt.legend()

        plt.show()
        plt.close()


def quasi_periodic_kernel(ti, tj):
    '''
    Computes values for quasi-periodic covariance function.

    Args: 
        ti, tj (floats) : any two times recorded within the data set

    Return:
        float
    '''
    return GP_amplitude**2 * np.exp(-((np.sin(np.pi * (ti-tj) / P_gp_test ))**2)/(2*lambda_p_test**2) - ((ti-tj)**2 / (lambda_e_test**2)))



###### dai et al 2017 plots ######

# ### importing test data ###
df = pd.read_csv('dai et al 2017 pfs.csv')

data_times_pfs = df['BJD_TDB'].to_numpy()
data_rv_pfs = df['RV (m/s)'].to_numpy()
data_rv_errors_pfs = df['Unc (m/s)'].to_numpy()


df = pd.read_csv('dai et al 2017 harps.csv')

data_times_harps = df['BJD_TDB'].to_numpy()
data_rv_harps = df['RV (m/s)'].to_numpy()
data_rv_errors_harps = df['Unc (m/s)'].to_numpy()

data_period = 0.3693038
data_transit_time = 2457582.9360
data_ecc = 0
data_kepler_semiamp = 6.77
data_omega = 0

GP_amplitude = 26
P_gp_test = 9.68
lambda_e_test = 11.6
lambda_p_test = 0.35

data_orbit_params = [data_transit_time, data_period, data_ecc, data_omega, data_kepler_semiamp]
t_predict_harps = np.linspace(np.min(data_times_harps), np.max(data_times_harps), int(1e3))
t_predict_pfs = np.linspace(np.min(data_times_pfs), np.max(data_times_pfs), int(1e3))


cv_harps = CrossValidation(data_times_harps, data_rv_harps, data_rv_errors_harps, quasi_periodic_kernel, data_orbit_params, -6694.7, 2)
cv_pfs = CrossValidation(data_times_pfs, data_rv_pfs, data_rv_errors_pfs, quasi_periodic_kernel, data_orbit_params, -1, 5.3)

cv_harps.Split(0.8)
cv_pfs.Split(1)

cv_harps.Plot(t_predict_harps)

# harps_predict_means, harps_predict_var = cv_harps.GP_Predict(t_predict_harps)
# pfs_predict_means, pfs_predict_var = cv_pfs.GP_Predict(t_predict_pfs)

# harps_fit, _ = cv_harps.GP_Predict(data_times_harps)

# pfs_fit, _ = cv_pfs.GP_Predict(data_times_pfs)

# fig, (ax1, ax2) = plt.subplots(2, sharex=True, height_ratios = [0.6, 0.4])
# fig.subplots_adjust(hspace=0)

# ax1.plot(t_predict_harps, harps_predict_means, color = 'b')
# ax1.plot(t_predict_pfs, pfs_predict_means, color = 'b')


# ax1.fill_between(t_predict_harps, harps_predict_means - np.sqrt(harps_predict_var), harps_predict_means + np.sqrt(harps_predict_var), color = 'r', alpha = 0.1)
# ax1.fill_between(t_predict_harps, harps_predict_means- 2*np.sqrt(harps_predict_var) , harps_predict_means + 2*np.sqrt(harps_predict_var) , color = 'r', alpha = 0.2)

# ax1.fill_between(t_predict_pfs, pfs_predict_means- np.sqrt(pfs_predict_var),  pfs_predict_means + np.sqrt(pfs_predict_var), color = 'tab:pink', alpha = 0.1)
# ax1.fill_between(t_predict_pfs, pfs_predict_means - 2*np.sqrt(pfs_predict_var),  pfs_predict_means + 2*np.sqrt(pfs_predict_var), color = 'tab:pink', alpha = 0.2)


# ax1.errorbar(data_times_harps, data_rv_harps + 6694.7, data_rv_errors_harps, ls="", marker="^", color="r", ecolor = '0.5', label="HARPS-N")
# ax1.errorbar(data_times_pfs, data_rv_pfs, data_rv_errors_pfs, ls="", marker="o", color="tab:pink", ecolor = '0.5',  label="PFS")

# ax1.legend(fontsize = 8)

# ax2.axhline(color="tab:grey", ls="--")

# ax2.errorbar(data_times_harps, data_rv_harps + 6694.7 - harps_fit, data_rv_errors_harps, ls="", marker="^", color="r", ecolor = '0.5', label="HARPS-N")
# ax2.errorbar(data_times_pfs, data_rv_pfs - pfs_fit, data_rv_errors_pfs, ls="", marker="o", color="tab:pink", ecolor = '0.5' , label="PFS")



# ax2.set_xlabel("Time")
# ax2.set_ylabel("RV residual")
# ax1.set_ylabel("RV")



# plt.show()





# ##### barragan et al 2024 plots ######

# ### importing test data ###
# df = pd.read_csv('harpstable.csv')

# data_times = df['#Time (BJD - 2457000)'].to_numpy()
# data_times += 2457000 ### revert to BJD
# data_rv_gaussian = df[' RV Gaussian [km/s]'].to_numpy()
# data_rv_GND = df['RV GND [km/s]'].to_numpy()
# data_rv_GND_28 = df[' RV GND_2.8 [km/s] '].to_numpy()
# data_rv_errors = df['eRV[km/s] '].to_numpy()


# data_period = 8.3249113
# data_transit_time = 2356.81398 + 2457000
# data_ecc = 0
# data_kepler_semiamp = 0.0346
# data_omega = 0

# GP_amplitude = 0.138
# P_gp_test = 3
# lambda_p_test = 0.42
# lambda_e_test = 69


# data_orbit_params = [data_transit_time, data_period, data_ecc, data_omega, data_kepler_semiamp]
# t_predict = np.linspace(np.min(data_times), np.max(data_times), int(1e3))




# my_cv = CrossValidation(data_times, data_rv_GND_28, data_rv_errors, quasi_periodic_kernel, data_orbit_params, -0.002, 0.026)
# my_cv.Split(0.7)
# my_cv.Plot(t_predict)


# stellar_signal_plus_kepler_means, stellar_signal_plus_kepler_variances = my_cv.GP_Predict(t_predict, False)


# stellar_signal_only_means, stellar_signal_only_variances =  my_cv.GP_Predict(t_predict, True)
# stellar_signal, _ = my_cv.Run_CV()







# keplerians = calc_keplerian_signal(t_predict, *data_orbit_params)
# keplerians2 = calc_keplerian_signal(data_times, *data_orbit_params)


# fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, height_ratios = [0.55, 0.25, 0.20])
# fig.subplots_adjust(hspace=0)

# ax1.plot(t_predict, stellar_signal_plus_kepler_means, label = "Stellar Signal + Keplerians", color = 'k')
# ax1.plot(t_predict, stellar_signal_only_means - 0.5, label = "Stellar Signal", color = 'b')
# ax1.plot(t_predict, keplerians - 1, label = "Keplerians", color = 'g')

# ax1.fill_between(t_predict, stellar_signal_plus_kepler_means- np.sqrt(stellar_signal_plus_kepler_variances), stellar_signal_plus_kepler_means + np.sqrt(stellar_signal_plus_kepler_variances), color = '#808080', alpha = 0.3)
# ax1.fill_between(t_predict, stellar_signal_plus_kepler_means- 2*np.sqrt(stellar_signal_plus_kepler_variances), stellar_signal_plus_kepler_means + 2*np.sqrt(stellar_signal_plus_kepler_variances), color = '#A9A9A9', alpha = 0.5)

# ax1.fill_between(t_predict, stellar_signal_only_means- np.sqrt(stellar_signal_only_variances) - 0.5, stellar_signal_only_means + np.sqrt(stellar_signal_only_variances) - 0.5, color = 'b', alpha = 0.1)
# ax1.fill_between(t_predict, stellar_signal_only_means- 2*np.sqrt(stellar_signal_only_variances) - 0.5, stellar_signal_only_means + 2*np.sqrt(stellar_signal_only_variances) - 0.5, color = 'b', alpha = 0.2)


# ax1.errorbar(data_times, data_rv_GND_28, data_rv_errors, ls="", marker="o", color="r", label="RV")

# ax1.legend(fontsize = 8)

# ax3.axhline(color="g", ls="--")


# ax2.plot(t_predict, keplerians,label = "Keplerians", color = 'g')
# ax2.errorbar(data_times, data_rv_GND_28 - stellar_signal, data_rv_errors, ls="", marker="o", color="r", label="RV - Stellar Signal")

# ax3.errorbar(data_times, data_rv_GND_28 - stellar_signal - keplerians2, data_rv_errors, ls="", marker="o", color="r")
# ax2.legend(fontsize = 8)


# ax3.set_xlabel("Time")
# ax3.set_ylabel("RV residual")
# ax1.set_ylabel("RV")
# ax2.set_ylabel("RV")

# plt.show()



