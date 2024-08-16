import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_cv(CrossValidation, times_predict, include_keplerian=True, inst=None):
    """
    Plots the GP predictions conditioned on the training data together with test split overplotted along with
    a plot of residuals beneath the GP prediction plot

    Args:
        CrossValidation (object): object holding CV data
        times_predict (np.array): set of times for the GPR to predict values for
        include_keplerian (bool): whether or not to include planetary signal as mean function in GP. Defaults to True.
        inst (str): which instrument parameters the GP is use in its prediction. If none passed, takes in the parameters of the first instrument given in the inst_params dictionary.

    Returns:
        matplotlib.pyplot.Figure
    """

    #### gathering GP mean predictions to plot
    predictive_means, predictive_variance = CrossValidation.GP_predict(
        times_predict, include_keplerian, inst
    )

    std_dev = np.sqrt(predictive_variance)

    #### gathering cross validation results to plot
    CV_means, _ = CrossValidation.run_CV(include_keplerian, inst)

    #### setting up figure
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, height_ratios=[0.7, 0.3])
    fig.subplots_adjust(hspace=0)

    #### plotting GP and cross validation results along with training and test set data points
    ax1.plot(
        times_predict, predictive_means, label="Mean Prediction", color="k", alpha=0.7
    )

    ax1.fill_between(
        times_predict,
        predictive_means - std_dev,
        predictive_means + std_dev,
        label="1 Std. dev",
        color="#808080",
        alpha=0.3,
    )
    ax1.fill_between(
        times_predict,
        predictive_means - 2 * std_dev,
        predictive_means + 2 * std_dev,
        label="2 Std. dev",
        color="#A9A9A9",
        alpha=0.5,
    )

    ax1.errorbar(
        CrossValidation.training_times,
        CrossValidation.training_rvs,
        CrossValidation.training_errors,
        ls="",
        marker="o",
        color="#009aa6",
        label="Training set",
    )
    ax1.errorbar(
        CrossValidation.test_times,
        CrossValidation.test_rvs,
        CrossValidation.test_errors,
        ls="",
        marker="^",
        color="#d82c24",
        label="Test set",
    )

    ax1.legend(fontsize=8)

    #### plotting residuals from the training and test set

    ax2.axhline(color="k", ls="--", alpha=0.7)

    ax2.errorbar(
        CrossValidation.training_times,
        CrossValidation.training_rvs - CV_means[: len(CrossValidation.training_times)],
        CrossValidation.training_errors,
        ls="",
        marker="o",
        color="#009aa6",
        label="Training set",
    )
    ax2.errorbar(
        CrossValidation.test_times,
        CrossValidation.test_rvs - CV_means[len(CrossValidation.training_times) :],
        CrossValidation.test_errors,
        ls="",
        marker="^",
        color="#d82c24",
        label="Test set",
    )

    ax2.legend(fontsize=8)

    #### axes
    ax2.set_xlabel("Time")
    ax2.set_ylabel("RV residual")
    ax1.set_ylabel("RV")

    return fig


def plot_histogram(
    CrossValidation, include_Gaussian=False, include_keplerian=True, inst=None
):
    """
    Plots the residuals of the training and test data from the model conditioned on the training set

    Args:
        CrossValidation (object): object holding CV data
        include_Gaussian (bool): whether or not to fit and plot Gaussians to residuals. Defaults to False.
        include_keplerian (bool): whether or not to include planetary signal as mean function in CV. Defaults to True.
        inst (str): which instrument parameters the CV is use in its prediction. If none passed, takes in the parameters of the first instrument given in the inst_params dictionary.


    Returns:
        matplotlib.pyplot.Figure
    """

    #### gathering cross validation results to plot
    CV_means, CV_variances = CrossValidation.run_CV(include_keplerian, inst)

    #### calculating results for training and test sets
    test_set_gp_resids = (
        CrossValidation.test_rvs - CV_means[len(CrossValidation.training_times) :]
    ) / np.sqrt(
        CV_variances[len(CrossValidation.training_times) :]
        + (CrossValidation.test_errors) ** 2
    )

    training_set_gp_resids = (
        CrossValidation.training_rvs - CV_means[: len(CrossValidation.training_times)]
    ) / np.sqrt(
        CV_variances[: len(CrossValidation.training_times)]
        + (CrossValidation.training_errors) ** 2
    )

    #### setting up figure
    fig = plt.figure()

    plt.hist(
        test_set_gp_resids,
        color="#d82c24",
        histtype="step",
        label="Test set",
        density=True,
        range=(-5, 5),
        bins=25,
    )
    plt.hist(
        training_set_gp_resids,
        color="#009aa6",
        alpha=0.5,
        range=(-5, 5),
        density=True,
        bins=25,
        label="Training set",
    )

    ### fitting and plotting Gaussian to residuals
    if include_Gaussian:
        mu_training, std_training = norm.fit(training_set_gp_resids)
        mu_test, std_test = norm.fit(test_set_gp_resids)

        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        plt.plot(
            x,
            norm.pdf(x, mu_training, std_training),
            color="#009aa6",
            linewidth=1,
            linestyle="--",
            alpha=0.5,
        )
        plt.plot(
            x,
            norm.pdf(x, mu_test, std_test),
            color="#d82c24",
            linewidth=1,
            linestyle="--",
            alpha=0.5,
        )

    plt.legend()

    return fig
