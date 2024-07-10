import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_cv(CrossValidation, times_predict, include_keplerian=True):

    predictive_means, predictive_variance = CrossValidation.GP_predict(
        times_predict, include_keplerian
    )

    std_dev = np.sqrt(predictive_variance)

    CV_means, _ = CrossValidation.run_CV(include_keplerian)

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, height_ratios=[0.7, 0.3])
    fig.subplots_adjust(hspace=0)

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
        CrossValidation.times[CrossValidation.training_mask],
        CrossValidation.rv_measurements[CrossValidation.training_mask],
        CrossValidation.rv_errors[CrossValidation.training_mask],
        ls="",
        marker="o",
        color="#009aa6",
        label="Training set",
    )
    ax1.errorbar(
        CrossValidation.times[CrossValidation.test_mask],
        CrossValidation.rv_measurements[CrossValidation.test_mask],
        CrossValidation.rv_errors[CrossValidation.test_mask],
        ls="",
        marker="^",
        color="#d82c24",
        label="Test set",
    )

    ax1.legend(fontsize=8)

    ax2.axhline(color="k", ls="--", alpha=0.7)

    ax2.errorbar(
        CrossValidation.times[CrossValidation.training_mask],
        CrossValidation.rv_measurements[CrossValidation.training_mask]
        - CV_means[CrossValidation.training_mask],
        CrossValidation.rv_errors[CrossValidation.training_mask],
        ls="",
        marker="o",
        color="#009aa6",
        label="Training set",
    )
    ax2.errorbar(
        CrossValidation.times[CrossValidation.test_mask],
        CrossValidation.rv_measurements[CrossValidation.test_mask]
        - CV_means[CrossValidation.test_mask],
        CrossValidation.rv_errors[CrossValidation.test_mask],
        ls="",
        marker="^",
        color="#d82c24",
        label="Test set",
    )

    ax2.legend(fontsize=8)

    ax2.set_xlabel("Time")
    ax2.set_ylabel("RV residual")
    ax1.set_ylabel("RV")

    return fig


def plot_histogram(CrossValidation, include_Gaussian=False):

    CV_means, CV_variances = CrossValidation.run_CV()

    fig = plt.figure()

    test_set_gp_resids = (
        CrossValidation.rv_measurements[CrossValidation.test_mask]
        - CV_means[CrossValidation.test_mask]
    ) / np.sqrt(
        CV_variances[CrossValidation.test_mask]
        + CrossValidation.rv_errors[CrossValidation.test_mask] ** 2
    )

    training_set_gp_resids = (
        CrossValidation.rv_measurements[CrossValidation.training_mask]
        - CV_means[CrossValidation.training_mask]
    ) / np.sqrt(
        CV_variances[CrossValidation.training_mask]
        + CrossValidation.rv_errors[CrossValidation.training_mask] ** 2
    )

    plt.hist(
        test_set_gp_resids,
        color="#d82c24",
        histtype="step",
        label="Test set",
        range=(-10, 10),
        bins=25,
        density=True,
    )
    plt.hist(
        training_set_gp_resids,
        color="#009aa6",
        alpha=0.5,
        range=(-10, 10),
        bins=25,
        density=True,
        label="Training set",
    )

    ### fitting Gaussian to residuals
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
