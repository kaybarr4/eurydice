import numpy as np
import eurydice.CV as CV
import pytest


threshold = 1e-5


def test_default_sqexp_kernel():
    """
    Test the default squared exponential kernel through simple test calculations
    """
    test_values = [0.1, 1, 10]

    true_outputs = [
        1,
        0.9839305143,
        0.1408302521,
        0.9839305143,
        1,
        0.1978986991,
        0.1408302521,
        0.1978986991,
        1,
    ]

    test_outputs = []
    for ii in test_values:
        for jj in test_values:
            test_outputs.append(CV.default_sqexp_kernel(ii, jj))

    for meas, truth in zip(test_outputs, true_outputs):
        assert meas == pytest.approx(truth, abs=threshold)


def test_build_covariance_matrix():
    """
    Test Covariance Matrix Builder by generating test points, finding true values
    through a test kernel and comparing to the output of build_covariance_matrix

    """
    ## generate values
    test_array_1 = np.random.randint(0, 100, 100)
    test_array_2 = np.random.randint(0, 100, 100)

    true_values = np.array(
        [[CV.default_sqexp_kernel(a, b) for b in test_array_2] for a in test_array_1]
    )

    covariance_matrix = CV.build_covariance_matrix(
        test_array_1, test_array_2, CV.default_sqexp_kernel
    )

    ## compare true values versus output of build_covariance_matrix func
    assert true_values.shape == covariance_matrix.shape
    assert np.allclose(true_values, covariance_matrix)


def test_split_random():
    """
    Test the split function of the CrossValidation object with randomization allowed
    """
    fake_times, fake_rv, fake_errs, fake_orbit = (
        np.ones(50),
        np.zeros(50),
        np.zeros(50),
        (0, 0, 0, 0, 0),
    )
    test_CV = CV.CrossValidation(fake_times, fake_rv, fake_errs, fake_orbit)

    test_splits = (0.25, 0.5, 0.85)
    split_nums = (12, 25, 42)

    for split, truth in zip(test_splits, split_nums):
        test_CV.split(split)
        assert len(test_CV.training_mask) == truth
        assert len(test_CV.test_mask) == (50 - truth)


def test_split_errors():
    """
    Test that the split function raises a proper ValueError when given a train_split out of bounds
    """
    fake_times, fake_rv, fake_errs, fake_orbit = (
        np.ones(50),
        np.zeros(50),
        np.zeros(50),
        (0, 0, 0, 0, 0),
    )
    test_CV = CV.CrossValidation(fake_times, fake_rv, fake_errs, fake_orbit)

    with pytest.raises(ValueError) as excinfo:
        test_CV.split(-0.5)
        assert (
            str(excinfo.value)
            == "The train_split value must be between 0 and 1 (inclusive). Your input train_split value was -0.5"
        )

    with pytest.raises(ValueError) as excinfo:
        test_CV.split(1.5)
        assert (
            str(excinfo.value)
            == "The train_split value must be between 0 and 1 (inclusive). Your input train_split value was 1.5"
        )


if __name__ == "__main__":
    test_build_covariance_matrix()
    test_default_sqexp_kernel()
    test_split_random()
    test_split_errors()
