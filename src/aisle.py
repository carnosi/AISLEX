"""The aisle.py logic.

Handles learning entropy computation in pure pythonic non accelerated way as published by Ivo Bukovsky.
"""

__author__ = "Ondrej Budik"
__copyright__ = "<2024> <University Southern Bohemia> <Czech Technical University>"
__credits__ = ["Ivo Bukovsky"]

__license__ = "MIT (X11)"
__version__ = "1.0.7"
__maintainer__ = ["Ondrej Budik"]
__email__ = ["obudik@prf.jcu.cz", "ondrej.budik@fs.cvut.cz"]
__status__ = "beta"

__python__ = "3.8.0"

import numpy as np


def aisle(weights: np.ndarray, alphas: np.ndarray, oles: np.ndarray) -> np.ndarray:
    """Approximate Individual Sample Learning Entropy (AISLE).

    Implements anomaly detection based on the concept of Learning Entropy, as introduced by Ivo Bukovsky [1].
    AISLE assesses the variability in weight adjustments during a learning process.
    Higher orders of Learning Entropy can reveal more subtle patterns in the learning process.

    Args:
        weights: A 2D array where each row represents the weights at a particular iteration of the
                learning process.
        alphas: A 1D array of thresholds that determine the sensitivity of the Learning Entropy calculation.
                Each alpha represents a different level of sensitivity.
        oles: A 1D array representing the orders of Learning Entropy to be calculated. Each order provides a
                different level of granularity in understanding the learning dynamics.

    Returns:
        A 1D array containing the calculated Learning Entropy values for each order specified in `oles`.
        Each value gives an approximation of the variability or stability in the learning process at that
        order.

    The function first ensures that all inputs are numpy arrays. It then iterates through each order of
    Learning Entropy (`oles`), calculating the Learning Entropy for that order based on the weight changes
    and the specified `alphas`. The result is a comprehensive profile of the learning process's stability
    across different orders and sensitivities.

    [1] I. Bukovsky, W. Kinsner, and N. Homma, “Learning Entropy as a Learning-Based Information Concept,”
    Entropy, vol. 21, no. 2, p. 166, Feb. 2019, doi: 10.3390/e21020166.

    """
    # Convert inputs to numpy arrays in case user inputs list or tuple.
    weights = np.array(weights)
    alphas = np.array(alphas)
    oles = np.array(oles)

    # Make sure that oles shape is correct
    oles = oles.reshape(-1)
    # Get number of weights
    nw = weights.shape[1]
    # Get number of alphas
    nalpha = alphas.shape[0]
    # Prepare memory space for learning entropy
    ea = np.zeros(len(oles))
    # Counter for LE position
    i = 0
    # Iterate over all oles
    for ole in range(0, np.max(oles) + 1):
        if ole == oles[i]:
            # Get latest weight change
            absdw = np.abs(weights[-1, :])
            # Get mean weight changes for provided weight window
            meanabsdw = np.mean(abs(weights[0 : weights.shape[0] - 1, :]), 0)
            # Evaluate alphas
            n_alpha = 0
            for alpha in alphas:
                n_alpha += np.sum(absdw > alpha * meanabsdw)
            # Save learning entropy
            ea[i] = float(n_alpha) / (nw * nalpha)
            i = i + 1
        # Prepare weights for next ole
        weights = weights[1:, :] - weights[0:-1, :]
    return ea


def aisle_window(
    window: int, weights: np.ndarray, alphas: np.ndarray, oles: np.ndarray
) -> np.ndarray:
    """Evaluate Approximate Individual Sample Learning Entropy over a moving window on the provided weights.

    This function calculates the Learning Entropy (LE) for each sub-window of the given size, effectively
    capturing the dynamics of the learning process over time. It is a way to analyze how the learning process
    evolves over consecutive segments of the learning sequence.

    Args:
        window: The size of the window over which to evaluate the LE.
        weights: A 2D array of weights, where each row corresponds to the weights at a particular iteration of
                the learning process.
        alphas: A 1D array of thresholds for the sensitivity of the LE calculation, with each
                value representing a different level of sensitivity.
        oles: A 1D array of the orders of LE to calculate, providing different granularity in
                understanding learning dynamics.

    Returns:
        A 2D array where each row contains the LE values for the corresponding window of weights.
        The number of columns corresponds to the number of orders in `oles`.

    The function first ensures that all inputs are numpy arrays, then initializes an array to hold the LE
    values for each window. It iterates over each window of the specified size within the weights, computing
    the Learning Entropy for that window using the `aisle` function.
    """
    # Convert inputs to numpy arrays for unified processing
    weights = np.array(weights)
    alphas = np.array(alphas)
    oles = np.array(oles)

    # Prepare holder for learning entropy window shift output
    ea_windowed = np.zeros((weights.shape[0], oles.shape[0]))
    # Iterate over weights of a window
    for shift in range(window, weights.shape[0]):
        # Evaluate learning entropy for given weindow
        ea_windowed[shift, :] = aisle(weights[shift - window : shift], alphas, oles)
    return ea_windowed


if __name__ == "__main__":
    msg = "The aisle.py is not meant to be run as a script. Do see examples for propper usage."
    raise OSError(msg)
