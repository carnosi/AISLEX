# -*- coding: utf-8 -*-
"""
aisle.py: Handles learning entropy computation in pure pythonic non accelerated way as published by Ivo Bukovsky.
__doc__ using Sphnix Style
"""
# Copyright 2022 University Southern Bohemia

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

__author__ = "Ondrej Budik"
__copyright__ = "<2023> <University Southern Bohemia>"
__credits__ = ["Ivo Bukovsky", "Czech Technical University"]

__license__ = "MIT (X11)"
__version__ = "1.0.6"
__maintainer__ = ["Ondrej Budik"]
__email__ = ["obudik@prf.jcu.cz", "ondrej.budik@fs.cvut.cz"]
__status__ = "alpha"

__python__ = "3.8.0"

import numpy as np


def aisle(weights: np.ndarray, alphas: np.ndarray, oles: np.ndarray) -> np.ndarray:
    """
    Approximate Individual Sample Learning Entropy
    Anomaly detection based on learning entropy as published by Ivo Bukovsky.

    Parameters
    ----------
    weights : np.ndarray
        Weights to be evaluated with Learning Entropy algorithm
    alphas : np.ndarray
        Sensitivity of learning entropy. List of various sensitivities to evaluate.
    oles : np.ndarray
        Orders of learning entropy evaluation.

    Returns
    -------
    np.ndarray
        evaluated learning entropy for all oles.
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
            Nalpha = 0
            for alpha in alphas:
                Nalpha += np.sum(absdw > alpha * meanabsdw)
            # Save learning entropy
            ea[i] = float(Nalpha) / (nw * nalpha)
            i = i + 1
        # Prepare weights for next ole
        weights = weights[1:, :] - weights[0:-1, :]
    return ea


def aisle_window(
    window: int, weights: np.ndarray, alphas: np.ndarray, oles: np.ndarray
) -> np.ndarray:
    """
    Evaluation of Approximate Individual Sample Learning Entropy
     with selected window over provided weights data.

    Parameters
    ----------
    window : int
        Window size to be evaluated with Learning Entropy
    weights : numpy.array
        Weights to be evaluated with Learning Entropy algorithm
    alphas : numpy.array
        Sensitivity of learning entropy. List of various sensitivities to evaluate.
    oles : numpy.array
        Orders of learning entropy evaluation.

    Returns
    -------
    ndarray
        Evaluation output for Learning Entropy for given settings.
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
    raise IOError("aisle.py is not meant to run as a script")
