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

##Learning entropy Author - IVO BUKOVSKY

__author__ = "Ondrej Budik"
__copyright__ = "<2023> <University Southern Bohemia> <Czech Technical University>"
__credits__ = ["Ivo Bukovsky"]

__license__ = "MIT (X11)"
__version__ = "1.0.0"
__maintainer__ = ["Ondrej Budik"]
__email__ = ["obudik@prf.jcu.cz"]
__status__ = "alpha"

__python__ = "3.8.0"

import numpy as np

def aisle(weights, alphas, oles):  # Wm ... recent window of weights including the very last weight updates
    """
    Approximate Individual Sample Learning Entropy
    Anomaly detection based on learning entropy as published by Ivo Bukovsky.

    Parameters
    ----------
    weights : numpy.array
        Weights to be evaluated with Learning Entropy algorithm
    alphas : numpy.array
        blab
    oles : numpy.array
        blab
    """
    weights = np.array(weights)
    alphas = np.array(alphas)
    oles = np.array(oles)

    oles = oles.reshape(-1)
    nw = weights.shape[1]
    nalpha = len(alphas)
    ea = np.zeros(len(oles))
    i = 0
    for ole in range(np.max(oles) + 1):
        if ole == oles[i]:  # assures the corresponding difference of Wm
            absdw = np.abs(weights[-1, :])  # very last updated weights
            meanabsdw = np.mean(abs(weights[0:weights.shape[0] - 1, :]), 0)
            Nalpha = 0
            for alpha in alphas:
                Nalpha += np.sum(absdw > alpha * meanabsdw)
            ea[i] = float(Nalpha) / (nw * nalpha)
            i += 1
        weights = weights[1:, :] - weights[0:(np.shape(weights)[0] - 1), :]  # difference Wm
    return (ea)


"""
def Learning_entropy(col_W,OLE,L,p,memory):
    OLEs = np.array([1, OLE])
    Wm = np.zeros((memory, col_W.shape[0]))
    EA = np.zeros((L + p, OLEs.shape[0]))
    EAP = np.zeros((L + p, OLEs.shape[0]))
    return Wm,EA,EAP,OLEs
"""