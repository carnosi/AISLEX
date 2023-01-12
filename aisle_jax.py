# -*- coding: utf-8 -*-
"""
aisle_jax.py: Handles learning entropy computational acceleration with JAX by Google

Prebuilt community packages of JAX are available on: https://github.com/cloudhan/jax-windows-builder
Other way you can also build jax on your own from https://github.com/google/jax

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
__copyright__ = "<2023> <University Southern Bohemia> <Czech Technical University>"
__credits__ = ["Ivo Bukovsky"]

__license__ = "MIT (X11)"
__version__ = "1.0.1"
__maintainer__ = ["Ondrej Budik"]
__email__ = ["obudik@prf.jcu.cz"]
__status__ = "alpha"

__python__ = "3.8.0"

from functools import partial
import jax
import jax.numpy as jnp
from jax import jit


@jit
def aisle(weights, alphas, oles):
    """_summary_
    """
    # Convert input arrays to jax arrays
    weights = jnp.array(weights)
    alphas = jnp.array(alphas)
    oles = jnp.array(oles)

    # Reshape oles to match alg req
    oles = oles.reshape(-1)
    # Get the amount of weights to be evaluated
    nw = weights.shape[1]
    # Get the amount of alphas to evaluate
    nalpha = alphas.shape[0]
    # Get scaler
    scaler = nw * nalpha
    # LE weights evaluation
    ea = __ole_loop__(weights, alphas, oles, scaler)

    return ea

@jit
def __ole_loop__(weights, alphas, oles, scaler):
    """
    i = 0
    for ole in range(np.max(oles) + 1):
        if ole == oles[i]:  # assures the corresponding difference of Wm
            ea = __ole__(weights, alphas, scaler, eval_weights)
            EA[i] = ea
            i += 1
        weights = weights[1:, :] - weights[0:(np.shape(weights)[0] - 1), :]  # difference Wm
    return (ea)
    """
    # Prepare the output array of learning entropy. Size matches LE orders.
    ea = jnp.zeros(oles.shape[0])
    # loop over OLES
    pos = 0
    for ole in oles:
        one_ea = __ole__(weights, alphas, oles, scaler)
        ea = ea.at[pos].set(one_ea)
        pos = pos + 1
    return ea

@jit
def __ole__(weights, alphas, oles, scaler):
    """
    Single OLE evaluation in LE algorithm
    Original code:
        absdw = np.abs(weights[-1, :])  # very last updated weights
        meanabsdw = np.mean(abs(weights[0:weights.shape[0] - 1, :]), 0)
        Nalpha = __alpha_loop__(alphas, absdw, meanabsdw)
        ea = float(Nalpha) / (nw * nalpha)

    Parameters
    ----------
    weights : _type_
        _description_
    alphas : _type_
        _description_
    scaler : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    """    runs = jnp.arange(0, ole, 1, jnp.int16)
    for nrun in runs:
        weights = weights[1:, :] - weights[0:-1, :]
    """
    absdw = jnp.abs(weights[-1, :])
    meanabsdw = jnp.mean(jnp.abs(weights[0:- 1, :]), 0)
    Nalpha = __alpha_loop__(alphas, absdw, meanabsdw)
    ea = jnp.float32(Nalpha) / (scaler)

    return ea

@jit
def __alpha_loop__(alphas, absdw, meanabsdw):
    """
    Loop of alphas in LE algorithm which should be flattened by JAX.
    Original loop:
                Nalpha = 0
                for alpha in alphas:
                     Nalpha += np.sum(absdw > alpha * meanabsdw)

    Parameters
    ----------
    start : _type_
        _description_
    stop : _type_
        _description_
    init_val : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    Nalpha = 0
    for alpha in alphas:
        temp = jnp.sum(absdw > alpha * meanabsdw)
        Nalpha = Nalpha + temp
    return Nalpha