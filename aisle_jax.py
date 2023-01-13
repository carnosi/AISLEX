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
__version__ = "1.0.3"
__maintainer__ = ["Ondrej Budik"]
__email__ = ["obudik@prf.jcu.cz"]
__status__ = "alpha"

__python__ = "3.8.0"

import jax
import jax.numpy as jnp
from jax import jit


@jit
def aisle(weights, alphas, oles):
    """
    Approximate Individual Sample Learning Entropy - accelerated with JAX framework.
    Should be run-able on CPU / GPU / TPU as long as JAX supports it.
    Calculations are limited to fp32 due to lib limitations.

    Parameters
    ----------
    weights : iterable
        Weights those should be evalueted with Learning Entropy
    alphas : iterable
        Sensitivity of learning entropy. List of various sensitivities to evaluate.
    oles : iterable
        Orders of learning entropy evaluation.

    Returns
    -------
    ndarray
        Evaluated learning entropy for all oles.
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
    # Get scaler, its same for all oles, no need to recalc it
    scaler = nw * nalpha
    # LE weights evaluation
    ea = __ole_loop__(weights, alphas, oles, scaler)

    return ea

@jit
def __ole_loop__(weights, alphas, oles, scaler):
    """
    Evaluate all oles in a loop

    Parameters
    ----------
    weights : jax.numpy.array
        Weights those should be evalueted with Learning Entropy
    alphas : jax.numpy.array
        Sensitivity of learning entropy. List of various sensitivities to evaluate.
    oles : jax.numpy.array
        Orders of learning entropy evaluation.
    scaler : jax.fp32
        Scaler for OLE/Alpha evaluated weights

    Returns
    -------
    ndarray
        Evaluated learning entropy for all oles.
    """

    # Prepare the output array of learning entropy. Size matches LE orders.
    ea = jnp.zeros(oles.shape[0])
    # loop over all OLES and get learning entropy
    pos = 0
    for ole in oles:
        # Calculate dws for given OLE order
        ole_weights, _, _ = __weight_ole__(weights, ole)
        # Evaluate learning entropy for given OLE
        one_ea = __ole__(ole_weights, alphas, scaler, ole)
        # Save it to array and move to the next one
        ea = ea.at[pos].set(one_ea)
        pos = pos + 1
    return ea

@jit
def __ole__(weights, alphas, scaler, ole):
    """
    Single OLE evaluation for given weights and alphas

    Parameters
    ----------
    weights : jax.numpy.array
        Weights those should be evalueted with Learning Entropy
    alphas : jax.numpy.array
        Sensitivity of learning entropy. List of various sensitivities to evaluate.
    scaler : jax.fp32
        Scaler for OLE/Alpha evaluated weights
    ole : jax.int32
        Single OLE which should be evaluated

    Returns
    -------
    jax.fp32
        Single Learning Entropy output for given OLE

    NOTICE
    ------
    meanabsdw is not calculated EXACT due to jax array manipulation limitations.
    Depending on the order of OLE last values contain 0s. These values should be sliced
    however JAX does not support dynamically sized arrays without large overhang.
    Error is very low tho, can be ignored for now.
    """
    # Get the latest dw change for given weights
    absdw = jnp.abs(weights[-ole-1, :])
    # Get mean weight changes from all weights. Compared to native python this is not EXACT and might cause unexpected values when len(weights)<10
    # Optimal solution would be to select weights[:-ole-1, :] which is not possible as its dynamic array
    meanabsdw = jnp.mean(jnp.abs(weights), 0)
    # Evaluate all alphas as sensitivity
    Nalpha = __alpha_loop__(alphas, absdw, meanabsdw)
    # Calculate learning entropy from alpha evaluation and scaler
    ea = jnp.float32(Nalpha) / (scaler)
    return ea

@jit
def __alpha_loop__(alphas, absdw, meanabsdw):
    """
    Evaluate all alphas for given latest weight change and mean weight behavior in provided window

    Parameters
    ----------
    alphas : jax.numpy.array
        Sensitivity of learning entropy. List of various sensitivities to evaluate.
    absdw : jax.numpy.array
        The latest dw change for given weights.
    meanabsdw : jax.numpy.array
        Mean weight changes from all weights.

    Returns
    -------
    ndarray
        Evaluated alphas
    """

    Nalpha = 0
    for alpha in alphas:
        temp = jnp.sum(absdw > alpha * meanabsdw)
        Nalpha = Nalpha + temp
    return Nalpha

@jit
def __weight_ole__(weights, ole):
    """
    Weight changes in accordance with different specified OLEs

    Parameters
    ----------
    weights : jax.numpy.array
        Weights those should be evalueted with Learning Entropy
    ole : jax.int32
        Numeric description position of ole. Changes weight change
    """
    def cond(arg):
        _ , step, ole = arg
        return (step < ole)

    def body(arg):
        weights, step, ole = arg
        temp_weights = weights[1:, :] - weights[0:-1, :]
        weights = jax.lax.dynamic_update_slice(weights, temp_weights, (0, 0))
        weights = weights.at[-1, :].set(0)
        return (weights, step + 1, ole)

    return jax.lax.while_loop(
        cond,
        body,
        (weights, 0, ole)
    )

if __name__ == "__main__":
    raise IOError("aisle_jax.py is not meant to be run as a script")