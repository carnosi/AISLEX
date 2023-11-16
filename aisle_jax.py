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
__copyright__ = "<2023> <University Southern Bohemia>"
__credits__ = ["Ivo Bukovsky", "Czech Technical University"]

__license__ = "MIT (X11)"
__version__ = "1.0.5"
__maintainer__ = ["Ondrej Budik"]
__email__ = ["obudik@prf.jcu.cz"]
__status__ = "alpha"

__python__ = "3.8.0"

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

import numpy as np


@jit
def aisle(weights: np.ndarray, alphas: np.ndarray, oles: np.ndarray) -> jnp.ndarray:
    """
    Computes the Approximate Individual Sample Learning Entropy (AISLE) using the JAX framework, which is compatible with CPUs, GPUs, and TPUs.

    Args:
        weights (np.ndarray): Weights to be evaluated with Learning Entropy.
        alphas (np.ndarray): Sensitivities of learning entropy, represented as a list of various sensitivities to evaluate.
        oles (np.ndarray): Orders of learning entropy evaluation.

    Returns:
        jnp.ndarray: Evaluated learning entropy for all OLEs.
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

    # Unification of output. Usefull for window processing
    ea = ea.reshape(1, -1)

    return ea


@jit
def __ole_loop__(
    weights: jnp.ndarray,
    alphas: jnp.ndarray,
    oles: jnp.ndarray,
    scaler: float,
) -> jnp.ndarray:
    """
    Evaluates learning entropy for all specified orders of learning entropy (OLEs) in a loop.

    Args:
        weights (jnp.ndarray): Weights to be evaluated with Learning Entropy.
        alphas (jnp.ndarray): Sensitivities of learning entropy, represented as a list of various sensitivities to evaluate.
        oles (jnp.ndarray): Orders of learning entropy evaluation.
        scaler (float): Scaler for OLE/Alpha evaluated weights.

    Returns:
        jnp.ndarray: The evaluated learning entropy values for all OLEs.
    """

    # Prepare the output array of learning entropy. Size matches LE orders.
    ea = jnp.zeros(oles.shape[0])
    # loop over all OLES and get learning entropy
    pos = 0
    for ole in oles:
        # Calculate dws for given OLE order. Oles usually are not many, no need to recompile into fori for now.
        ole_weights, _, _ = __weight_ole__(weights, ole)
        # Evaluate learning entropy for given OLE
        one_ea = __ole__(ole_weights, alphas, scaler, ole)
        # Save it to array and move to the next one
        ea = ea.at[pos].set(one_ea)
        pos = pos + 1
    return ea


@jit
def __ole__(
    weights: jnp.ndarray,
    alphas: jnp.ndarray,
    scaler: float,
    ole: int,
) -> float:
    """
    Performs a single Order of Learning Entropy (OLE) evaluation for given weights and alphas.

    Args:
        weights (jnp.ndarray): Weights to be evaluated with Learning Entropy.
        alphas (jnp.ndarray): Sensitivities of learning entropy, represented as a list of various sensitivities to evaluate.
        scaler (float): Scaler for OLE/Alpha evaluated weights.
        ole (int): A single OLE value to be evaluated.

    Returns:
        float: The Learning Entropy output for the given OLE.

    Notice:
        Mean absolute dw is not calculated exactly due to JAX array manipulation limitations. Depending on the OLE order, last values may contain zeros.
        These values should ideally be sliced; however, JAX does not support dynamically sized arrays without significant overhead. The error is
        minimal as long as the window is not too small and can typically be ignored.
    """
    # Get the latest dw change for given weights
    absdw = jnp.abs(weights[-ole - 1, :])
    # Get mean weight changes from all weights. Compared to native python this is not EXACT and might cause unexpected values when len(weights)<10
    # Optimal solution would be to select weights[:-ole-1, :] which is not possible as its dynamic array and it would increase calc time
    meanabsdw = jnp.mean(jnp.abs(weights), 0)
    # Evaluate all alphas as sensitivity
    Nalpha = __alpha_loop__(alphas, absdw, meanabsdw)
    # Calculate learning entropy from alpha evaluation and scaler
    ea = jnp.float32(Nalpha) / (scaler)
    return ea


@jit
def __alpha_loop__(
    alphas: jnp.ndarray, absdw: jnp.ndarray, meanabsdw: jnp.ndarray
) -> jnp.ndarray:
    """
    Evaluates all alpha values for the given latest weight change and mean weight behavior in a specified window.

    Args:
        alphas (jnp.ndarray): Sensitivities of learning entropy, represented as a list of various sensitivities to be evaluated.
        absdw (jnp.ndarray): The latest weight change delta for the given weights.
        meanabsdw (jnp.ndarray): Mean weight changes calculated from all weights.

    Returns:
        jnp.ndarray: The evaluated alpha values.
    """

    def body(shift, arg):
        Nalpha = arg
        Nalpha = Nalpha + jnp.sum(absdw > alphas[shift] * meanabsdw)
        return Nalpha

    return jax.lax.fori_loop(0, alphas.shape[0], body, (0))


@jit
def __weight_ole__(weights: jnp.ndarray, ole: jnp.ndarray) -> jnp.ndarray:
    """
    Computes weight changes according to different specified Orders of Learning Entropy (OLEs).

    Args:
        weights (jnp.ndarray): Weights to be evaluated with Learning Entropy.
        ole (int): Numeric description indicating the position of the OLE, which dictates the weight change process.

    Returns:
        jnp.ndarray: Adjusted weights in accordance with the specified OLE.
    """

    def cond(arg):
        _, step, ole = arg
        return step < ole

    def body(arg):
        weights, step, ole = arg
        temp_weights = weights[1:, :] - weights[0:-1, :]
        weights = jax.lax.dynamic_update_slice(weights, temp_weights, (0, 0))
        weights = weights.at[-1, :].set(0)
        return (weights, step + 1, ole)

    return jax.lax.while_loop(cond, body, (weights, 0, ole))


@partial(jit, static_argnums=(0,))
def aisle_window(
    window: int, weights: jnp.ndarray, alphas: jnp.ndarray, oles: jnp.ndarray
) -> jnp.ndarray:
    """
    Evaluates Approximate Individual Sample Learning Entropy (AISLE) over a selected window size using provided weights data.

    Args:
        window (int): The window size to be evaluated with Learning Entropy.
        weights (jnp.ndarray): Weights to be evaluated with the Learning Entropy algorithm.
        alphas (jnp.ndarray): Sensitivity of learning entropy, represented as a list of various sensitivities to evaluate.
        oles (jnp.ndarray): Orders of learning entropy evaluation.

    Returns:
        jnp.ndarray: The evaluation output for Learning Entropy for the given settings.
    """

    # Convert input arrays to jax arrays
    weights = jnp.array(weights)
    alphas = jnp.array(alphas)
    oles = jnp.array(oles)

    # Prepare holder for learning entropy window shift output
    ea_windowed = jnp.zeros((weights.shape[0], oles.shape[0]))

    # Prepare windows to be processed
    weight_windows = moving_window(weights, window)

    def body(shift, arg):
        ea_windowed = arg
        ea = aisle(weight_windows[shift], alphas, oles)
        ea_windowed = jax.lax.dynamic_update_slice(
            ea_windowed, ea, (shift + window, 0)
        )  # not best, but it works
        return ea_windowed

    ea_windowed = jax.lax.fori_loop(0, weight_windows.shape[0], body, (ea_windowed))

    return ea_windowed


@partial(jit, static_argnums=(1,))
def moving_window(array: jnp.ndarray, size: int) -> jnp.ndarray:
    """Creates moving average arrays to be processed.

    This function uses a flattened loop approach for each portion of data, resulting in high speed but significant memory usage.

    Args:
        array (jnp.ndarray): Input array from which moving windows should be generated.
        size (int): Size of the moving window.

    Returns:
        jnp.ndarray: Array containing all moving windows.
    """
    starts = jnp.arange(array.shape[0] - size + 1)
    return vmap(
        lambda start: jax.lax.dynamic_slice(array, (start, 0), (size, array.shape[1]))
    )(starts)


@partial(jit, static_argnums=(0,))
def aisle_window_mem_restricted(
    window: int, weights: np.ndarray, alphas: np.ndarray, oles: np.ndarray
) -> jnp.ndarray:
    """
    Evaluation of Approximate Individual Sample Learning Entropy
    with selected window over provided weights data with memory contrains.

    """
    raise NotImplementedError("This method is not implemented yet.")


if __name__ == "__main__":
    raise IOError("aisle_jax.py is not meant to be run as a script")
