"""The aisle_jax.py logic.

Handles learning entropy computation with JAX acceleration.

Prebuilt community packages of JAX are available on: https://github.com/cloudhan/jax-windows-builder
Other way you can also build jax on your own from https://github.com/google/jax
"""

__author__ = "Ondrej Budik"
__copyright__ = "<2024> <University Southern Bohemia> <Czech Technical University>"
__credits__ = ["Ivo Bukovsky", "Czech Technical University"]

__license__ = "MIT (X11)"
__version__ = "1.0.6"
__maintainer__ = ["Ondrej Budik"]
__email__ = ["obudik@prf.jcu.cz", "ondrej.budik@fs.cvut.cz"]
__status__ = "alpha"

__python__ = "3.9.0"

import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap


@partial(jit, static_argnums=(1, 2))
def aisle(weights: np.ndarray, alphas: np.ndarray, oles: np.ndarray) -> jnp.ndarray:
    """Computes the Approximate Individual Sample Learning Entropy (AISLE) using the JAX framework.

    This function is optimized for performance on various hardware including CPUs, GPUs, and TPUs.
    It evaluates the Learning Entropy (LE) of given weights, reflecting the stability and convergence behavior
    of a learning process.

    Args:
        weights: A 2D numpy array of weights, with each row representing the weights at a particular iteration
                of the learning process.
        alphas: A 1D numpy array of thresholds determining the sensitivity of the LE calculation, with each
                element representing a different level of sensitivity.
        oles: A 1D numpy array representing the orders of LE to be calculated, providing different levels of
                granularity in understanding the learning dynamics.

    Returns:
        A 1D jax.numpy array containing the calculated LE values for each order specified in `oles`.
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
    ea = _ole_loop_(weights, alphas, oles, scaler)

    return ea


@partial(jit, static_argnums=(1, 2, 3))
def _ole_loop_(
    weights: jnp.ndarray,
    alphas: jnp.ndarray,
    oles: jnp.ndarray,
    scaler: float,
) -> jnp.ndarray:
    """Evaluates Learning Entropy (LE) for all specified Orders of Learning Entropy (OLEs) in a loop.

    This internal function is part of the AISLE implementation. It iterates over the specified OLEs,
    calculating the learning entropy for each OLE.

    Args:
        weights: A 2D jax.numpy array where each row represents the weights at a particular iteration of the
                learning process.
        alphas: A 1D jax.numpy array of thresholds for the sensitivity of the LE calculation.
        oles: A 1D jax.numpy array of the orders of LE to calculate.
        scaler: A float that acts as a scaler for the OLE/Alpha evaluated weights.

    Returns:
        A 1D jax.numpy array containing the evaluated LE values for each order specified in `oles`.
    """
    # Prepare the output array of learning entropy. Size matches LE orders.
    ea = jnp.zeros(oles.shape[0])
    # loop over all OLES and get learning entropy
    pos = 0
    step = 0
    for ole in oles:
        # Calculate dws for given OLE order. Oles usually are not many, no need to recompile for now.
        weights, step, _ = _weight_ole_(weights, ole, step)
        # Evaluate learning entropy for given OLE
        one_ea = _ole_(weights, alphas, scaler, ole)
        # Save it to array and move to the next one
        ea = ea.at[pos].set(one_ea)
        pos = pos + 1
    return ea


@partial(jit, static_argnums=(2,))
def _ole_(
    weights: jnp.ndarray,
    alphas: jnp.ndarray,
    scaler: float,
    ole: int,
) -> float:
    """Performs a single Order of Learning Entropy (OLE) evaluation for given weights and alphas.

    This function calculates the Learning Entropy (LE) for a specific order, considering the provided weights
    and sensitivity thresholds.

    Args:
        weights: A 2D jax.numpy array where each row represents the weights at a particular iteration of the
                learning process.
        alphas: A 1D jax.numpy array of thresholds for the sensitivity of the LE calculation.
        scaler: A float that acts as a scaler for the OLE/Alpha evaluated weights.
        ole: An integer representing the specific order of LE to be calculated.

    Returns:
        A float representing the LE output for the given OLE.
    """
    # Get the latest dw change for given weights
    absdw = jnp.abs(weights[-ole - 1, :])

    # Get mean dw excluding the nans
    meanabsdw = jnp.nanmean(jnp.abs(weights[:-1, :]), axis=0)

    # Evaluate all alphas as sensitivity
    n_alpha = _alpha_loop_(alphas, absdw, meanabsdw)

    # Calculate learning entropy from alpha evaluation and scaler
    ea = jnp.float32(n_alpha) / (scaler)
    return ea


@partial(jit, static_argnums=(0,))
def _alpha_loop_(alphas: jnp.ndarray, absdw: jnp.ndarray, meanabsdw: jnp.ndarray) -> jnp.ndarray:
    """Evaluates all alpha values for the given weight change and mean weight behavior in a specified window.

    This function iterates through each alpha value, calculating the sensitivity of the learning entropy.

    Args:
        alphas: A 1D jax.numpy array of thresholds for the sensitivity of the learning entropy calculation.
        absdw: A 1D jax.numpy array representing the latest weight change delta for the given weights.
        meanabsdw: A 1D jax.numpy array representing the mean weight changes calculated from all weights.

    Returns:
        A jax.numpy array representing the evaluated alpha values.
    """

    def body(shift, arg):
        n_alpha = arg
        n_alpha = n_alpha + jnp.nansum(absdw > alphas[shift] * meanabsdw)
        return n_alpha

    return jax.lax.fori_loop(0, alphas.shape[0], body, (0))


@jit
def _weight_ole_(weights: jnp.ndarray, ole: int, step: int) -> jnp.ndarray:
    """Computes weight changes according to different specified Orders of Learning Entropy (OLEs).

    This function iterates through the weights array, adjusting the weights based on the specified order of
    learning entropy.

    Args:
        weights: A 2D jax.numpy array where each row represents the weights at a particular iteration of the
                learning process.
        ole: An integer indicating the order of Learning Entropy, which dictates the weight change process.
        step: An integer indicating at which shift state are the weights.

    Returns:
        A 2D jax.numpy array of adjusted weights in accordance with the specified OLE.
    """

    def cond(arg):
        _, step, ole = arg
        return step < ole

    def body(arg):
        weights, step, ole = arg
        temp_weights = weights[1:, :] - weights[0:-1, :]
        weights = jax.lax.dynamic_update_slice(weights, temp_weights, (0, 0))
        weights = weights.at[-1, :].set(jnp.nan)
        return (weights, step + 1, ole)

    return jax.lax.while_loop(cond, body, (weights, step, ole))


@partial(jit, static_argnums=(0, 2, 3))
def aisle_window(window: int, weights: jnp.ndarray, alphas: jnp.ndarray, oles: jnp.ndarray) -> jnp.ndarray:
    """Evaluates Approximate Individual Sample Learning Entropy (AISLE) over a selected window.

    This function slides a window over the provided weights, computing the Learning Entropy for each window
    segment.

    Args:
        window: The size of the window over which to evaluate the Learning Entropy.
        weights: A 2D jax.numpy array of weights, where each row corresponds to the weights at a particular
                iteration of the learning process.
        alphas: A 1D jax.numpy array of thresholds for the sensitivity of the Learning Entropy calculation.
        oles: A 1D jax.numpy array of the orders of Learning Entropy to calculate.

    Returns:
        A 2D jax.numpy array where each row contains the Learning Entropy values for the corresponding window
        of weights. The number of columns corresponds to the number of orders in `oles`.
    """
    # Convert input arrays to jax arrays
    weights = jnp.array(weights)
    alphas = jnp.array(alphas)
    oles = jnp.array(oles)

    # Prepare holder for learning entropy window shift output
    window_zeros = jnp.zeros((window, oles.shape[0]))

    # Prepare windows to be processed
    weight_windows = moving_window(weights, window)

    # Vectorize aisle function to operate on batches of weight_windows
    vectorized_aisle = jax.vmap(aisle, in_axes=(0, None, None), out_axes=0)

    # Compute the results for the specified windows
    updates = vectorized_aisle(weight_windows, alphas, oles)

    # Create a new array with the updates
    ea_windowed_updated = jnp.concatenate([window_zeros, updates], axis=0)

    return ea_windowed_updated


@partial(jit, static_argnums=(0, 1))
def moving_window(array: jnp.ndarray, size: int) -> jnp.ndarray:
    """Creates moving window segments from the provided array.

    This function generates segments of the specified window size from the input array. It's designed for
    high-speed processing but requires significant memory, especially for large datasets.

    Args:
        array: A 2D jax.numpy array from which moving windows are to be generated.
        size: The size of each moving window.

    Returns:
        A 3D jax.numpy array where each slice along the first dimension represents a moving window of
        the input array.

    The function computes the starting indices for each window and uses the `vmap` function, along with
    `jax.lax.dynamic_slice`, to efficiently extract each window segment from the input array.
    """
    starts = jnp.arange(array.shape[0] - size)
    return vmap(lambda start: jax.lax.dynamic_slice(array, (start, 0), (size, array.shape[1])))(starts)


if __name__ == "__main__":
    filename = os.path.basename(__file__)
    MSG = f"The {filename} is not meant to be run as a script."
    raise OSError(MSG)
