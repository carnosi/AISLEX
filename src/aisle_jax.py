"""The aisle_jax.py logic.

Handles learning entropy computation with JAX acceleration.

Prebuilt community packages of JAX are available on: https://github.com/cloudhan/jax-windows-builder
Other way you can also build jax on your own from https://github.com/google/jax
"""

__author__ = "Ondrej Budik"
__copyright__ = "<2024> <University Southern Bohemia> <Czech Technical University>"
__credits__ = ["Ivo Bukovsky", "Czech Technical University"]

__license__ = "MIT (X11)"
__version__ = "1.0.8"
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

    ea = ea.reshape(1, -1)

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

    def body_fn(carry: tuple[jnp.ndarray, int], ole: int) -> tuple[tuple[jnp.ndarray, int], int]:
        weights, step = carry
        weights, step, _ = _weight_ole_(weights, ole, step)
        one_ea = _ole_(weights, alphas, scaler, ole)
        return (weights, step), one_ea

    # Initial carry values
    initial_carry = (weights, 0)

    # Use jax.lax.scan for the loop
    _, ea = jax.lax.scan(body_fn, initial_carry, oles)

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

    def single_alpha_evaluation(alpha: int) -> jnp.array:
        return jnp.nansum(absdw > alpha * meanabsdw)

    # Use jax.vmap to vectorize the single_alpha_evaluation function
    n_alpha_per_alpha = jax.vmap(single_alpha_evaluation)(alphas)

    # Sum the results to get a single value
    n_alpha = jnp.sum(n_alpha_per_alpha)

    return n_alpha


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

    # Prepare windows to be processed
    weight_windows = moving_window(weights, window)

    # Define the function to process each window shift
    def process_window(weight_window):
        return aisle(weight_window, alphas, oles)

    # Vectorize the function over weight_windows
    # ea_windowed_updates = vmap(process_window)(weight_windows) #vmap shows worse perf than lax.map
    ea_windowed_updates = jax.lax.map(process_window, weight_windows)
    # Get rid of the extra dimension caused by parallel processing
    ea_windowed_updates = ea_windowed_updates.squeeze(axis=1)

    # Concatenate with zero padding to match the input size
    zero_padding = jnp.zeros((window - 1, ea_windowed_updates.shape[1]))
    ea_windowed = jnp.vstack([zero_padding, ea_windowed_updates])

    return ea_windowed


@partial(jit, static_argnums=(0, 2, 3, 4))
def aisle_window_chunked(
    window: int, weights: jnp.ndarray, alphas: jnp.ndarray, oles: jnp.ndarray, chunk_size: int
) -> jnp.ndarray:
    """Evaluates Approximate Individual Sample Learning Entropy (AISLE) over a selected window using chunking.

    Args:
        window: The size of the window over which to evaluate the Learning Entropy.
        weights: A 2D jax.numpy array of weights, where each row corresponds to the weights at a particular
                iteration of the learning process.
        alphas: A 1D jax.numpy array of thresholds for the sensitivity of the Learning Entropy calculation.
        oles: A 1D jax.numpy array of the orders of Learning Entropy to calculate.
        chunk_size: The number of windows to process per chunk for memory efficiency.

    Returns:
        A 2D jax.numpy array where each row contains the Learning Entropy values for the corresponding window
        of weights. The number of columns corresponds to the number of orders in `oles`.
    """

    def moving_window_chunk(
        weights: jnp.ndarray, window_size: int, start_idx: int, num_windows: int
    ) -> jnp.ndarray:
        """Creates a chunk of moving window segments from the provided array."""
        starts = jnp.arange(start_idx, start_idx + num_windows)
        return vmap(
            lambda start: jax.lax.dynamic_slice(weights, (start, 0), (window_size, weights.shape[1]))
        )(starts)

    def process_chunk(start_idx, num_windows):
        # Generate the windows for this chunk
        weight_windows_chunk = moving_window_chunk(weights, window, start_idx, num_windows)
        # Process each window in the chunk using aisle
        result_chunk = vmap(lambda w: aisle(w, alphas, oles))(weight_windows_chunk)

        return result_chunk

    num_windows = weights.shape[0] - window + 1
    ea_windowed = jnp.zeros((weights.shape[0], 1, len(oles)))

    # Process the weights in chunks
    for start_idx in range(0, num_windows, chunk_size):
        end_idx = min(start_idx + chunk_size, num_windows)
        num_windows_in_chunk = end_idx - start_idx

        # Process the current chunk of windows
        result_chunk = process_chunk(start_idx, num_windows_in_chunk)
        ea_windowed = ea_windowed.at[start_idx + window - 1 : end_idx + window - 1, :, :].set(result_chunk)

    ea_windowed = jnp.squeeze(ea_windowed, axis=1)

    return ea_windowed


@partial(jit, static_argnums=(1,))
def moving_window(weights: jnp.ndarray, window_size: int) -> jnp.ndarray:
    """Creates moving window segments from the provided array.

    This function generates segments of the specified window size from the input array. It's designed for
    high-speed processing but requires significant memory, especially for large datasets.

    Args:
        weights: A 2D jax.numpy array from which moving windows are to be generated.
        window_size: The size of each moving window.

    Returns:
        A 3D jax.numpy array where each slice along the first dimension represents a moving window of
        the input array.

    The function computes the starting indices for each window and uses the `vmap` function, along with
    `jax.lax.dynamic_slice`, to efficiently extract each window segment from the input array.
    """
    starts = jnp.arange(weights.shape[0] - window_size + 1)
    return vmap(lambda start: jax.lax.dynamic_slice(weights, (start, 0), (window_size, weights.shape[1])))(
        starts
    )


if __name__ == "__main__":
    filename = os.path.basename(__file__)
    MSG = f"The {filename} is not meant to be run as a script."
    raise OSError(MSG)
