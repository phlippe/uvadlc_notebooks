"""MIT License.

Copyright (c) 2024 Phillip Lippe

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass
from flax.training import train_state

# Type aliases
PyTree = Any
Metrics = Dict[str, Tuple[jax.Array, ...]]


class TrainState(train_state.TrainState):
    rng: jax.Array


@dataclass
class Batch:
    inputs: jax.Array
    labels: jax.Array


def accumulate_gradients_loop(
    state: TrainState,
    batch: Batch,
    rng: jax.random.PRNGKey,
    num_minibatches: int,
    loss_fn: Callable,
) -> Tuple[PyTree, Metrics]:
    """Calculate gradients and metrics for a batch using gradient accumulation.

    Args:
        state: Current training state.
        batch: Full training batch.
        rng: Random number generator to use.
        num_minibatches: Number of minibatches to split the batch into. Equal to the number of gradient accumulation steps.
        loss_fn: Loss function to calculate gradients and metrics.

    Returns:
        Tuple with accumulated gradients and metrics over the minibatches.
    """
    batch_size = batch.inputs.shape[0]
    minibatch_size = batch_size // num_minibatches
    rngs = jax.random.split(rng, num_minibatches)
    # Define gradient function for single minibatch.
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    # Prepare loop variables.
    grads = None
    metrics = None
    for minibatch_idx in range(num_minibatches):
        with jax.named_scope(f"minibatch_{minibatch_idx}"):
            # Split the batch into minibatches.
            start = minibatch_idx * minibatch_size
            end = start + minibatch_size
            minibatch = jax.tree_map(lambda x: x[start:end], batch)
            # Calculate gradients and metrics for the minibatch.
            (_, step_metrics), step_grads = grad_fn(
                state.params, state.apply_fn, minibatch, rngs[minibatch_idx]
            )
            # Accumulate gradients and metrics across minibatches.
            if grads is None:
                grads = step_grads
                metrics = step_metrics
            else:
                grads = jax.tree_map(jnp.add, grads, step_grads)
                metrics = jax.tree_map(jnp.add, metrics, step_metrics)
    # Average gradients over minibatches.
    grads = jax.tree_map(lambda g: g / num_minibatches, grads)
    return grads, metrics


def accumulate_gradients_scan(
    state: TrainState,
    batch: Batch,
    rng: jax.random.PRNGKey,
    num_minibatches: int,
    loss_fn: Callable,
) -> Tuple[PyTree, Metrics]:
    """Calculate gradients and metrics for a batch using gradient accumulation.

    In this version, we use `jax.lax.scan` to loop over the minibatches. This is more efficient in terms of compilation time.

    Args:
        state: Current training state.
        batch: Full training batch.
        rng: Random number generator to use.
        num_minibatches: Number of minibatches to split the batch into. Equal to the number of gradient accumulation steps.
        loss_fn: Loss function to calculate gradients and metrics.

    Returns:
        Tuple with accumulated gradients and metrics over the minibatches.
    """
    batch_size = batch.inputs.shape[0]
    minibatch_size = batch_size // num_minibatches
    rngs = jax.random.split(rng, num_minibatches)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def _minibatch_step(minibatch_idx: jax.Array | int) -> Tuple[PyTree, Metrics]:
        """Determine gradients and metrics for a single minibatch."""
        minibatch = jax.tree_map(
            lambda x: jax.lax.dynamic_slice_in_dim(  # Slicing with variable index (jax.Array).
                x, start_index=minibatch_idx * minibatch_size, slice_size=minibatch_size, axis=0
            ),
            batch,
        )
        (_, step_metrics), step_grads = grad_fn(
            state.params, state.apply_fn, minibatch, rngs[minibatch_idx]
        )
        return step_grads, step_metrics

    def _scan_step(
        carry: Tuple[PyTree, Metrics], minibatch_idx: jax.Array | int
    ) -> Tuple[Tuple[PyTree, Metrics], None]:
        """Scan step function for looping over minibatches."""
        step_grads, step_metrics = _minibatch_step(minibatch_idx)
        carry = jax.tree_map(jnp.add, carry, (step_grads, step_metrics))
        return carry, None

    # Determine initial shapes for gradients and metrics.
    grads_shapes, metrics_shape = jax.eval_shape(_minibatch_step, 0)
    grads = jax.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), grads_shapes)
    metrics = jax.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), metrics_shape)
    # Loop over minibatches to determine gradients and metrics.
    (grads, metrics), _ = jax.lax.scan(
        _scan_step, init=(grads, metrics), xs=jnp.arange(num_minibatches), length=num_minibatches
    )
    # Average gradients over minibatches.
    grads = jax.tree_map(lambda g: g / num_minibatches, grads)
    return grads, metrics


def accumulate_gradients(
    state: TrainState,
    batch: Batch,
    rng: jax.random.PRNGKey,
    num_minibatches: int,
    loss_fn: Callable,
    use_scan: bool = False,
) -> Tuple[PyTree, Metrics]:
    """Calculate gradients and metrics for a batch using gradient accumulation.

    This function supports scanning over the minibatches using `jax.lax.scan` or using a for loop.

    Args:
        state: Current training state.
        batch: Full training batch.
        rng: Random number generator to use.
        num_minibatches: Number of minibatches to split the batch into. Equal to the number of gradient accumulation steps.
        loss_fn: Loss function to calculate gradients and metrics.
        use_scan: Whether to use `jax.lax.scan` for looping over the minibatches.

    Returns:
        Tuple with accumulated gradients and metrics over the minibatches.
    """
    if use_scan:
        return accumulate_gradients_scan(
            state=state, batch=batch, rng=rng, num_minibatches=num_minibatches, loss_fn=loss_fn
        )
    else:
        return accumulate_gradients_loop(
            state=state, batch=batch, rng=rng, num_minibatches=num_minibatches, loss_fn=loss_fn
        )


def print_metrics(metrics: Metrics, title: str | None = None) -> None:
    """Prints metrics with an optional title."""
    metrics = jax.device_get(metrics)
    lines = [f"{k}: {v[0] / v[1]:.6f}" for k, v in metrics.items()]
    if title:
        title = f" {title} "
        max_len = max(len(title), max(map(len, lines)))
        lines = [title.center(max_len, "=")] + lines
    print("\n".join(lines))


def get_num_params(state: TrainState) -> int:
    """Calculate the number of parameters in the model."""
    return sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(state.params))
