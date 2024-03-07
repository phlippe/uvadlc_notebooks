"""MIT License.

Copyright (c) 2024 Phillip Lippe

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import functools
from typing import Any, Callable, Dict, List, Literal, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from pipeline_parallel import ModelParallelismWrapper
from tensor_parallel import MLPBlockInput, MLPBlockOutput, scale_init

PyTree = Any
Parameter = jax.Array | nn.Partitioned
Metrics = Dict[str, Tuple[jax.Array, ...]]


def async_gather(x: PyTree, axis_name: str, shift_up: bool = True) -> List[PyTree]:
    """All gather using ring permutation.

    Args:
        x: The input to gather.
        axis_name: The axis name to gather along.
        shift_up: Whether to shift up (device 0 send to device 1) or down (device 1 send to device 0).

    Returns:
        List of gathered inputs.
    """
    tp_size = jax.lax.psum(1, axis_name)
    # Determine communication permutation.
    if shift_up:
        shift_perm = [(j, (j + 1) % tp_size) for j in range(tp_size)]
    else:
        shift_perm = [(j, (j - 1) % tp_size) for j in range(tp_size)]
    ps = [x]
    p = x
    # Perform all-gather using ring permutation.
    for _ in range(1, tp_size):
        p = jax.lax.ppermute(p, axis_name, perm=shift_perm)
        ps.append(p)
    return ps


def async_gather_bidirectional(
    x: jax.Array, axis_name: str, shift_up: bool = True
) -> List[jax.Array]:
    """All gather using ring permutation with bidirectional communication.

    Args:
        x: The input to gather.
        axis_name: The axis name to gather along.
        shift_up: Whether to return the order of tensors that complies with the unidrectional version of shift up (device 0 send to device 1) or down (device 1 send to device 0).

    Returns:
        List of gathered inputs.
    """
    tp_size = jax.lax.psum(1, axis_name)
    shift_up_perm = [(j, (j + 1) % tp_size) for j in range(tp_size)]
    shift_down_perm = [(j, (j - 1) % tp_size) for j in range(tp_size)]
    ps_up = []
    ps_down = []
    p_up = x
    p_down = x
    for i in range(1, tp_size):
        if i % 2 == 0:
            p_down = jax.lax.ppermute(p_down, axis_name=axis_name, perm=shift_down_perm)
            ps_down.append(p_down)
        else:
            p_up = jax.lax.ppermute(p_up, axis_name=axis_name, perm=shift_up_perm)
            ps_up.append(p_up)
    # Combine communication in both directions.
    # This list will have the same order as the unidirectional up version.
    if shift_up:
        ps = [x] + ps_up + ps_down[::-1]
    else:
        ps = [x] + ps_down + ps_up[::-1]
    return ps


def async_gather_split(x: jax.Array, axis_name: str) -> List[jax.Array]:
    """All gather using ring permutation with features split for bidirectional communication.

    Args:
        x: The input to gather.
        axis_name: The axis name to gather along.

    Returns:
        List of gathered inputs. Length is 2 * axis size - 1.
    """
    x1, x2 = jax.tree_map(lambda x: jnp.split(x, 2, axis=-1), x)
    return async_gather(x1, axis_name, shift_up=True) + async_gather(x2, axis_name, shift_up=False)


def async_scatter(xs: Sequence[PyTree], axis_name: str, shift_up: bool = True) -> PyTree:
    """Scatter sum using ring permutation.

    Args:
        xs: The inputs to scatter sum. The length of the list should match the size of the axis.
        axis_name: The axis name to scatter sum along.
        shift_up: Whether to shift up (device 0 send to device 1) or down (device 1 send to device 0).

    Returns:
        The scatter summed output.
    """
    tp_size = jax.lax.psum(1, axis_name)
    assert (
        len(xs) == tp_size
    ), f"Number of shards needs to match axis size, but got {len(xs)} with {axis_name} axis size {tp_size}."
    if shift_up:
        shift_perm = [(j, (j + 1) % tp_size) for j in range(tp_size)]
    else:
        shift_perm = [(j, (j - 1) % tp_size) for j in range(tp_size)]
    y = xs[0]
    for x in xs[1:]:
        y = jax.lax.ppermute(y, axis_name, perm=shift_perm)
        y = jax.tree_map(jnp.add, y, x)
    return y


def async_scatter_split(xs: Sequence[PyTree], axis_name: str) -> PyTree:
    """Scatter sum using ring permutation with features split for bidirectional communication.

    Args:
        xs: The inputs to scatter sum. The length of the list should match the size of the axis.
        axis_name: The axis name to scatter sum along.

    Returns:
        The scatter summed output.
    """

    def _split(x: PyTree) -> Tuple[PyTree, PyTree]:
        return (
            jax.tree_map(lambda x: x[..., : x.shape[-1] // 2], x),
            jax.tree_map(lambda x: x[..., x.shape[-1] // 2 :], x),
        )

    tp_size = jax.lax.psum(1, axis_name)
    assert (
        len(xs) == tp_size
    ), f"Number of shards needs to match axis size, but got {len(xs)} with {axis_name} axis size {tp_size}."
    shift_perm_up = [(j, (j + 1) % tp_size) for j in range(tp_size)]
    shift_perm_down = [(j, (j - 1) % tp_size) for j in range(tp_size)]
    y_up, y_down = _split(xs[0])
    for x in xs[1:]:
        y_up = jax.lax.ppermute(y_up, axis_name, perm=shift_perm_up)
        y_down = jax.lax.ppermute(y_down, axis_name, perm=shift_perm_down)
        x_up, x_down = _split(x)
        y_up = jax.tree_map(jnp.add, y_up, x_up)
        y_down = jax.tree_map(jnp.add, y_down, x_down)
    return jax.tree_map(lambda y1, y2: jnp.concatenate([y1, y2], axis=-1), y_up, y_down)


class TPAsyncDense(nn.Module):
    """Tensor-Parallel Dense Layer with Asynchronous Communication.

    This layer can be used to perform a dense layer with Tensor Parallelism support, and overlaps communication with computation whenever possible.

    Attributes:
        dense_fn: Constructor function of the dense layer to use. Needs to support the keyword argument `kernel_init`.
        model_axis_name: The name of the model axis.
        tp_mode: The Tensor Parallelism mode to use. Can be "scatter", "gather", or "none".
        kernel_init: The initializer to use for the kernel of the dense layer.
        kernel_init_adjustment: The adjustment factor to use for the kernel initializer.
        dense_name: The name of the dense layer module.
        use_bidirectional_gather: Whether to use bidirectional or unidirectional gather over the device ring for communication.
        use_bidirectional_scatter: Whether to use bidirectional or unidirectional scatter over the device ring for communication.
    """

    dense_fn: Any
    model_axis_name: str
    tp_mode: Literal["scatter", "gather", "none"] = "none"
    kernel_init: Callable = nn.initializers.lecun_normal()
    kernel_init_adjustment: float = 1.0
    dense_name: str = "module"
    use_bidirectional_gather: bool = True
    use_bidirectional_scatter: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        tp_size = jax.lax.psum(1, self.model_axis_name)
        tp_mode = self.tp_mode if tp_size > 1 else "none"

        dense_fn = functools.partial(
            ModelParallelismWrapper,
            model_axis_name=self.model_axis_name,
            module_fn=functools.partial(
                self.dense_fn,
                kernel_init=scale_init(self.kernel_init, self.kernel_init_adjustment),
            ),
            name=self.dense_name,
        )

        if tp_mode == "none":
            y = self.dense_fn(kernel_init=self.kernel_init, name="shard_0")(x)
        elif tp_mode == "gather":
            # Async gathering of all inputs.
            async_op = (
                async_gather_bidirectional if self.use_bidirectional_gather else async_gather
            )
            xs = async_op(x, axis_name=self.model_axis_name)
            # Compute output per input (scheduled as communication makes inputs available).
            ys = [
                dense_fn(
                    module_kwargs={
                        "use_bias": (i == 0)
                    },  # Only need a single per final output feature.
                    name=f"shard_{i}",
                )(x)
                for i, x in enumerate(xs)
            ]
            # Final sum of all outputs.
            y = jax.tree_map(lambda *args: sum(args), *ys)
        elif tp_mode == "scatter":
            # Calculate all outputs per device.
            ys = [
                dense_fn(
                    module_kwargs={
                        "use_bias": (i == 0)
                    },  # Only need a single per final output feature.
                    name=f"shard_{i}",
                )(x)
                for i in range(tp_size)
            ]
            # Async scatter sum of all outputs (communication already starts after first output is ready).
            async_op = async_scatter_split if self.use_bidirectional_scatter else async_scatter
            y = async_op(ys, axis_name=self.model_axis_name)
        else:
            raise ValueError(f"Unknown Tensor Parallel mode: {tp_mode}")
        return y


class TPNorm(nn.Module):
    config: ConfigDict

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = ModelParallelismWrapper(
            model_axis_name=self.config.model_axis_name,
            module_fn=functools.partial(
                nn.RMSNorm,
                dtype=self.config.dtype,
                axis_name=self.config.model_axis_name,
            ),
            name="norm",
        )(x)
        return x


class TPAsyncMLPBlock(nn.Module):
    config: ConfigDict
    train: bool

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        tp_size = jax.lax.psum(1, self.config.model_axis_name)
        input_features = x.shape[-1]
        # Normalize across devices before the input layer.
        x = TPNorm(config=self.config, name="pre_norm")(x)
        # Input dense layer with async gather.
        x = TPAsyncDense(
            dense_fn=functools.partial(
                MLPBlockInput,
                config=self.config,
                features=self.config.hidden_size * self.config.mlp_expansion // tp_size,
                use_norm=False,
            ),
            model_axis_name=self.config.model_axis_name,
            tp_mode="gather",
            kernel_init_adjustment=tp_size**-0.5,
            name="input",
        )(x)
        # Output dense layer with async scatter.
        x = TPAsyncDense(
            dense_fn=functools.partial(
                MLPBlockOutput,
                config=self.config,
                features=input_features,
            ),
            model_axis_name=self.config.model_axis_name,
            tp_mode="scatter",
            kernel_init_adjustment=tp_size**-0.5,
            name="output",
        )(x)
        return x
