"""MIT License.

Copyright (c) 2024 Phillip Lippe

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import functools
from typing import Any, Callable, Dict, Literal, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from data_parallel import fold_rng_over_axis, sync_gradients
from ml_collections import ConfigDict
from pipeline_parallel import ModelParallelismWrapper
from single_gpu import Batch, TrainState, accumulate_gradients

PyTree = Any
Parameter = jax.Array | nn.Partitioned
Metrics = Dict[str, Tuple[jax.Array, ...]]


def scale_init(init_fn: Callable, scale_factor: float = 1.0):
    """Scales the output of the given init function by the given factor.

    Args:
        init_fn: The init function to scale.
        scale_factor: The factor to scale the output of the init function by.

    Returns:
        A new init function that scales the output of the given init function by the given factor.
    """

    def _init_fn(rng, *args, **kwargs):
        return scale_factor * init_fn(rng, *args, **kwargs)

    return _init_fn


class TPDense(nn.Module):
    """Dense layer with Tensor Parallelism support.

    This layer can be used to perform a dense layer with Tensor Parallelism support.

    Attributes:
        dense_fn: Constructor function of the dense layer to use. Needs to support the keyword argument `kernel_init`.
        model_axis_name: The name of the model axis.
        tp_mode: The Tensor Parallelism mode to use. Can be "scatter", "gather", or "none".
        skip_communication: Whether to skip communication in the Tensor Parallelism strategy. Useful for layers with custom communication or where input has been already gathered beforehand.
        kernel_init: The initializer to use for the kernel of the dense layer.
        kernel_init_adjustment: The adjustment factor to use for the kernel initializer.
        dense_name: The name of the dense layer module.
    """

    dense_fn: Any
    model_axis_name: str
    tp_mode: Literal["scatter", "gather", "none"] = "none"
    skip_communication: bool = False
    kernel_init: Callable = nn.initializers.lecun_normal()
    kernel_init_adjustment: float = 1.0
    dense_name: str = "module"

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        tp_size = jax.lax.psum(1, self.model_axis_name)
        tp_mode = self.tp_mode if tp_size > 1 else "none"
        # Wrap the dense layer in a ModelParallelismWrapper to shard the parameters.
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
            # Vanilla dense layer.
            x = self.dense_fn(kernel_init=self.kernel_init)(x)
        elif tp_mode == "gather":
            # Gather strategy: communicate all the inputs to all the devices, then perform the dense layer.
            if not self.skip_communication:
                x = jax.lax.all_gather(x, self.model_axis_name, axis=-1, tiled=True)
            x = dense_fn()(x)
        elif tp_mode == "scatter":
            # Scatter strategy: perform the dense layer on each device, then communicate the outputs to all the devices.
            x = dense_fn()(x)
            if not self.skip_communication:
                x = jax.lax.psum_scatter(
                    x, axis_name=self.model_axis_name, scatter_dimension=x.ndim - 1, tiled=True
                )
        else:
            raise ValueError(f"Unknown Tensor Parallel mode: {tp_mode}")
        return x


####################################################
# Modules and functions for notebook examples.     #
# Not needed for general tensor parallelism usage. #
####################################################


class MLPBlockInput(nn.Module):
    config: ConfigDict
    features: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    use_bias: bool = True
    use_norm: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        if self.use_norm:
            x = nn.RMSNorm(dtype=self.config.dtype, name="pre_norm")(x)
        x = nn.Dense(
            features=self.features,
            kernel_init=self.kernel_init,
            use_bias=self.use_bias,
            dtype=self.config.dtype,
            name="dense",
        )(x)
        return x


class MLPBlockOutput(nn.Module):
    config: ConfigDict
    features: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.silu(x)
        x = nn.Dense(
            features=self.features,
            kernel_init=self.kernel_init,
            use_bias=self.use_bias,
            dtype=self.config.dtype,
            name="dense",
        )(x)
        return x


class TPMLPBlock(nn.Module):
    config: ConfigDict
    train: bool

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        tp_size = jax.lax.psum(1, self.config.model_axis_name)
        input_features = x.shape[-1]
        # Input layer
        x = TPDense(
            dense_fn=functools.partial(
                MLPBlockInput,
                config=self.config,
                features=self.config.hidden_size * self.config.mlp_expansion // tp_size,
            ),
            model_axis_name=self.config.model_axis_name,
            tp_mode="gather",
            name="input",
        )(x)
        # Output layer
        x = TPDense(
            dense_fn=functools.partial(
                MLPBlockOutput,
                config=self.config,
                features=input_features * tp_size,
            ),
            model_axis_name=self.config.model_axis_name,
            tp_mode="scatter",
            kernel_init_adjustment=tp_size**-0.5,  # fan-in with tp_size fewer inputs.
            name="output",
        )(x)
        return x


class TPMLPLayers(nn.Module):
    config: ConfigDict
    train: bool
    block_class: Callable[..., nn.Module] = TPMLPBlock

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        module = self.block_class(config=self.config, train=self.train, name="block")
        x, _ = nn.scan(
            lambda module, carry, _: (module(carry) + carry, None),
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            length=self.config.num_layers,
            metadata_params={
                "partition_name": None
            },  # We do not need to partition the parameters over the layer axis.
        )(module, x, ())
        return x


class TPClassifier(nn.Module):
    config: ConfigDict
    block_class: Callable[..., nn.Module] = TPMLPBlock

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        tp_size = jax.lax.psum(1, self.config.model_axis_name)
        # Input layer
        x = TPDense(
            dense_fn=functools.partial(
                nn.Dense,
                features=self.config.hidden_size // tp_size,
                dtype=self.config.dtype,
            ),
            model_axis_name=self.config.model_axis_name,
            tp_mode="gather",
            skip_communication=True,  # Input already gathered.
            name="input_layer",
        )(x)
        # Backbone MLP blocks
        x = TPMLPLayers(config=self.config, train=train, name="mlp", block_class=self.block_class)(
            x
        )
        # Output layer
        x = TPDense(
            dense_fn=functools.partial(
                nn.Dense,
                features=self.config.num_classes,
                dtype=self.config.dtype,
            ),
            model_axis_name=self.config.model_axis_name,
            tp_mode="scatter",
            skip_communication=True,  # Manual communication.
            name="output_layer",
            kernel_init_adjustment=tp_size**-0.5,  # fan-in with tp_size fewer inputs.
        )(x)
        x = jax.lax.psum(x, axis_name=self.config.model_axis_name)
        x = x.astype(jnp.float32)
        return x


def get_default_tp_classifier_config():
    data_config = ConfigDict(
        dict(
            batch_size=128,
            num_classes=10,
            input_size=784,
        )
    )
    model_config = ConfigDict(
        dict(
            hidden_size=512,
            dropout_rate=0.1,
            mlp_expansion=1,
            num_layers=3,
            dtype=jnp.bfloat16,
            num_classes=data_config.num_classes,
            data_axis_name="data",
            model_axis_name="model",
            model_axis_size=4,
        )
    )
    optimizer_config = ConfigDict(
        dict(
            learning_rate=1e-3,
            num_minibatches=1,
        )
    )
    config = ConfigDict(
        dict(
            model=model_config,
            optimizer=optimizer_config,
            data=data_config,
            data_axis_name=model_config.data_axis_name,
            model_axis_name=model_config.model_axis_name,
            model_axis_size=model_config.model_axis_size,
            seed=42,
        )
    )
    return config


def init_tp(
    rng: jax.random.PRNGKey, x: jax.Array, model: nn.Module, optimizer: Callable
) -> TrainState:
    init_rng, rng = jax.random.split(rng)
    variables = model.init({"params": init_rng}, x, train=False)
    params = variables.pop("params")
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        rng=rng,
    )
    return state


def loss_fn_tp(
    params: PyTree,
    apply_fn: Any,
    batch: Batch,
    rng: jax.Array,
    config: ConfigDict,
) -> Tuple[jax.Array, Dict[str, Any]]:
    # Since dropout masks vary across the batch dimension, we want each device to generate a
    # different mask. We can achieve this by folding the rng over the data axis, so that each
    # device gets a different rng and thus mask.
    dropout_rng = fold_rng_over_axis(rng, (config.data_axis_name, config.model_axis_name))
    # Remaining computation is the same as before for single device.
    logits = apply_fn(
        {"params": params},
        batch.inputs,
        train=True,
        rngs={"dropout": dropout_rng},
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch.labels)
    correct_pred = jnp.equal(jnp.argmax(logits, axis=-1), batch.labels)
    batch_size = np.prod(batch.labels.shape)
    # Mask out loss and accuracy for model devices except first one.
    model_idx = jax.lax.axis_index(config.model_axis_name)
    loss = jnp.where(model_idx != 0, 0.0, loss)
    correct_pred = jnp.where(model_idx != 0, False, correct_pred)
    batch_size = jnp.where(model_idx != 0, 0, batch_size)
    # Collect metrics and return loss.
    step_metrics = {
        "loss": (loss.sum(), batch_size),
        "accuracy": (correct_pred.sum(), batch_size),
    }
    loss = loss.mean()
    return loss, step_metrics


def train_step_tp(
    state: TrainState,
    metrics: Metrics | None,
    batch: Batch,
    config: ConfigDict,
    loss_fn: Callable = loss_fn_tp,
) -> Tuple[TrainState, Metrics]:
    rng, step_rng = jax.random.split(state.rng)
    grads, step_metrics = accumulate_gradients(
        state,
        batch,
        step_rng,
        config.optimizer.num_minibatches,
        loss_fn=functools.partial(loss_fn, config=config),
    )
    # Update parameters. We need to sync the gradients across devices before updating.
    with jax.named_scope("sync_gradients"):
        grads = sync_gradients(grads, (config.data_axis_name, config.model_axis_name))
    new_state = state.apply_gradients(grads=grads, rng=rng)
    # Sum metrics across replicas. Alternatively, we could keep the metrics separate
    # and only synchronize them before logging. For simplicity, we sum them here.
    with jax.named_scope("sync_metrics"):
        step_metrics = jax.tree_map(
            lambda x: jax.lax.psum(x, axis_name=(config.data_axis_name, config.model_axis_name)),
            step_metrics,
        )
    if metrics is None:
        metrics = step_metrics
    else:
        metrics = jax.tree_map(jnp.add, metrics, step_metrics)
    return new_state, metrics
