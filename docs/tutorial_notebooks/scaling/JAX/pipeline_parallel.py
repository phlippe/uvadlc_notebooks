"""MIT License.

Copyright (c) 2024 Phillip Lippe

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import functools
from typing import Any, Callable, Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from data_parallel import fold_rng_over_axis, sync_gradients
from flax.core.frozen_dict import FrozenDict
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from ml_collections import ConfigDict
from single_gpu import Batch, TrainState, accumulate_gradients

PyTree = Any
Parameter = jax.Array | nn.Partitioned
Metrics = Dict[str, Tuple[jax.Array, ...]]


def stack_params(
    params: PyTree, axis_name: str, axis: int = 0, mask_except: jax.Array | int | None = None
) -> PyTree:
    """Stacks sharded parameters along a given axis name.

    Args:
        params: PyTree of parameters.
        axis_name: Name of the axis to stack along.
        axis: Index of the axis to stack along.
        mask_except: If not None, only the `mask_except`-th shard will be non-zero.

    Returns:
        PyTree of parameters with the same structure as `params`, but with the leaf
        nodes replaced by `nn.Partitioned` objects with sharding over axis name added
        to `axis`-th axis of parameters.
    """

    def _stack(x: Parameter) -> Parameter:
        if isinstance(x, nn.Partitioned):
            value, names = x.value, x.names
        else:
            value, names = x, (None,) * x.ndim
        if mask_except is not None:
            axis_index = jax.lax.axis_index(axis_name)
            value = jnp.where(axis_index == mask_except, value, 0.0)
        value = jnp.expand_dims(value, axis)
        names = names[:axis] + (axis_name,) + names[axis:]
        return nn.Partitioned(value, names=names)

    return jax.tree_map(_stack, params, is_leaf=lambda x: isinstance(x, nn.Partitioned))


def unstack_params(params: PyTree, axis_name: str) -> PyTree:
    """Unstacks parameters along a given axis name.

    Inverse operation to `stack_params`.

    Args:
        params: PyTree of parameters.
        axis_name: Name of the axis to unstack along.

    Returns:
        PyTree of parameters with the same structure as `params`, but
        with the leaf nodes having the sharding over the axis name removed.
    """

    def _unstack(x: Parameter) -> Parameter:
        if isinstance(x, nn.Partitioned) and axis_name in x.names:
            value = x.value
            names = x.names
            axis_idx = names.index(axis_name)
            value = value.squeeze(axis_idx)
            names = names[:axis_idx] + names[axis_idx + 1 :]
            if all([n is None for n in names]):
                return value
            else:
                return nn.Partitioned(value, names=names)
        else:
            return x

    return jax.tree_map(_unstack, params, is_leaf=lambda x: isinstance(x, nn.Partitioned))


def execute_pipeline_step(
    module: nn.Module,
    state: jax.Array,
    input: jax.Array,
    *args,
    model_axis_name: str,
    **kwargs,
) -> Tuple[jax.Array, jax.Array]:
    """Single micro-batch pipeline step.

    Args:
        module: Flax module representing the stage to execute.
        state: Last communicated features between stages. Used as input to the module for all stages except the first.
        input: Original micro-batch input to the pipeline stage. Used as input to the module for the first stage.
        *args: Additional arguments to the module.
        model_axis_name: Name of the model axis in the mesh/shard_map.
        **kwargs: Additional keyword arguments to the module.

    Returns:
        Tuple of the new state (after communication) and the output of the module.
    """
    num_stages = jax.lax.psum(1, model_axis_name)
    stage_index = jax.lax.axis_index(model_axis_name)
    # For the first stage, we use the microbatches as input.
    # For all other stages, we use the last state from the
    # previous stage as input.
    state = jnp.where(stage_index == 0, input, state)
    state = module(state, *args, **kwargs)
    # For the last stage, we return the state as output.
    # For all other stages, we return zeros.
    output = jnp.where(
        stage_index == num_stages - 1,
        state,
        jnp.zeros_like(state),
    )
    # Communicate the last state to the next stage.
    state = jax.lax.ppermute(
        state,
        model_axis_name,
        perm=[(i, (i + 1) % num_stages) for i in range(num_stages)],
    )
    return (state, output)


@jax.named_scope("pipeline")  # Naming scope for profiling.
def execute_pipeline(
    module: nn.Module, x: jax.Array, *args, num_microbatches: int, model_axis_name: str, **kwargs
) -> jax.Array:
    """Execute a pipeline of stages on a batch of data.

    Uses the principle of GPipe in splitting the batch into micro-batches
    and running the pipeline stages in parallel.

    Args:
        module: Flax module representing the pipeline stage to execute.
        x: Batch of input data, only needed on device of the first stage. Data will be split into micro-batches.
        *args: Additional arguments to the module.
        num_microbatches: Number of micro-batches to split the batch into.
        model_axis_name: Name of the model axis in the mesh/shard_map.
        **kwargs: Additional keyword arguments to the module.

    Returns:
        Output of the last stage of the pipeline. For devices that are not
        the last stage, the output is zeros.
    """
    num_stages = jax.lax.psum(1, model_axis_name)
    # Structure the input data into micro-batches.
    batch_size = x.shape[0]
    assert (
        batch_size % num_microbatches == 0
    ), f"Batch size {batch_size} must be divisible by number of microbatches {num_microbatches}"
    microbatch_size = batch_size // num_microbatches
    microbatches = jnp.reshape(x, (num_microbatches, microbatch_size, *x.shape[1:]))
    inputs = jnp.concatenate(  # Add zeros for unused computation blocks in first stage.
        [
            microbatches,
            jnp.zeros((num_stages - 1, *microbatches.shape[1:]), dtype=microbatches.dtype),
        ],
        axis=0,
    )
    state = jnp.zeros_like(microbatches[0])
    num_iterations = inputs.shape[0]
    # Run loop over pipeline steps.
    _, outputs = nn.scan(
        functools.partial(
            execute_pipeline_step,
            *args,
            model_axis_name=model_axis_name,
            **kwargs,
        ),
        variable_broadcast={"params": True},
        split_rngs={"params": False, "dropout": True},
        length=num_iterations,
        in_axes=0,
        out_axes=0,
    )(module, state, inputs)
    # Take last N outputs (first ones are zeros from unused computation blocks in last stage).
    outputs = jnp.concatenate(outputs[-num_microbatches:], axis=0)
    return outputs


class PipelineModule(nn.Module):
    model_axis_name: str
    num_microbatches: int
    module_fn: Callable[..., nn.Module]

    @nn.compact
    def __call__(self, *args, **kwargs):
        module = self.module_fn()
        return execute_pipeline(
            module,
            *args,
            **kwargs,
            num_microbatches=self.num_microbatches,
            model_axis_name=self.model_axis_name,
        )


class ModelParallelismWrapper(nn.Module):
    """Wrapper for adding model parallelism to a module.

    This wrapper adds sharding over the model axis to the parameters of the module
    and initializes the module with different parameters across the model axis.

    Args:
        model_axis_name: Name of the model axis to shard over.
        module_fn: Function that returns the Flax module to wrap.
        mask_except_model_idx: If not None, only the `mask_except_model_idx`-th shard will be non-zero.
        split_rngs: If True, split the random number generators across the model axis.
        module_kwargs: Additional keyword arguments to pass to the module function.
    """

    model_axis_name: str
    module_fn: Callable[..., nn.Module]
    mask_except_model_idx: int | None = None
    split_rngs: bool = True
    module_kwargs: FrozenDict[str, Any] = FrozenDict({})

    @nn.compact
    def __call__(self, *args, **kwargs):
        if self.is_initializing() and self.split_rngs:
            # Initialize each module across the model axis with different parameters.
            self.scope.rngs["params"] = self.scope.rngs["params"].replace(
                rng=fold_rng_over_axis(self.scope.rngs["params"].rng, self.model_axis_name)
            )
        # Wrap variables in nn.Partitioned objects to add sharding over the model axis.
        module = nn.map_variables(
            target=functools.partial(
                self.module_fn,
                name="sharded",
                **self.module_kwargs,
            ),
            trans_in_fn=functools.partial(unstack_params, axis_name=self.model_axis_name),
            trans_out_fn=functools.partial(
                stack_params,
                axis_name=self.model_axis_name,
                mask_except=self.mask_except_model_idx,
            ),
            mapped_collections="params",
            mutable=True,
        )()
        return module(
            *args,
            **kwargs,
        )


######################################################
# Modules and functions for notebook examples.       #
# Not needed for general pipeline parallelism usage. #
######################################################


class MLPBlock(nn.Module):
    config: ConfigDict
    train: bool

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        input_features = x.shape[-1]
        residual = x
        x = nn.LayerNorm(dtype=self.config.dtype, name="pre_norm")(x)
        x = nn.Dense(
            features=self.config.hidden_size * self.config.mlp_expansion,
            dtype=self.config.dtype,
            name="input_dense",
        )(x)
        x = nn.silu(x)
        x = nn.Dropout(rate=self.config.dropout_rate, deterministic=not self.train)(x)
        x = nn.Dense(features=input_features, dtype=self.config.dtype, name="output_dense")(x)
        return x + residual


class MLPLayers(nn.Module):
    config: ConfigDict
    train: bool

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # Scan version
        block_class = MLPBlock
        if "MLP" in self.config.remat:
            block_class = nn.remat(block_class, prevent_cse=False)
        block = block_class(config=self.config, train=self.train, name="block")
        x, _ = nn.scan(
            lambda module, carry, _: (module(carry), ()),
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            length=self.config.num_layers,
        )(block, x, ())
        # Non-scanned version
        # for i in range(self.config.num_layers):
        #     x = block_class(self.config, train=train, name=f"block_{i}")(x)
        return x


class PPClassifier(nn.Module):
    config: ConfigDict
    pipeline_module_class: Callable[..., nn.Module] = PipelineModule

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        # Input layer. Only needed in the first stage.
        x = ModelParallelismWrapper(
            module_fn=functools.partial(
                nn.Dense,
                features=self.config.hidden_size,
                dtype=self.config.dtype,
            ),
            model_axis_name=self.config.model_axis_name,
            mask_except_model_idx=0,
            name="input_dense",
        )(x)
        # Pipeline
        stage_module_fn = functools.partial(
            MLPLayers, config=self.config, train=train, name="mlp_layers"
        )
        pipeline_module_fn = functools.partial(
            self.pipeline_module_class,
            model_axis_name=self.config.model_axis_name,
            num_microbatches=self.config.num_microbatches,
            module_fn=stage_module_fn,
        )
        module = ModelParallelismWrapper(
            module_fn=pipeline_module_fn,
            model_axis_name=self.config.model_axis_name,
            name="pipeline",
        )
        x = module(x)
        # Output layer. Only needed in the last stage.
        output_wrapper = functools.partial(
            ModelParallelismWrapper,
            model_axis_name=self.config.model_axis_name,
            mask_except_model_idx=self.config.model_axis_size - 1,
        )
        x = output_wrapper(
            module_fn=functools.partial(nn.LayerNorm, dtype=self.config.dtype), name="output_norm"
        )(x)
        x = output_wrapper(
            module_fn=functools.partial(
                nn.Dense, features=self.config.num_classes, dtype=self.config.dtype
            ),
            name="output_dense",
        )(x)
        x = x.astype(jnp.float32)
        return x


def get_default_pp_classifier_config() -> ConfigDict:
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
            mlp_expansion=1,
            dropout_rate=0.1,
            num_layers=8,
            dtype=jnp.float32,
            num_classes=data_config.num_classes,
            remat=(),
            data_axis_name="data",
            model_axis_name="model",
            model_axis_size=4,
            num_microbatches=8,
        )
    )
    model_config.num_layers //= model_config.model_axis_size  # Layers distributed over model axis.
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


def loss_fn_pp(
    params: PyTree, apply_fn: Any, batch: Batch, rng: jax.Array, config: ConfigDict
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
    batch_size = batch.inputs.shape[0]
    # Mask out loss and accuracy for pipeline stages except last one.
    model_idx = jax.lax.axis_index(config.model_axis_name)
    model_size = jax.lax.psum(1, config.model_axis_name)
    loss = jnp.where(model_idx != model_size - 1, 0.0, loss)
    correct_pred = jnp.where(model_idx != model_size - 1, False, correct_pred)
    batch_size = jnp.where(model_idx != model_size - 1, 0, batch_size)
    # Collect metrics and return loss.
    step_metrics = {
        "loss": (loss.sum(), batch_size),
        "accuracy": (correct_pred.sum(), batch_size),
    }
    loss = loss.mean()
    return loss, step_metrics


def train_step_pp(
    state: TrainState,
    metrics: Metrics | None,
    batch: Batch,
    config: ConfigDict,
) -> Tuple[TrainState, Metrics]:
    rng, step_rng = jax.random.split(state.rng)
    grads, step_metrics = accumulate_gradients(
        state,
        batch,
        step_rng,
        config.optimizer.num_minibatches,
        loss_fn=functools.partial(loss_fn_pp, config=config),
    )
    # Update parameters. We need to sync the gradients across data devices before updating.
    with jax.named_scope("sync_gradients"):
        grads = sync_gradients(grads, (config.data_axis_name, config.model_axis_name))
    new_state = state.apply_gradients(grads=grads, rng=rng)
    # Sum metrics across replicas (both model and data axes).
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


def train_pipeline_model(
    config: ConfigDict,
    mesh: Mesh,
    batch: Batch,
    model_init_rng: jax.Array,
    num_steps: int,
) -> TrainState:
    """Train a pipeline model on a given batch.

    Reproduces the training loop for the Part 3.1 notebook.
    """
    model = PPClassifier(config=config.model)
    optimizer = optax.adamw(
        learning_rate=config.optimizer.learning_rate,
    )

    def init_fn(rng: jax.random.PRNGKey, x: jax.Array) -> TrainState:
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

    init_pp_fn = shard_map(
        init_fn,
        mesh,
        in_specs=(P(), P(config.data_axis_name)),
        out_specs=P(),
        check_rep=False,
    )
    state_pp_shapes = jax.eval_shape(init_pp_fn, model_init_rng, batch.inputs)
    state_pp_specs = nn.get_partition_spec(state_pp_shapes)
    init_pp_fn = jax.jit(
        shard_map(
            init_fn,
            mesh,
            in_specs=(P(), P(config.data_axis_name)),
            out_specs=state_pp_specs,
            check_rep=False,
        ),
    )
    state_pp = init_pp_fn(model_init_rng, batch.inputs)

    train_step_pp_fn = jax.jit(
        shard_map(
            functools.partial(train_step_pp, config=config),
            mesh,
            in_specs=(state_pp_specs, P(), P(config.data_axis_name)),
            out_specs=(state_pp_specs, P()),
            check_rep=False,
        ),
        donate_argnames=("state", "metrics"),
    )
    _, metric_shapes = jax.eval_shape(
        train_step_pp_fn,
        state_pp,
        None,
        batch,
    )
    metrics_pp = jax.tree_map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), metric_shapes)
    for _ in range(num_steps):
        state_pp, metrics_pp = train_step_pp_fn(state_pp, metrics_pp, batch)
    return state_pp
