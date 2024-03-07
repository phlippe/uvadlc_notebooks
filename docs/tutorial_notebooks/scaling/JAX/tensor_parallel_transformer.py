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
import numpy as np
from data_parallel import shard_module_params
from ml_collections import ConfigDict
from pipeline_parallel import ModelParallelismWrapper
from tensor_parallel import MLPBlockInput, MLPBlockOutput
from tensor_parallel_async import TPAsyncDense, TPAsyncMLPBlock, TPNorm

PyTree = Any
Parameter = jax.Array | nn.Partitioned
Metrics = Dict[str, Tuple[jax.Array, ...]]


class QKVDense(nn.Module):
    config: ConfigDict
    num_heads: int
    head_dim: int
    kernel_init: Callable
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        q = nn.DenseGeneral(
            (self.num_heads, self.head_dim),
            kernel_init=self.kernel_init,
            use_bias=False,
            dtype=self.config.dtype,
            name="query",
        )(x)
        k = nn.DenseGeneral(
            (self.num_heads, self.head_dim),
            kernel_init=self.kernel_init,
            use_bias=False,
            dtype=self.config.dtype,
            name="key",
        )(x)
        v = nn.DenseGeneral(
            (self.num_heads, self.head_dim),
            kernel_init=self.kernel_init,
            use_bias=False,
            dtype=self.config.dtype,
            name="value",
        )(x)

        if self.config.normalize_qk:
            q = nn.RMSNorm(
                dtype=self.config.dtype,
                name="query_norm",
            )(q)
            k = nn.RMSNorm(
                dtype=self.config.dtype,
                name="key_norm",
            )(k)
        return q, k, v


def dot_product_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    mask: jax.Array | None,
    softmax_dtype: jnp.dtype = jnp.float32,
):
    """Dot-product attention.

    Follows the setup of https://flax.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.dot_product_attention,
    but supports switch to float32 for numerical stability during softmax.

    Args:
        query: The query array, shape [..., num queries, num heads, hidden size].
        key: The key array, shape [..., num keys, num heads, hidden size].
        value: The value array, shape [..., num keys, num heads, hidden size].
        mask: The boolean mask array (0 for masked values, 1 for non-masked). If None, no masking is applied.
        softmax_dtype: The dtype to use for the softmax operation.

    Returns:
        The attention output array, shape [..., num queries, num heads, hidden size].
    """
    num_features = query.shape[-1]
    dtype = query.dtype
    scale = num_features**-0.5
    query = query * scale
    # Switch dtype right before the dot-product for numerical stability.
    query = query.astype(softmax_dtype)
    key = key.astype(softmax_dtype)
    weights = jnp.einsum("...qhd,...khd->...hqk", query, key)
    if mask is not None:
        weights = jnp.where(mask, weights, jnp.finfo(softmax_dtype).min)
    weights = nn.softmax(weights, axis=-1)
    # After softmax, switch back to the original dtype
    weights = weights.astype(dtype)
    new_vals = jnp.einsum("...hqk,...khd->...qhd", weights, value)
    new_vals = new_vals.astype(dtype)
    return new_vals


class AttnOut(nn.Module):
    config: ConfigDict
    features: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.DenseGeneral(
            features=self.features,
            axis=(-2, -1),
            kernel_init=self.kernel_init,
            use_bias=self.use_bias,
            dtype=self.config.dtype,
            name="out",
        )(x)
        return x


class TPMultiHeadAttn(nn.Module):
    config: ConfigDict
    train: bool
    mask: jax.Array | None = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        tp_size = jax.lax.psum(1, self.config.model_axis_name)
        input_features = x.shape[-1]
        head_dim = self.config.head_dim
        num_heads = self.config.num_heads
        # Normalize across devices before the input layer.
        x = TPNorm(config=self.config, name="pre_norm")(x)
        # Calculate Q, K, V using async dense layers.
        q, k, v = TPAsyncDense(
            dense_fn=functools.partial(
                QKVDense,
                config=self.config,
                num_heads=num_heads // tp_size,
                head_dim=head_dim,
            ),
            model_axis_name=self.config.model_axis_name,
            tp_mode="gather",
            kernel_init_adjustment=tp_size**-0.5,
            name="qkv",
        )(x)
        # Attention calculation.
        x = dot_product_attention(q, k, v, self.mask)
        # Output layer with async scatter.
        x = TPAsyncDense(
            dense_fn=functools.partial(
                AttnOut,
                config=self.config,
                features=input_features,
            ),
            model_axis_name=self.config.model_axis_name,
            tp_mode="scatter",
            kernel_init_adjustment=tp_size**-0.5,
            name="out",
        )(x)
        return x


def prepare_module(
    layer: Callable[..., nn.Module], layer_name: str, config: ConfigDict
) -> Callable[..., nn.Module]:
    """Remats and shards layer if needed.

    This function wraps the layer function in a remat and/or sharding function if its layer name is present in the remat and fsdp configuration, respectively.

    Args:
        layer: The layer to prepare.
        layer_name: The name of the layer.
        config: The configuration to use.

    Returns:
        The layer with remat and sharding applied if needed.
    """
    # Shard parameters over model axis. Performed before remat, such that the gathered parameters would not be kept under remat.
    if config.get("fsdp", None) is not None and layer_name in config.fsdp.modules:
        layer = shard_module_params(
            layer, axis_name=config.data_axis_name, min_weight_size=config.fsdp.min_weight_size
        )
    if config.get("remat", None) is not None and layer_name in config.remat:
        layer = nn.remat(layer, prevent_cse=False)
    return layer


class TPTransformerBlock(nn.Module):
    config: ConfigDict
    train: bool
    mask: jax.Array | None = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # Attention layer
        attn_layer = prepare_module(TPMultiHeadAttn, "Attn", self.config)
        attn_out = attn_layer(
            config=self.config,
            train=self.train,
            mask=self.mask,
            name="attn",
        )(x)
        attn_out = nn.Dropout(rate=self.config.dropout_rate, deterministic=not self.train)(
            attn_out
        )
        x = x + attn_out
        # MLP layer
        mlp_layer = prepare_module(TPAsyncMLPBlock, "MLP", self.config)
        mlp_out = mlp_layer(
            config=self.config,
            train=self.train,
            name="mlp",
        )(x)
        mlp_out = nn.Dropout(rate=self.config.dropout_rate, deterministic=not self.train)(mlp_out)
        x = x + mlp_out
        return x


class QKVMLPDense(nn.Module):
    config: ConfigDict
    num_heads: int
    head_dim: int
    mlp_dim: int
    kernel_init: Callable
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]:
        h = MLPBlockInput(
            config=self.config,
            features=self.mlp_dim,
            kernel_init=self.kernel_init,
            use_bias=self.use_bias,
            use_norm=False,
            name="mlp",
        )(x)
        q, k, v = QKVDense(
            config=self.config,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            kernel_init=self.kernel_init,
            use_bias=self.use_bias,
            name="qkv",
        )(x)
        return h, (q, k, v)


class AttnMLPOut(nn.Module):
    config: ConfigDict
    features: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: Tuple[jax.Array, jax.Array]) -> jax.Array:
        mlp_h, attn_v = x
        mlp_out = MLPBlockOutput(
            config=self.config,
            features=self.features,
            kernel_init=self.kernel_init,
            use_bias=self.use_bias,
            name="mlp",
        )(mlp_h)
        attn_out = AttnOut(
            config=self.config,
            features=self.features,
            kernel_init=self.kernel_init,
            use_bias=self.use_bias,
            name="attn",
        )(attn_v)
        out = mlp_out + attn_out
        return out


class TPTransformerParallelBlock(nn.Module):
    config: ConfigDict
    train: bool
    mask: jax.Array | None = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        tp_size = jax.lax.psum(1, self.config.model_axis_name)
        input_features = x.shape[-1]
        residual = x
        # Normalize across devices before the input layer.
        x = TPNorm(config=self.config, name="pre_norm")(x)
        # Calculate MLP hidden and q, k, v using async dense layers.
        h, (q, k, v) = TPAsyncDense(
            dense_fn=functools.partial(
                QKVMLPDense,
                config=self.config,
                num_heads=self.config.num_heads // tp_size,
                head_dim=self.config.head_dim,
                mlp_dim=self.config.hidden_size * self.config.mlp_expansion // tp_size,
            ),
            model_axis_name=self.config.model_axis_name,
            tp_mode="gather",
            kernel_init_adjustment=tp_size**-0.5,
            name="hqkv",
        )(x)
        # Attention calculation.
        v = dot_product_attention(q, k, v, self.mask)
        # MLP and attention layer with async scatter.
        block_out = TPAsyncDense(
            dense_fn=functools.partial(
                AttnMLPOut,
                config=self.config,
                features=input_features,
            ),
            model_axis_name=self.config.model_axis_name,
            tp_mode="scatter",
            kernel_init_adjustment=tp_size**-0.5,
            name="out",
        )((h, v))
        # Apply dropout and add residual.
        block_out = nn.Dropout(rate=self.config.dropout_rate, deterministic=not self.train)(
            block_out
        )
        out = residual + block_out
        return out


class TransformerBackbone(nn.Module):
    config: ConfigDict
    train: bool
    mask: jax.Array | None = None
    block_fn: Any = TPTransformerBlock

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        block_fn = prepare_module(
            self.block_fn,
            "Block",
            self.config,
        )
        block = block_fn(config=self.config, train=self.train, mask=self.mask, name="block")
        # Scan version
        x, _ = nn.scan(
            lambda module, carry, _: (module(carry), None),
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            length=self.config.num_layers,
            metadata_params={
                "partition_name": None
            },  # We do not need to partition over the layer axis.
        )(block, x, ())
        return x


class PositionalEncoding(nn.Module):
    config: ConfigDict

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        tp_size = jax.lax.psum(1, self.config.model_axis_name)
        tp_index = jax.lax.axis_index(self.config.model_axis_name)
        seq_len, num_feats = x.shape[-2:]
        if self.config.positional_encoding_type == "learned":
            pos_emb = self.param(
                "pos_emb",
                nn.initializers.normal(stddev=0.02),
                (seq_len, num_feats),
            )
        elif self.config.positional_encoding_type == "sinusoidal":
            # Adjusted to multi-device setting.
            position = jnp.arange(0, seq_len, dtype=jnp.float32)[:, None]
            div_term = jnp.exp(
                jnp.arange(tp_index * num_feats, (tp_index + 1) * num_feats, 2)
                * (-np.log(10000.0) / (tp_size * num_feats))
            )
            pos_emb = jnp.stack(
                [jnp.sin(position * div_term), jnp.cos(position * div_term)], axis=-1
            )
            pos_emb = jnp.reshape(pos_emb, (seq_len, num_feats))
        else:
            raise ValueError(
                f"Unknown positional encoding type: {self.config.positional_encoding_type}"
            )
        pos_emb = pos_emb.astype(
            x.dtype
        )  # Cast to the same dtype as the input, e.g. support bfloat16.
        pos_emb = jnp.expand_dims(pos_emb, axis=range(x.ndim - 2))
        x = x + pos_emb
        return x


class InputEmbedding(nn.Module):
    config: ConfigDict

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        tp_size = jax.lax.psum(1, self.config.model_axis_name)
        x = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size // tp_size,
            embedding_init=nn.initializers.normal(stddev=1.0),
            dtype=self.config.dtype,
            name="token_emb",
        )(x)
        x = PositionalEncoding(config=self.config, name="pos_enc")(x)
        return x


class TPInputEmbedding(nn.Module):
    config: ConfigDict

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return ModelParallelismWrapper(
            model_axis_name=self.config.model_axis_name,
            module_fn=functools.partial(InputEmbedding, config=self.config),
            name="module",
        )(x)


def split_array_over_mesh(x: jax.Array, axis_name: str, split_axis: int) -> jax.Array:
    axis_size = jax.lax.psum(1, axis_name)
    axis_index = jax.lax.axis_index(axis_name)
    slice_size = x.shape[split_axis] // axis_size
    x = jax.lax.dynamic_slice_in_dim(
        x,
        axis_index * slice_size,
        slice_size,
        axis=split_axis,
    )
    return x


class TPOutputLayer(nn.Module):
    config: ConfigDict

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # Gather outputs over feature dimension and split over sequence length.
        x = jax.lax.all_gather(x, axis_name=self.config.model_axis_name, axis=-1, tiled=True)
        x = split_array_over_mesh(x, axis_name=self.config.model_axis_name, split_axis=1)
        # Shard parameters over model axis.
        norm_fn = shard_module_params(
            nn.RMSNorm,
            axis_name=self.config.model_axis_name,
            min_weight_size=self.config.fsdp.min_weight_size,
        )
        dense_fn = shard_module_params(
            nn.Dense,
            axis_name=self.config.model_axis_name,
            min_weight_size=self.config.fsdp.min_weight_size,
        )
        # Apply normalization and output layer.
        x = norm_fn(dtype=self.config.dtype, name="out_norm")(x)
        x = dense_fn(
            features=self.config.num_outputs,
            dtype=jnp.float32,
            name="output_layer",
        )(x)
        return x


class Transformer(nn.Module):
    config: ConfigDict
    block_fn: Any = TPTransformerBlock

    @nn.compact
    def __call__(self, x: jax.Array, train: bool, mask: jax.Array | None = None) -> jax.Array:
        if mask is None and self.config.causal_mask:
            mask = nn.make_causal_mask(x, dtype=jnp.bool_)
        x = TPInputEmbedding(
            config=self.config,
            name="input_embedding",
        )(x)
        x = TransformerBackbone(
            config=self.config,
            train=train,
            mask=mask,
            block_fn=self.block_fn,
            name="backbone",
        )(x)
        x = TPOutputLayer(
            config=self.config,
            name="output_layer",
        )(x)
        return x
