"""A compact DCN-v1 rating scorer for time-safe MovieLens experiments.

The model estimates a scalar for user/item pairs whose explicit target ratings
are in the 0.5--5 star range. The output itself is unbounded, as in standard
MSE rating-regression models.
It is not a click-through-rate model. Feature values and ID vocabularies must
be fitted from the training period only by the caller.
"""
from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="nexvia")
class CrossLayer(tf.keras.layers.Layer):
    """Vector DCN cross layer: x_(l+1) = x0 * (x_l · w) + b + x_l."""

    def build(self, input_shape: tuple[tf.TensorShape, tf.TensorShape]) -> None:
        dim = int(input_shape[0][-1])
        self.kernel = self.add_weight(
            name="kernel", shape=(dim,), initializer="glorot_uniform", trainable=True
        )
        self.bias = self.add_weight(
            name="bias", shape=(dim,), initializer="zeros", trainable=True
        )
        super().build(input_shape)

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        x0, x = inputs
        projection = tf.reduce_sum(x * self.kernel, axis=-1, keepdims=True)
        return x0 * projection + self.bias + x

    def get_config(self) -> dict:
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="nexvia")
class DCNRatingRanker(tf.keras.Model):
    """Cross network plus an MLP over user/item IDs and numeric rank features.

    Inputs are integer `user_id` and `item_id` indices in frozen training-only
    vocabularies, and a float `numeric` matrix. Use zero-based dense indices;
    reserve index 0 for unknown IDs and pass cardinalities including that row.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        numeric_features: int,
        embedding_dim: int = 32,
        cross_layers: int = 3,
        deep_units: tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if min(num_users, num_items, numeric_features, embedding_dim, cross_layers) < 1:
            raise ValueError("Model cardinalities, dimensions and cross depth must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.numeric_features = int(numeric_features)
        self.embedding_dim = int(embedding_dim)
        self.cross_depth = int(cross_layers)
        self.deep_units = tuple(int(x) for x in deep_units)
        self.dropout_rate = float(dropout)

        self.user_embedding = tf.keras.layers.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = tf.keras.layers.Embedding(self.num_items, self.embedding_dim)
        self.cross = [CrossLayer(name=f"cross_{i}") for i in range(self.cross_depth)]
        self.deep = [tf.keras.layers.Dense(width, activation="relu") for width in self.deep_units]
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.output_layer = tf.keras.layers.Dense(1, name="predicted_rating")

    def call(self, inputs: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        user_id = tf.cast(inputs["user_id"], tf.int32)
        item_id = tf.cast(inputs["item_id"], tf.int32)
        numeric = tf.cast(inputs["numeric"], tf.float32)
        tf.debugging.assert_all_finite(numeric, "numeric features must be finite")
        tf.debugging.assert_greater_equal(user_id, 0, message="user_id indices must be nonnegative")
        tf.debugging.assert_less(user_id, self.num_users, message="user_id index is out of vocabulary")
        tf.debugging.assert_greater_equal(item_id, 0, message="item_id indices must be nonnegative")
        tf.debugging.assert_less(item_id, self.num_items, message="item_id index is out of vocabulary")
        tf.debugging.assert_equal(tf.shape(numeric)[-1], self.numeric_features)

        x0 = tf.concat(
            [self.user_embedding(user_id), self.item_embedding(item_id), numeric], axis=-1
        )
        crossed = x0
        for layer in self.cross:
            crossed = layer((x0, crossed))
        deep = x0
        for layer in self.deep:
            deep = layer(deep)
            deep = self.dropout(deep, training=training)
        return tf.squeeze(self.output_layer(tf.concat([crossed, deep], axis=-1)), axis=-1)

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "num_users": self.num_users,
            "num_items": self.num_items,
            "numeric_features": self.numeric_features,
            "embedding_dim": self.embedding_dim,
            "cross_layers": self.cross_depth,
            "deep_units": self.deep_units,
            "dropout": self.dropout_rate,
        }
