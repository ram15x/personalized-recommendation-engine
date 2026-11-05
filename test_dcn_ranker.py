import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from scripts.dcn_ranker import DCNRatingRanker


def _inputs(user_ids=(0, 1, 2), item_ids=(1, 2, 3)):
    return {
        "user_id": tf.constant(user_ids, dtype=tf.int32),
        "item_id": tf.constant(item_ids, dtype=tf.int32),
        "numeric": tf.constant([[0.1, 1.0], [0.2, 0.0], [0.3, -1.0]], dtype=tf.float32),
    }


def test_dcn_outputs_one_finite_rating_score_per_pair():
    model = DCNRatingRanker(num_users=4, num_items=5, numeric_features=2)
    output = model(_inputs())
    assert output.shape == (3,)
    assert np.isfinite(output.numpy()).all()


def test_dcn_has_finite_gradients_for_rating_regression():
    model = DCNRatingRanker(num_users=4, num_items=5, numeric_features=2)
    labels = tf.constant([4.5, 3.0, 1.5], dtype=tf.float32)
    with tf.GradientTape() as tape:
        predictions = model(_inputs(), training=True)
        loss = tf.reduce_mean(tf.square(labels - predictions))
    gradients = tape.gradient(loss, model.trainable_variables)
    assert gradients and all(g is not None for g in gradients)
    assert all(np.isfinite(tf.convert_to_tensor(g).numpy()).all() for g in gradients)


def test_dcn_rejects_unknown_id_indices():
    model = DCNRatingRanker(num_users=4, num_items=5, numeric_features=2)
    with pytest.raises(tf.errors.InvalidArgumentError):
        model(_inputs(user_ids=(4, 1, 2)))


def test_dcn_keras_save_load_preserves_predictions(tmp_path):
    model = DCNRatingRanker(num_users=4, num_items=5, numeric_features=2)
    inputs = _inputs()
    expected = model(inputs).numpy()
    path = tmp_path / "ranker.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path)
    actual = restored(inputs).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
