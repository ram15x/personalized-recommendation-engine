import pandas as pd

from streaming_data import load_streaming_sample


def test_strict_interaction_cap_preserves_full_history_and_popularity(tmp_path):
    path = tmp_path / "ratings.csv"
    pd.DataFrame([
        {"userId": user, "movieId": user, "rating": 5.0, "timestamp": 10}
        for user in range(1, 6)
    ] + [{"userId": 5, "movieId": 10, "rating": 5.0, "timestamp": 20}]).to_csv(path, index=False)
    args = (path, 15, 25)
    result = load_streaming_sample(*args, seed=42, max_interactions=2,
                                   eval_users=0, interactions_per_user=3, chunksize=2)
    sample, history, validation, popularity, counts, _ = result
    assert len(sample) <= 2 and counts["training_users"] == 5
    assert counts["training_users_selected_for_sampling"] == 2
    assert counts["retained_training_rows"] == len(sample)
    assert set(popularity.movieId) == {1, 2, 3, 4, 5}
    assert set(history.loc[history.userId == 5, "movieId"]) == {5}
    assert set(validation.loc[validation.userId == 5, "movieId"]) == {10}
    repeated = load_streaming_sample(*args, seed=42, max_interactions=2,
                                     eval_users=0, interactions_per_user=3, chunksize=2)[0]
    pd.testing.assert_frame_equal(sample, repeated)
