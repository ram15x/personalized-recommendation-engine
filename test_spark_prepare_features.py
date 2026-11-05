"""Small point-in-time contract test for the optional Spark preparation job."""

from pathlib import Path

import pytest

pytest.importorskip("pyspark")

from pyspark.sql import SparkSession, functions as F

from scripts.spark_prepare_features import build_tables


@pytest.fixture(scope="module")
def spark():
    session = (SparkSession.builder.master("local[2]")
               .appName("nexvia-spark-feature-tests")
               .config("spark.sql.shuffle.partitions", "2")
               .getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def test_feature_tables_use_train_only_aggregates_and_time_safe_catalog(spark, tmp_path):
    ratings = tmp_path / "ratings.csv"
    movies = tmp_path / "movies.csv"
    output = tmp_path / "features"
    ratings.write_text(
        "userId,movieId,rating,timestamp\n"
        "1,1,5.0,100\n"
        "1,2,3.5,110\n"
        "2,2,4.5,119\n"
        "2,3,5.0,120\n"
        "3,4,5.0,130\n",
        encoding="utf-8",
    )
    movies.write_text(
        "movieId,title,genres\n"
        "1,One (2000),Adventure|Comedy\n"
        "2,Two (2001),(no genres listed)\n"
        "3,Three (2002),Drama\n"
        "4,Four (2003),Action\n",
        encoding="utf-8",
    )

    counts = build_tables(spark, ratings, movies, output, 120, 130, 4.0)

    assert counts == {
        "train_rows": 3,
        "validation_rows": 1,
        "train_users": 2,
        "train_items_with_interactions": 2,
        "movie_catalog_rows": 2,
    }
    interactions = spark.read.parquet(str(output / "interactions"))
    assert interactions.count() == 4
    assert interactions.agg(F.max("event_ts")).first()[0] == 120
    train = interactions.where(F.col("split") == "train")
    assert train.where(F.col("is_positive")).count() == 2

    categories = {
        row.item_id: row.categories
        for row in spark.read.parquet(str(output / "item_features")).collect()
    }
    assert categories == {1: ["Adventure", "Comedy"], 2: []}
    assert 3 not in categories and 4 not in categories


def test_invalid_temporal_bounds_fail_before_spark_actions(tmp_path):
    with pytest.raises(ValueError, match="train-end-ts"):
        build_tables(None, Path("unused"), Path("unused"), tmp_path / "out", 20, 20)


def test_failed_run_removes_staging_output(tmp_path, spark):
    ratings = tmp_path / "ratings_with_duplicate.csv"
    movies = tmp_path / "movies.csv"
    output = tmp_path / "failed_features"
    ratings.write_text(
        "userId,movieId,rating,timestamp\n"
        "1,1,5.0,100\n"
        "1,1,4.5,110\n",
        encoding="utf-8",
    )
    movies.write_text("movieId,title,genres\n1,One (2000),Drama\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate user/item"):
        build_tables(spark, ratings, movies, output, 120, 130, 4.0)

    assert not output.exists()
    assert not list(tmp_path.glob(".failed_features.partial-*"))
