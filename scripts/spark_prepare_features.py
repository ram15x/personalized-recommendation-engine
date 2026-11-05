"""Build time-safe MovieLens interaction and aggregate feature tables with Spark.

This is a data-preparation job, not model training. By default it uses the
previously explored 60/70 development window. Rows outside that interval are
excluded from the output and all aggregates; Spark still scans the CSV source
to apply its timestamp filter.
"""

from __future__ import annotations

import argparse
import shutil
import uuid
from pathlib import Path

from pyspark.sql import SparkSession, functions as F, types as T


DEFAULT_TRAIN_END = 1_198_868_375
DEFAULT_VALIDATION_END = 1_288_571_552

RATINGS_SCHEMA = T.StructType([
    T.StructField("userId", T.LongType(), False),
    T.StructField("movieId", T.LongType(), False),
    T.StructField("rating", T.DoubleType(), False),
    T.StructField("timestamp", T.LongType(), False),
])
MOVIES_SCHEMA = T.StructType([
    T.StructField("movieId", T.LongType(), False),
    T.StructField("title", T.StringType(), True),
    T.StructField("genres", T.StringType(), True),
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--movies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-end-ts", type=int, default=DEFAULT_TRAIN_END)
    parser.add_argument("--validation-end-ts", type=int, default=DEFAULT_VALIDATION_END)
    parser.add_argument("--positive-rating", type=float, default=4.0)
    parser.add_argument("--master", default="local[*]")
    return parser.parse_args()


def build_tables(spark: SparkSession, ratings_path: Path, movies_path: Path,
                 output_path: Path, train_end_ts: int, validation_end_ts: int,
                 positive_rating: float = 4.0) -> dict[str, int]:
    """Create immutable Parquet tables; validation rows never enter features."""
    if train_end_ts >= validation_end_ts:
        raise ValueError("train-end-ts must be earlier than validation-end-ts")
    if output_path.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output_path}")
    if not ratings_path.is_file() or not movies_path.is_file():
        raise FileNotFoundError("Both ratings and movies CSV files must exist")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = output_path.with_name(
        f".{output_path.name}.partial-{uuid.uuid4().hex}"
    )

    ratings = (spark.read.option("header", True).schema(RATINGS_SCHEMA)
               .csv(str(ratings_path)))
    movies = (spark.read.option("header", True).schema(MOVIES_SCHEMA)
              .csv(str(movies_path)))

    # Tag and materialize the selected window in one CSV scan. Spark must scan
    # the source to apply the filter, but later rows do not enter outputs.
    window = ratings.where(F.col("timestamp") < validation_end_ts)
    scoped = window.withColumn(
        "split", F.when(F.col("timestamp") < train_end_ts, "train").otherwise("validation")
    ).select(
        F.col("userId").alias("user_id"),
        F.col("movieId").alias("item_id"),
        F.col("rating"),
        F.col("timestamp").alias("event_ts"),
        (F.col("rating") >= F.lit(positive_rating)).alias("is_positive"),
        "split",
    )

    try:
        staging_path.mkdir(parents=False, exist_ok=False)
        interactions_path = staging_path / "interactions"
        (scoped.write.mode("errorifexists").partitionBy("split")
         .parquet(str(interactions_path)))
        materialized = spark.read.parquet(str(interactions_path))
        train = materialized.where(F.col("split") == "train")
        validation = materialized.where(F.col("split") == "validation")

        duplicate_pairs = (train.groupBy("user_id", "item_id").count()
                           .where(F.col("count") > 1).limit(1).count())
        if duplicate_pairs:
            raise ValueError("Duplicate user/item rows found in training window")

        # All aggregates use training rows only. Ratings below threshold are
        # retained in interactions but do not contribute to positive popularity.
        user_features = train.groupBy("user_id").agg(
            F.count(F.lit(1)).alias("rating_count"),
            F.avg("rating").alias("mean_rating"),
            F.sum(F.when(F.col("rating") >= positive_rating, 1).otherwise(0))
             .cast("long").alias("positive_count"),
            F.max("event_ts").alias("last_event_ts"),
        )

        popularity = train.groupBy("item_id").agg(
            F.sum(F.when(F.col("rating") >= positive_rating, 1).otherwise(0))
             .cast("long").alias("positive_interaction_count"),
            F.count(F.lit(1)).cast("long").alias("rating_count"),
            F.avg("rating").alias("mean_rating"),
        )

        training_catalog = popularity.select("item_id").distinct()
        item_features = (movies
            .select(
                F.col("movieId").alias("item_id"),
                "title",
                F.when(F.col("genres").isNull() | (F.col("genres") == "(no genres listed)"),
                       F.array().cast(T.ArrayType(T.StringType())))
                 .otherwise(F.split("genres", "\\|")).alias("categories"),
            )
            .join(training_catalog, on="item_id", how="inner"))

        (user_features.write.mode("errorifexists")
         .parquet(str(staging_path / "user_features")))
        (item_features.write.mode("errorifexists")
         .parquet(str(staging_path / "item_features")))
        (popularity.write.mode("errorifexists")
         .parquet(str(staging_path / "popularity_features")))

        counts = {
            "train_rows": train.count(),
            "validation_rows": validation.count(),
            "train_users": user_features.count(),
            "train_items_with_interactions": popularity.count(),
            "movie_catalog_rows": item_features.count(),
        }
        if output_path.exists():
            raise FileExistsError(f"Output appeared during run; refusing to overwrite: {output_path}")
        staging_path.rename(output_path)
        return counts
    except BaseException:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise


def main() -> None:
    args = parse_args()
    spark = (SparkSession.builder.master(args.master)
             .appName("nexvia-movielens-feature-preparation")
             .config("spark.sql.shuffle.partitions", "8")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    try:
        counts = build_tables(
            spark, args.ratings, args.movies, args.output,
            args.train_end_ts, args.validation_end_ts, args.positive_rating,
        )
        print({
            "status": "success",
            "output": str(args.output.resolve()),
            "train_end_timestamp_exclusive": args.train_end_ts,
            "validation_end_timestamp_exclusive": args.validation_end_ts,
            "positive_rating_threshold": args.positive_rating,
            "counts": counts,
        })
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
