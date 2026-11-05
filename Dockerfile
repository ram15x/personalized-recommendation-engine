FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-api.txt .
RUN python -m pip install --no-cache-dir -r requirements-api.txt

COPY scripts/ scripts/
COPY demo.html .
COPY data/ml-25m/movies.csv data/ml-25m/movies.csv
COPY reports/ml-25m/popularity.csv reports/ml-25m/popularity.csv
COPY reports/development/itemknn-60-70-corrected-cohort-2k-v2/item_similarity_top100.npz reports/development/itemknn-60-70-corrected-cohort-2k-v2/
COPY reports/development/itemknn-60-70-corrected-cohort-2k-v2/positive_item_ids.npy reports/development/itemknn-60-70-corrected-cohort-2k-v2/
COPY reports/development/itemknn-60-70-corrected-cohort-2k-v2/artifact_manifest.json reports/development/itemknn-60-70-corrected-cohort-2k-v2/
COPY reports/development/itemknn-60-70-corrected-cohort-2k-v2/training_popularity.csv reports/development/itemknn-60-70-corrected-cohort-2k-v2/
COPY reports/development/itemknn-60-70-corrected-cohort-2k-v2/training_popularity_manifest.json reports/development/itemknn-60-70-corrected-cohort-2k-v2/

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "scripts.recommendation_api:app", "--host", "0.0.0.0", "--port", "8000"]
