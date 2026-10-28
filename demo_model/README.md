# Demo model

A ready-to-use recommendation model so the project works on a fresh clone with
no setup. The app finds this directory automatically; `MODEL_DIR` is only
needed to point somewhere else.

## What's here

| File | Purpose |
|------|---------|
| `movie_metadata.parquet` | Title, year, genres, rating, overview, poster path |
| `neighbors_idx.npy` | Top-50 most similar movies per title (indices) |
| `neighbors_scores.npy` | Matching similarity scores |
| `title_to_idx.json` | Title → row index lookup |
| `config.json` | Movie count, format, top-K |

Nothing here is a pickle. The app reads parquet, JSON and numpy arrays only,
so loading a model never executes code.

## How it was built

```bash
python training/train.py ./TMDB_movie_dataset_v11.csv -o demo_model -q high -m 10000
```

1,493,255 rows filtered to 6,427 movies with 500+ votes, then 6,248 after
dropping entries with too little text to model. TF-IDF over genres, keywords,
production companies, overview and tagline; reduced to 500 SVD components;
top-50 cosine neighbours per movie.

The training run also produces `tfidf_vectorizer.pkl` and `svd_model.pkl`
(~40 MB). Those are only needed to retrain, never to serve, so `*.pkl` is
git-ignored and they are not committed.

## Data attribution

Derived from the
[TMDB Movies Dataset](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies),
originating from [TMDB](https://www.themoviedb.org/).

> This product uses data from TMDB but is not endorsed or certified by TMDB.

The project's MIT licence covers its source code, not this movie data.

## Replacing it

Train a larger or different model into this directory and commit the result,
or keep yours in git-ignored `models/` and set `MODEL_DIR=models`. See
[training/guide.md](../training/guide.md).
