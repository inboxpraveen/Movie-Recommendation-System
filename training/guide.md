# Training Guide

## 🚀 Quickest start: the command line

`train.py` is a normal CLI. You do not need to edit the file to train a model.

```bash
pip install -r requirements-train.txt

# Default: movies with 50+ votes (about 26,000 on the TMDB dataset), 50 neighbours each
python training/train.py ./TMDB_movie_dataset_v11.csv -o models

# Smaller and much faster
python training/train.py ./TMDB_movie_dataset_v11.csv -o models -q high -m 10000

# Every option
python training/train.py --help
```

| Flag | Meaning | Default |
|------|---------|---------|
| `data_path` | CSV file (or directory containing `TMDB_movie_dataset_v11.csv`) | required |
| `-o, --output-dir` | Where artifacts are written | `./models` |
| `-q, --quality` | Minimum votes: `low`=5, `medium`=50, `high`=500 | `medium` |
| `-m, --max-movies` | Cap the catalogue (`0` = no cap) | `50000` |
| `-k, --top-k` | Neighbours stored per movie | `50` |
| `--components` | SVD latent dimensions | `500` |
| `--no-svd` | Use raw TF-IDF vectors instead of SVD | off |
| `--chunk-size` | Rows compared at a time; lower it if memory is tight | `1024` |
| `--legacy-matrix` | Write the full N x N similarity matrix | off |

### Why top-K neighbours

By default the trainer keeps only the **K** most similar movies for each title
rather than the whole similarity matrix. The matrix grows with the square of
the catalogue: at 50,000 movies it is about 10 GB, which no small host can
hold. The same model as top-50 neighbours is roughly 20 MB, and the
recommendations are identical for the first K results.

`--legacy-matrix` still produces the old format if you need it, and the
inference code reads either.

---

## 🎬 Dataset Overview

**TMDB Movies Dataset 2023** - 1.3M+ movies with rich metadata

### Key Features:
- ✅ **Single CSV file** (no merging needed)
- ✅ **1,332,407 movies** total
- ✅ **Rich metadata**: genres, keywords, production companies, countries
- ✅ **Quality metrics**: vote_average, vote_count, popularity
- ✅ **IMDB integration**: imdb_id for cross-referencing
- ✅ **Poster images**: Direct URLs to movie posters

### Dataset Columns Used:
```
✅ title               - Movie title
✅ genres              - Multiple genres per movie
✅ keywords            - Descriptive keywords
✅ production_companies - Production studios
✅ production_countries - Countries of production
✅ overview            - Plot summary (up to 50 words)
✅ tagline             - Movie tagline
✅ vote_average        - Rating (0-10)
✅ vote_count          - Number of votes
✅ popularity          - TMDB popularity score
✅ release_date        - Release date
✅ imdb_id             - IMDB identifier
✅ poster_path         - Poster image path
```

## 🚀 Major Improvements Over Original Code

### 1. **Dataset Handling**
```python
# Before: Multiple CSV files with complex merging
movies = pd.read_csv('movies_metadata.csv')
credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')
movies = movies.merge(credits).merge(keywords)

# After: Single CSV with all data
df = pd.read_csv('TMDB_movie_dataset_v11.csv')
# No merging needed - everything in one place!
```

### 2. **Scalability**
Measured on the TMDB v11 dataset (1.49 million rows):

| Run | Movies in model | Training time | Peak memory | Model files |
|-----|-----------------|---------------|-------------|-------------|
| `-q high -m 10000` | 6,248 | 31 s | 2.5 GB | 3.8 MB |
| default (`-q medium`) | 26,326 | about 1 min | 2.5 GB | 16.1 MB |
| `-q low -m 100000` | 100,000 | 4.3 min | 3.9 GB | 61.5 MB |

Quality filters (three vote tiers) are described below.

### 3. **Quality Filtering**

**Three quality tiers:**
```python
# Low Quality: 5+ votes (maximum dataset size)
trainer.train(quality_threshold='low')      # about 113,000 movies on the TMDB v11 dataset

# Medium Quality: 50+ votes (recommended - balanced)
trainer.train(quality_threshold='medium')   # about 26,000 movies

# High Quality: 500+ votes (highest quality)
trainer.train(quality_threshold='high')     # about 6,000 movies
```

### 4. **Enhanced Features**

#### Original Features:
- Cast (top 3)
- Director
- Genres
- Keywords

#### New Features:
- ✅ **Production Companies** (weighted)
- ✅ **Production Countries**
- ✅ **Plot Overview** (first 50 words)
- ✅ **Taglines**
- ✅ **Quality Score** (vote_average × log(vote_count))
- ✅ **IMDB Integration**
- ✅ **Poster URLs**

### 5. **Advanced Filtering**

```python
# Filter by year range
recommender.get_recommendations(
    "Inception",
    min_year=2015,
    max_year=2023
)

# Filter by rating
recommender.get_recommendations(
    "The Matrix",
    min_rating=7.5  # Only highly rated
)

# Filter by genres (multiple)
recommender.get_recommendations(
    "Interstellar",
    genres=['Science Fiction', 'Drama']
)

# Exclude same production company
recommender.get_recommendations(
    "Avatar",
    exclude_same_company=True
)

# Combine all filters
recommender.get_recommendations(
    "The Dark Knight",
    n_recommendations=10,
    min_year=2010,
    max_year=2023,
    genres=['Action', 'Thriller'],
    min_rating=7.0,
    exclude_same_company=True
)
```

## 📊 Recommended Configurations

All three use CPU-only scikit-learn; no GPU is involved. Peak memory is mostly the
658 MB CSV being read into pandas.

### Configuration 1: Default (recommended)
```bash
python training/train.py ./TMDB_movie_dataset_v11.csv -o models
```
- Movies in model: 26,326 (50+ votes)
- Training time: about 1 minute
- Peak memory: 2.5 GB
- Model files: 16 MB
- Best for: production deployment on a small host

### Configuration 2: Large catalogue
```bash
python training/train.py ./TMDB_movie_dataset_v11.csv -o models_large -q low -m 100000
```
- Movies in model: 100,000 (5+ votes, capped)
- Training time: 4.3 minutes
- Peak memory: 3.9 GB
- Model files: 61.5 MB
- Best for: a catalogue that includes lesser-known films

### Configuration 3: Fast and small
```bash
python training/train.py ./TMDB_movie_dataset_v11.csv -o models_fast -q high -m 10000
```
- Movies in model: 6,248 (500+ votes)
- Training time: 31 seconds
- Peak memory: 2.5 GB
- Model files: 3.8 MB
- Best for: testing, development, and the committed `demo_model/`

## 🎯 Complete Usage Example

### Step 1: Install Dependencies
```bash
pip install pandas numpy scikit-learn scipy nltk kagglehub
```

### Step 2: Download Dataset & Train
```python
import kagglehub
from training.train import MovieRecommenderTrainer

# Download dataset
path = kagglehub.dataset_download("asaniczka/tmdb-movies-dataset-2023-930k-movies")

# Train model (recommended config)
trainer = MovieRecommenderTrainer(
    output_dir='./models',
    use_dimensionality_reduction=True,
    n_components=500
)

df, neighbors = trainer.train(
    path,
    quality_threshold='medium',
    max_movies=50000
)
```

### Step 3: Load & Use Recommender
```python
from training.infer import MovieRecommender

# Load trained model
recommender = MovieRecommender(model_dir='./models')

# Get recommendations
results = recommender.get_recommendations(
    "Inception",
    n_recommendations=10,
    min_rating=7.0
)

# Print results
recommender.print_recommendations(results, show_scores=True)
```

## 🎨 Output Examples

### Basic Recommendations:
```
====================================================================================================
🎬 Recommendations for: Inception
   Production: Warner Bros. Pictures | Rating: 8.4/10 | Genres: Action, ScienceFiction, Adventure
====================================================================================================

 1. Shutter Island
    ⭐ 8.2/10 (22,527 votes) | 📅 13-02-2010
    🎭 Drama, Thriller, Mystery | 🏢 Paramount Pictures [Similarity: 0.847]
    🔗 https://www.imdb.com/title/tt1130884

 2. The Prestige
    ⭐ 8.2/10 (13,562 votes) | 📅 17-10-2006
    🎭 Drama, Mystery, ScienceFiction | 🏢 Touchstone Pictures [Similarity: 0.832]
    🔗 https://www.imdb.com/title/tt0482571
```

### Movie Details:
```python
details = recommender.get_movie_details("Interstellar")
```
```
Title: Interstellar
Rating: 8.4/10 (32,571 votes)
Genres: Adventure, Drama, Science Fiction
Production: Legendary Pictures
Overview: The adventures of a group of explorers who make use of a newly 
          discovered wormhole to surpass the limitations on human space...
IMDb: tt0816692
Poster: https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg
```

### Top Rated Movies:
```python
top_scifi = recommender.get_top_rated(
    n=5,
    min_votes=5000,
    genres=['Science Fiction']
)
```
```
1. The Shawshank Redemption - 8.7/10 (25,567 votes)
2. Interstellar - 8.4/10 (32,571 votes)
3. Inception - 8.4/10 (34,495 votes)
4. The Matrix - 8.2/10 (24,117 votes)
5. WALL·E - 8.1/10 (17,258 votes)
```

## 🔧 Advanced Features

### 1. Diverse Recommendations (MMR)
Prevents echo chamber effect - recommends similar but diverse movies:
```python
results = recommender.get_diverse_recommendations(
    "The Matrix",
    n_recommendations=10,
    diversity_weight=0.5  # 0=similar, 1=diverse
)
```

### 2. Search Movies
```python
# Partial title search
movies = recommender.search_movies("dark knight", n=5)
# Returns: ['The Dark Knight', 'The Dark Knight Rises', ...]

# With rating filter
movies = recommender.search_movies("avengers", min_rating=7.0)
```

### 3. Movie Details
```python
details = recommender.get_movie_details("Inception")
# Returns: Full metadata including poster URL, IMDb link, etc.
```

## 📈 Performance Metrics

### Inference (in-process, 300 random titles)
```
                        6,248 films   26,326 films   100,000 films
Load model:             40 ms         75 ms          210 ms
Recommendation, median: 1.1 ms        1.1 ms         1.1 ms
Recommendation, p95:    1.6 ms        1.3 ms         1.4 ms
```
A recommendation is one read of a precomputed row, so it does not grow with the
catalogue. Title autocomplete scans titles and does (p95 1.0 ms at 6,248 films,
15.9 ms at 100,000).

### Storage (100,000 films)
```
neighbors_idx.npy:       20.0 MB
neighbors_scores.npy:    20.0 MB
movie_metadata.parquet:  18.5 MB
title_to_idx.json:        2.9 MB
─────────────────────────────────
Served at runtime:       61.5 MB
```
`tfidf_vectorizer.pkl` and `svd_model.pkl` (about 81 MB more here) are written for
retraining only and are never loaded when serving.

## 🐛 Troubleshooting

### Issue: Out of Memory during Training
**Solution 1:** Reduce dataset size
```python
trainer.train(max_movies=50000)  # Limit to 50K
```

**Solution 2:** Reduce SVD components
```python
trainer = MovieRecommenderTrainer(n_components=300)
```

**Solution 3:** Lower quality threshold
```python
trainer.train(quality_threshold='high')  # Only 500+ votes
```

### Issue: Training Too Slow
**Solution 1:** Use smaller dataset
```python
trainer.train(max_movies=10000)
```

**Solution 2:** Disable SVD for small datasets
```python
trainer = MovieRecommenderTrainer(use_dimensionality_reduction=False)
```

### Issue: Poor Recommendation Quality
**Solution 1:** Increase quality threshold
```python
trainer.train(quality_threshold='high', min_votes=1000)
```

**Solution 2:** Increase SVD components
```python
trainer = MovieRecommenderTrainer(n_components=600)
```

**Solution 3:** Use larger dataset
```python
trainer.train(max_movies=200000)  # More data = better patterns
```

## 🚀 Production Deployment Tips

### 1. Build REST API
```python
from fastapi import FastAPI
from training.infer import MovieRecommender

app = FastAPI()
recommender = MovieRecommender('./models')

@app.get("/recommend/{movie_title}")
def recommend(movie_title: str, n: int = 10):
    return recommender.get_recommendations(movie_title, n)
```

### 2. Add Caching
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_recommendations(movie_title, n):
    return recommender.get_recommendations(movie_title, n)
```

### 3. Batch Processing
For processing multiple recommendations efficiently:
```python
def batch_recommend(movie_titles, n=10):
    results = {}
    for title in movie_titles:
        results[title] = recommender.get_recommendations(title, n)
    return results
```

## 📚 Key Differences: Original vs TMDB Dataset

| Feature | Original Dataset | TMDB Dataset |
|---------|------------------|--------------|
| **Files** | 7 separate CSVs | 1 single CSV |
| **Movies** | ~45K | 1.3M+ |
| **Data Quality** | Mixed | Excellent |
| **Metadata** | Basic | Rich |
| **Director Info** | ✅ Separate crew file | ❌ Use production company |
| **Cast Info** | ✅ Detailed | ❌ Not available |
| **Plot Summary** | ❌ Limited | ✅ Full overview |
| **Posters** | ❌ No | ✅ Direct URLs |
| **IMDB Links** | ✅ Via mapping | ✅ Direct IDs |
| **Companies** | ❌ Limited | ✅ Multiple per movie |
| **Countries** | ❌ Limited | ✅ Multiple per movie |
| **Keywords** | ✅ Yes | ✅ Yes |
| **Ratings** | ✅ Yes | ✅ Enhanced |

## 🎓 What's Changed in Code

### Feature Engineering:
```python
# Original: Used director from crew
df['director'] = df['crew'].apply(get_director)

# TMDB: Use production company (no crew data)
df['primary_company'] = df['companies'].apply(lambda x: x[0] if x else None)
```

### Soup Creation:
```python
# Original
soup = keywords + cast + director + genres

# TMDB (Enhanced)
soup = (
    keywords +
    genres * 2 +              # Weight genres more
    companies_weighted +       # Production companies
    companies_clean +
    countries_clean +         # Production countries
    overview_words +          # Plot summary words
    tagline_words            # Tagline words
)
```

### Quality Filtering:
```python
# Original: No quality filters
# Used all movies

# TMDB: Multiple quality levels
quality_score = vote_average * log(vote_count + 1)
df = df[df['vote_count'] >= threshold]  # Filter by votes
df = df.sort_values('quality_score', ascending=False)
```

## Summary

You now have a recommender that:
- Reads a **single CSV** (no merging) of up to 1.49 million rows
- Keeps films above a **vote threshold** (three tiers) and ranks them by a quality score
- Builds a model of 6,000 to 100,000 films in **30 seconds to about 4 minutes**
- Stores **top-50 neighbours** per film, so a 26,000 film model is 16 MB
- Serves a recommendation in about **1 millisecond**
- Provides **rich metadata** (posters, IMDb ids, plots) and year, rating and genre filters through `training/infer.py`
