"""
Movie Recommendation System views.

The recommendation model is a set of files on disk produced by
``training/train.py``. It can take a while to load, so it is loaded once per
process in a background thread and the UI polls ``/api/model-status/`` for
progress. Everything in this module is read-only once the model is loaded.
"""
import json
import logging
import threading
import time
from difflib import get_close_matches
from pathlib import Path
from urllib.parse import quote, quote_plus

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.cache import cache
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

logger = logging.getLogger(__name__)

# Files that must exist for a directory to count as a trained model.
REQUIRED_ARTIFACTS = ('config.json', 'title_to_idx.json', 'movie_metadata.parquet')

# Upper bounds on user-supplied input, applied before any expensive work.
MAX_QUERY_LENGTH = 200
MAX_AUTOCOMPLETE_RESULTS = 20

SEARCH_CACHE_TIMEOUT = 300  # seconds


class ModelLoadError(RuntimeError):
    """Raised when the recommendation model cannot be loaded."""


def _as_list(value) -> list:
    """Normalise a metadata cell to a plain list.

    Parquet list columns come back as numpy arrays rather than Python lists,
    so an ``isinstance(value, list)`` test silently discards every genre.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return []


# The trainer lowercases genres and strips spaces so they work as TF-IDF tokens,
# and that same column is saved for display. These two are the only TMDB genres
# that need more than title-casing to read properly.
_GENRE_LABELS = {'sciencefiction': 'Science Fiction', 'tvmovie': 'TV Movie'}


def _genre_label(genre) -> str:
    """Display form of a stored genre: ``sciencefiction`` becomes ``Science Fiction``."""
    text = str(genre).strip()
    if text != text.lower():
        return text  # already display-cased
    return _GENRE_LABELS.get(text, text.title())


def _clean(value, default='Unknown'):
    """Return a display-safe scalar, collapsing NaN/None to a default."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return value


def _number(value) -> float | None:
    """Return a float for a numeric cell, or None when it is missing."""
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _year(release_date) -> str:
    """Extract a 4-digit year from a release date cell."""
    text = str(_clean(release_date, ''))
    return text[:4] if len(text) >= 4 and text[:4].isdigit() else ''


class MovieRecommender:
    """Content-based recommender backed by pre-computed similarity artifacts."""

    def __init__(self, model_dir, progress_callback=None):
        self.model_dir = Path(model_dir)
        self.metadata = None
        self.config = None
        self.title_to_idx = {}
        # Lowercased title -> canonical title, for fast case-insensitive lookup.
        self._lower_to_title = {}
        self._titles: list[str] = []
        # Exactly one of these is populated, cheapest representation first.
        self._neighbor_idx = None      # (n, k) int array of pre-computed neighbours
        self._neighbor_scores = None   # (n, k) float array of matching scores
        self._similarity_sparse = None  # scipy sparse matrix, never densified
        self._similarity_dense = None   # memory-mapped (n, n) array
        self._load(progress_callback)

    # ------------------------------------------------------------------ load

    def _load(self, progress_callback=None):
        def report(pct):
            if progress_callback:
                progress_callback(pct)

        logger.info("Loading recommendation model from %s", self.model_dir)
        report(10)

        self.metadata = pd.read_parquet(self.model_dir / 'movie_metadata.parquet')
        report(35)

        self._load_similarity()
        report(70)

        with open(self.model_dir / 'title_to_idx.json', encoding='utf-8') as handle:
            self.title_to_idx = json.load(handle)
        self._titles = list(self.title_to_idx.keys())
        self._lower_to_title = {title.lower(): title for title in self._titles}
        report(90)

        with open(self.model_dir / 'config.json', encoding='utf-8') as handle:
            self.config = json.load(handle)
        report(100)

        logger.info("Loaded %s movies successfully", f"{self.movie_count:,}")

    def _load_similarity(self):
        """Load similarity data in the most memory-efficient form available.

        Densifying a full similarity matrix costs ``n * n * 4`` bytes -- 10 GB
        for a 50k-movie model -- which is why neither the sparse matrix nor the
        dense one is ever read into memory whole.
        """
        neighbors_idx = self.model_dir / 'neighbors_idx.npy'
        neighbors_scores = self.model_dir / 'neighbors_scores.npy'
        sparse_path = self.model_dir / 'similarity_matrix.npz'
        dense_path = self.model_dir / 'similarity_matrix.npy'

        if neighbors_idx.exists() and neighbors_scores.exists():
            self._neighbor_idx = np.load(neighbors_idx, mmap_mode='r')
            self._neighbor_scores = np.load(neighbors_scores, mmap_mode='r')
            logger.info("Using pre-computed top-%d neighbours", self._neighbor_idx.shape[1])
        elif sparse_path.exists():
            from scipy.sparse import load_npz

            # Kept sparse: a single row is materialised per request.
            self._similarity_sparse = load_npz(sparse_path).tocsr()
            logger.info("Using sparse similarity matrix %s", self._similarity_sparse.shape)
        elif dense_path.exists():
            # Memory-mapped so the OS pages in only the row being read.
            self._similarity_dense = np.load(dense_path, mmap_mode='r')
            logger.info("Using memory-mapped similarity matrix %s", self._similarity_dense.shape)
        else:
            raise ModelLoadError(
                f"No similarity data found in {self.model_dir}. Expected one of "
                "neighbors_idx.npy, similarity_matrix.npz or similarity_matrix.npy."
            )

    # --------------------------------------------------------------- lookups

    @property
    def movie_count(self) -> int:
        if self.config and 'n_movies' in self.config:
            return int(self.config['n_movies'])
        return len(self._titles)

    def find_movie(self, title: str) -> str | None:
        """Resolve user input to a known title, cheapest strategy first."""
        query = title.strip()
        if not query:
            return None

        lowered = query.lower()

        # 1. Exact match (case-insensitive) -- what autocomplete selections give us.
        exact = self._lower_to_title.get(lowered)
        if exact:
            return exact

        # 2. Prefix, then substring. Prefer the shortest match, which is almost
        #    always the base film rather than a sequel or a making-of feature.
        prefix, substring = [], []
        for candidate in self._titles:
            candidate_lower = candidate.lower()
            if candidate_lower.startswith(lowered):
                prefix.append(candidate)
            elif lowered in candidate_lower:
                substring.append(candidate)
        if prefix:
            return min(prefix, key=len)
        if substring:
            return min(substring, key=len)

        # 3. Fuzzy match as a last resort, narrowed to titles of a similar
        #    length so difflib is not run across the whole catalogue.
        window = range(max(1, len(query) - 6), len(query) + 7)
        narrowed = [t for t in self._titles if len(t) in window]
        matches = get_close_matches(query, narrowed or self._titles, n=1, cutoff=0.6)
        return matches[0] if matches else None

    def suggest(self, query: str, n: int = 5) -> list[str]:
        """Titles to offer after a failed search.

        find_movie already resolves anything matching by prefix, substring or a
        close fuzzy match, so by the time we get here only a looser threshold
        has any chance of helping -- hence the lower cutoff.
        """
        cleaned = query.strip()
        if not cleaned:
            return []
        matches = get_close_matches(cleaned, self._titles, n=n, cutoff=0.4)
        if matches:
            return matches
        # Last resort: match on any single word the visitor typed.
        words = [w for w in cleaned.lower().split() if len(w) > 2]
        hits = []
        for candidate in self._titles:
            lowered = candidate.lower()
            if any(word in lowered for word in words):
                hits.append(candidate)
                if len(hits) >= n:
                    break
        return hits

    def search_movies(self, query: str, n: int = MAX_AUTOCOMPLETE_RESULTS) -> list[str]:
        """Search titles by prefix first, then substring, for autocomplete."""
        lowered = query.strip().lower()
        if not lowered:
            return []

        prefix, substring = [], []
        for candidate in self._titles:
            candidate_lower = candidate.lower()
            if candidate_lower.startswith(lowered):
                prefix.append(candidate)
            elif lowered in candidate_lower:
                substring.append(candidate)
            if len(prefix) >= n:
                break

        prefix.sort(key=len)
        substring.sort(key=len)
        return (prefix + substring)[:n]

    # ------------------------------------------------------- recommendations

    def _candidate_neighbours(self, movie_idx: int, limit: int):
        """Return (indices, scores) of the most similar movies, best first."""
        if self._neighbor_idx is not None:
            indices = np.asarray(self._neighbor_idx[movie_idx])
            scores = np.asarray(self._neighbor_scores[movie_idx])
            keep = indices != movie_idx
            return indices[keep][:limit], scores[keep][:limit]

        if self._similarity_sparse is not None:
            row = self._similarity_sparse[movie_idx].toarray().ravel()
        else:
            row = np.asarray(self._similarity_dense[movie_idx], dtype=np.float32)

        row = row.copy()
        row[movie_idx] = -np.inf  # never recommend the movie itself

        # argpartition is O(n) against the O(n log n) of a full sort, which
        # matters a great deal when n is in the tens of thousands.
        take = int(min(limit, row.size - 1))
        if take <= 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=np.float32)
        if take < row.size:
            candidates = np.argpartition(-row, take - 1)[:take]
        else:
            candidates = np.arange(row.size)
        candidates = candidates[np.argsort(-row[candidates], kind='stable')]
        return candidates, row[candidates]

    def _movie_payload(self, idx: int, score: float) -> dict:
        movie = self.metadata.iloc[int(idx)]
        title = str(movie['title'])
        imdb_id = _clean(movie.get('imdb_id'), None)
        poster_path = _clean(movie.get('poster_path'), None)
        rating = _number(movie.get('vote_average'))
        votes = _number(movie.get('vote_count'))
        overview = str(_clean(movie.get('overview'), ''))
        genres = [_genre_label(g) for g in _as_list(movie.get('genres'))[:3]]

        return {
            'title': title,
            'release_date': _clean(movie.get('release_date')),
            'year': _year(movie.get('release_date')),
            'production': _clean(movie.get('primary_company')),
            'genre_list': genres,
            'genres': ', '.join(genres) or 'N/A',
            'rating': f"{rating:.1f}/10" if rating is not None else 'N/A',
            'rating_value': round(rating, 1) if rating is not None else None,
            'votes': f"{int(votes):,}" if votes is not None else 'N/A',
            'similarity_score': f"{float(score):.3f}",
            'similarity_percent': max(0, min(100, round(float(score) * 100))),
            'overview': (overview[:180].rstrip() + '...') if len(overview) > 180 else overview,
            'imdb_id': imdb_id,
            'poster_url': f"https://image.tmdb.org/t/p/w342{poster_path}" if poster_path else None,
            'google_link': 'https://www.google.com/search?q=' + quote_plus(f"{title} movie"),
            'imdb_link': f"https://www.imdb.com/title/{quote(str(imdb_id))}/" if imdb_id else None,
        }

    def get_recommendations(self, movie_title: str, n: int = 15,
                            min_rating: float | None = None) -> dict:
        """Recommend movies similar to ``movie_title``."""
        matched_title = self.find_movie(movie_title)
        if not matched_title:
            return {
                'error': f"We couldn't find a movie matching “{movie_title}”.",
                'suggestions': self.suggest(movie_title, 5),
            }

        movie_idx = int(self.title_to_idx[matched_title])
        source = self.metadata.iloc[movie_idx]

        # Over-fetch so post-filtering still has enough candidates to fill n.
        pool = n * 4 + 25 if min_rating is not None else n + 5
        indices, scores = self._candidate_neighbours(movie_idx, pool)

        recommendations = []
        for idx, score in zip(indices, scores, strict=False):
            if len(recommendations) >= n:
                break
            if min_rating is not None:
                rating = _number(self.metadata.iloc[int(idx)].get('vote_average'))
                if rating is None or rating < min_rating:
                    continue
            recommendations.append(self._movie_payload(idx, score))

        source_rating = _number(source.get('vote_average'))
        return {
            'query_movie': matched_title,
            'source_movie': {
                'title': matched_title,
                'year': _year(source.get('release_date')),
                'production': _clean(source.get('primary_company')),
                'rating': f"{source_rating:.1f}/10" if source_rating is not None else 'N/A',
                'genres': ', '.join(_genre_label(g) for g in _as_list(source.get('genres'))[:3]) or 'N/A',
                'poster_url': (
                    f"https://image.tmdb.org/t/p/w342{_clean(source.get('poster_path'), '')}"
                    if _clean(source.get('poster_path'), None) else None
                ),
            },
            'recommendations': recommendations,
        }


# ---------------------------------------------------------------------------
# Process-wide model loading
#
# Loading happens once per worker process, in a daemon thread, behind a lock so
# concurrent first requests cannot each kick off their own load.
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
_RECOMMENDER: MovieRecommender | None = None
_LOADING_THREAD: threading.Thread | None = None
_PROGRESS = 0
_LOAD_ERROR: str | None = None
_GENERATION = 0
_LAST_ATTEMPT = 0.0

# How long to wait before looking for a model again after finding none. This
# lets a model trained while the server is running be picked up on a refresh,
# without retrying the filesystem on every single request.
MISSING_MODEL_RETRY_SECONDS = 15.0


def _resolve_model_dir() -> Path | None:
    """Return the first directory that actually contains a trained model."""
    candidates = [Path(settings.MODEL_DIR)]
    # Kept for backwards compatibility with earlier documented locations.
    for extra in ('models', 'demo_model', 'static', 'training/models'):
        candidate = Path(settings.BASE_DIR) / extra
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        if all((candidate / artifact).exists() for artifact in REQUIRED_ARTIFACTS):
            return candidate

    logger.warning(
        "No model found. Looked in: %s. Set MODEL_DIR or run training/train.py.",
        ', '.join(str(c) for c in candidates),
    )
    return None


def _load_model():
    """Background worker: load the model and publish progress."""
    global _RECOMMENDER, _PROGRESS, _LOAD_ERROR, _GENERATION

    model_dir = _resolve_model_dir()
    if model_dir is None:
        with _LOCK:
            _LOAD_ERROR = 'missing'
        return

    try:
        def progress(value):
            global _PROGRESS
            _PROGRESS = value

        recommender = MovieRecommender(model_dir, progress)
    except Exception:
        logger.exception("Failed to load recommendation model from %s", model_dir)
        with _LOCK:
            _LOAD_ERROR = 'error'
        return

    with _LOCK:
        _RECOMMENDER = recommender
        _PROGRESS = 100
        _LOAD_ERROR = None
        _GENERATION += 1
    logger.info("Recommendation model ready")


def _start_model_loading():
    """Kick off the background load once, and only once, per process."""
    global _LOADING_THREAD, _PROGRESS, _LOAD_ERROR, _LAST_ATTEMPT

    with _LOCK:
        if _RECOMMENDER is not None:
            return
        if _LOADING_THREAD is not None and _LOADING_THREAD.is_alive():
            return
        if _LOAD_ERROR == 'error':
            # A load was attempted and blew up. Retrying on every request would
            # just hammer the logs; a restart is the right recovery here.
            return
        if _LOAD_ERROR == 'missing':
            if time.monotonic() - _LAST_ATTEMPT < MISSING_MODEL_RETRY_SECONDS:
                return
            _LOAD_ERROR = None
        _LAST_ATTEMPT = time.monotonic()
        _PROGRESS = 0
        _LOADING_THREAD = threading.Thread(
            target=_load_model, name='model-loader', daemon=True
        )
        logger.info("Starting model load in the background")
        _LOADING_THREAD.start()


def _get_recommender() -> MovieRecommender | None:
    """Return the loaded recommender, or None while it is unavailable."""
    _start_model_loading()
    return _RECOMMENDER


def _model_state() -> dict:
    """Describe the model's availability for the status and health endpoints."""
    if _RECOMMENDER is not None:
        return {'loaded': True, 'progress': 100, 'status': 'ready'}
    if _LOAD_ERROR == 'missing':
        return {
            'loaded': False,
            'progress': 0,
            'status': 'missing',
            'message': (
                'No recommendation model found. Train one with training/train.py '
                'or point the MODEL_DIR environment variable at an existing model.'
            ),
        }
    if _LOAD_ERROR is not None:
        return {
            'loaded': False,
            'progress': 0,
            'status': 'error',
            'message': 'The recommendation model could not be loaded. Check the server logs.',
        }
    return {'loaded': False, 'progress': _PROGRESS, 'status': 'loading'}


def _reset_for_tests():
    """Clear cached module state. Used by the test suite only."""
    global _RECOMMENDER, _LOADING_THREAD, _PROGRESS, _LOAD_ERROR, _GENERATION
    global _LAST_ATTEMPT
    with _LOCK:
        _RECOMMENDER = None
        _LOADING_THREAD = None
        _PROGRESS = 0
        _LOAD_ERROR = None
        _LAST_ATTEMPT = 0.0
        _GENERATION += 1
    cache.clear()


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def _base_context(**extra) -> dict:
    state = _model_state()
    context = {
        'model_ready': state['status'] == 'ready',
        'model_status': state['status'],
        'total_movies': _RECOMMENDER.movie_count if _RECOMMENDER else 0,
    }
    context.update(extra)
    return context


@require_http_methods(["GET", "POST"])
def main(request):
    """Search form (GET) and recommendation results (POST)."""
    recommender = _get_recommender()

    if recommender is None:
        state = _model_state()
        error = None
        if request.method == 'POST':
            error = state.get(
                'message', 'The movie database is still loading. Please try again in a moment.'
            )
        return render(request, 'recommender/index.html', _base_context(error_message=error))

    if request.method == 'GET':
        return render(request, 'recommender/index.html', _base_context())

    movie_name = request.POST.get('movie_name', '').strip()[:MAX_QUERY_LENGTH]
    if not movie_name:
        return render(request, 'recommender/index.html', _base_context(
            error_message='Please enter a movie name.',
        ))

    try:
        result = recommender.get_recommendations(
            movie_name, n=getattr(settings, 'RECOMMENDATION_COUNT', 15)
        )
    except Exception:
        logger.exception("Recommendation failed for %r", movie_name)
        return render(request, 'recommender/index.html', _base_context(
            input_movie_name=movie_name,
            error_message='Something went wrong generating recommendations. Please try again.',
        ), status=500)

    if 'error' in result:
        return render(request, 'recommender/index.html', _base_context(
            input_movie_name=movie_name,
            error_message=result['error'],
            suggestions=result.get('suggestions', []),
        ))

    return render(request, 'recommender/result.html', {
        'input_movie_name': result['query_movie'],
        'source_movie': result['source_movie'],
        'recommended_movies': result['recommendations'],
        'total_recommendations': len(result['recommendations']),
    })


@require_http_methods(["GET"])
def search_movies(request):
    """Autocomplete endpoint used by the search box."""
    query = request.GET.get('q', '').strip()[:MAX_QUERY_LENGTH]

    if len(query) < 2:
        return JsonResponse({'movies': [], 'count': 0})

    recommender = _get_recommender()
    if recommender is None:
        return JsonResponse({'movies': [], 'count': 0, 'loading': True})

    cache_key = f"movie-search:{_GENERATION}:{query.lower()}"
    matches = cache.get(cache_key)
    if matches is None:
        try:
            matches = recommender.search_movies(query, n=MAX_AUTOCOMPLETE_RESULTS)
        except Exception:
            logger.exception("Autocomplete failed for %r", query)
            return JsonResponse({'error': 'Search failed'}, status=500)
        cache.set(cache_key, matches, SEARCH_CACHE_TIMEOUT)

    return JsonResponse({'movies': matches, 'count': len(matches)})


@require_http_methods(["GET"])
def model_status(request):
    """Report model loading progress so the UI can show a real progress bar."""
    _start_model_loading()
    return JsonResponse(_model_state())


@require_http_methods(["GET"])
def health_check(request):
    """Liveness/readiness probe.

    Returns 200 while the model is still loading: a platform health check that
    fails during a slow first load would kill the deploy before it can finish.
    """
    _start_model_loading()
    state = _model_state()

    if state['status'] == 'ready':
        return JsonResponse({
            'status': 'healthy',
            'model_loaded': True,
            'movies_loaded': _RECOMMENDER.movie_count,
            'model_dir': _RECOMMENDER.model_dir.name,
        })

    if state['status'] == 'loading':
        return JsonResponse({
            'status': 'starting',
            'model_loaded': False,
            'progress': state['progress'],
        })

    return JsonResponse({
        'status': 'unhealthy',
        'model_loaded': False,
        'reason': state['status'],
        'message': state.get('message', ''),
    }, status=503)


def handler404(request, exception=None):
    return render(request, 'recommender/error.html', {
        'error_icon': '🔍',
        'error_title': 'Page not found',
        'error_message': "That page doesn't exist. Let's get you back to the movies.",
    }, status=404)


def handler500(request):
    return render(request, 'recommender/error.html', {
        'error_icon': '⚠️',
        'error_title': 'Something went wrong',
        'error_message': 'An unexpected error occurred. Please try again in a moment.',
    }, status=500)
