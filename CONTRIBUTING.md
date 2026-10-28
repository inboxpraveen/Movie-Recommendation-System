# Contributing

Thanks for considering a contribution. This guide covers getting set up,
running the checks, and the few things about this project that are not obvious
from the code.

---

## Getting set up

```bash
git clone https://github.com/inboxpraveen/Movie-Recommendation-System.git
cd Movie-Recommendation-System

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Open <http://localhost:8000>. A 6,248-movie demo model ships in `demo_model/`
and is picked up automatically, so search works immediately, so you do **not**
need a dataset or a training run to work on the app.

Python 3.11 or newer is required (numpy 2.3 dropped 3.10).

---

## Before you open a pull request

Run these three. CI runs the same ones.

```bash
python manage.py check                 # configuration and startup problems
python manage.py test                  # the test suite
ruff check .                           # lint (pip install ruff)
```

`ruff check . --fix` applies the mechanical fixes. Style is configured in
`pyproject.toml`; there is no separate style guide to read.

### Why `manage.py check` matters

It catches a whole class of bug the tests cannot: anything that breaks before
Django finishes loading settings. This project shipped a release where a
logging handler pointed at a directory that did not exist, and *every*
`manage.py` command failed, while the test suite could not have caught it,
because the tests never got to run. Please don't skip it.

---

## Working on the different pieces

### The web app: `recommender/`

`views.py` holds everything: model loading, matching, and the four endpoints.
There are no database models; the recommender is read-only once loaded.

The model loads once per process in a background thread, and the UI polls
`/api/model-status/` for progress. If you touch that machinery, exercise all
four states (`loading`, `ready`, `missing`, `error`), not just the happy path.

Templates are `recommender/templates/recommender/`. CSS and JS are inline by
design (no build step). Two rules to keep:

- **Nothing that depends on the catalogue size goes into the page.** Titles are
  fetched from `/api/search/`. An earlier version inlined every title into
  every render, which grew the page to megabytes.
- **Never `innerHTML` with server data.** The autocomplete builds DOM nodes.

### The model pipeline: `training/`

Needs extra dependencies:

```bash
pip install -r requirements-train.txt
```

`train.py` builds a model from a TMDB-shaped CSV; `infer.py` is a standalone
inference/CLI example. Run `python training/train.py --help` for the options.

Two things worth knowing:

- **Top-K neighbours, not a similarity matrix.** Storing the full N×N matrix
  costs `n*n*4` bytes, about 10 GB at 50,000 movies. The trainer stores the
  50 nearest neighbours per movie instead (~20 MB) and computes them in
  bounded-memory chunks. `--legacy-matrix` still writes the old format, and
  inference reads either.
- **Parquet list columns load as numpy arrays, not lists.** `isinstance(x, list)`
  on a genres cell is always `False` and silently drops every genre. Use the
  `as_list()` / `_as_list()` helpers.

### Model files

Model artifacts are not source. `models/` and `*.pkl` are git-ignored; the
committed `demo_model/` is deliberately not. If you change the model format,
update the loader in `views.py`, `training/infer.py`, and the artifact list in
`demo_model/README.md` together.

---

## Tests

```bash
python manage.py test                          # everything
python manage.py test recommender              # the web app
python manage.py test training                 # trainer and inference helpers
```

The suite builds a small synthetic model in a temp directory, so it is fast and
needs no dataset. The training tests skip automatically when scikit-learn and
nltk are not installed.

Please add a test with a bug fix. The ones worth copying as examples are the
regression tests for bugs that were invisible until real data arrived: genres
being dropped by an `isinstance` check, and poster artwork rendering underneath
its own placeholder.

---

## Troubleshooting

**`ImportError: DLL load failed while importing _moduleTNC`** (or another scipy
import error) when running the trainer, typically on Windows: a security policy
is blocking the pip-installed scipy binary. Use conda for the scientific stack:

```bash
conda create -n movierec python=3.12 numpy scipy pandas pyarrow scikit-learn nltk
conda activate movierec
pip install -r requirements.txt
```

The web app is unaffected (it imports scipy lazily), so only training breaks.

**`ValueError: Missing staticfiles manifest entry`**: you are running with
`DEBUG=False` without having run `python manage.py collectstatic`.

**`/api/health/` returns 503 with `"reason": "missing"`**: no model was found.
Check that `demo_model/` exists or that `MODEL_DIR` points at a directory
containing `config.json`, `title_to_idx.json` and `movie_metadata.parquet`.

---

## Pull requests

- Branch from `master`.
- Keep the change focused; separate refactors from behaviour changes.
- Update `CHANGELOG.md` under `[Unreleased]`.
- Say what you tested. If you changed the UI, a screenshot helps.

Bug reports are most useful with the output of `python manage.py check`, your
Python version, and whether a model is loaded (`curl localhost:8000/api/health/`).

---

## Data attribution

Movie data comes from [TMDB](https://www.themoviedb.org/). This product uses
data from TMDB but is not endorsed or certified by TMDB. The MIT licence covers
this project's source code, not the movie data.
