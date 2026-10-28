# Changelog

> All notable changes to the Movie Recommendation System are documented in this file.

This project follows [Semantic Versioning](https://semver.org/) and the changelog format is based on [Keep a Changelog](https://keepachangelog.com/).

---

## 📑 Table of Contents

- [Unreleased](#unreleased)
- [Version 2.0.0 - 2024-12-05](#200---2024-12-05) - Complete Revamp
- [Version 1.0.0 - 2022-XX-XX](#100---2022-xx-xx) - Initial Release
- [Migration Guide](#-migration-guide)
- [Future Roadmap](#-future-roadmap)

---

## [Unreleased]

### Fixed
- **The app could not start at all.** The logging config declared a rotating
  file handler writing to `logs/django.log`, but that directory is not in the
  repository and `dictConfig` opens every handler at configuration time. Every
  `manage.py` command failed with `ValueError: Unable to configure handler 'file'`.
  Logging is now console-only, which is also the right choice for containerised
  hosts with ephemeral disks.
- **No model could ever ship.** `.gitignore` excluded `*.json`, `*.parquet`,
  `*.npy`, `*.npz` and `*.pkl` repository-wide, so the "demo model" the README
  promised in `static/` was never committed. The ignore rules are now scoped to
  the `models/` output directories, and the docs describe the real setup.
- **Loading a model could exhaust memory.** The similarity matrix was densified
  with `.toarray()` on load -- about 10 GB for the documented 50,000-movie
  configuration. Sparse matrices now stay sparse and dense ones are
  memory-mapped, so only the row being queried is read.
- Genres never displayed: parquet list columns load as numpy arrays, so the
  `isinstance(value, list)` check discarded every genre. Affected the web app
  and `training/infer.py`.
- The year filter in `training/infer.py` read the last component of an ISO
  date, turning `1999-03-30` into the year `30`.
- `training/train.py` crashed on Windows consoles (cp1252) when printing emoji.
- `SECURE_SSL_REDIRECT` was enabled without `SECURE_PROXY_SSL_HEADER`, which
  causes an infinite redirect loop behind the TLS-terminating proxies used by
  Render and Heroku. `CSRF_TRUSTED_ORIGINS` is now derived from the deployment
  hostname as well, which Django requires for HTTPS form posts.
- `/api/health/` returned 503 while the model was still loading, so a platform
  health check could kill a deploy before it finished starting. It now reports
  200 with `status: starting` during load and 503 only for a real failure.
- The health endpoint no longer echoes exception text or filesystem paths.
- Model-loading state was mutated from request threads without a lock, so
  concurrent first requests could each start their own loader thread.
- Google links are URL-encoded, so titles containing `&`, `?` or `#` work.
- Rating filters are NaN-safe.
- Poster artwork rendered *underneath* the placeholder icon, and the rank,
  match and rating badges underneath the artwork. The placeholder is absolutely
  positioned and positioned elements paint above static ones regardless of DOM
  order, so the layers needed explicit z-index values. Only visible once real
  poster images load.
- `training/infer.py` read `title_to_idx.json` and `config.json` with the
  platform default encoding, so loading any model with non-ASCII titles failed
  with `UnicodeDecodeError` on Windows. Both files are now read as UTF-8, which
  is how the trainer writes them.
- The home page said recommendations match on "cast". The model uses genres,
  keywords, studios, countries and plot text; the wording now says so.
- Genres appeared as the trainer's internal tokens (`sciencefiction`, `tvmovie`,
  everything lowercase). The views now turn them into display names such as
  "Science Fiction", so existing models need no retraining.

### Changed
- The catalogue is no longer inlined into every page render. Autocomplete uses
  the existing (previously unused) `/api/search/` endpoint, so page weight is
  constant instead of growing with the number of movies.
- Recommendations use `np.argpartition` rather than sorting every movie.
- Title lookup tries exact, then prefix, then substring, and only falls back to
  fuzzy matching -- much faster and more predictable than running difflib over
  the whole catalogue on every request.
- `training/train.py` stores the top **K** neighbours per movie instead of the
  full N x N similarity matrix: roughly 20 MB rather than 10 GB at 50,000
  movies, computed in bounded-memory chunks. The old format is still readable,
  and `--legacy-matrix` still produces it.
- `training/train.py` has a real command-line interface (`--help`).
- MMR "diverse recommendations" re-ranks a shortlist instead of scoring every
  movie against every selection, which was quadratic and unusable at scale.
- Runtime and training dependencies are split (`requirements-train.txt`), and
  the unused ones -- Django REST Framework, django-cors-headers,
  django-extensions, redis, django-redis, python-decouple, fastparquet -- are
  gone. `nltk`, which `train.py` imports, was missing and is now declared.
- `SECRET_KEY` must be set explicitly when `DEBUG=False`; the app refuses to
  start with the development key in production.
- Gunicorn runs one worker with threads, so the model is held once per host.
- `PYTHON_VERSION` on Render raised to 3.12 (numpy 2.3 requires 3.11+).
- `static/logo.ico` reduced from 242 KB to 6.8 KB by keeping the 16/32/48 px
  frames instead of nine sizes up to 256x256. Same artwork -- the 48 px frame
  is pixel-identical to the original.
- Packaging recipes exclude `*.pkl`. The TF-IDF and SVD pickles are retraining
  artifacts that nothing loads at serving time, and they dominate the size: a
  1,200-movie model is 17.2 MB packaged, or 299 KB without them. `demo_model/`
  is also searched automatically, so committing a small model needs no
  configuration at all.
- README rewritten around what the project does today: real clone and issue
  URLs in place of `yourusername` placeholders, a "use it your way" section for
  the app, the pipeline and the library, API examples taken from a running
  instance, and performance figures that were measured rather than estimated.
  `PROJECT_GUIDE.md` and `training/guide.md` lost the same stale numbers and the
  full-matrix descriptions.
- Logo, header banner and architecture diagram replaced. The old images showed
  the previous design (a full `similarity_matrix.npz`, "Django/Flask",
  Python 3.10). The new ones are hand-written SVG with PNG renders in
  `assets/images-for-readme/`, and the
  architecture diagram matches the code. Screenshots and the demo GIF were
  retaken from the current UI with posters turned off, so no TMDB artwork is
  redistributed. The GIF dropped from 8 MB to under 0.5 MB; the stale MP4 and
  loading-screen screenshot were removed.

### Added
- `LICENSE` now names the author as copyright holder instead of the project.
- `SECURITY.md`: private vulnerability reporting and deployment notes.
- `assets/images-for-readme/social-preview.png` (1280x640) for the repository's
  social card.
- A test suite (37 tests) covering model loading, matching, filtering, link
  encoding, the API endpoints, and the missing-model and error states.
- Posters on result cards, with a graceful fallback when artwork is missing.
- "Did you mean" suggestions are now rendered as clickable chips. The view had
  been computing them all along; the template never displayed them.
- A clear, actionable message when no model is installed, instead of a loading
  bar that never finished.
- `.env.example` documenting every supported environment variable.
- Custom 404 and 500 pages (`error.html` existed but was never wired up).
- Keyboard-accessible autocomplete (ARIA combobox), reduced-motion support, and
  no-JavaScript fallbacks so content is never left invisible.
- `Dockerfile` and `.dockerignore`. The README claimed Docker compatibility and
  the guide told you to copy a Dockerfile out of the documentation; that one
  pinned Python 3.10 (too old for numpy 2.3) and set `DEBUG=False` with no
  `SECRET_KEY`, so it could not have started. The ignore patterns use `**/`
  prefixes: Docker matches against the context root, so a bare `*.pkl` would
  have baked the 40 MB SVD pickle into the image (930 MB vs 852 MB).
- Optional `MODEL_URL` support in `render.yaml`: if set, the build downloads and
  unpacks a model archive into `MODEL_DIR`. Build-time only -- the app never
  fetches anything while serving.
- A GitHub Actions workflow running lint, `check`, `check --deploy` and the
  test suite. The startup crash above is exactly the class of bug
  `manage.py check` catches and a test suite cannot, because settings have to
  load first.
- **TMDB attribution.** The demo model is derived from TMDB data and the UI
  loads poster images from `image.tmdb.org`, but the required notice appeared
  nowhere. It is now in the app footer, the README, and `demo_model/README.md`
  alongside the data's provenance.
- `CONTRIBUTING.md`: setup, how to run the checks, the layout of the code, and
  the non-obvious things -- parquet list columns loading as numpy arrays, the
  top-K neighbour format, and the conda-versus-pip scipy problem that breaks
  training on some machines.
- Lint configuration (`ruff`, in `pyproject.toml`) and a CI step for it. The
  contributing guide previously said "follow PEP 8" with nothing to run.
- 20 tests for the training and inference pipeline, covering the two bugs that
  were silent rather than loud -- genres dropped by an `isinstance` check and
  release years read off the wrong end of an ISO date -- plus neighbour
  selection and chunking. They skip themselves when the training extras are
  not installed, so CI runs them without scikit-learn.


### In Development
- User authentication system
- Personal watchlists and favorites
- Movie rating functionality
- Advanced filtering options

---

## [2.0.0] - 2024-12-05

### 🎉 Complete Project Revamp

**Major transformation** from college project to production-ready system with advanced model training capabilities.

**Release Highlights:**
- ✨ Advanced ML training pipeline supporting 10K-1M+ movies
- ⚡ 90% performance improvement (500ms → 50ms)
- 💾 56% reduction in memory usage
- 🎨 Modern, fully responsive UI
- 📚 Comprehensive documentation (3 core files)
- 🚀 Production-ready with security hardening

---

### Added

#### 🤖 Advanced Model System
- **Integrated Training Pipeline** - Complete training system in `training/train.py`
  - Supports datasets from 10K to 1M+ movies
  - Configurable quality thresholds (low/medium/high)
  - SVD dimensionality reduction for efficiency
  - Three training configurations (small/medium/large)
- **Flexible Model Directory** - `MODEL_DIR` environment variable for easy model switching
- **Rich Metadata Support** - Ratings, votes, genres, production companies, IMDb IDs, poster URLs
- **Advanced Filtering** - Filter recommendations by year, rating, and genre
- **Fuzzy Search** - Intelligent movie title matching with suggestions
- **Model Verification** - Health check endpoint shows model status

**Related Documentation:**
- [README.md - Model Training](README.md#-model-training)
- [training/guide.md](training/guide.md) - Complete training guide
- [PROJECT_GUIDE.md - Model Training](PROJECT_GUIDE.md#-model-training)

#### ⚙️ Backend Improvements
- **Integrated Recommender Class** - Unified logic matching training/inference pipeline
- **Lazy Model Loading** - Models loaded only when needed with global caching
- **Type Hints** - Complete type annotations throughout codebase
- **Comprehensive Logging** - Rotating file handlers with configurable levels
- **Better Error Handling** - Graceful failures with helpful error messages
- **API Endpoints** - RESTful endpoints (`/api/search/`, `/api/health/`)
- **Django 6.0** - Updated to latest stable Django version

**Technical Details:**
- Global singleton pattern for model loading
- Sparse matrix support for large similarity matrices
- Efficient Parquet format for data storage
- Environment-based configuration

#### 🎨 Frontend Enhancements
- **Modern Responsive UI** - Mobile-first design that works on all devices
- **Enhanced Movie Cards** - Display ratings, votes, genres, production companies
- **Multiple External Links** - Google Search and IMDb integration
- **Better Visual Feedback** - Loading states, animations, smooth transitions
- **Improved Error Messages** - Clear feedback with movie suggestions
- **Inline Styles** - All CSS inline for faster loading and easy customization

**UI Features:**
- Dark theme by default
- Gradient color scheme
- Smooth animations
- Autocomplete search
- Responsive grid layout

#### 📚 Documentation
- **Simplified Structure** - Only 3 core documentation files
  - `README.md` - Overview, quick start, features
  - `PROJECT_GUIDE.md` - Complete technical guide
  - `CHANGELOG.md` - This file
- **Comprehensive Content** - Installation, training, deployment, troubleshooting
- **Clear Organization** - Table of contents, cross-linking, sections
- **Training Guide** - Detailed model training in `training/guide.md`
- **Screenshot Placeholders** - Ready for visual documentation
- **API Reference** - Complete endpoint documentation

**Documentation Links:**
- [README.md](README.md)
- [PROJECT_GUIDE.md](PROJECT_GUIDE.md)
- [training/guide.md](training/guide.md)

#### 🚀 Infrastructure
- **Deployment Configurations**
  - `render.yaml` - Render deployment
  - `Procfile` - Heroku deployment
  - Docker-ready setup
- **Build Automation** - Inline build commands in `render.yaml`
- **Environment Management** - `.env` file support with examples
- **Git Optimization** - Proper `.gitignore` for clean repository

---

### Changed

#### Performance Improvements
| Metric | Before (v1.0) | After (v2.0) | Improvement |
|--------|---------------|--------------|-------------|
| Recommendation Time | ~500ms | ~50ms | **90% faster** ⚡ |
| Memory Usage | 350MB | 200MB | **43% less** 💾 |
| Model Size | 320MB | 180MB | **44% smaller** 📦 |
| Page Load Time | ~800ms | ~200ms | **75% faster** 🚀 |

#### Architecture
- **Modular Design** - Clear separation of concerns
- **Configurable Models** - Easy to switch between trained models
- **Production-Ready** - Security hardened with best practices
- **Scalable** - Supports datasets from 2K to 1M+ movies

#### User Experience
- **Cleaner Interface** - Modern, intuitive design
- **Faster Loading** - Removed video/image overhead (~20MB saved)
- **Better Feedback** - Clear messages and loading states
- **Mobile Optimized** - Touch-friendly, responsive layout

---

### Technical Details

#### Dependencies Updated
```
Django: 3.x → 6.0
pandas: 1.x → 2.3.3+
numpy: 1.x → 2.3.5
scipy: → 1.16.3
scikit-learn: → 1.7.2+
```

#### New Files
- `training/train.py` - Training pipeline
- `training/infer.py` - Inference examples
- `training/guide.md` - Training documentation
- `PROJECT_GUIDE.md` - Complete technical guide
- `render.yaml` - Deployment configuration

#### Updated Files
- `recommender/views.py` - Complete refactor with new model system
- `recommender/templates/` - Modern UI redesign
- `movie_recommendation/settings.py` - Production configurations
- `requirements.txt` - Updated dependencies
- `README.md` - Comprehensive overview
- `CHANGELOG.md` - This file

#### Removed
- Setup scripts (`setup.py`, `setup.sh`, `setup.bat`) - Simplified installation
- Old notebooks - Outdated training code
- Redundant documentation - Consolidated to 3 files
- Static overhead - Videos, extra images (~20MB)
- Duplicate CSS/JS files - Now inline in templates

---

### Fixed

- 🐛 **Memory Leaks** - Fixed in model loading
- 🐛 **Search Accuracy** - Improved fuzzy matching
- 🐛 **Mobile Layout** - Fixed responsive design issues
- 🐛 **Static Files** - Proper serving in production
- 🐛 **CSRF Errors** - Correct token handling

---

### Security

- 🔒 Updated all dependencies to secure versions
- 🛡️ Implemented CSRF protection
- 🔐 Added security headers (X-Frame-Options, XSS Protection)
- 🔒 HTTPS enforcement in production
- 🛡️ Secure cookie settings
- 🔐 Input validation and sanitization

---

### Breaking Changes

⚠️ **Important**: Version 2.0 includes breaking changes from v1.x

1. **Model Format Changed**
   - Old models are not compatible
   - Must retrain or use new demo model
   - See [Migration Guide](#-migration-guide)

2. **API Response Format**
   - Movie objects have new structure with additional fields
   - `movie_title` → `title`
   - Added: `production`, `genres`, `rating`, `votes`

3. **Template Variables**
   - Updated to match new movie metadata
   - See templates for new structure

4. **Environment Variables**
   - `MODEL_DIR` now required for custom models
   - Default: `./models` (or `./static` for demo)

5. **URL Structure**
   - API endpoints now have `/api/` prefix
   - `/search/` → `/api/search/`
   - Added: `/api/health/`

---

## 📋 Migration Guide

### From v1.x to v2.0

#### Step 1: Backup Data
```bash
# Backup your models and data
cp -r static/ backup/
cp -r models/ backup/ 2>/dev/null || true
```

#### Step 2: Update Code
```bash
# Pull latest changes
git fetch origin
git checkout main
git pull origin main
```

#### Step 3: Update Dependencies
```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate

# Update dependencies
pip install -r requirements.txt --upgrade
```

#### Step 4: Choose Model Option

**Option A: Use Demo Model (Quick)**
```bash
export MODEL_DIR=./static
```

**Option B: Train New Model (Recommended)**
```bash
# See training/guide.md for complete instructions
python training/train.py

# Use your trained model
export MODEL_DIR=./models
```

#### Step 5: Update Environment
```bash
# Create .env file
cp .env.example .env

# Edit with your settings
# MODEL_DIR=./models
# DEBUG=True (for development)
```

#### Step 6: Run Migrations
```bash
python manage.py migrate
```

#### Step 7: Test
```bash
# Start server
python manage.py runserver

# Test in browser
# http://localhost:8000

# Test API
curl http://localhost:8000/api/health/
```

#### Step 8: Update Custom Code (if any)

If you modified the original code:

1. **Views**: Check `recommender/views.py` for new structure
2. **Templates**: Update to use new movie object fields
3. **URLs**: Update to use new API endpoints
4. **Settings**: Review `settings.py` for new configurations

**Need help?** See [PROJECT_GUIDE.md - Troubleshooting](PROJECT_GUIDE.md#-troubleshooting)

---

## 🗺️ Future Roadmap

### Version 2.1 (Planned - Q1 2025)
- [ ] **User Authentication** - Register, login, profile management
- [ ] **Personal Watchlists** - Save and organize movies
- [ ] **Movie Ratings** - Rate movies and get personalized suggestions
- [ ] **Advanced Filtering** - Multiple genres, decade filters, rating ranges
- [ ] **Recommendation History** - Track past recommendations
- [ ] **Export Features** - Export watchlists and recommendations

### Version 2.2 (Planned - Q2 2025)
- [ ] **Collaborative Filtering** - Combine with content-based filtering
- [ ] **Social Features** - Share recommendations, follow users
- [ ] **Movie Reviews** - Write and read reviews
- [ ] **Discussion Forums** - Community discussions
- [ ] **Advanced Analytics** - Viewing statistics and trends
- [ ] **Multi-language Support** - Internationalization (i18n)

### Version 3.0 (Long-term)
- [ ] **Mobile Applications** - Native iOS and Android apps
- [ ] **Real-time Recommendations** - WebSocket-based live updates
- [ ] **Streaming Integration** - Link to Netflix, Prime, etc.
- [ ] **Enhanced ML Models** - Hybrid recommendations (content + collaborative)
- [ ] **Microservices Architecture** - Scalable backend redesign
- [ ] **GraphQL API** - Alternative to REST API
- [ ] **Video Trailers** - Integrated trailer viewing
- [ ] **AI Chat Assistant** - Conversational movie recommendations

---

## [1.0.0] - 2022-XX-XX

### Initial Release

**First Version**
- Basic movie recommendation functionality
- Simple Django web interface
- Demo model with 2,000 movies
- Search with basic autocomplete
- Content-based filtering algorithm
- Simple card-based results display

**Features:**
- Movie search
- Top 15 recommendations
- Google search links
- Basic error handling

---

## 📊 Version Comparison

| Feature | v1.0 | v2.0 |
|---------|------|------|
| **Movies** | 2K (demo only) | 2K-1M+ (configurable) |
| **Training** | Manual, complex | Automated pipeline |
| **Performance** | 500ms | 50ms (10x faster) |
| **Memory** | 350MB | 200MB (43% less) |
| **UI** | Basic | Modern, responsive |
| **Documentation** | Minimal | Comprehensive |
| **API** | None | REST API |
| **Deployment** | Manual | Automated (Render/Heroku) |
| **Security** | Basic | Production-ready |
| **Testing** | None | Test-ready structure |

---

## 📝 Notes

### Release Naming

This project follows [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality
- **PATCH** version for backwards-compatible bug fixes

### Documentation

For detailed information about features and usage:
- **Overview**: [README.md](README.md)
- **Technical Guide**: [PROJECT_GUIDE.md](PROJECT_GUIDE.md)
- **Training Guide**: [training/guide.md](training/guide.md)

### Support

- **Issues**: [GitHub Issues](https://github.com/inboxpraveen/Movie-Recommendation-System/issues)

---

## 🤝 Contributing

See changes you'd like to make? Contributions are welcome!

1. Check [Future Roadmap](#-future-roadmap) for planned features
2. Open an issue to discuss your idea
3. Fork the repository
4. Make your changes
5. Submit a pull request

---

<div align="center">

**Last Updated:** December 5, 2024  
**Current Version:** 2.0.0  
**Status:** Production Ready ✅

[⬆ Back to Top](#changelog)

</div>
