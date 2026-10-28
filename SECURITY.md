# Security policy

## Supported versions

Security fixes go into the latest release on `master`. Older releases are not patched.

## Reporting a vulnerability

Please report security issues privately, not in a public issue.

Open the repository's **Security** tab and choose **Report a vulnerability**. This creates a private advisory visible only to the maintainer.

Include what you found, the steps to reproduce it, and the version or commit you tested. You can expect an acknowledgement within a week. Once a fix is released, the report can be disclosed publicly with credit if you want it.

## Deployment notes

The app is designed to be safe by default, but a deployment is only as safe as its configuration:

- Set `SECRET_KEY` and `DEBUG=False` in production. The app refuses to start with the development key when `DEBUG=False`.
- Set `ALLOWED_HOSTS` to your real domain.
- Serve over HTTPS. HSTS and secure cookies are enabled when `DEBUG=False`.
- Only load model directories you trained or trust. The app reads parquet, JSON and NumPy arrays and never unpickles anything, but the retraining artifacts (`*.pkl`) are Python pickles and must only be opened from sources you trust.
