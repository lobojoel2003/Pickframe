"""Build metadata.

Release builds overwrite this file in CI (see ``tools/stamp_build.py``) with
the CI run number, the commit and the build date. Source checkouts keep the
placeholder values below.
"""

BUILD_NUMBER = "0"
COMMIT = ""
BUILD_DATE = ""
CHANNEL = "dev"
