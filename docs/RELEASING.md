# Releasing Pickframe

## Versions and build numbers

* **Version**: [Semantic Versioning](https://semver.org) `MAJOR.MINOR.PATCH`,
  stored once in `src/pickframe/__init__.py` (`__version__`). Pre-releases use
  suffixes such as `1.1.0rc1` and are published as GitHub pre-releases.
* **Tag**: `v` + version, e.g. `v1.0.0`. The release workflow refuses to run
  if the tag and `__version__` differ, or if `CHANGELOG.md` lacks a
  `## [1.0.0]` section.
* **Build number**: the run number of the Release workflow. CI stamps it into
  `src/pickframe/_build.py` together with the commit and date
  (`tools/stamp_build.py`), so it appears in:
  * `pickframe --version` → `Pickframe 1.0.0 (build 7, 1a2b3c4, 2026-11-05)`
  * the app's About box,
  * Windows file properties (`1.0.0.7`) and the installer,
  * macOS `CFBundleVersion`.

Source checkouts report `(dev)` instead of a build number.

## Cutting a release

1. Move the entries under `## [Unreleased]` in `CHANGELOG.md` to a new
   `## [X.Y.Z] - YYYY-MM-DD` section and update the links at the bottom.
2. Set `__version__ = "X.Y.Z"` in `src/pickframe/__init__.py`.
3. Commit, then tag and push:

   ```bash
   git commit -am "Release vX.Y.Z"
   git tag -a vX.Y.Z -m "Pickframe X.Y.Z"
   git push origin main vX.Y.Z
   ```

4. The [Release workflow](../.github/workflows/release.yml) then:
   * verifies the version, changelog and tests,
   * builds the desktop app with PyInstaller on Windows, macOS and Linux and
     smoke-tests each build,
   * packages a Windows installer and zip, a macOS disk image, a Linux AppImage
     and tarball, plus the Python wheel and sdist,
   * writes `SHA256SUMS.txt` and publishes the GitHub release with notes taken
     from the changelog.

To rebuild an existing tag, run the workflow manually ("Run workflow") and
enter the tag.

## Publishing to PyPI (optional)

1. On PyPI, add a trusted publisher for this repository, workflow
   `release.yml` and environment `pypi`.
2. In the GitHub repository settings, create the `pypi` environment and set
   the repository variable `PYPI_PUBLISH` to `true`.

The next release will then also upload the wheel and sdist to PyPI.
