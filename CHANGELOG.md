Changelog
=========

[0.5.0] - 2023-09-18
--------------------

### Added

- `CHANGELOG.md`
- `Tagger` now supports taking a `dict[str, int]` as the spec, instead of just
  `list[str]`.
- Version is now available as the `__version__` attribute

### Changed

- Bumped required Python to 3.8


[0.4.1] - 2022-10-10
--------------------

### Changed

- Wrap casting to `int` in `Tagger.tag` so that it always throws `TypeError` if
  casting fails. Previously, NumPy could raise both `TypeError` and `ValueError`
  during casting.

### Added

- Improve documentation
  - Add examples to `Tagger.tag`
  - Add "Raises" block to `Tagger.tag`
  - Update README for changes introduced in v0.4.0
