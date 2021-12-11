# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Most recent change on the bottom.

## [Unreleased]
### Added
- Added dtype autocasting based on example input trace

## 0.1.4 - 2021-11-7
### Added
- `opt_einsum_fx.__version__`
- Partially symbolic shape propagation for efficient einsum optimization (#15)

## 0.1.3 - 2021-10-29
### Added
- PyTorch 1.10 compatability

### Fixed
- Added `packaging` to dependency list

## 0.1.2 - 2021-06-28
### Added
- PyTorch 1.9 compatibility

## 0.1.1 - 2021-05-27
### Added
- Docs
- PyPI package

### Fixed
- `jitable` no longer makes some FX nodes' `.args` lists (technically not allowed) instead of tuples

## [0.1.0] - 2021-05-17