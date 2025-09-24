# Changelog

## [Unreleased]
- Added `run_pipeline.py` to orchestrate loader → normalizer → matcher → issues → UI dataset in a single command.
- Removed legacy `src/reconciler.py` in favor of the matcher → issues → UI dataset pipeline documented in the README.
- Rebuilt data pipeline modules (`loader`, `normalizer`, `matcher`, `ui_dataset_builder`, exports, UI server) with ASCII-safe implementations, structured logging, and CLI compatibility with VS Code tasks.
- Added refined matching configuration and regex catalogs in `cfg/` and cleaned repository tree (`TREE.md`).
- Implemented end-to-end fixtures and pytest suite covering loader -> normalizer -> matcher -> UI datasets.
- Updated VS Code tasks to align with new pipeline outputs and relocated configuration files.

