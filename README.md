## crashrisk project

temp readme

### Setup
python -m venv .venv

.venv\Scripts\activate
pip install -U pip
pip install -e .

### Scripts
Run `pip install -e .` once after setup. Re-run it only after dependency or package metadata changes.

## Build OSM road graph
python scripts/build_network.py

## Pull data
python scripts/download_data.py

## Assin data to graph
python scripts/align_data.py
