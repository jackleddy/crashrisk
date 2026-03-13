## crashrisk project

temp readme

### Setup
python -m venv .venv

.venv\Scripts\activate
pip install -U pip
pip install -e .

### Scripts
for now, gotta run ```pip install -e . ``` before each script when edited

## Build OSM road graph
python scripts/build_network.py

## Pull data
python scripts/download_data.py

## Assin data to graph
python scripts/align_data.py