# Piece of Crop

Does multispectral (13-band) Sentinel-2 imagery reduce the land-cover classification errors
that are most costly to crop insurers, compared with RGB imagery?

Team: Estelle Arnander, Manvi Gupta, Rishi Gupta, Loraine Nieto, Faris Selimovic, Aoi Ueno.

## Setup

Run these from the project root (the folder that contains pyproject.toml):

```bash
conda env create -f environment.yml
conda activate piece-of-crop
python -m pip install -e .
python -c "import piece_of_crop; print('Package works')"
pytest -q
```

## Data

See data/README.md.

## Project structure

- notebooks/: 01_data_exploration, 02_split_and_preprocessing, 03_baseline_rf, 04_cnn,
  05_transfer_resnet, 06_evaluation, 07_haze (optional). Run them in this order.
- src/piece_of_crop/: reusable code imported by the notebooks
- results/predictions/: test predictions, one CSV per model, input and seed,
  named <model>_<input>_seed<n>.csv with columns file, true_label, pred_label, prob_0 to prob_9
- results/figures/: saved figures
- tests/: pytest tests

## How to reproduce

To be completed.

## Results and limitations

To be completed.
