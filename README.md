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

See data/README.md. The first training notebook you run builds `data/eurosat_cache.npy`
(about 2.9 GB, not in Git) from the GeoTIFFs. This happens once; later runs only load it.

## Project structure

- notebooks/: 01_data_exploration, 02_split_and_preprocessing, 03_baseline_rf, 04_cnn,
  05_transfer_resnet, 06_evaluation, 07_haze (optional). Run them in this order.
- src/piece_of_crop/: reusable code imported by the notebooks
- results/predictions/: test predictions, one CSV per model, input and seed,
  named <model>_<input>_seed<n>.csv (inspect saved CSV headers for the exact prediction columns)
- results/figures/: saved figures
- tests/: pytest tests

## How to reproduce

1. Clone the repository and follow the Setup instructions above.
2. Obtain the multispectral EuroSAT GeoTIFF dataset using
   the instructions in data/README.md.
3. Extract the dataset into data/EuroSAT_MS/, with class
   folders such as AnnualCrop directly inside that directory.
4. Open the repository root in VS Code and select the
   configured Python environment as the Jupyter kernel.
5. Run notebooks 01 through 07 in numerical order.
   Check that each notebook resolves the project root
   correctly before running all cells.

Notebook execution order:

- 01: Explore data and generate descriptive figures.
- 02: Create the train/validation/test split, training-only
  band statistics and the reusable EuroSAT cache.
- 03: Train and evaluate Random Forest baselines.
- 04: Train RGB and 13-band CNN models.
- 05: Fine-tune RGB and 13-band ResNet18 models.
- 06: Compare models, confusion matrices and business costs.
- 07: Test model robustness under synthetic haze.

Notebook 01 generates results/file_summary.csv locally.
Notebook 02 creates data/eurosat_cache.npy, which is reused
by subsequent notebooks.

Model checkpoints and raw data are excluded from Git.
To reproduce the models from scratch, the training notebooks
must be executed. Pretrained ResNet weights may require
an initial download.

Results and visualizations are saved under results/.


## Results and limitations

The project compares RGB and 13-band multispectral
Sentinel-2 imagery using Random Forest, CNN and ResNet18
classifiers on EuroSAT land-cover categories.

Evaluation outputs include accuracy, macro-F1, confusion
matrices, class-level performance and cost-sensitive analysis.

Business cost matrices represent evaluation scenarios, not
verified insurance claim payments. Their assumptions should
be reviewed before operational use.

The haze experiment uses synthetic perturbations rather
than real atmospheric haze. Results should not be interpreted
as proof of robustness under actual weather conditions.

Random seeds are used to assess model variability.
The trained models are research prototypes and are not
validated for real-world insurance decision-making.

Saved CSV files and figures reflect the executed experiments.
Raw imagery, image caches and model weights are deliberately
excluded from Git because of their size.
