# Piece of Crop — Streamlit prototype

A small academic prototype using the team's existing **EuroSAT 13-band** dataset, **Random Forest**, **SmallCNN** and **ResNet18** experiments. It demonstrates three workflows:

1. **Satellite Classification:** live inference for RGB **and** 13 bands on the *same* full multispectral patch, or an explicitly labelled *saved test prediction* if model files are unavailable.
2. **Model Comparison:** accuracy, macro-F1 and mean insurer severity computed from saved test predictions across up to 3 seeds. Only prediction files that match the *current* `results/split.csv` are included.
3. **Insurance Impact:** cost matrix, rank of cost-weighted model errors and most consequential misclassifications. The numbers are **illustrative severity points, not euros or actual claims**.

## Repository layout

Place the whole `prototype/` directory inside the **existing** `piece_of_crop` repository, next to `src/`, `notebooks/`, `models/` (local only) and `results/`:

```
piece_of_crop/
├── prototype/
│   ├── app.py
│   ├── core.py
│   ├── analytics.py
│   ├── preflight.py
│   ├── requirements.txt
│   ├── README.md
│   └── tests/
├── src/piece_of_crop/     # use your team's newest main branch code
├── results/               # current split.csv, band_stats.csv, saved prediction CSVs
├── models/                # local only: checkpoints, NOT committed
└── data/EuroSAT_MS/       # local only: original multispectral TIFFs
```

## Installation (Mac / VS Code terminal)

Use the existing conda environment documented in the main repository. From the project root:

```bash
conda env create -f environment.yml  # omit if environment already exists
conda activate piece-of-crop
python -m pip install -e .
python -m pip install -r prototype/requirements.txt
```

If conda says the environment already exists, simply `conda activate piece-of-crop`. Confirm `python` belongs to that environment with `which python` (your previous Mac setup had a broken unrelated pyenv executable).

Run a quick read-only resource check:

```bash
python prototype/preflight.py
```

Launch:

```bash
python -m streamlit run prototype/app.py
```

Open the local address printed by Streamlit, normally `http://localhost:8501`.

## Live inference and saved result fallback

- **Live inference** needs `results/band_stats.csv`, a full multispectral GeoTIFF and the corresponding trusted checkpoint in `models/`. Files use these names for seed 0, 1 or 2:
  - `rf_rgb_seed0.joblib`, `rf_13band_seed0.joblib`
  - `cnn_rgb_seed0.pt`, `cnn_13band_seed0.pt`
  - `resnet_rgb_seed0.pt`, `resnet_13band_seed0.pt`
- RGB uses the B04/B03/B02 channels from the SAME original 13-band image, not a separate RGB version of EuroSAT. The preview stretches bands to make them visible; the model receives training-normalized raw values.
- **RF** uses the project's feature extraction (6 RGB vs 28 multispectral features, including NDVI). **CNN** uses 64×64 input. **ResNet18** uses training-equivalent antialiased resizing to 224×224.
- **Test patch from EuroSAT** requires raw files in `data/EuroSAT_MS/<class_name>/<file>.tif` to show the image and run live inference. If raw data or weights are absent, the app can *still* show a **saved prediction** for a selected held-out test patch, but only if the saved file **matches the current test split exactly**. The UI labels these as `NOT live inference`. No saved-prediction fallback is offered for a new uploaded image.
- **Upload 13-band GeoTIFF** takes a 13-channel, 64×64 GeoTIFF with the EuroSAT band ordering. A normal RGB JPG or PNG is not accepted, because the 10 missing channels cannot be reconstructed.

**No retraining or download of pretrained weights is performed by the app.** The team’s weights must be from the same experiment as the supplied training statistics. `torch.load(..., weights_only=True)` is used where available; `joblib.load` can run arbitrary Python code, so only load **trusted, team-generated** checkpoints in the local `models/` folder.

## How to verify that it works on your Mac

1. Run `python -m pytest -q prototype/tests`. These are **synthetic compatibility tests**, not proof of the real model's accuracy.
2. Run `python prototype/preflight.py` and check which checkpoints exist and how many prediction CSVs match the current split (ideally 18/18).
3. In the Streamlit app open **Satellite Classification**, select `AnnualCrop` and a real test patch, choose **ResNet18, seed 0**, and press **Compare RGB vs 13 bands**. If both weights exist, both cards must state **Live inference**, display a class and a confidence in 0–100%, and probabilities that sum to ~100%.
4. Repeat with **Random Forest** and **Small CNN**. Verify that each input changes to 3 or 13 bands internally without changing the original sample.
5. Open **Model Comparison**: confirm number of verified experiment files and that the displayed test count equals the current `split.csv` (not a previous split from another experiment).
6. Open **Insurance Impact**: verify severity matrix (diagonal zeros) and the rankings of real errors. Never label points as money.
7. As a final smoke test, try `Upload 13-band GeoTIFF` with a known test TIFF and confirm predictions match the corresponding live-prediction result from the test selector.

**If weights are missing**: charts and verified saved predictions can still be presented, but you must not claim the application can run live inference. Add a limited set of trusted seed-0 checkpoints outside regular Git (e.g. a group-shared archive or a release asset) and document how the instructor can obtain them if live demo on another machine is required. Do **not** commit the 1GB `models/` directory or 7GB dataset to Git.

## Deliverable / GitHub

- Commit the source-only **`prototype/`** directory to the team's GitHub repository on its own branch and create a pull request. Do not overwrite the group's source modules or notebooks.
- Add a short `Prototype` section and launch command to the project's top-level README via your group's normal PR process.
- In the report include: objective, usage steps, a screenshot of the pairwise prediction, one model-comparison figure and a short limitations paragraph. In the presentation show a screenshot or a short live demo. The report/PPTX **are not substitutes** for the working app.
- Streamlit Community Cloud hosting is **optional**, not required. Without local weights or an external weight source the hosted app can only show verified saved results, not execute full model inference.

## Limitations

EuroSAT covers 10 land-cover classes; it is not a real insurance claims dataset. The severity matrix is an illustrative stakeholder scenario and cannot substitute for actuarial evidence. Spatial holdouts and seeds improve scientific rigor but do not guarantee performance on unseen geographic regions. Displayed confidence is a softmax or forest probability, **not necessarily calibrated**. The prototype does not identify crops at parcel level, assess a person's eligibility or estimate premium prices.
