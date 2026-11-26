# Piece of Crop

**Research question.** Does 13-band Sentinel-2 imagery reduce the land-cover classification errors that are most costly to crop insurers, compared with RGB imagery?

We compare a random forest, a small CNN and a pretrained ResNet18, each trained on RGB and on all 13 bands of the same EuroSAT patches, and weigh their mistakes with an insurer cost matrix. The sub-questions are:

- **SQ1.** Does the 13-band advantage hold for all three models?
- **SQ2.** Which costly confusions do the models make?
- **SQ3.** Do the 13 bands reduce those costly confusions?
- **SQ4.** Does the 13-band advantage hold in hazy images?

**Team:** Estelle Arnander, Manvi Gupta, Rishi Gupta, Loraine Nieto, Faris Selimovic, Aoi Ueno.
Course: AI, Machine Learning & Analytics, IE University, fall 2026 (Prof. P. Delgado-Santos).

## 1. Setup

Run these from the project root (the folder that contains `pyproject.toml`):

```bash
conda env create -f environment.yml
conda activate piece-of-crop
python -m pip install -e .
python -c "import piece_of_crop; print('Package works')"
pytest -q
```

If the environment already exists, only run `conda activate piece-of-crop`. `pytest -q` runs the tests in `tests/`, which check the data split, training helpers, cost matrix and haze functions.

## 2. Data

We use the multispectral version of EuroSAT (Helber et al., 2019): 27,000 Sentinel-2 patches of 64 x 64 pixels with 13 bands, in 10 land-cover classes.

- **Source:** https://github.com/phelber/EuroSAT (downloads on Zenodo, https://zenodo.org/records/7711810).
- **Licence and permitted use:** MIT licence, plus the Copernicus Sentinel data terms. The data is public, contains no personal information and may be used for research and teaching.
- The data is not stored in this repository (about 2 GB).

**How to download.**

1. Download `EuroSAT_MS.zip` (2.1 GB) from https://zenodo.org/records/7711810 into the `data/` folder (create it with `mkdir data` if it does not exist). Do not download `EuroSAT_RGB.zip`: we build the RGB input from bands 4, 3 and 2 of the 13-band files, so RGB and 13 bands come from exactly the same images.
2. Check that the download is complete: `md5 EuroSAT_MS.zip` must give `091174add3c8e680a49244acf185b9f0` (the checksum listed on Zenodo).
3. Unzip it in `data/`: `unzip -q EuroSAT_MS.zip`.

You should now have `data/EuroSAT_MS/` with 10 class folders (AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake) and 27,000 `.tif` files of 13 x 64 x 64 values (uint16).

The first notebook that needs the patches builds `data/eurosat_cache.npy` (about 2.9 GB, not in Git) from the GeoTIFFs. This happens once; later notebooks only load it.

## 3. How to reproduce the results

1. Follow the setup above and download the data.
2. Open the project in Jupyter or PyCharm and select the `piece-of-crop` environment as the notebook kernel.
3. Run the notebooks in order, each with a restarted kernel (Restart and Run All):

| Notebook | What it does |
|---|---|
| 01_data_exploration | Checks the files, class balance, band values and patch locations |
| 02_split_and_preprocessing | Spatial train/validation/test split (0.5° blocks with a buffer) and training-only band statistics |
| 03_baseline_rf | Random forest on band means, standard deviations and NDVI, RGB and 13 bands |
| 04_cnn | Small CNN trained from scratch, RGB and 13 bands |
| 05_transfer_resnet | ImageNet-pretrained ResNet18, fully fine-tuned, RGB and 13 bands |
| 06_evaluation | Comparison of all models, costly confusions, cost matrix, block-bootstrap intervals and human review |
| 07_haze | Robustness to simulated haze, and which classes hazy patches are sent to |

Every model is trained with three seeds (0, 1 and 2) and results are reported as the mean.

`results/split.csv` and `results/band_stats.csv` are fixed files. Do not delete or regenerate them: every model is trained and tested on them, and a different scikit-learn version can produce a different split.

Model weights (`models/`), the raw data and the cache are not in Git because of their size. The training notebooks (03 to 05) recreate the weights; notebook 05 downloads the ImageNet weights for ResNet18 from torchvision the first time it runs. All notebooks in the repository are saved with their outputs, so the results can be checked without running them.

## 4. Project structure

```
piece_of_crop/
├── README.md
├── environment.yml          conda environment
├── pyproject.toml           package definition (pip install -e .)
├── data/                    EuroSAT_MS (local only, see section 2)
├── notebooks/               01 to 07, run in order
├── src/piece_of_crop/
│   ├── data.py              file checks, spatial split, normalisation, cache, Dataset and DataLoaders
│   ├── features.py          random forest features
│   ├── cnn.py               the small CNN
│   ├── resnet.py            ResNet18 adapted to 3 or 13 bands
│   ├── train.py             training loop, evaluation and prediction files
│   ├── evaluate.py          cost matrix, loading predictions, block bootstrap, human review
│   └── haze.py              simulated haze and predictions on hazy patches
├── results/
│   ├── predictions/         test predictions, one CSV per model, input and seed (<model>_<input>_seed<n>.csv)
│   ├── figures/             saved figures
│   └── *.csv                split, band statistics and result tables
├── prototype/               Streamlit app (section 5)
├── tests/                   pytest tests for the package
└── models/                  trained weights (local only)
```

## 5. Prototype

`prototype/` contains a local Streamlit app that shows the project to a user. It has four tabs:

1. **Satellite Classification.** Choose a test patch (or upload a 13-band 64 x 64 EuroSAT GeoTIFF), a model and a seed, and compare the RGB and 13-band predictions on the same patch, with the class probabilities and the cost of the mistake. Simulated haze can be added (the same haze as notebook 07), and SeaLake can be forbidden as an answer to see whether a hazy patch still points to the right class.
2. **Model Comparison.** Accuracy, macro-F1 and mean insurer cost of all models, from the saved test predictions.
3. **Insurance Impact.** The cost matrix and the costliest confusions of each model.
4. **Haze Robustness.** The notebook 07 results: accuracy and critical-error rate against haze strength, RGB against 13 bands.

**Install and run** (from the project root, after the setup in section 1):

```bash
conda activate piece-of-crop
python -m pip install -r prototype/requirements.txt
python prototype/preflight.py
python -m streamlit run prototype/app.py
```

`preflight.py` checks, without changing anything, that the split, band statistics, data, saved predictions and model weights are in place. Streamlit then opens the app in the browser, normally at http://localhost:8501. Stop it with Ctrl+C in the terminal. The prototype tests run with `python -m pytest -q prototype/tests`.

**What needs what.** Live predictions need the model weights in `models/` and the EuroSAT GeoTIFFs in `data/`, which are not in Git; run notebooks 03 to 05 to create the weights. Without them, the Model Comparison, Insurance Impact and Haze Robustness tabs still work from the saved results, and the Satellite Classification tab shows the saved test prediction for a chosen patch, labelled as not live. Haze needs live predictions, because the saved predictions are of clear images.

The app runs locally and uploads nothing. It is an academic demonstration, not a validated insurance underwriting system.

## 6. Results

Test set of 3,854 patches in 134 spatial blocks, mean of three seeds. A critical error is a confusion with cost 8 or more in the insurer cost matrix.

| Model | Accuracy RGB | Accuracy 13 bands | Critical-error rate RGB | Critical-error rate 13 bands |
|---|---|---|---|---|
| Random forest | 0.699 | 0.857 | 16.4% | 9.7% |
| Small CNN | 0.918 | 0.954 | 5.0% | 3.5% |
| ResNet18 (pretrained) | 0.972 | 0.972 | 2.4% | 2.5% |

- **SQ1.** The 13 bands raise accuracy for the random forest (+15.8 points, 95% block-bootstrap interval +12.1 to +19.9) and the CNN (+3.6 points, +2.0 to +5.9). For the pretrained ResNet there is no difference (interval -0.7 to +0.6).
- **SQ2.** The costly mistakes are concentrated in a few pairs: PermanentCrop and HerbaceousVegetation confused with each other, PermanentCrop called AnnualCrop and HerbaceousVegetation called Forest.
- **SQ3.** The 13 bands reduce critical errors for the random forest and the CNN, but not for the ResNet, where fewer grassland-to-permanent-crop mistakes are offset by more grassland-to-forest mistakes. The conclusions hold under all five cost assumptions tested.
- **SQ4.** The 13-band advantage holds under light haze for the random forest and the CNN. The pretrained ResNet is the most robust model (85 to 88% accuracy at 20% haze). Under heavy haze the ResNets and the RGB models fall back on SeaLake, a cheap answer in the cost matrix that hides critical errors.
- Sending the 10% least confident patches to a human reviewer catches 71 to 73% of the ResNet's critical errors on clear images, but only 39 to 51% at 20% haze, so hazy images need to be screened before the model is used.
- **For an insurer**, the pretrained ResNet is the best model, and with it RGB is as good as 13 bands. The extra bands pay off for models that do not start from ImageNet features.

## 7. Limitations

- EuroSAT covers 10 broad land-cover classes in Europe only. It is not crop-level or parcel-level data and not an insurance claims dataset.
- The cost matrix is an assumed scenario in severity points, not real claim costs. We tested five cost assumptions, but they still need to be checked with an insurer.
- The spatial split keeps neighbouring patches apart, but the results do not prove that the models work in other regions or seasons.
- The haze is simulated, not real atmospheric haze or clouds.
- The model confidence is a softmax or forest probability and is not calibrated.
- The models are research prototypes and are not validated for real insurance decisions.

## 8. AI-use declaration

*[Placeholder: to be completed by the team, consistent with section 10 of the report.]*

## References

The full reference list is in the report, and each notebook ends with the references it uses. Main external sources:

- Helber, P., Bischke, B., Dengel, A. & Borth, D. (2019). EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 12(7), 2217–2226. https://doi.org/10.1109/JSTARS.2019.2918242
- He, K., Zhang, X., Ren, S. & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, 770–778. Pretrained weights: torchvision `ResNet18_Weights.IMAGENET1K_V1` (ImageNet, Deng et al., 2009).
