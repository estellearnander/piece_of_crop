"""piece_of_crop: EuroSAT land-cover classification, RGB vs 13 bands."""

from .data import (
    BAND_NAMES,
    CLASS_NAMES,
    RGB_BAND_INDICES,
    TRAIN_TRANSFORM,
    EuroSATDataset,
    apply_buffer,
    compute_band_stats,
    find_duplicates,
    get_cache,
    list_files,
    make_loaders,
    nearest_distance_km,
    normalise,
    read_bands,
    read_split,
    select_bands,
    spatial_split,
    summarise_files,
    to_rgb_image,
)
from .features import MS_FEATURES, RGB_FEATURES, extract_rf_features, tune_random_forest
from .cnn import SmallCNN
from .resnet import IMAGE_SIZE, build_resnet, first_layer_norms, resnet_param_groups
from .train import predict, predictions_table, seed_all, train_model, train_one_epoch
from .evaluate import (
    MODEL_FILES,
    block_bootstrap_ci,
    insurer_cost_matrix,
    listed_critical_confusions,
    load_predictions,
)
from .haze import (
    ANGSTROM_EXPONENT,
    WAVELENGTHS_NM,
    add_haze,
    haze_share,
    hazy_test_loader,
    rf_hazy_probabilities,
)
