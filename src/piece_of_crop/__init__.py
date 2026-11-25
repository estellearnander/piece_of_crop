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
from .resnet import IMAGE_SIZE, build_resnet, resnet_param_groups
from .train import predict, predictions_table, seed_all, train_model, train_one_epoch
from .evaluate import (
    MODEL_FILES,
    cost_scenarios,
    critical_matrix,
    is_critical,
    load_predictions,
    make_cost_matrices,
    paired_block_bootstrap,
)
from .haze import (
    WAVELENGTH_FACTORS,
    HazyTestDataset,
    load_deep_model,
    predict_deep,
    predict_rf,
    simulate_haze,
)
