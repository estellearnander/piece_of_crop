"""piece_of_crop: EuroSAT land-cover classification, RGB vs 13 bands."""

from .data import (
    BAND_NAMES,
    CLASS_NAMES,
    RGB_BAND_INDICES,
    EuroSATDataset,
    apply_buffer,
    assign_blocks,
    augment,
    build_cache,
    check_file,
    compute_band_stats,
    find_duplicates,
    find_project_root,
    get_cache,
    list_files,
    load_cache,
    make_loaders,
    nearest_distance_km,
    normalise,
    read_bands,
    read_center_lonlat,
    read_split,
    select_bands,
    spatial_split,
    summarise_files,
    to_rgb_image,
)
from .features import RGB_INDICES, extract_rf_features, tune_random_forest
from .cnn import SmallCNN, build_cnn, count_parameters
from .resnet import build_resnet, resnet_param_groups
from .train import (
    check_test_order,
    evaluate_model,
    fit,
    get_device,
    predictions_table,
    resize_images,
    set_seed,
    train_epoch,
)
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
