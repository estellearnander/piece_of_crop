"""piece_of_crop: EuroSAT land-cover classification, RGB vs 13 bands."""

from .data import (
    BAND_NAMES,
    CLASS_NAMES,
    RGB_BAND_INDICES,
    check_file,
    find_duplicates,
    list_files,
    read_bands,
    read_center_lonlat,
    summarise_files,
    to_rgb_image,
)
