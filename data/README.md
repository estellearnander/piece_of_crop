# Data

We use the multispectral version of EuroSAT (Helber et al., 2019): 27,000 Sentinel-2 patches,
64 x 64 pixels, 13 bands, 10 land-cover classes. Licence: MIT, plus the Copernicus Sentinel data terms.

Source: https://github.com/phelber/EuroSAT (the official page points to Zenodo for downloads)

The data is not stored in this repository (about 2 GB).

## How to download

1. Download EuroSAT_MS.zip (2.1 GB) from https://zenodo.org/records/7711810
   Do not download EuroSAT_RGB.zip: we build the RGB input from bands 4, 3 and 2 of the
   13-band files, so RGB and 13 bands come from exactly the same images.
2. Put the zip in this data/ folder and check that it downloaded completely:
   md5 EuroSAT_MS.zip
   The result must be 091174add3c8e680a49244acf185b9f0 (the checksum listed on Zenodo).
3. Unzip it here:
   unzip -q EuroSAT_MS.zip

You should now have data/EuroSAT_MS/ with 10 class folders and 27,000 .tif files:
AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop,
Residential, River, SeaLake. Each file has 13 bands of 64 x 64 pixels (uint16) and is
geo-referenced.
