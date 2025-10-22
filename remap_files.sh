#!/bin/bash

# Directory containing the NetCDF files
INPUT_DIR="/p/project1/cslts/miaari1/python_scripts/global_TL/spatio-temporal-LSTM/inputs/EU_74_-48_69_20/raw/ERA5-Land"
GRID_FILE="/p/project1/cslts/miaari1/python_scripts/global_TL/spatio-temporal-LSTM/inputs/EU_74_-48_69_20/raw/EUR11_TSMP_grid.txt"

# Loop over all .nc files in the directory
for file in "$INPUT_DIR"/*.nc; do
    filename=$(basename "$file")
    
    # Check if the filename contains t2m, d2m, or swvl3
    if [[ "$filename" == *"t2m"* || "$filename" == *"d2m"* || "$filename" == *"swvl3"* ]]; then
        echo "Processing (bilinear): $filename"
        cdo remapbil,"$GRID_FILE" "$file" "${INPUT_DIR}/regridded_to_EUR11/BonA_bilinear_${filename}"

    # Check if the filename contains tp
    elif [[ "$filename" == *"tp"* ]]; then
        echo "Processing (conservative): $filename"
        cdo remapcon,"$GRID_FILE" "$file" "${INPUT_DIR}/regridded_to_EUR11/BonA_conservative_${filename}"

    # If none match, skip the file
    else
        echo "Skipping file (no match): $filename"
    fi
done

echo "✅ All processing completed."
