#!/bin/bash

# Directory containing the NetCDF files
INPUT_DIR="/p/project1/cslts/miaari1/python_scripts/global_TL/spatio-temporal-LSTM/inputs/EU_74_-48_69_20/raw/ERA5-Land"
OUTPUT_DIR="${INPUT_DIR}/daily_23h"
mkdir -p "$OUTPUT_DIR"

# Loop over all NetCDF files in the input directory
for file in "$INPUT_DIR"/*.nc; do
    filename=$(basename "$file")
    output_file="${OUTPUT_DIR}/at23h_${filename}"
    
    echo "Processing: $filename"

    # Extract only 23:00 timesteps (hour 23)
    cdo selhour,23 "$file" "$output_file"
done

echo "✅ Extraction completed. Files saved in: $OUTPUT_DIR"
