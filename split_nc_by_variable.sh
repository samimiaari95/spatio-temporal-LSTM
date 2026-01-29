#!/bin/bash

# Directory containing NetCDF files
DIR="/p/project1/cslts/miaari1/python_scripts/global_TL/spatio-temporal-LSTM/inputs/EU_74_-48_69_20/raw/ERA5-Land"

# List of variables to extract
VARS=("t2m" "d2m" "swvl3")

if [[ -z "$DIR" ]]; then
    echo "Usage: $0 <directory_with_nc_files>"
    exit 1
fi

for file in "$DIR"/*.nc; do
    echo "Processing $file"
    [[ -f "$file" ]] || continue

    base=$(basename "$file" .nc)
    cdo daymean "$file" "${DIR}/${base}_daily_mean.nc"
    for var in "${VARS[@]}"; do
        echo "  Extracting variable: $var"
        newbase="${base/t2m_d2m_swvl3/$var}"
        out="${DIR}/${newbase}.nc"
        # ncks -v "$var" "${DIR}/${base}_daily_mean.nc" "$out"        
        cdo selname,"$var" "${DIR}/${base}_daily_mean.nc" "$out"
    done
done
