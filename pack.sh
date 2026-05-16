#!/bin/bash

# Exit automatically if a command fails
set -e

if [ "$1" == "--unpack" ]; then
    FILE="$2"
    if [ -z "$FILE" ]; then
        # If no file is provided, try to find the latest archive matching the pattern
        FILE=$(ls -t data_*.tar.* 2>/dev/null | head -n 1)
        if [ -z "$FILE" ]; then
            echo "Error: No archive file specified and no default data_*.tar.* found."
            echo "Usage: $0 --unpack [filename]"
            exit 1
        fi
        echo "No file specified. Automatically selected the latest archive: $FILE"
    fi

    if [ ! -f "$FILE" ]; then
        echo "Error: File '$FILE' does not exist."
        exit 1
    fi

    echo "Unpacking '$FILE'..."
    if [[ "$FILE" == *.tar.zst ]]; then
        if command -v zstd >/dev/null 2>&1; then
            tar -I "zstd -T0" -xf "$FILE"
        else
            echo "zstd not found, trying default tar extraction..."
            tar -xf "$FILE"
        fi
    elif [[ "$FILE" == *.tar.gz ]]; then
        if command -v pigz >/dev/null 2>&1; then
            tar -I pigz -xf "$FILE"
        else
            tar -xzf "$FILE"
        fi
    else
        tar -xf "$FILE"
    fi
    echo "Done! Unpacked '$FILE'."
    exit 0
fi

# Packing logic
# Generate filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="data_${TIMESTAMP}.tar.zst"
FALLBACK_FILE="data_${TIMESTAMP}.tar.gz"

echo "Packing and compressing the 'data' directory..."

# Try to use zstd for high efficiency and speed, fallback to pigz, then gzip
if command -v zstd >/dev/null 2>&1; then
    echo "zstd detected. Using zstd for highly efficient compression..."
    tar -I "zstd -T0" -cf "${OUTPUT_FILE}" data/
    echo "Done! Output: ${OUTPUT_FILE}"
elif command -v pigz >/dev/null 2>&1; then
    echo "pigz detected. Using multi-threaded gzip (pigz) for faster compression..."
    tar -I pigz -cf "${FALLBACK_FILE}" data/
    echo "Done! Output: ${FALLBACK_FILE}"
else
    echo "Using standard gzip..."
    tar -czf "${FALLBACK_FILE}" data/
    echo "Done! Output: ${FALLBACK_FILE}"
fi


# tar -I "zstd -T0" -cf notes.tar.zst notes/
# tar -I "zstd -T0" -cf processed.tar.zst data/processed/rqvae