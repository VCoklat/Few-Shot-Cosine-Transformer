#!/bin/bash
# Processing script for HAM10000 skin cancer dataset
# This script generates base.json, val.json, and novel.json files
# for use with the Few-Shot Cosine Transformer framework

echo "========================================"
echo "HAM10000 Dataset Processing"
echo "========================================"
echo ""
echo "Prerequisites:"
echo "  1. Download HAM10000 dataset from Kaggle:"
echo "     https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000"
echo "  2. Prepare your image list CSV file (e.g., final_1000_image_list.csv)"
echo "  3. Update paths in write_HAM10000_filelist.py if needed"
echo ""
echo "Running dataset processing..."
echo ""

python write_HAM10000_filelist.py

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ Processing completed successfully!"
    echo "========================================"
    echo ""
    echo "Generated files:"
    ls -lh base.json val.json novel.json 2>/dev/null || echo "  (Sample files created - replace with actual data)"
    echo ""
    echo "Next steps:"
    echo "  1. Verify the JSON files contain correct paths"
    echo "  2. Update configs.py to include HAM10000 dataset"
    echo "  3. Run training: python train.py --dataset HAM10000 --method FSCT_cosine"
else
    echo ""
    echo "❌ Processing failed. Please check the error messages above."
    exit 1
fi
