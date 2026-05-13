#!/bin/bash
# Preprocess the Crossdocked dataset.
# Run from the project root: bash scripts/01_preprocess.sh
#
# Before running this script, manually download the following files
# from https://drive.google.com/drive/folders/1CzwxmTpjbrt83z_wBzcQncq84OVDPurM
# and place them in voxbind/dataset/data/:
#   - split_by_name.pt
#   - crossdocked_pocket10.tar.gz
# Then decompress: tar xvzf voxbind/dataset/data/crossdocked_pocket10.tar.gz -C voxbind/dataset/data/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/dataset/data"

# Verify required files exist
if [ ! -f "$DATA_DIR/split_by_name.pt" ]; then
    echo "ERROR: $DATA_DIR/split_by_name.pt not found."
    echo "Please download it from https://drive.google.com/drive/folders/1CzwxmTpjbrt83z_wBzcQncq84OVDPurM"
    exit 1
fi

if [ ! -d "$DATA_DIR/crossdocked_pocket10" ]; then
    echo "ERROR: $DATA_DIR/crossdocked_pocket10/ not found."
    echo "Please download crossdocked_pocket10.tar.gz and decompress it into $DATA_DIR/"
    exit 1
fi

echo "==> Running preprocessing (this may take a couple of hours)..."
cd "$PROJECT_ROOT/dataset"
conda run -n voxbind python preprocess_crossdocked.py --data_dir "$DATA_DIR"

echo "==> Done. Output files:"
echo "    $DATA_DIR/data_train.pt"
echo "    $DATA_DIR/data_test.pt"
