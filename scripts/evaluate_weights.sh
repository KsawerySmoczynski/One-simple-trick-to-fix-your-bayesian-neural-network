# !/bin/bash
# TODO add parameters verification
NET_PATH=$1
NET_CONFIG_PATH=$2
SAVE_DIR="test"

if [ -z ${NET_PATH} ] || [ -z ${NET_PATH} ]; then
    echo "Not properly set arguments"
    exit 1
fi

python scripts/evaluate_weights_2d.py ${NET_PATH} ${NET_CONFIG_PATH} 12 --n-weights 2 --window 2 --rate 5 --n-random-layers 2 \
    --save-dir ${SAVE_DIR} --seed 48 --batch-size 6000 --in-memory
