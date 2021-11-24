# !/bin/bash
NET_PATH="/home/ksawery/Documents/Bayesian/Code/lightning_logs/version_1/checkpoints/epoch=29-step=11489.ckpt"
NET_CONFIG_PATH="/home/ksawery/Documents/Bayesian/Code/lightning_logs/version_1/config.yaml"
SAVE_DIR="test"

#Evaluate weights 1d and 2d
# python scripts/evaluate_weights_1d.py "$NET_PATH" "$NET_CONFIG_PATH" 16 --save-dir "$SAVE_DIR" & \
python scripts/evaluate_weights_2d.py "$NET_PATH" "$NET_CONFIG_PATH" 8 --save-dir "$SAVE_DIR" --seed 73
