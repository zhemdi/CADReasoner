SRC_DATASET="/path/to/original_dataset"
DST_DATASET="/path/to/new_defected_dataset"
CONFIG_PATH="/path/to/config.json"

NUM_WORKERS=2

python3 -u run_generate.py \
  --src_dataset "$SRC_DATASET" \
  --dst_dataset "$DST_DATASET" \
  --config "$CONFIG_PATH" \
  --num_workers $NUM_WORKERS \
  --overwrite 