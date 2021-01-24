#!/bin/bash

DOMAIN="fashion"
ROOT="../data/simmc_${DOMAIN}/"

# Input files.
TRAIN_JSON_FILE="${ROOT}${DOMAIN}_train_dials.json"
DEV_JSON_FILE="${ROOT}${DOMAIN}_dev_dials.json"
DEVTEST_JSON_FILE="${ROOT}${DOMAIN}_devtest_dials.json"


METADATA_FILE="${ROOT}fashion_metadata.json"


# Output files.
VOCAB_FILE="${ROOT}${DOMAIN}_vocabulary.json"
METADATA_EMBEDS="${ROOT}${DOMAIN}_asset_embeds.npy"
ATTR_VOCAB_FILE="${ROOT}${DOMAIN}_attribute_vocabulary.json"

printf "[step1] Extract assistant API\n"
# Step 1: Extract assistant API.
INPUT_FILES="${TRAIN_JSON_FILE} ${DEV_JSON_FILE} ${DEVTEST_JSON_FILE}"
python ../tools/extract_actions_fashion.py \
    --json_path="${INPUT_FILES}" \
    --save_root="${ROOT}" \
    --metadata_path="${METADATA_FILE}"

read -p "Press any key to continue ..."

printf "\n[step2] Extract vocabulary from train\n"
# Step 2: Extract vocabulary from train.
python ../tools/extract_vocabulary.py \
    --train_json_path="${TRAIN_JSON_FILE}" \
    --vocab_save_path="${VOCAB_FILE}" \
    --threshold_count=5

read -p "Press any key to continue ..."

printf "\n[step3] Read and embed shopping assets\n"
# Step 3: Read and embed shopping assets.
python ../tools/embed_fashion_assets.py \
    --input_asset_file="${METADATA_FILE}" \
    --embed_path="${METADATA_EMBEDS}"

read -p "Press any key to continue ..."

printf "\n[step4] Convert all the splits into npy files for dataloader\n"
# Step 4: Convert all the splits into npy files for dataloader.
SPLIT_JSON_FILES=("${TRAIN_JSON_FILE}" "${DEV_JSON_FILE}" "${DEVTEST_JSON_FILE}")
for SPLIT_JSON_FILE in "${SPLIT_JSON_FILES[@]}" ; do
    python ../tools/build_multimodal_inputs.py \
        --json_path="${SPLIT_JSON_FILE}" \
        --vocab_file="${VOCAB_FILE}" \
        --save_path="$ROOT" \
        --action_json_path="${SPLIT_JSON_FILE/.json/_api_calls.json}" \
        --retrieval_candidate_file="${SPLIT_JSON_FILE/.json/_retrieval_candidates.json}" \
        --domain="${DOMAIN}"
done
read -p "Press any key to continue ..."

printf "\n[step5] Extract vocabulary for attributes from train npy file\n"
# Step 5: Extract vocabulary for attributes from train npy file.
python ../tools/extract_attribute_vocabulary.py \
    --train_npy_path="${TRAIN_JSON_FILE/.json/_mm_inputs.npy}" \
    --vocab_save_path="${ATTR_VOCAB_FILE}" \
    --domain="${DOMAIN}"

read -p "Press any key to continue ..."
