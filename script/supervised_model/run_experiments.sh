#!/bin/bash

# List of models to train and infer
MODELS=("AttentionUnet" "FRNet" "Unet")

LOSS=("MonaiDiceCELoss" "MonaiDiceFocalLoss")
# List of datasets
DATASETS=("OCTA500_6M" "OCTA500_3M" "ROSSA")

# Training parameters
TRAIN_BATCH_SIZE=16
VALID_BATCH_SIZE=8
TRAIN_NUM_STEPS=10 # 10000 steps is equivalent to more than 100 epochs
VALID_EVERY=1
SAVE_EVERY=5

# Inference parameters
INFER_BATCH_SIZE=8

for MODEL in "${MODELS[@]}"; do
    for LOSS in "${LOSS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            echo "Training $MODEL on $DATASET"
            
            # Run training
            accelerate launch script/supervised_model/train.py \
                --dset $DATASET \
                --network $MODEL \
                --loss $LOSS \
                --train-batch-size $TRAIN_BATCH_SIZE \
                --valid-batch-size $VALID_BATCH_SIZE \
                --train-num-steps $TRAIN_NUM_STEPS \
                --valid-every $VALID_EVERY \
                --save-every $SAVE_EVERY

            # Check if the training command was successful
            if [ $? -ne 0 ]; then
                echo "Error occurred during training. Exiting..."
                exit 1
            fi
            
            # Get the result folder name
            RESULT_FOLDER="results/${MODEL}__${LOSS}__${DATASET}"

            echo "Processing: $RESULT_FOLDER"
            
            echo "Inferring $MODEL on $DATASET"
            
            # Run inference
            accelerate launch script/supervised_model/infer.py \
                $RESULT_FOLDER \
                --batch-size $INFER_BATCH_SIZE

            # Check if the inference command was successful
            if [ $? -ne 0 ]; then
                echo "Error occurred during inference. Exiting..."
                exit 1
            fi
            
            echo "Completed $MODEL on $DATASET"
            echo "-----------------------------"
        done
    done
done

echo "All experiments completed!"

