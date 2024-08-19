#!/bin/bash

# List of models to train and infer
MODELS=("AttentionUnet" "FRNet" "Unet")

# List of datasets
DATASETS=("OCTA500_6M" "OCTA500_3M" "ROSSA")

# Training parameters
TRAIN_BATCH_SIZE=16
VALID_BATCH_SIZE=8
TRAIN_NUM_STEPS=10000 # 10000 steps is equivalent to more than 100 epochs
VALID_EVERY=100
SAVE_EVERY=500

# Inference parameters
INFER_BATCH_SIZE=8

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        echo "Training $MODEL on $DATASET"
        
        # Run training
        accelerate launch script/supervised_model/train.py \
            --dset $DATASET \
            --network $MODEL \
            --loss MonaiDiceCELoss \
            --train-batch-size $TRAIN_BATCH_SIZE \
            --valid-batch-size $VALID_BATCH_SIZE \
            --train-num-steps $TRAIN_NUM_STEPS \
            --valid-every $VALID_EVERY \
            --save-every $SAVE_EVERY
        
        # Get the result folder name
        RESULT_FOLDER=$(ls -td results/${MODEL}_MonaiDiceCELoss_${DATASET}_* | head -1)
        
        echo "Inferring $MODEL on $DATASET"
        
        # Run inference
        accelerate launch script/supervised_model/infer.py \
            $RESULT_FOLDER \
            --batch-size $INFER_BATCH_SIZE
        
        echo "Completed $MODEL on $DATASET"
        echo "-----------------------------"
    done
done

echo "All experiments completed!"
