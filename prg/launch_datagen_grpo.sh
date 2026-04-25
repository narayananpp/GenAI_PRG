#!/bin/bash

MODEL_PATH="./save/humanml_trans_enc_512/model000475000.pt"
SAVE_DIR="/data3/npalghat/reward_dataset"
PYTHONPATH="/home/npalghat/projects/GenAI/motion-diffusion-model"
SCRIPT="prg/datagen_grpo_style.py"
NUM_SAMPLES=5000

COMMON_ARGS="--model_path $MODEL_PATH \
             --save_dir $SAVE_DIR \
             --dataset humanml \
             --diffusion_steps 50 \
             --num_frames 196 \
             --num_samples $NUM_SAMPLES"

for run_idx in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$run_idx \
    PYTHONPATH=$PYTHONPATH \
    python $SCRIPT \
        $COMMON_ARGS \
        --run_idx $run_idx \
        --device 0 \
        > /tmp/run${run_idx}.log 2>&1 &
    echo "Launched run $run_idx on GPU $run_idx (pid $!)"
done

wait
echo "All 4 runs complete."