#!/bin/bash

# Simple script to run inference with the specified model checkpoint.

MODEL_PATH="./models/epoch_34.pth"
INPUT_DIR="test/"
OUTPUT_PATH="./output/results.json"
DEVICE="cuda"

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    exit 1
fi

# Run inference
echo "Running inference with model: $MODEL_PATH"
echo "Input directory: $INPUT_DIR"
echo "Output path: $OUTPUT_PATH"
echo "Command: python process.py --checkpoint $MODEL_PATH --input_dir $INPUT_DIR --output_path $OUTPUT_PATH --device $DEVICE"
echo "------------------------------------------------------------"

# Execute the inference command
python process.py \
    --checkpoint "$MODEL_PATH" \
    --input_dir "$INPUT_DIR" \
    --output_path "$OUTPUT_PATH" \
    --device "$DEVICE"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "Inference completed successfully!"
    echo "Results saved in: $OUTPUT_PATH"
else
    echo "Error during inference"
    exit 1
fi