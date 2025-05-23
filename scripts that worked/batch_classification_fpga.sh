#!/bin/bash
# batch_classification_fpga.sh
# Processes all .jpg images in test_images on ZCU104 with proper labeling and logging

# Check if the model argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: ./batch_classification_fpga.sh <model_name>"
    exit 1
fi

# Define directories
INPUT_DIR="/home/root/Vitis-AI/examples/vai_library/samples/classification/test_images"
RESULTS_DIR="/home/root/Vitis-AI/examples/vai_library/samples/classification/test_results"
MODEL="$1"  # Model is now set via terminal input

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Clear previous results log
> "$RESULTS_DIR/classification_results.log"

# Counter for file numbering
count=0

# Process each .jpg image
for img in "$INPUT_DIR"/*.jpg; do
    count=$((count + 1))
    filename=$(basename "$img" .jpg)
    output_img="$RESULTS_DIR/${filename}_result.jpg"

    echo "Processing Image: $filename.jpg with model: $MODEL..."

    # Run the classification with the specified model
    ./test_jpeg_classification "$MODEL" "$img" -t 4 > "$RESULTS_DIR/${filename}_log.txt" 2>&1

    # Check if the result image was generated successfully
    if [ -f "$RESULTS_DIR/${filename}_result.jpg" ]; then
        echo "Image $count: $filename.jpg (Model: $MODEL)" >> "$RESULTS_DIR/classification_results.log"
        echo "Top-K Classification Results:" >> "$RESULTS_DIR/classification_results.log"
        grep "Class:" "$RESULTS_DIR/${filename}_log.txt" >> "$RESULTS_DIR/classification_results.log"
        echo "----------------------------------------" >> "$RESULTS_DIR/classification_results.log"
    else
        echo "Error: No result image generated for $filename.jpg" | tee -a "$RESULTS_DIR/classification_results.log"
    fi
done

echo "✅ Batch Classification Completed!"
echo "Results and images saved in $RESULTS_DIR"


