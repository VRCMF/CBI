#!/bin/bash
# Run in the training environment. Set HF_TOKEN externally for gated models.
if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Configuration ----
# MODEL_NAME="meta-llama/Llama-3.1-8B"
MODEL_NAME="Qwen/Qwen2.5-7B"
# MODEL_NAME="mistralai/Mistral-7B-v0.1"
# MODEL_NAME="microsoft/Phi-3.5-mini-instruct"
# MODEL_NAME="google/gemma-2b"
BATCH_SIZE=1                  # per-GPU batch size; keep small to avoid OOM
GRAD_ACCUM=1                  # effective batch = BATCH_SIZE * NGPU * GRAD_ACCUM
NUM_EPOCHS=5
LR=2e-5
NPROC_PER_NODE=2              # number of GPUs on this node

# DATASETS=("matres" "tbd" "tcr" "tdd_auto" "tdd_man")
# DATASETS=("matres" "tbd" "tcr")
DATASETS=("tdd_auto" "tdd_man")

for DATASET in "${DATASETS[@]}"; do
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_DIR="outputs/ft/${MODEL_NAME}/${DATASET}_${TIMESTAMP}"
    mkdir -p "$OUTPUT_DIR"

    echo "========================================"
    echo "Dataset: $DATASET"
    echo "Output:  $OUTPUT_DIR"
    echo "========================================"

    # Save config
    cat > "$OUTPUT_DIR/config.txt" << EOF
Timestamp:   $TIMESTAMP
MODEL_NAME:  $MODEL_NAME
DATASET:     $DATASET
BATCH_SIZE:  $BATCH_SIZE (per GPU)
GRAD_ACCUM:  $GRAD_ACCUM
NUM_EPOCHS:  $NUM_EPOCHS
LR:          $LR
NPROC:       $NPROC_PER_NODE
EOF

    torchrun \
        --standalone \
        --nproc_per_node=$NPROC_PER_NODE \
        main_ft.py \
        --model_name "$MODEL_NAME" \
        --dataset "$DATASET" \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --num_epochs $NUM_EPOCHS \
        --learning_rate $LR \
        --output_dir "$OUTPUT_DIR"

    echo "Dataset $DATASET done. Results in: $OUTPUT_DIR"
    echo "----------------------------------------"
done

echo "All datasets completed!"
