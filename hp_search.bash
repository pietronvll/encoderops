#!/bin/bash

# Define hyperparameter options
MAX_GRAD_NORMS=("0.2" "none")
NORMALIZE_LINS=("true" "false")
REGULARIZATIONS=("0.0001" "0.001" "0.01")
MIN_ENCODER_LRS=("none" "0.0001")

# Count total runs
TOTAL_RUNS=$((${#MAX_GRAD_NORMS[@]} * ${#NORMALIZE_LINS[@]} * ${#REGULARIZATIONS[@]} * ${#MIN_ENCODER_LRS[@]}))
echo "Starting hyperparameter search with $TOTAL_RUNS total runs"

# Run counter
RUN_COUNT=0

# Loop over all combinations
for MAX_GRAD_NORM in "${MAX_GRAD_NORMS[@]}"; do
    for NORMALIZE_LIN in "${NORMALIZE_LINS[@]}"; do
        for REGULARIZATION in "${REGULARIZATIONS[@]}"; do
            for MIN_ENCODER_LR in "${MIN_ENCODER_LRS[@]}"; do
                RUN_COUNT=$((RUN_COUNT + 1))
                
                # Prepare arguments
                ARGS="chignolin-prod"
                
                # Add overrides based on hyperparameter values
                if [[ "$MAX_GRAD_NORM" == "none" ]]; then
                    ARGS="$ARGS --model_args.max_grad_norm=None"
                else
                    ARGS="$ARGS --model_args.max_grad_norm=$MAX_GRAD_NORM"
                fi
                
                # Handle boolean flag properly
                if [[ "$NORMALIZE_LIN" == "true" ]]; then
                    ARGS="$ARGS --model_args.normalize-lin"
                else
                    ARGS="$ARGS --model_args.no-normalize-lin"
                fi
                
                ARGS="$ARGS --model_args.regularization=$REGULARIZATION"
                
                if [[ "$MIN_ENCODER_LR" == "none" ]]; then
                    ARGS="$ARGS --model_args.min-encoder-lr=None"
                else
                    ARGS="$ARGS --model_args.min-encoder-lr=$MIN_ENCODER_LR"
                fi
                
                
                echo "[$RUN_COUNT/$TOTAL_RUNS] Running with: $ARGS"
                
                # Launch training job
                uv run --env-file=.env -- python -m trainers.single_task $ARGS
                
                # Optional: add a short delay between jobs if needed
            done
        done
    done
done

echo "Hyperparameter search completed. $TOTAL_RUNS runs executed."