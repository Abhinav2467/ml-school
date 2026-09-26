# Makefile for Business Entity Resolution Pipeline
# =================================================

# Automatically detect Python interpreter (prefers python3.12 if present)
PYTHON ?= $(shell command -v python3.12 2>/dev/null || command -v python3 2>/dev/null || echo python3)
SHELL  := /usr/bin/env bash

# Directory paths
ROOT_DIR     := $(shell pwd)
CODE_DIR     := $(ROOT_DIR)/code/business_entity_resolution
DATA_DIR     := $(ROOT_DIR)/dataset
TRAIN_DIR    := $(DATA_DIR)/train
TEST_DIR     := $(DATA_DIR)/test
OUTPUT_DIR   := $(ROOT_DIR)/output
ARTIFACTS_DIR:= $(CODE_DIR)/artifacts

# Output files
MATCHING_OUT  := $(OUTPUT_DIR)/matching_results.tsv
CANDIDATE_OUT := $(OUTPUT_DIR)/candidate_pairs.tsv
GOLD_TRAIN    := $(TRAIN_DIR)/train_ground_truth.tsv
MODEL_BUNDLE  := $(ARTIFACTS_DIR)/matcher.joblib

.PHONY: all help install dummy train infer validate eval smoke clean

# Default target
all: smoke

## help: Show available make targets with description
help:
	@echo "Business Entity Resolution Pipeline - Makefile"
	@echo "==============================================="
	@echo "Python interpreter: $(PYTHON)"
	@echo "Usage: make [target] [PYTHON=python3.12] [TRAIN_DIR=path] ..."
	@echo ""
	@echo "Targets:"
	@grep -E '^## [a-zA-Z_-]+:.*?$$' $(MAKEFILE_LIST) | sed -e 's/## //' | awk 'BEGIN {FS = ":"}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

## install: Install Python dependencies
install:
	$(PYTHON) -m pip install -r $(CODE_DIR)/requirements.txt

## dummy: Generate synthetic dummy dataset (5 S1 sample records)
dummy:
	cd $(CODE_DIR) && $(PYTHON) src/make_dummy_data.py

## train: Train blocking + pairwise matcher and export model bundle
train:
	cd $(CODE_DIR) && $(PYTHON) src/train.py --train-dir $(TRAIN_DIR)

## infer: Run inference on test dataset and emit TSVs
infer:
	@mkdir -p $(OUTPUT_DIR)
	cd $(CODE_DIR) && $(PYTHON) src/infer.py --data-dir $(TEST_DIR) --prefix test --out-dir $(OUTPUT_DIR)

## validate: Validate submission TSV formatting and constraints
validate:
	cd $(CODE_DIR) && $(PYTHON) utils/validate_submission.py \
		--matching $(MATCHING_OUT) \
		--candidate $(CANDIDATE_OUT) \
		--test-dir $(TEST_DIR)

## eval: Evaluate generated predictions against ground truth (macro F0.5)
eval:
	cd $(CODE_DIR) && $(PYTHON) src/eval_split.py \
		--matching $(MATCHING_OUT) \
		--candidate $(CANDIDATE_OUT) \
		--gold $(GOLD_TRAIN)

## smoke: Run end-to-end smoke test pipeline (dummy -> train -> infer -> validate -> eval)
smoke: dummy train infer validate eval
	@echo ""
	@echo "=== Pipeline Completed Successfully ==="
	@echo "--- matching_results.tsv preview ---"
	@head -n 10 $(MATCHING_OUT)
	@echo ""
	@echo "--- candidate_pairs.tsv preview ---"
	@head -n 10 $(CANDIDATE_OUT)

## clean: Remove generated outputs, artifacts, and Python bytecode caches
clean:
	rm -rf $(OUTPUT_DIR)/* $(ARTIFACTS_DIR)/*
	find $(ROOT_DIR) -type d -name "__pycache__" -exec rm -rf {} +
	find $(ROOT_DIR) -type f -name "*.pyc" -delete
	@echo "Cleaned generated files and caches."
