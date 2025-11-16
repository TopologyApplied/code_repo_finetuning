# Makefile
.PHONY: all config analyze generate train merge evaluate clean

all: config analyze generate train merge evaluate

config:
	@echo "Step 0: Updating configuration..."
	python utils/config_manager.py

analyze:
	@echo "Step 1: Analyzing repository..."
	python scripts/01_analyze_repo.py

generate:
	@echo "Step 2: Generating training data..."
	python scripts/02_generate_data.py

train:
	@echo "Step 3: Fine-tuning model..."
	deepspeed --num_gpus=2 scripts/03_train_model.py

merge:
	@echo "Step 4: Merging LoRA weights..."
	python scripts/04_merge_weights.py

evaluate:
	@echo "Step 5: Evaluating model..."
	python scripts/05_evaluate.py

clean:
	@echo "Cleaning output files..."
	rm -rf output/finetuned_model/checkpoints/*
	rm -rf data/training_data/*

help:
	@echo "Available targets:"
	@echo "  make all       - Run complete pipeline"
	@echo "  make config    - Update repository config"
	@echo "  make analyze   - Analyze code repository"
	@echo "  make generate  - Generate training data"
	@echo "  make train     - Fine-tune model"
	@echo "  make merge     - Merge LoRA weights"
	@echo "  make evaluate  - Evaluate model"
	@echo "  make clean     - Clean output files"