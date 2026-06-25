# Cancer Mutation Pathogenicity Predictor — Makefile
#
# Usage:
#   make setup       — create venv, install requirements
#   make download    — download ClinVar + cBioPortal data
#   make train       — train the model with default config
#   make evaluate    — evaluate a trained model
#   make ablation    — run the ablation study
#   make figures     — generate publication-quality figures
#   make test        — run the test suite
#   make lint        — run ruff + mypy
#   make all         — download → train → evaluate → ablation → figures

.PHONY: setup download train evaluate ablation figures test lint format \
        all clean docker-build docker-train help

PYTHON     ?= python
VENV       := .venv
PIP        := $(VENV)/Scripts/pip
PYTHON_BIN := $(VENV)/Scripts/python
CONFIG     := configs/default.yaml
CHECKPOINT := results/checkpoints/best_model.ckpt
DOCKER_IMG := cancer-pathogenicity

# Detect OS for venv activation path
ifeq ($(OS),Windows_NT)
    ACTIVATE := $(VENV)\Scripts\activate
    PIP      := $(VENV)\Scripts\pip
    PYTHON_BIN := $(VENV)\Scripts\python
else
    ACTIVATE := $(VENV)/bin/activate
    PIP      := $(VENV)/bin/pip
    PYTHON_BIN := $(VENV)/bin/python
endif

# Default target
.DEFAULT_GOAL := help

## help: Show this help message
help:
	@echo.
	@echo Cancer Mutation Pathogenicity Predictor
	@echo ========================================
	@echo.
	@echo Available targets:
	@echo   make setup          Create venv and install requirements
	@echo   make download       Download ClinVar + cBioPortal data
	@echo   make train          Train the model with default config
	@echo   make evaluate       Evaluate a trained model
	@echo   make ablation       Run the ablation study
	@echo   make figures        Generate publication-quality figures
	@echo   make test           Run the test suite (pytest)
	@echo   make lint           Run ruff + mypy checks
	@echo   make format         Auto-format code with ruff
	@echo   make all            Full pipeline: download, train, evaluate, ablation, figures
	@echo   make docker-build   Build Docker image
	@echo   make docker-train   Train inside Docker container (GPU)
	@echo   make clean          Remove build artifacts and caches
	@echo.

## setup: Create venv and install all requirements
setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install matplotlib-venn
	$(PIP) install -e .
	@echo.
	@echo Setup complete. Activate the venv:
	@echo   Windows:  $(VENV)\Scripts\Activate.ps1
	@echo   Linux:    source $(VENV)/bin/activate

## download: Download ClinVar and cBioPortal data
download:
	$(PYTHON_BIN) scripts/download_data.py --config $(CONFIG)

## train: Train the model with the default config
train:
	$(PYTHON_BIN) scripts/train.py --config $(CONFIG)

## evaluate: Evaluate the trained model on the test set
evaluate:
	$(PYTHON_BIN) scripts/evaluate.py --checkpoint $(CHECKPOINT) --config $(CONFIG)

## ablation: Run the full ablation study
ablation:
	$(PYTHON_BIN) scripts/run_ablation.py --base-config $(CONFIG)

## figures: Generate all publication-quality figures
figures:
	$(PYTHON_BIN) scripts/generate_figures.py

## test: Run the full test suite
test:
	$(PYTHON_BIN) -m pytest tests/ -v --tb=short

## lint: Run ruff linter and mypy type checker
lint:
	$(PYTHON_BIN) -m ruff check src/ scripts/
	$(PYTHON_BIN) -m mypy src/ --ignore-missing-imports

## format: Auto-format code with ruff
format:
	$(PYTHON_BIN) -m ruff format src/ scripts/

## all: Full pipeline — download → train → evaluate → ablation → figures
all: download train evaluate ablation figures
	@echo.
	@echo Full pipeline complete. Results are in results/

## docker-build: Build the Docker image
docker-build:
	docker build -t $(DOCKER_IMG) .

## docker-train: Train the model inside a Docker container (GPU required)
docker-train: docker-build
	docker run --gpus all \
		-v $(CURDIR)/data:/app/data \
		-v $(CURDIR)/results:/app/results \
		-v $(CURDIR)/configs:/app/configs \
		$(DOCKER_IMG) scripts/train.py --config configs/default.yaml

## clean: Remove build artifacts, caches, and temporary files
clean:
	@echo Cleaning build artifacts...
	-rmdir /s /q __pycache__ 2>nul
	-rmdir /s /q .pytest_cache 2>nul
	-rmdir /s /q .mypy_cache 2>nul
	-rmdir /s /q .ruff_cache 2>nul
	-rmdir /s /q build 2>nul
	-rmdir /s /q dist 2>nul
	-rmdir /s /q *.egg-info 2>nul
	-del /s /q *.pyc 2>nul
	@echo Done.
