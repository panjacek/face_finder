IMAGES ?= $(PWD)
SERVICE_NAME ?= face_finder

ifeq ($(USE_GPU), 1)
	SERVICE_NAME := $(SERVICE_NAME)_gpu
endif

.PHONY: help setup test test-gpu lint format run-cli up down shell run clean build

help: ## Show available targets
	@awk -F ':.*?## ' '/^#########/ {printf "\n%s\n", $$0} /^[a-zA-Z_-]+:.*?## / {printf "%-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

######### SETUP #########

setup: ## Sync uv env (dev + web groups) and fetch the face-detection model
	uv sync --group dev --group web
	uv run python scripts/fetch_model.py

######### TEST #########

test: ## All uv-runnable tests (unit, smoke, web) with coverage; gpu tests excluded
	uv run --group dev --group web pytest tests -m "not gpu" --cov=face_finder --cov-report=term-missing --cov-report=xml

test-gpu: ## Container: DRI smoke tests + cpu/gpu delegate benchmarks (pytest-benchmark)
	docker compose run --rm \
		-e UV_PROJECT_ENVIRONMENT=/tmp/test-venv \
		-e UV_CACHE_DIR=/tmp/uv-cache \
		face_finder_gpu \
		uv run --group dev --frozen pytest tests/test_gpu.py -m gpu -v

######### LINT #########

lint: ## Ruff, mypy and bandit over the codebase
	uv run --group dev ruff check .
	uv run --group dev ruff format --check .
	uv run --group dev --group web mypy src tests
	uv run --group dev bandit -q -r src

format: ## Auto-format everything
	uv run --group dev ruff format .
	uv run --group dev ruff check --fix .

######### RUN #########

run-cli: ## Run CLI locally: CALC_ARGS="path/to/images -l DEBUG"
	uv run face_finder $(CALC_ARGS)

run-web: ## Launch streamlit web UI on :8501
	uv run --group web streamlit run src/web_app/app.py

######### DOCKER #########

build: ## Build docker image
	docker compose build face_finder

shell: ## Bash inside container (USE_GPU=1 for gpu variant)
	docker compose run --rm $(SERVICE_NAME) /bin/bash

run: ## Run container CLI: IMAGES=/path/to/images
	docker compose run --rm -e IMAGES=$(IMAGES) $(SERVICE_NAME)

up: ## Launch web UI via docker on :8080
	docker compose up face_finder_web

down: ## Stop web UI containers
	docker compose down --remove-orphans --volumes face_finder_web

######### CLEAN #########

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist coverage.xml
	rm -rf src/face_finder/__pycache__ src/web_app/__pycache__ tests/__pycache__ .benchmarks
