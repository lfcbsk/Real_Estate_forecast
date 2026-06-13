.PHONY: help install train api streamlit orchestrate-fast orchestrate-full \
        test test-cov lint docker-up docker-up-gpu docker-down clean

UV ?= uv

help:
	@echo "Targets:"
	@echo "  install          - tạo venv & cài dependencies (uv pip install -e .[dev])"
	@echo "  train            - chạy full training pipeline"
	@echo "  api              - chạy FastAPI (uvicorn, reload)"
	@echo "  streamlit        - chạy Streamlit dashboard"
	@echo "  orchestrate-fast - retrain nhanh (không Optuna), promote nếu pass gate"
	@echo "  orchestrate-full - retrain full (Optuna), promote nếu pass gate"
	@echo "  test             - chạy pytest (bỏ e2e)"
	@echo "  test-cov         - chạy pytest với coverage (fail-under=50)"
	@echo "  lint             - chạy ruff/format check (nếu có cấu hình)"
	@echo "  docker-up        - docker compose up --build (trong docker/)"
	@echo "  docker-up-gpu    - docker compose --profile gpu up --build"
	@echo "  docker-down      - docker compose down"
	@echo "  clean            - xoá cache, __pycache__, .pytest_cache"

install:
	$(UV) python install 3.10
	$(UV) venv --python 3.10
	$(UV) pip install -e ".[dev]"

train:
	$(UV) run python -m src.pipeline.training

api:
	$(UV) run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

streamlit:
	$(UV) run streamlit run src/app/streamlit_app.py

orchestrate-fast:
	$(UV) run python -m src.pipeline.orchestrator --tune false --promote true

orchestrate-full:
	$(UV) run python -m src.pipeline.orchestrator --tune true --promote true --n-trials 10

test:
	$(UV) run pytest tests/ -v -m "not e2e"

test-cov:
	$(UV) run pytest tests/ --cov=src --cov-fail-under=50

lint:
	$(UV) run flake8 src tests
	$(UV) run black --check src tests

docker-up:
	cd docker && docker compose up --build

docker-up-gpu:
	cd docker && docker compose --profile gpu up --build

docker-down:
	cd docker && docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov