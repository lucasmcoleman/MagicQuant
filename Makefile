.PHONY: install test lint format docker-build docker-up clean

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check --select F magicquant/ tools/ tests/

format:
	$(PYTHON) -m ruff format magicquant/ tests/

docker-build:
	docker build -f docker/Dockerfile -t magicquant:latest .

docker-up:
	docker run --rm -it magicquant:latest

clean:
	rm -rf build/ dist/ *.egg-info/ __pycache__/ .pytest_cache/ .ruff_cache/
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
