.PHONY: install test lint format build docker-build docker-up clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

lint:
	python -m py_compile magicquant/__init__.py
	python -m py_compile magicquant/__main__.py
	python -m py_compile magicquant/orchestrator.py
	python -m py_compile magicquant/gguf/writer.py
	python -m py_compile magicquant/quant/converters.py

format:
	@echo "No formatter configured — add ruff or black to dev deps"

build:
	python -m build

docker-build:
	docker build -t magicquant:latest .

docker-up:
	docker run --rm -it magicquant:latest

clean:
	rm -rf build/ dist/ *.egg-info/ __pycache__/ .pytest_cache/
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
