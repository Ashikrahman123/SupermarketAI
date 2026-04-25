SHELL := /bin/zsh

VENV := .venv
PYTHON_BIN := $(shell command -v python3.11 2>/dev/null || command -v python3.10 2>/dev/null)
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help venv install run api train predict clean

help:
	@echo "Available targets:"
	@echo "  make venv     - Create local virtual environment"
	@echo "  make install  - Install Python dependencies"
	@echo "  make run      - Run Streamlit app"
	@echo "  make api      - Run FastAPI service"
	@echo "  make train    - Train and save model"
	@echo "  make predict  - Run image prediction script"
	@echo "  make clean    - Remove virtual environment and cache files"
	@echo ""
	@echo "Note: TensorFlow on macOS requires Python 3.11 or 3.10."

venv:
	@if [ -z "$(PYTHON_BIN)" ]; then \
		echo "Error: python3.11 or python3.10 is required for TensorFlow on macOS."; \
		echo "Install with: brew install python@3.11"; \
		exit 1; \
	fi
	@if [ -x "$(PYTHON)" ] && ! $(PYTHON) -c 'import sys; raise SystemExit(0 if sys.version_info[:2] in [(3, 11), (3, 10)] else 1)'; then \
		echo "Existing .venv uses unsupported Python version. Recreating with $(PYTHON_BIN)..."; \
		rm -rf $(VENV); \
	fi
	@if [ ! -x "$(PYTHON)" ]; then \
		$(PYTHON_BIN) -m venv $(VENV); \
	fi

install: venv requirements.txt
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run: install
	$(PYTHON) -m streamlit run app.py

api: install
	$(PYTHON) -m uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8000 --reload

train: install
	$(PYTHON) train.py

predict: install
	$(PYTHON) predict.py

clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +