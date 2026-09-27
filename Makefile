# Uses .venv/bin/python when a venv exists in the repo, else whatever `python` is on PATH.
PY := $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

.PHONY: install test data demo run docker-build docker clean

install:   ## editable install with all extras
	$(PY) -m pip install -e ".[api,experiments,dev]"

test:      ## 6 tests in tests/test_core.py; all fail until core.py is implemented
	$(PY) -m pytest -q

data:      ## download SMD machine-1-1 and machine-1-2 into data/smd/ (pass MACHINES="..." to override)
	bash scripts/get_smd.sh $(MACHINES)

demo:      ## host-shift demo: prints 3 numbers, writes experiments/host_shift.png and .json
	$(PY) experiments/host_shift.py $(DEMO_ARGS)

run:       ## dev server with reload on http://127.0.0.1:8000 (Swagger at /docs)
	$(PY) -m uvicorn alertcal.api:app --reload

docker-build:
	docker build -t alertcal:local .

docker: docker-build   ## build + run locally on :8000
	docker run --rm -p 8000:8000 alertcal:local

clean:
	rm -rf .pytest_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
