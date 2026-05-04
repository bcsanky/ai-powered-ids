.PHONY: dataset clean-data train-ae eval final-ae-minimal final-eval-stat final-validate final-plot-ae-minimal final-plot-ae-context final-compare final-day4 final-day5

BASELINE ?= stat
PYTHON ?= python3
CONFIG ?= experiments/experiment.yaml
DATA_DIR ?= data/processed
RESULTS_DIR ?= results

dataset:
	$(PYTHON) ml/src/build_dataset.py --config $(CONFIG)

clean-data:
	rm -f data/processed/train.parquet \
	      data/processed/val.parquet \
	      data/processed/calib.parquet \
	      data/processed/test.parquet \
	      data/processed/preprocess.pkl \
	      data/processed/dataset_metadata.json

train-ae:
	$(PYTHON) -m ml.src.train_ae --config $(CONFIG)

eval:
	$(PYTHON) -m ml.src.eval --baseline $(BASELINE) --data-dir $(DATA_DIR) --results-dir $(RESULTS_DIR)

final-ae-minimal:
	$(MAKE) dataset CONFIG=experiments/final/ae_minimal.yaml
	$(MAKE) train-ae CONFIG=experiments/final/ae_minimal.yaml

final-eval-stat:
	$(MAKE) eval BASELINE=stat DATA_DIR=data/processed/final/ae_minimal RESULTS_DIR=results/final/final-baseline-stat-v1

final-validate:
	$(PYTHON) -c 'import yaml; from pathlib import Path; [yaml.safe_load(open(p, encoding="utf-8")) for p in sorted(Path("experiments/final").glob("*.yaml"))]; print("Final YAML configs OK")'
	$(PYTHON) -m py_compile ml/src/build_dataset.py ml/src/train_ae.py ml/src/eval.py ml/src/plot_final_results.py ml/src/compare_final_results.py
	$(PYTHON) -m pytest ml/tests

final-plot-ae-minimal:
	@RUN_DIR=$$(find results/final/final-ae-minimal-v1 -mindepth 1 -maxdepth 1 -type d -name 'ae_v1_*' | sort | tail -n 1); \
	if [ -z "$$RUN_DIR" ]; then \
		echo "No AE-Minimal run directory found under results/final/final-ae-minimal-v1"; \
		exit 1; \
	fi; \
	echo "Plotting AE-Minimal results from $$RUN_DIR"; \
	$(PYTHON) -m ml.src.plot_final_results --run-dir "$$RUN_DIR"

final-plot-ae-context:
	@if [ ! -d results/final/final-ae-context-v1 ]; then \
		echo "No AE-Context run directory found under results/final/final-ae-context-v1"; \
		exit 1; \
	fi; \
	RUN_DIR=$$(find results/final/final-ae-context-v1 -mindepth 1 -maxdepth 1 -type d -name 'ae_v1_*' | sort | tail -n 1); \
	if [ -z "$$RUN_DIR" ]; then \
		echo "No AE-Context run directory found under results/final/final-ae-context-v1"; \
		exit 1; \
	fi; \
	echo "Plotting AE-Context results from $$RUN_DIR"; \
	$(PYTHON) -m ml.src.plot_final_results --run-dir "$$RUN_DIR"

final-compare:
	$(PYTHON) -m ml.src.compare_final_results --results-root results/final --output-dir results/final/comparison

final-day4:
	$(MAKE) final-validate
	$(MAKE) final-plot-ae-minimal
	@if [ -d results/final/final-ae-context-v1 ] && find results/final/final-ae-context-v1 -mindepth 1 -maxdepth 1 -type d -name 'ae_v1_*' | grep . >/dev/null 2>&1; then \
		$(MAKE) final-plot-ae-context; \
	else \
		echo "Skipping AE-Context plot: no run directory found under results/final/final-ae-context-v1"; \
	fi
	$(MAKE) final-compare

final-day5:
	$(MAKE) final-validate
	$(MAKE) dataset CONFIG=experiments/final/ae_context.yaml
	$(MAKE) train-ae CONFIG=experiments/final/ae_context.yaml
	$(MAKE) final-plot-ae-context
	$(MAKE) final-plot-ae-minimal
	$(MAKE) final-compare
