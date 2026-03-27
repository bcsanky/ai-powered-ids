.PHONY: dataset clean-data

dataset:
	python3 ml/src/build_dataset.py --config experiments/experiment.yaml

clean-data:
	rm -f data/processed/train.parquet \
	      data/processed/val.parquet \
	      data/processed/test.parquet \
	      data/processed/preprocess.pkl \
	      data/processed/dataset_metadata.json

BASELINE ?= stat
PYTHON ?= python3

eval:
	$(PYTHON) -m ml.src.eval --baseline $(BASELINE)
