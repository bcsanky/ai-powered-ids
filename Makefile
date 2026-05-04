.PHONY: dataset clean-data train-ae eval final-ae-minimal final-eval-stat final-validate final-plot-ae-minimal final-plot-ae-context final-compare final-rule-proxy final-hybrid wazuh-parse-alerts wazuh-correlate wazuh-eval-real lab-ae-validate-features lab-ae-score lab-ae-eval hybrid-real-eval real-compare real-plot final-real-hybrid score-sample-events score-lab-events generate-security-report generate-case-studies benchmark-scoring plot-performance generate-performance-report export-performance-artifacts collect-thesis-figures final-day4 final-day5 final-day6 final-day7 final-day8 final-day9 final-day10-performance

BASELINE ?= stat
PYTHON ?= python3
CONFIG ?= experiments/experiment.yaml
DATA_DIR ?= data/processed
RESULTS_DIR ?= results
WAZUH_ALERTS ?= data/wazuh/alerts.jsonl
WAZUH_GROUND_TRUTH ?= data/lab/lab_ground_truth.csv
WAZUH_WORK_DIR ?= data/processed/wazuh_real
WAZUH_RESULTS_DIR ?= results/wazuh_real
WAZUH_WINDOW_SECONDS ?= 60
LAB_FEATURES ?= data/lab/lab_features.csv
LAB_AE_WORK_DIR ?= data/processed/lab_ae
LAB_AE_RESULTS_DIR ?= results/ae_lab
HYBRID_REAL_RESULTS_DIR ?= results/hybrid_real
REAL_COMPARISON_DIR ?= results/real_comparison
HYBRID_WEIGHTED_THRESHOLD ?= 0.5

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
	$(PYTHON) -m py_compile ml/src/build_dataset.py ml/src/train_ae.py ml/src/eval.py ml/src/plot_final_results.py ml/src/compare_final_results.py ml/src/create_rule_proxy_export.py ml/src/hybrid_eval.py ml/src/scoring_runtime.py ml/src/score_events.py ml/src/generate_security_report.py ml/src/generate_case_studies.py ml/src/benchmark_scoring.py ml/src/plot_performance_results.py ml/src/generate_performance_report.py ml/src/export_performance_outputs.py ml/src/collect_thesis_figures.py ml/src/wazuh_baseline/build_ground_truth.py ml/src/wazuh_baseline/parse_wazuh_alerts.py ml/src/wazuh_baseline/correlate_alerts.py ml/src/wazuh_baseline/evaluate_wazuh_baseline.py ml/src/lab_ae_eval/validate_lab_features.py ml/src/lab_ae_eval/score_lab_features.py ml/src/lab_ae_eval/evaluate_ae_lab.py ml/src/hybrid_real/evaluate_hybrid_real.py ml/src/hybrid_real/compare_real_results.py ml/src/hybrid_real/plot_real_comparison.py infra/mlservice/app/config.py infra/mlservice/app/schemas.py infra/mlservice/app/scoring.py infra/mlservice/app/main.py
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

final-rule-proxy:
	$(PYTHON) -m ml.src.create_rule_proxy_export --data-dir data/processed/final/ae_minimal --output-dir data/processed/final/rule_proxy --threshold-quantile 0.95
	$(PYTHON) -m ml.src.eval --baseline wazuh --data-dir data/processed/final/rule_proxy --results-dir results/final/final-rule-proxy-v1 --wazuh-input data/processed/final/rule_proxy/wazuh_like_rule_eval.csv

final-hybrid:
	$(PYTHON) -m ml.src.hybrid_eval --ae-root results/final/final-ae-minimal-v1 --rule-root results/final/final-rule-proxy-v1 --output-dir results/final/final-hybrid-v1

wazuh-parse-alerts:
	$(PYTHON) -m ml.src.wazuh_baseline.parse_wazuh_alerts --input $(WAZUH_ALERTS) --output $(WAZUH_WORK_DIR)/alerts_parsed.csv

wazuh-correlate:
	$(PYTHON) -m ml.src.wazuh_baseline.build_ground_truth --input $(WAZUH_GROUND_TRUTH) --output $(WAZUH_WORK_DIR)/lab_ground_truth_validated.csv
	$(PYTHON) -m ml.src.wazuh_baseline.correlate_alerts --ground-truth $(WAZUH_WORK_DIR)/lab_ground_truth_validated.csv --alerts $(WAZUH_WORK_DIR)/alerts_parsed.csv --output $(WAZUH_WORK_DIR)/wazuh_correlated.csv --window-seconds $(WAZUH_WINDOW_SECONDS)

wazuh-eval-real:
	$(PYTHON) -m ml.src.wazuh_baseline.evaluate_wazuh_baseline --input $(WAZUH_WORK_DIR)/wazuh_correlated.csv --results-dir $(WAZUH_RESULTS_DIR)

lab-ae-validate-features:
	$(PYTHON) -m ml.src.lab_ae_eval.validate_lab_features --input $(LAB_FEATURES) --output $(LAB_AE_WORK_DIR)/lab_features_validated.csv

lab-ae-score:
	$(PYTHON) -m ml.src.lab_ae_eval.score_lab_features --features $(LAB_AE_WORK_DIR)/lab_features_validated.csv --ground-truth $(WAZUH_WORK_DIR)/lab_ground_truth_validated.csv --output $(LAB_AE_WORK_DIR)/ae_lab_predictions.csv --model-root artifacts/final/final-ae-minimal-v1 --preprocess data/processed/final/ae_minimal/preprocess.pkl

lab-ae-eval:
	$(PYTHON) -m ml.src.lab_ae_eval.evaluate_ae_lab --input $(LAB_AE_WORK_DIR)/ae_lab_predictions.csv --results-dir $(LAB_AE_RESULTS_DIR)

hybrid-real-eval:
	$(PYTHON) -m ml.src.hybrid_real.evaluate_hybrid_real --wazuh-predictions $(WAZUH_RESULTS_DIR)/predictions.csv --ae-predictions $(LAB_AE_RESULTS_DIR)/predictions.csv --results-dir $(HYBRID_REAL_RESULTS_DIR) --weighted-threshold $(HYBRID_WEIGHTED_THRESHOLD)

real-compare:
	$(PYTHON) -m ml.src.hybrid_real.compare_real_results --wazuh-metrics $(WAZUH_RESULTS_DIR)/metrics_summary.csv --ae-metrics $(LAB_AE_RESULTS_DIR)/metrics_summary.csv --hybrid-metrics $(HYBRID_REAL_RESULTS_DIR)/metrics_summary.csv --output-dir $(REAL_COMPARISON_DIR)

real-plot:
	$(PYTHON) -m ml.src.hybrid_real.plot_real_comparison --comparison $(REAL_COMPARISON_DIR)/metrics_comparison.csv --output-dir $(REAL_COMPARISON_DIR)

final-real-hybrid:
	$(MAKE) wazuh-parse-alerts
	$(MAKE) wazuh-correlate
	$(MAKE) wazuh-eval-real
	$(MAKE) lab-ae-validate-features
	$(MAKE) lab-ae-score
	$(MAKE) lab-ae-eval
	$(MAKE) hybrid-real-eval
	$(MAKE) real-compare
	$(MAKE) real-plot

score-sample-events:
	$(PYTHON) -m ml.src.score_events --input examples/scoring/sample_events.jsonl --output reports/scored_events.jsonl --model-root artifacts/final/final-ae-minimal-v1 --preprocess data/processed/final/ae_minimal/preprocess.pkl --thresholds-auto

score-lab-events:
	$(PYTHON) -m ml.src.score_events --input examples/lab/lab_events.jsonl --output reports/lab/lab_scored_events.jsonl --csv-output reports/lab/lab_scored_events.csv --model-root artifacts/final/final-ae-minimal-v1 --preprocess data/processed/final/ae_minimal/preprocess.pkl --thresholds-auto

generate-case-studies:
	$(PYTHON) -m ml.src.generate_case_studies --scored-events reports/lab/lab_scored_events.jsonl --output-dir reports/lab

benchmark-scoring:
	$(PYTHON) -m ml.src.benchmark_scoring --input examples/lab/lab_events.jsonl --output-dir reports/performance --model-root artifacts/final/final-ae-minimal-v1 --preprocess data/processed/final/ae_minimal/preprocess.pkl --event-counts 100 500 1000 5000 10000 --batch-sizes 1 10 50 100 --repeats 3

plot-performance:
	$(PYTHON) -m ml.src.plot_performance_results --input reports/performance/benchmark_results.csv --output-dir reports/performance

generate-performance-report:
	$(PYTHON) -m ml.src.generate_performance_report --benchmark reports/performance/benchmark_results.csv --system-info reports/performance/system_info.json --output-dir reports/performance

export-performance-artifacts:
	$(PYTHON) -m ml.src.export_performance_outputs --benchmark reports/performance/benchmark_results.csv --latency-figure reports/performance/latency_by_batch_size.png --throughput-figure reports/performance/throughput_by_batch_size.png --resource-figure reports/performance/resource_usage_by_batch_size.png --results-dir results/performance --figures-dir figures/final

collect-thesis-figures:
	$(PYTHON) -m ml.src.collect_thesis_figures --output-dir reports/final/thesis_figures --ae-minimal-root results/final/final-ae-minimal-v1 --ae-context-root results/final/final-ae-context-v1 --comparison-dir results/final/comparison --lab-dir reports/lab --performance-dir reports/performance

generate-security-report:
	$(PYTHON) -m ml.src.generate_security_report --comparison results/final/comparison/metrics_comparison.csv --scored-events reports/scored_events.jsonl --case-summary reports/lab/scenario_summary.csv --output-dir reports/final

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

final-day6:
	$(MAKE) final-validate
	$(MAKE) final-rule-proxy
	$(MAKE) final-hybrid
	$(MAKE) final-compare

final-day7:
	$(MAKE) final-validate
	$(MAKE) score-sample-events
	$(MAKE) generate-security-report

final-day8:
	$(MAKE) final-validate
	$(MAKE) score-lab-events
	$(MAKE) generate-case-studies
	$(MAKE) collect-thesis-figures
	$(MAKE) generate-security-report

final-day9:
	$(MAKE) final-validate
	$(MAKE) benchmark-scoring
	$(MAKE) plot-performance
	$(MAKE) generate-performance-report
	$(MAKE) collect-thesis-figures

final-day10-performance:
	$(MAKE) final-validate
	$(MAKE) benchmark-scoring
	$(MAKE) plot-performance
	$(MAKE) generate-performance-report
	$(MAKE) export-performance-artifacts
