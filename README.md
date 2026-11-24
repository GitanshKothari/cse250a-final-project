# Reproduce the CSE 250A final project

The main study uses the 12-symbol parser, frozen issue splits, and five cross-validation folds. Two additional implementations in this archive produce appendix results. Each has its own package named hmm_agents. Run each section from the indicated directory so its imports, data and outputs stay together.

## Setup and data

Use Python 3.12 or newer with pip, Git on PATH (the runner invokes it to record provenance), internet access for the public data, and space for 12 downloaded shards (about 1.1 GB), the processed table, model caches and predictions. The commands below use a POSIX shell. Start in the extracted project directory. Setup installs the package in editable mode because it resolves configs and data relative to the source tree.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q tests
python scripts/01_download.py --all
python scripts/02_preprocess.py
```

The downloader reads the public nebius/SWE-agent-trajectories dataset on Hugging Face. Keep configs/splits.csv and configs/folds.csv as supplied. Check that data/processed/runs.parquet contains 79,929 retained runs from 80,036 raw runs. Training combines hmm_train and clf_train (55,259 runs, 2,513 issues), validation has 12,094 runs and 539 issues, and test has 12,576 runs and 539 issues. These counts catch partial downloads. The original downloader did not record an immutable dataset revision. The included requirements describe compatible versions. verification-versions.txt records the direct library versions used for submission checks, rather than the original training environment.

## Main tables

Table 1 describes model choices. For the headline validation and test results in Tables 3 and 13, use the exact final specification below. It includes five model families and seeds 0, 1 and 2. Run this in a fresh extracted copy to keep the submitted logs intact.

```sh
python scripts/run_experiment.py configs/experiments/final_comparison.yaml --mode val
python scripts/run_experiment.py configs/experiments/final_comparison.yaml --mode test --final
python scripts/summarize.py results/final_comparison/val --t 5 10 15 20 30
python scripts/summarize.py results/final_comparison/test --t 5 10 15 20 30
```

Read results/final_comparison/{val,test}/metrics.csv for pooled AUROC (auroc), within-issue AUROC (auroc_wi), calibration and oracle metrics. Use cohort=alive and the indicated t. Average stochastic methods over the three seeds for the model rows. Use paired.csv for each seed’s paired difference and interval, and tradeoff.csv for the 10% stopping rule. The reported test GRU pooled AUROCs are 0.747 at 10 steps, 0.771 at 20 and 0.794 at 30. At 10 steps the test HMM-LR is 0.722. On validation the GRU is 0.723. Compare rounded values and use these as checkpoints.

For Tables 2 and 4–11 and the development results, run the following sequentially. Model fitting across five folds and multiple seeds is substantially more expensive than the tests. Preserve models/cache between experiments.

```sh
for exp in confirm_trap_features diag_hmm_vs_ngram hmm_full_data zoo_classifiers zoo_hmm_family hsmm_capacity k14_check zoo_neural zoo_cnn; do
  python scripts/run_experiment.py configs/experiments/$exp.yaml --mode cv || break
done
python scripts/run_experiment.py configs/experiments/confirm_trap_features.yaml --mode val
python scripts/summarize.py results/zoo_classifiers/cv results/zoo_hmm_family/cv results/zoo_neural/cv results/zoo_cnn/cv --t 5 10 15 20 30
```

| Report material                | Inputs and outputs                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| §§7.1–7.2, protocol and parser | docs/PROTOCOL.md, docs/parser_validation.md and dev_docs/02_parser_revamp.md record the protocol and hand audit. Run python scripts/experiments/exp01_parser_decisions.py after preprocessing for the two parser decisions. The second pipeline’s two manual review sheets are in archive/parallel_dev/docs. The cross-parser 3,000-run comparison is recorded in dev_docs/16_parallel_dev_integration.md. Human correctness judgments are recorded evidence. |
| §7.3, calibration and restarts | Use the historical calibration recipe below.                                                                                                                                                                                                                                                                                                                                                                                                                  |
| §§7.4–7.6, Tables 2, 4         | confirm_trap_features, diag_hmm_vs_ngram and hmm_full_data. Read paired.csv and metrics.csv. dev_docs/10–13 describe the cohorts, training budgets and comparisons.                                                                                                                                                                                                                                                                                           |
| §§7.7–7.8, Tables 5–7          | zoo_classifiers, zoo_hmm_family and hsmm_capacity. For Table 7 compare the 24-state control with the 3-phase HMM at seed 0.                                                                                                                                                                                                                                                                                                                                   |
| §7.9, Tables 8–9               | k14_check and exp04 below, plus the supplied 14-state reference model. The separate K-selection CV is in supplements/alternative_models.                                                                                                                                                                                                                                                                                                                      |
| §7.10, Tables 10–11            | zoo_neural and zoo_cnn, with zoo_classifiers and zoo_hmm_family for feature baselines. Use each seed’s predictions for cross-experiment intervals. The text and Transformer checks are in supplements/alternative_models.                                                                                                                                                                                                                                     |
| §7.11, Table 12                | exp02 and exp02b below. Use exp03 in the alternative implementation for the issue-adjusted trap analysis.                                                                                                                                                                                                                                                                                                                                                     |
| §§7.12–7.14, Tables 13–14      | final_comparison for held-out results. For Table 14 compare confirm_trap_features with the archived validation pipeline. The equal-budget tuning grid was stopped and is outside the reported comparison.                                                                                                                                                                                                                                                     |

For a paired comparison across experiment folders, the order is B minus A. For example, run the following for Table 11 and repeat for seeds 1 and 2. Each comparison writes a CSV next to the second input. For the remaining contrasts use the method identifiers in each experiment YAML and predictions.parquet.

```sh
python scripts/compare.py results/zoo_neural/cv:gru@s0 results/zoo_cnn/cv:cnn@s0 --cohort alive
for seed in 0 1 2; do
  python scripts/experiments/exp02_model_explanations.py --seed "$seed" || break
done
python scripts/experiments/exp02b_stability.py
python scripts/experiments/exp04_k14_modes.py
```

See results/experiments/exp02/seed/ and stability.csv for transition and dwell statistics, and results/experiments/exp04/ for mode matching. The K=14 model in archive/parallel_dev/models_snapshot is an input to exp04 and must stay at that path.
