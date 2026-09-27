# SPEC — alertcal v0.1

## Problem
Label-free anomaly detectors are evaluated by ranking (AUROC/AUPRC) but deployed with a threshold.
A threshold set as a quantile of benign scores in the source environment realises a different
false-positive rate in the deployment environment, because it depends on the location and scale of
the score distribution there, and ranking does not. Observed in the thesis on host, network and sensor
data across six detector families. Recalibrating on a benign deployment slice repaired it.

## Core API (src/alertcal/core.py) — Tanay implements
| function | maths | returns |
|---|---|---|
| `source_threshold(benign_scores, alpha)` | Q_{1-alpha} of benign scores, source env | float |
| `realised_fpr(benign_scores, threshold)` | mean(scores >= threshold) | float in [0,1] |
| `recalibrate(benign_deployment_scores, alpha)` | Q_{1-alpha} of benign scores, deployment env | float |
| `sample_size_curve(scores, alpha, sizes, draws, seed)` | for each n: draw n w/o replacement `draws` times, recalibrate, measure realised FPR on the full set | `{n: (mean, p5, p95)}` |

Validation: `alpha` outside (0,1) -> ValueError. `recalibrate` with fewer than `ceil(1/alpha)`
samples -> ValueError (the quantile is meaningless below that). Sizes larger than the data are skipped.
Quantile method: `np.quantile(x, 1 - alpha)` default (linear). Match what the thesis used if
HANDOFF.md says otherwise.

## CLI (done)
`alertcal calibrate|evaluate|curve <csv> [--alpha] [--threshold] [--column score]`

## API (done)
`GET /health`, `POST /calibrate`, `POST /evaluate`, `POST /curve`. Pydantic-validated. Swagger at `/docs`.

## Demo (experiments/host_shift.py, skeleton done)
1. Fit StandardScaler + IsolationForest on 85% of SMD machine-1-1 train (benign).
2. `source_threshold` on the held-out 15% at alpha = 0.05.
3. Score machine-1-2 test with the same scaler + model. Split by label into benign / anomalous.
4. Report realised FPR and TPR of the fixed threshold.
5. Take 200 random benign deployment points, `recalibrate`, remove them from evaluation, report again.
6. `sample_size_curve` on the deployment benign set. Plot both panels.
If machine-1-2 shows a weak shift, try another machine as deployment host and say which in the README.

## Data
SMD: 28 machines, 38 metrics, ~1 min sampling, train files benign, test files labelled.
Source repo: NetManAIOps/OmniAnomaly on GitHub, folder `ServerMachineDataset/{train,test,test_label}`.
**Verify the raw URLs in scripts/get_smd.sh before running.** Known caveat: SMD labels have been
criticised (Wu & Keogh 2023); irrelevant here because the claim is about the threshold, not the ranking.

## Deployment
Azure Container Apps, region germanywestcentral, image built by `az acr build` (no local push).
Teardown command is at the bottom of azure/deploy.sh.

## Non-goals for v0.1
Contamination study, rolling recalibration, conformal baseline, Prometheus/Azure Monitor adapters,
any detector other than IsolationForest in the demo. All belong in the README roadmap.

## Definition of done
`make test` green; `make demo` prints three numbers where fixed FPR >> 0.05 and recalibrated ~0.05;
public `/docs` URL responds; README has the plot, the numbers and the quickstart.
