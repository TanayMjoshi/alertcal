# alertcal

Recalibrate anomaly-alert thresholds after deployment shift. Scores in, thresholds out.

A detector can rank anomalies perfectly on a new server and still fire 4 to 11 times the intended
alerts, because the threshold encodes the environment it was set in. `alertcal` re-estimates the
(1 - alpha) quantile on a small benign sample from the deployment environment. It is
model-agnostic: it consumes **scores**, never raw data, so it cannot refit anything.

Three faces of the same four functions: a Python library, a CLI, and a FastAPI service in Docker
on Azure Container Apps.

## Why

In my bachelor thesis I compared anomaly detectors on network, host and sensor data, and the
result that stuck with me was not about which model wins. It was that every detector ranked
anomalies on an unseen host with an AUROC above 0.97 and still raised between 22 and 54 percent
false alarms when I carried the threshold over from the training host. Same model, same ranking,
wrong cutoff. On sensor data the same nominal percentile gave a 3.4-fold range of false-positive
rates just from changing the window length. Recalibrating the threshold on a small benign sample
from the deployment environment brought the classical baselines back to between 4.2 and 5.7
percent in every seed. The paper I am writing with my supervisor looks at how much deployment
data that correction needs and how much contamination it tolerates; alertcal is the small,
dataset-agnostic tool version of the same idea, so that you can measure your own realised
false-positive rate, recalibrate on a slice, and see the sample-size curve before you trust a
threshold that was set somewhere else.

## Results

_Filled in by `make demo`._ Source host: SMD machine-1-1. Deployment host: machine-1-2.
Plot: `experiments/host_shift.png`. Numbers: `experiments/host_shift.json`.

## Quickstart

```bash
python3.12 -m venv .venv && source .venv/bin/activate
make install   # pip install -e ".[api,experiments,dev]"
make test      # 6 tests in tests/test_core.py
make data      # downloads SMD machine-1-1 and machine-1-2 (about 35 MB) into data/smd/
make demo      # prints three numbers, writes experiments/host_shift.png
```

### Library

```python
from alertcal import source_threshold, realised_fpr, recalibrate, sample_size_curve

thr = source_threshold(source_benign_scores, alpha=0.05)      # set once, at home
realised_fpr(deploy_benign_scores, thr)                        # measure there: probably not 0.05
thr_new = recalibrate(deploy_benign_slice, alpha=0.05)         # 200 clean points from the new host
realised_fpr(deploy_benign_rest, thr_new)                      # about 0.05 again (slice excluded)
sample_size_curve(deploy_benign_scores, 0.05, sizes=(50, 200, 1000), draws=20, seed=0)
```

### CLI

```
alertcal calibrate scores.csv --alpha 0.05 --column score
alertcal evaluate  scores.csv --threshold 0.62
alertcal curve     scores.csv --alpha 0.05 --sizes 50,200,1000 --draws 20 --seed 0
```

### API

`make run`, then open http://127.0.0.1:8000/docs. Endpoints: `GET /health`, `POST /calibrate`,
`POST /evaluate`, `POST /curve`.

```bash
curl -s -X POST localhost:8000/calibrate -H 'content-type: application/json' \
  -d '{"scores": [0.1, 0.4, 0.2, 0.9, 0.3, 0.5, 0.2, 0.1, 0.6, 0.3, 0.2, 0.4, 0.1, 0.3, 0.5, 0.2, 0.7, 0.1, 0.3, 0.2], "alpha": 0.05}'
```

### Docker and Azure

`make docker` builds and runs the API locally on :8000. `azure/deploy.sh` builds the image with
`az acr build` and deploys it to Azure Container Apps in germanywestcentral. The teardown command
is at the bottom of that script.

## Conventions and decisions

- Higher score = more anomalous. An event alerts when `score >= threshold`, so tied scores at the
  cutoff alert. Scikit-learn's `score_samples` points the other way; the demo negates it once.
- `alpha` is a rate in (0, 1), not a percent.
- Quantile: numpy default (linear interpolation), the same as the thesis. No finite-sample correction.
- `recalibrate` refuses fewer than ceil(1/alpha) scores.
- The calibration slice is removed from every set a realised rate is reported on.
- Seeds are explicit everywhere; no global RNG state.
- Data: Server Machine Dataset from NetManAIOps/OmniAnomaly. Train files are benign, test files are
  labelled. SMD's labels have been criticised (Wu and Keogh 2023); that does not matter here
  because the claim is about the threshold, not the ranking.

## Roadmap (not in v0.1)

- Conformal quantile: threshold = k-th smallest score with k = ceil((n + 1)(1 - alpha)).
- Wilson interval and sample count on every realised rate.
- Contamination sensitivity: how far the cutoff moves when a fraction of the slice is not benign.
- Rolling recalibration.
- Prometheus and Azure Monitor score adapters.
- Detectors other than IsolationForest in the demo. A warning when many scores tie at the cutoff.
- A LICENSE file.
