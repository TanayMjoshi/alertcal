# alertcal: handoff from the thesis codebase (16 Sep 2026)

For the new `alertcal` chat. Written by reading the thesis repository at
`/Users/tanayjoshi/Desktop/Thesis` without modifying anything. alertcal
packages one thesis finding: a fixed anomaly threshold set in the source
environment does not transfer to the deployment environment even when ranking
does, and recalibrating the quantile on a small benign deployment slice repairs
it. It must reuse ideas and conventions only, never thesis data or
thesis-specific code. Line numbers below are from the repository as of today.

Where the conventions live:

| Topic | File | Lines |
|---|---|---|
| Score sign, flows (time series) | `code/timeseries/method1/run_method1.py` | 180 to 189 |
| Score sign, flows (static) | `code/static/experiments/nslkdd_full_comparison.py` | 361 to 368, 485 to 487 |
| Score sign, IF / OCSVM / AE | `code/timeseries/beth/run_beth_baselines.py` | 105 to 112, 164 to 177 |
| Score, FFJORD (sign fix) | `code/static/cnf/ffjord_core.py`, commit `654d9dc` | see section 6 |
| Score, Latent ODE | `code/timeseries/method3/latent_ode_core.py` | 327 to 356 |
| Threshold sweep (time series) | `code/timeseries/method1/run_method1.py` | 192 to 218 |
| Threshold sweep (static) | `code/static/experiments/nslkdd_full_comparison.py` | 418 to 433 |
| Validation split (static) | `code/static/experiments/nslkdd_full_comparison.py` | 225 to 226 |
| Validation split (time series) | `code/timeseries/method1/run_method1.py` 324 to 330; `run_latent_ode.py` 459 to 462; `run_beth_baselines.py` 124 to 125 | |
| Equal realised FPR (cross-method) | `code/slides/build_equal_fpr_restatement.py` | 47 to 54 |
| Equal realised FPR (analysis device on test normals) | `code/static/nslkdd/analysis/fp_overlap_study.py` | 115 to 124 |
| Recalibration on BETH | `code/timeseries/beth/run_beth_baselines.py` | 69, 133 to 147, 179 to 189 |
| Standardisation fix | `code/timeseries/nab/sliding_window_nsf.py` 391 to 403; commit `5faeeb3` | |
| Window harness | `code/timeseries/analysis/a9_window_confusion.py` | whole file |
| The audits | `docs/audits/red_team_audits.md` | audit 1 section 1, audit 2 sections 1 to 3 |

---

## 1. Score convention: higher means more anomalous, enforced at the call site

The thesis has one rule, "higher score = more anomalous", and applies it by
hand wherever a detector's native output points the other way. There is no
shared wrapper; each runner negates at the call site. That is fragile and is
the first thing alertcal should centralise.

Flows produce a log-likelihood, negated to make the score:

```python
# code/timeseries/method1/run_method1.py:180-189
def score_dataset(flow, ds, batch_size: int, device: torch.device) -> np.ndarray:
    """Return per-window anomaly score (= -log p_θ). Higher → more anomalous."""
    ...
            scores.append((-flow.log_prob(batch)).cpu().numpy())
```

The static runner keeps `compute_log_probs` returning `log p(x)` and negates
when it calls the sweep:

```python
# code/static/experiments/nslkdd_full_comparison.py:485-487
val_lp = compute_log_probs(flow_v2, X_val, DEVICE)
test_lp = compute_log_probs(flow_v2, X_test, DEVICE)
seed_results['nsf_v2'] = threshold_sweep(-test_lp, -val_lp, y_test_binary, y_test_category)
```

The classical baselines are negated because scikit-learn's `score_samples`
and `decision_function` are "higher = more normal":

```python
# code/timeseries/beth/run_beth_baselines.py:164-177
    if model_name == "isolation_forest":
        mdl = IsolationForest(n_estimators=200, contamination=0.001,
                              random_state=seed, n_jobs=-1).fit(Xfit)
        val_s, test_s = -mdl.score_samples(Xval), -mdl.score_samples(Xte)
    elif model_name == "one_class_svm":
        ...
        val_s = -mdl.decision_function(Xval)
        test_s = -mdl.decision_function(Xte)
    else:
        ae = train_ae(Xfit, Xfit.shape[1], seed, epochs=5 if smoke else 50)
        val_s, test_s = ae_score(ae, Xval), ae_score(ae, Xte)   # MSE, already higher = worse
```

The Latent ODE score is a negated Gaussian log-density summed over the
window, with an optional normalisation by the number of observed points
(`latent_ode_core.py:350`: `score = -log_p_xt.sum(dim=(1, 2))   # higher = more anomalous`).
FFJORD's `per_sample_nll` is `-log p(x)` with `log p(x) = LogPrior - DeltaLogProb`
after the sign fix (section 6).

For alertcal: one function `to_anomaly_scores(detector, X)` that knows the
sign of every supported detector (for `IsolationForest`: `-score_samples`),
plus a smoke assertion in the tests that AUROC on a labelled toy set is above
0.5. The sign flip is the single most common silent bug in this whole family
of tools, and the thesis found its own version of it inside FFJORD.

## 2. Threshold sweep: the (100 − p)th percentile of validation-normal scores

Both pillars do the same thing: take the scores of the held-out normal
validation split, take the `(100 - p)`th percentile with numpy's default
method, and flag test samples above it. Nominal FPR is therefore `p` percent.

```python
# code/timeseries/method1/run_method1.py:192-218
def matched_fpr_metrics(test_scores, test_labels, val_scores,
                        percentiles=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)):
    ...
    for pct in percentiles:
        thr = float(np.percentile(val_scores, 100 - pct))
        pred = (test_scores >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(test_labels, pred, labels=[0, 1]).ravel()
        ...
        out['threshold_sweep'].append({
            'percentile': pct, 'threshold': thr,
            'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
            'TPR': float(tpr), 'TNR': float(tnr),
            'FPR': float(fp / (fp + tn) if (fp + tn) else 0.0),
            'F1': ..., 'FOR': ..., 'G_mean': ...})
```

```python
# code/static/experiments/nslkdd_full_comparison.py:418-433
def threshold_sweep(scores_test, scores_val, y_binary, y_category):
    ...
    for p in PERCENTILES:
        threshold = float(np.percentile(scores_val, 100.0 - p))
        y_pred = (scores_test > threshold).astype(int)
        normal_mask = y_binary == 0
        fpr = float(y_pred[normal_mask].mean())
```

Facts to carry over:

- Quantile method: `np.percentile` default, which is `method="linear"`
  (interpolation between the two nearest order statistics). No finite-sample
  correction. The realised rate is therefore only approximately `p` even on
  the validation data itself.
- Data: the validation split of the NORMAL training data, never the fit split
  and never test data. Static: `train_test_split(df_normal, test_size=0.15,
  random_state=seed)` (`nslkdd_full_comparison.py:225`). Time series: the
  LAST 15 percent of the training windows in time order,
  `torch.utils.data.Subset(train_ds_full, range(train_count, train_count + val_count))`
  (`run_method1.py:324-330`), and in the BETH baselines
  `Xfit, Xval = Xtr_full[:-n_val], Xtr_full[-n_val:]` (`run_beth_baselines.py:124-125`).
  The same validation split is used for early stopping, which the thesis
  discloses as an optimistic early-stopping signal; alertcal has no training
  loop so this does not apply, but keep calibration data and any model
  selection data separate anyway.
- Comparison operator is INCONSISTENT across the thesis: `>=` in the
  time-series sweeps (`run_method1.py:205`, `run_latent_ode.py:271`,
  `run_ffjord_nslkdd.py:206`) and `>` in every static sweep
  (`nslkdd_full_comparison.py:425`, `fp_overlap_study.py:122`,
  `threshold_tweaking.py:48`). With continuous scores it never mattered; with
  tied scores it does (section 9). alertcal must pick one and document it.
- The sweep grid is `(0.1, 0.5, 1, 2, 5, 10, 20)` percent and the thesis
  operating point is 5 percent. alertcal's `alpha` should be a float in
  (0, 1), not a percent, to avoid the `100 - p` inversion appearing in user
  code.

## 3. Realised FPR and the equal-realised-FPR protocol

Two different devices in the thesis, and only one of them belongs in alertcal.

Device A, the cross-method comparison. Each method's per-seed sweep is a set
of points `(realised fpr, per-category rate)`; the rate at a common target FPR
is obtained by linear interpolation, with no extrapolation:

```python
# code/slides/build_equal_fpr_restatement.py:47-54
def interp_at_fpr(sweep: list[dict], cat: str, target: float) -> float:
    """Linear interpolation of detection rate vs realized FPR. NaN outside range."""
    pts = sorted((r["fpr"], r[cat]) for r in sweep)
    fprs = np.array([p[0] for p in pts]); rates = np.array([p[1] for p in pts])
    if target < fprs[0] or target > fprs[-1]:
        return float("nan")
    return float(np.interp(target, fprs, rates))
```

The docstring's caveat is honest: this is linear interpolation of a step
function, so values are approximate to the sweep resolution. It exists to
compare six methods fairly (audit 2 section 3 had found the earlier
"matched" handout comparing two methods at 4.26 and 6.47 percent). alertcal
compares a threshold before and after recalibration for ONE detector, so it
does not need this. Do not copy it.

Device B, an analysis device that calibrates on TEST normals:

```python
# code/static/nslkdd/analysis/fp_overlap_study.py:115-124
def equal_fpr_predictions(scores_test, y_bin, target_fpr):
    """Threshold calibrated so realized FPR on test-normal == target (analysis point)."""
    normal_scores = scores_test[y_bin == 0]
    threshold = float(np.quantile(normal_scores, 1.0 - target_fpr))
    y_pred = (scores_test > threshold).astype(int)
    realized_fpr = float(y_pred[y_bin == 0].mean())
    return threshold, y_pred, realized_fpr
```

The thesis is explicit that this is "an analysis device calibrated on the
test normals for set comparability, not a deployable threshold". alertcal's
`realised_fpr` is the measurement half of this: `mean(scores_benign > thr)`
on benign deployment data that was NOT used to set `thr`. Report it with the
count and a binomial (Wilson) interval, because at 5 percent and a few
hundred samples the interval is wide. Do not offer a "set the threshold on
the evaluation data" path unless it is named as an oracle.

## 4. The BETH recalibration protocol, verbatim

```python
# code/timeseries/beth/run_beth_baselines.py:69
RECAL_FRAC = 0.20          # benign test-host slice used ONLY for the cutoff

# code/timeseries/beth/run_beth_baselines.py:133-147
def recalibrated_sweep(test_scores, yte, seed):
    """Second sweep: cutoff percentiles taken on a benign TEST-host slice
    (removed from the evaluated set — no leakage)."""
    rng = np.random.default_rng(seed)
    benign_idx = np.flatnonzero(yte == 0)
    n_cal = max(200, int(len(benign_idx) * RECAL_FRAC))
    cal_idx = rng.choice(benign_idx, size=min(n_cal, len(benign_idx)), replace=False)
    cal_mask = np.zeros(len(yte), dtype=bool)
    cal_mask[cal_idx] = True
    m = matched_fpr_metrics(test_scores[~cal_mask], yte[~cal_mask],
                            test_scores[cal_mask])
    m["calibration"] = {"source": "test-host benign slice",
                        "n_calibration": int(cal_mask.sum()),
                        "n_eval": int((~cal_mask).sum())}
    return m
```

How it works: the slice is drawn per seed with `np.random.default_rng(seed)`,
uniformly at random from the benign windows of the deployment host, size
`max(200, 20 percent of benign)`, without replacement; the calibration
windows are masked out of the evaluated set; the cutoff percentiles are
taken on the slice's scores exactly as in section 2; only the cutoff changes,
the model and the scaler are untouched. The output records `n_calibration`
and `n_eval` so the arithmetic can be checked.

Three things the thesis learned that alertcal inherits:

- Only the three classical baselines have a stored recalibrated sweep; the
  flow runners never stored one. The thesis and the paper draft scope the
  claim accordingly ("for the three classical baselines, whose recalibrated
  sweeps were stored"). Store everything, always, from the first run.
- With stride-1 windows, calibration windows share raw events with about 96
  percent of the evaluated benign windows (thesis Section 6.5). It touches
  only the cutoff scalar, but it is a disclosure the thesis had to make. If
  alertcal windows SMD, draw the slice as a contiguous time block with a gap
  before the evaluation block, or score rows rather than windows.
- The slice must be benign. The thesis had that by label; a deployment has
  it by operational knowledge ("this hour was clean"). The paper plans a
  contamination-tolerance experiment; alertcal should at least expose how
  much the threshold moves when a fraction of the slice is contaminated.

## 5. Seeds and RNG: what the thesis does, and the one pattern to adopt

Inventory of what the thesis actually does (it is a mixture):

- Global seeding at run start: `torch.manual_seed(seed); np.random.seed(seed)`
  (`run_method1.py:592`, `run_latent_ode.py:790`, `run_beth_baselines.py:159`,
  `nslkdd_full_comparison.py:469`).
- Local legacy generators for subsampling: `np.random.RandomState(seed)`
  (`run_beth_baselines.py:169`, `nslkdd_full_comparison.py:238, 520`).
- Modern generators for analysis steps: `np.random.default_rng(seed)` for the
  calibration slice (`run_beth_baselines.py:136`) and tuple seeds
  `np.random.default_rng((drop_seed, int(round(keep_rate * 1000))))` for the
  stress-test drops (`run_irregularity_stress.py:166`), so each keep rate has
  its own reproducible stream.
- scikit-learn objects get `random_state=seed` (`IsolationForest`,
  `train_test_split`).
- Two seed families, static `{42, 123, 456, 789, 999, 1234}` and time-series
  `{7, 13, 23, 42, 101, 202}`, never mixed in a paired comparison.

Pattern for alertcal: never touch global state. Create one
`rng = np.random.default_rng(seed)` at the API boundary and pass it down;
derive scikit-learn's `random_state` from it (`int(rng.integers(2**31 - 1))`);
for repeated draws in `sample_size_curve`, spawn independent child streams with
`np.random.SeedSequence(seed).spawn(n_draws)` so draw k is reproducible on its
own; write the seed into every result dictionary. The thesis's tuple-seed
trick is the right idea for "one stream per design point".

## 6. The three implementation errors, and which ones bite alertcal

All three were found by adversarial review after results existed, and each
fix made a claim smaller. The paper draft's limitations say exactly that.

### Error 1: test-set self-standardisation (commit `5faeeb3`, 4 Jul 2026)

Audit 1, finding 1(a): "`sliding_window_nsf.py:379-383` z-scores each BETH
file with its own mean/std. The test file is 83.8 percent attack, so attack
events dominate the test standardization statistics. Attacks are mapped near
zero mean / unit variance, exactly where the model saw training data, while
the benign minority is pushed into the tails." Rescoring the same model with
train statistics moved AUROC from 0.0098 to 0.677; the "structural host-shift
inversion" was an artefact. The fix:

```python
# code/timeseries/nab/sliding_window_nsf.py:391-403 (after the fix)
        if cfg.standardize:
            # Test sets must use TRAIN stats — self-standardizing an 84%-attack
            # test set maps attacks onto the training manifold (inversion artifact).
            if ext_stats is not None:
                mu, sigma = ext_stats
            else:
                mu = X.mean(axis=0)
                sigma = X.std(axis=0) + 1e-8
            X = (X - mu) / sigma
            self.mu, self.sigma = mu, sigma
```

and the caller passes the training statistics explicitly:
`te = SlidingWindowBETH(TEST_CSV, ..., ext_stats=(tr.mu, tr.sigma))`
(`run_beth_baselines.py:119-120`). The static pillar had always done it
right: `scaler = StandardScaler().fit(X)` only on the normal training split
(`nslkdd_full_comparison.py:253-259`).

This one bites alertcal directly, in two forms. First, any preprocessing
fitted on the deployment data: a scaler, a quantile transform, a min-max
scaler, even the "contamination" parameter of an Isolation Forest refitted on
deployment data. Fit on the source training data, freeze, apply. On SMD, each
machine is its own file; fit on the source machine's train split only. Second,
the recalibration slice must change the cutoff and nothing else. The moment
`recalibrate` refits anything, the result is a different detector, not a
recalibrated one, and the "ranking survives, threshold does not" story is
gone. Write a test that asserts `AUROC` on the deployment set is identical
before and after `recalibrate`.

### Error 2: FFJORD detached ODE states (commit `654d9dc`, 4 Jul 2026)

Audit 2, finding 1: "The ODE solver detaches the state at every integration
step, so the only gradient path to the parameters is the Hutchinson trace
term; the prior term contributes zero gradient. Training minimizes the
integrated divergence along frozen trajectories, a degenerate surrogate, not
the NLL." The fix removed every `.detach()` in `_ComputeAugmentedDerivative`,
`_EulerStep`, `_RungeKutta4Step` and `Integrate`, and was verified on a 2-D
Gaussian where the analytic NLL optimum is known (2.819 against 2.838).

Not applicable to alertcal mechanically (no ODE, no training), but the
method of catching it is: verify the core computation on a toy problem with a
closed-form answer. For alertcal: `source_threshold` on uniform(0, 1) scores
must return about `1 - alpha`; on standard normal scores about
`scipy.stats.norm.ppf(1 - alpha)`; `realised_fpr` on synthetic benign scores
drawn from the same distribution must land at `alpha` within the binomial
interval. Put those in the tests before writing the CLI.

### Error 3: FFJORD log-determinant sign (commit `654d9dc`, same day)

From the commit message: "log p(x) added DeltaLogProb instead of subtracting
it; a contracting flow was rewarded twice, hence the −1850-nat collapse.
Proven against an exact 2-D Jacobian." The change was one character:

```python
-        return LogPrior + DeltaLogProb
+        return LogPrior - DeltaLogProb
```

with the proof in `code/static/cnf/tests/test_ffjord_sign_convention.py`. The
alertcal analogue is the score sign of section 1. Negate scikit-learn's
outputs once, in one place, and assert on a labelled toy set that anomalies
score higher. A sign error here does not crash; it produces a threshold that
flags the most normal samples and a realised FPR near 100 percent, which a
tired reader will explain away as "host shift".

### Two near-misses from the same audits that are exactly alertcal's territory

- Audit 2, finding 3: the retired "matched FPR" handout chose each method's
  sweep point nearest 5 percent realised FPR on the TEST set, using test
  labels. That is threshold selection on evaluation data. alertcal must
  never select a threshold on the data it then reports a realised FPR on; the
  slice is removed from evaluation for exactly this reason.
- Audit 1, finding 8: an aggregation glob (`summary_v2_w32*`) silently
  pulled a window-64 run into the window-32 figure. Encode every knob (alpha,
  n, seed, method) in result file names and assert on read.

## 7. What NOT to copy

Everything below is tied to the thesis datasets or models. alertcal should
not import, adapt, or re-implement any of it.

- Dataset loaders and windowing: `code/timeseries/nab/sliding_window_nsf.py`
  (`SlidingWindowNAB`, `SlidingWindowBETH`, `SlidingWindowMetro`, per-process
  windows, anchor-point labelling, the contiguous 200,000-event cap, the BETH
  feature engineering with the signed log-clip of `returnValue` and the
  `is_root_uid` / `is_low_pid` flags), `code/timeseries/method3/adapters/*`.
- NSL-KDD preprocessing: `preprocess_for_seed` in
  `code/static/experiments/nslkdd_full_comparison.py` (log1p of heavy-tailed
  counts, one-hot with U(0, 1) dequantisation noise, scaler on normal train).
- Every model: `code/static/nslkdd/v2/*` (NSF), `code/static/cnf/*` (FFJORD),
  `code/timeseries/method3/latent_ode_core.py`, the small autoencoder in the
  runners.
- The stress-test harness `code/timeseries/stress/run_irregularity_stress.py`
  (random drops, re-binning, the Δt-feature arm, the w = 104 span control)
  and `code/timeseries/analysis/interpolation_at_dropped.py`.
- The cross-method analyses: `build_equal_fpr_restatement.py`,
  `fp_overlap_study.py`, `a1_ensemble_mitigation.py`,
  `a5_counterpath_prevalence.py`, `threshold_tweaking.py`.
- Figure and slide builders under `code/slides/`.
- The Colab checkpoint and resume machinery in the runners (alertcal runs in
  seconds on a CPU).
- The dead-number and never-write lists in `docs/HANDOFF_MASTER.md` are
  thesis discipline, not alertcal's concern, except for one habit worth
  keeping: never quote a number that is not regenerated by the code.

## 8. The NAB window harness, and what generalises to a sample-size curve

`code/timeseries/analysis/a9_window_confusion.py` is a tabulator, not a
library. It reads the saved sweeps of the window-NSF runs at window sizes 8,
16, 32, 64 and 128 (seed 42, one file), picks the sweep entry nearest the 5
percent nominal point, and tabulates TP, FN, FP, TN, TPR, FPR, F1, AUROC and
the anomaly base rate per window size (`row_for_window`, lines 59 to 83). Its
result: the base rate is flat across window sizes (so counts are comparable),
but the same nominal percentile realises 0.031 to 0.105 FPR, a 3.4-fold
spread, and TPR tracks the realised FPR rather than the window size.

No code in it is reusable (it parses thesis JSON), but its shape is exactly
`sample_size_curve`:

1. Fix the nominal operating point (alpha).
2. Vary one design knob (there: window size; here: calibration slice size n).
3. At each knob value, measure the realised FPR on a fixed evaluation set and
   check the invariant that makes the points comparable (there: the base
   rate; here: the evaluation set being benign-only and disjoint from the
   slice).
4. Report the realised rate and its spread, and the threshold-free metric
   beside it, so the reader sees that ranking did not move while the
   realised rate did.

The harness has one seed and one file, and the thesis marks it "illustrative"
for that reason. alertcal's curve should have many draws per n from the
start (section 9).

## 9. Recommendations for the alertcal API

- One sign contract. `to_anomaly_scores(detector, X)` returns "higher = more
  anomalous"; for `IsolationForest` that is `-detector.score_samples(X)`.
  Test it with AUROC above 0.5 on a labelled toy set.
- `source_threshold(scores, alpha, method)`: `scores` are benign
  validation scores from the source environment, never the fit data.
  Offer two methods and document both: `"linear"`, numpy's default
  `np.quantile(scores, 1 - alpha)` (what the thesis did), and `"conformal"`,
  the finite-sample rule `k = ceil((n + 1) * (1 - alpha))`, threshold = k-th
  smallest score, which gives a realised rate at most alpha in expectation
  for exchangeable data. Return the threshold, n, the method, and
  `min_alpha = 1 / (n + 1)`.
- Alpha below the resolution. If `alpha < 1 / (n + 1)` the conformal rule
  asks for an order statistic beyond the sample; raise a clear error, do not
  return the maximum score silently. The linear method would happily
  interpolate; refuse there too.
- Ties and constant scores. Define the decision as `score > threshold`
  (strictly above), because with `>=` a slice of constant scores flags
  everything (realised FPR 1.0) and with `>` it flags nothing (0.0). Warn
  when more than about 1 percent of the calibration scores are tied at the
  chosen cutoff; Isolation Forests on small data produce many ties.
- `realised_fpr(scores_benign, threshold)`: `mean(scores > threshold)` with
  n and a Wilson interval. Reject NaN or non-finite scores explicitly
  (`np.quantile` on NaN returns NaN without complaint). Make it impossible to
  pass the calibration slice as the evaluation set by accident: take an
  explicit `exclude_mask` or separate arrays and assert they are disjoint.
- `recalibrate(scores_benign_deploy_slice, alpha, method, min_n=200)`:
  changes only the cutoff; takes scores, not data, so it cannot refit
  anything; returns the new threshold with n and `min_alpha`; refuses below
  `min_n` unless the caller overrides. The thesis floor was 200 or 20 percent
  of the benign set, whichever was larger.
- `sample_size_curve(scores_benign_deploy, alpha, sizes, n_draws, seed)`:
  for each n in `sizes`, draw `n_draws` slices without replacement from
  independent child generators, recalibrate on each, measure the realised
  FPR on the remainder, and report per n: mean, standard deviation, the 5th
  and 95th percentiles across draws, `|mean - alpha|`, the fraction of draws
  inside a tolerance band, `1 / (n + 1)`, and the source threshold's realised
  FPR as the horizontal reference line. Note in the output that the
  remainder shrinks as n grows, so the evaluation noise
  `sqrt(alpha (1 - alpha) / n_eval)` grows with n.
- Windowed data. If SMD is scored in windows, draw calibration slices as
  contiguous time blocks with a gap before the evaluation block; the thesis's
  random-window slice shared raw events with 96 percent of the evaluated
  windows through stride-1 overlap. Row-level scoring avoids the issue.
- Output discipline. One JSON per run with a fixed schema: alpha, method,
  seed, n_calibration, n_eval, threshold, realised FPR with its interval,
  AUROC before and after (identical by construction; assert it), and a
  version string. Aggregate in a separate step; never hand-edit a table.
- Contamination. The slice is benign by assumption; add
  `contamination_sensitivity(scores_slice, scores_anomalous, fractions)` or
  at least a warning in the README that a contaminated slice moves the
  cutoff upward and lowers the realised FPR below the target while missing
  more anomalies.

## 10. README "Why" paragraph (casual voice; numbers only from the paper draft)

Why does this exist. In my bachelor thesis I compared anomaly detectors on
network, host and sensor data, and the result that stuck with me was not
about which model wins. It was that every detector ranked anomalies on an
unseen host with an AUROC above 0.97 and still raised between 22 and 54
percent false alarms when I carried the threshold over from the training
host. Same model, same ranking, wrong cutoff. On sensor data the same nominal
percentile gave a 3.4-fold range of false-positive rates just from changing
the window length. Recalibrating the threshold on a small benign sample from
the deployment environment brought the classical baselines back to between
4.2 and 5.7 percent in every seed. The paper I am writing with my supervisor
looks at how much deployment data that correction needs and how much
contamination it tolerates; alertcal is the small, dataset-agnostic tool
version of the same idea, so that you can measure your own realised
false-positive rate, recalibrate on a slice, and see the sample-size curve
before you trust a threshold that was set somewhere else.

## Numbers you may cite in alertcal's README (all in the paper draft abstract)

AUROC above 0.97 on the unseen host; a nominal 5 percent threshold realising
between 22 and 54 percent; a 3.4-fold range across window lengths; the three
classical baselines returning to between 4.2 and 5.7 percent per seed after
recalibration. Nothing else from the thesis or its result files.
