"""Six tests for alertcal.core. All fail until core.py is implemented.

Conventions under test (CLAUDE.md):
- higher score = more anomalous; an event alerts when score >= threshold
- alpha is the target false-positive rate, strictly inside (0, 1)
- quantiles are numpy default (linear); seeds are explicit

Closed-form checks follow HANDOFF section 6 (error 2): verify the core computation on a toy
problem whose answer is known before trusting it on real scores.
"""
import math

import numpy as np
import pytest

from alertcal.core import realised_fpr, recalibrate, sample_size_curve, source_threshold

ALPHA = 0.05
Z_95 = 1.6448536269514722  # scipy.stats.norm.ppf(0.95); hard-coded so scipy is not a dependency


def test_realised_fpr_counts_scores_at_or_above_threshold():
    scores = np.array([0.1, 0.5, 0.5, 0.9])
    assert realised_fpr(scores, 0.5) == pytest.approx(0.75)  # ties alert: >= not >
    assert realised_fpr(scores, 0.91) == 0.0
    assert realised_fpr(scores, 0.1) == 1.0
    assert isinstance(realised_fpr(scores, 0.5), float)


def test_source_threshold_matches_closed_form_quantiles():
    rng = np.random.default_rng(0)
    uniform = rng.uniform(0.0, 1.0, size=200_000)
    assert source_threshold(uniform, ALPHA) == pytest.approx(1 - ALPHA, abs=0.01)
    normal = rng.standard_normal(200_000)
    assert source_threshold(normal, ALPHA) == pytest.approx(Z_95, abs=0.03)
    # pins the quantile method: linear interpolation, 1 - alpha = 0.5 is the median
    assert source_threshold(np.array([1.0, 2.0, 3.0, 4.0]), 0.5) == pytest.approx(2.5)


def test_alpha_and_sample_size_are_validated():
    scores = np.arange(100, dtype=float)
    for bad_alpha in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            source_threshold(scores, bad_alpha)
        with pytest.raises(ValueError):
            recalibrate(scores, bad_alpha)
    need = math.ceil(1 / ALPHA)  # 20 at alpha = 0.05
    with pytest.raises(ValueError):
        recalibrate(scores[: need - 1], ALPHA)
    recalibrate(scores[:need], ALPHA)  # exactly enough: must not raise


def test_source_threshold_realises_alpha_without_shift():
    rng = np.random.default_rng(1)
    calibration = rng.standard_normal(5_000)
    evaluation = rng.standard_normal(50_000)  # same environment, disjoint sample
    thr = source_threshold(calibration, ALPHA)
    assert realised_fpr(evaluation, thr) == pytest.approx(ALPHA, abs=0.01)


def test_recalibrate_repairs_realised_fpr_after_shift():
    rng = np.random.default_rng(2)
    source_benign = rng.standard_normal(5_000)
    deploy_benign = 1.0 + 1.5 * rng.standard_normal(20_200)  # location and scale shift
    calibration_slice, evaluation = deploy_benign[:200], deploy_benign[200:]

    thr_fixed = source_threshold(source_benign, ALPHA)
    fpr_fixed = realised_fpr(evaluation, thr_fixed)
    assert fpr_fixed > 0.15  # the carried-over threshold over-alerts badly

    thr_new = recalibrate(calibration_slice, ALPHA)
    fpr_new = realised_fpr(evaluation, thr_new)
    assert fpr_new == pytest.approx(ALPHA, abs=0.03)  # binomial noise at n = 200
    assert thr_new > thr_fixed


def test_sample_size_curve_shape_and_convergence():
    rng = np.random.default_rng(3)
    scores = 1.0 + 1.5 * rng.standard_normal(5_000)
    sizes = [50, 200, 1_000, 10_000]  # 10_000 > len(scores): must be skipped
    curve = sample_size_curve(scores, ALPHA, sizes=sizes, draws=20, seed=0)

    assert set(curve) == {50, 200, 1_000}
    for n, (mean, p5, p95) in curve.items():
        assert 0.0 <= p5 <= mean <= p95 <= 1.0
    spread = {n: p95 - p5 for n, (_, p5, p95) in curve.items()}
    assert spread[1_000] < spread[50]  # more calibration data, tighter realised FPR
    assert curve[1_000][0] == pytest.approx(ALPHA, abs=0.02)
    assert sample_size_curve(scores, ALPHA, sizes=sizes, draws=20, seed=0) == curve  # explicit seed
