"""alertcal.core: the four functions. Everything else in the package is plumbing around these.

Conventions (CLAUDE.md, do not change):
- Higher score = more anomalous. An event alerts when ``score >= threshold`` (ties alert).
- ``alpha`` is the target false-positive rate, strictly inside (0, 1). A rate, not a percent.
- Quantiles use numpy's default ``method="linear"``, which is what the thesis used
  (HANDOFF section 2).
- Seeds are explicit: ``np.random.default_rng(seed)``. Never global state.

All functions take arrays of *scores*, never raw data or a model, so nothing here can refit
anything. That is deliberate (HANDOFF section 6, error 1): recalibration must change the
cutoff and nothing else.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np

__all__ = ["source_threshold", "realised_fpr", "recalibrate", "sample_size_curve"]


def source_threshold(benign_scores, alpha: float) -> float:
    """Threshold at the (1 - alpha) quantile of benign scores from the *source* environment.

    Maths: ``Q_{1-alpha}(benign_scores)``, i.e. ``np.quantile(x, 1 - alpha)`` (linear).
    Alerting at ``score >= threshold`` then flags about ``alpha`` of benign events *in the
    environment the scores came from*. The number encodes the location and scale of the score
    distribution there, which is exactly why it does not transfer.

    Args:
        benign_scores: 1-D array of scores of benign events. Use a held-out benign validation
            slice, never the data the detector was fitted on (HANDOFF section 2).
        alpha: target false-positive rate, ``0 < alpha < 1``.

    Returns:
        The threshold as a plain Python float.

    Raises:
        ValueError: if ``alpha`` is not strictly inside (0, 1).
    """
    raise NotImplementedError


def realised_fpr(benign_scores, threshold: float) -> float:
    """Fraction of benign scores that alert at ``threshold``: ``mean(benign_scores >= threshold)``.

    This is a measurement, not a calibration. Pass benign scores that were NOT used to set
    ``threshold``; HANDOFF section 3 and section 6: never report a realised rate on the data
    the cutoff was chosen on. Applied to *anomalous* scores the same formula is the
    true-positive rate, which the demo uses.

    Args:
        benign_scores: 1-D array of scores of benign events.
        threshold: the cutoff to test.

    Returns:
        A float in [0, 1].
    """
    raise NotImplementedError


def recalibrate(benign_deployment_scores, alpha: float) -> float:
    """Threshold at the (1 - alpha) quantile of benign scores from the *deployment* environment.

    Same quantile as :func:`source_threshold`. What differs is what goes in (a small benign
    slice from where the detector now runs) and the sample-size guard. Only the cutoff changes;
    scaler and model stay frozen, so the ranking (AUROC) is untouched by construction.

    Args:
        benign_deployment_scores: 1-D array of scores of events known to be benign in the
            deployment environment. Remove these from any set you later evaluate on.
        alpha: target false-positive rate, ``0 < alpha < 1``.

    Returns:
        The new threshold as a plain Python float.

    Raises:
        ValueError: if ``alpha`` is outside (0, 1), or if there are fewer than
            ``math.ceil(1 / alpha)`` scores. Below that the (1 - alpha) quantile is
            meaningless: at alpha = 0.05 you need at least 20 scores (the thesis used a
            floor of 200, or 20 percent of the benign set, whichever was larger).
    """
    raise NotImplementedError


def sample_size_curve(
    scores,
    alpha: float,
    sizes: Sequence[int] = (50, 100, 200, 500, 1000, 2000),
    draws: int = 20,
    seed: int = 0,
) -> Dict[int, Tuple[float, float, float]]:
    """How the realised FPR of a recalibrated threshold depends on the calibration slice size.

    For each ``n`` in ``sizes``, ``draws`` times: draw ``n`` scores without replacement,
    :func:`recalibrate` on the draw, measure :func:`realised_fpr` on the scores *not* in the
    draw, and summarise the ``draws`` realised rates as ``(mean, p5, p95)``.

    Args:
        scores: 1-D array of benign scores from the deployment environment.
        alpha: target false-positive rate.
        sizes: calibration slice sizes to try. Sizes larger than the data are skipped
            (so is ``n == len(scores)``, which would leave nothing to evaluate on).
        draws: number of independent draws per size.
        seed: explicit seed. Hint: ``np.random.SeedSequence(seed).spawn(len(sizes))`` gives
            one independent child stream per size (HANDOFF section 5); a single
            ``np.random.default_rng(seed)`` used in order also reproduces the whole curve.

    Returns:
        ``{n: (mean, p5, p95)}`` with plain Python ints and floats, one entry per size kept.

    Decision pending: SPEC's table says "measure realised FPR on the full set". This docstring
    says the remainder, which is what the demo does for its 200-point slice and what HANDOFF
    section 6 insists on. Pick one, update SPEC, delete this paragraph.
    """
    raise NotImplementedError
