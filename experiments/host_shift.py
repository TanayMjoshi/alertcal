#!/usr/bin/env python
"""Host-shift demo on the Server Machine Dataset (SPEC, section Demo).

1. Fit StandardScaler + IsolationForest on the first 85 percent (time order) of the source
   machine's train file, which is benign. The last 15 percent is the benign validation slice
   (HANDOFF section 2: calibrate on held-out benign data, never on the fit split).
2. source_threshold on the validation scores at alpha.
3. Score the deployment machine's test file with the SAME frozen scaler and model
   (HANDOFF section 6, error 1). Split by label into benign and anomalous.
4. Realised FPR and TPR of the fixed threshold.
5. Draw n_cal random benign deployment points, recalibrate on them, remove them from the
   evaluation set, report FPR and TPR again on the same evaluation set for both thresholds.
6. sample_size_curve on the deployment benign set. Plot both panels.

Prints three numbers: AUROC on the deployment host (unchanged by recalibration), the realised
FPR of the fixed threshold, and the realised FPR after recalibration.
Writes experiments/host_shift.png and experiments/host_shift.json.

Usage:  python experiments/host_shift.py [--source machine-1-1] [--deploy machine-1-2]
                                         [--alpha 0.05] [--n-cal 200] [--seed 42] [--val-frac 0.15]
Data:   `make data` first (scripts/get_smd.sh puts the files in data/smd/).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from alertcal import __version__, realised_fpr, recalibrate, sample_size_curve, source_threshold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "smd"
CURVE_SIZES = (50, 100, 200, 500, 1000, 2000, 5000)
CURVE_DRAWS = 20

# Colours: dataviz reference palette, categorical slots 1 and 2 (validated CVD-safe pair).
# Blue = source side / fixed threshold. Orange = deployment side / recalibrated threshold.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def load(split: str, machine: str) -> np.ndarray:
    path = DATA / split / f"{machine}.txt"
    if not path.exists():
        sys.exit(f"missing {path.relative_to(ROOT)}: run `make data` (or scripts/get_smd.sh {machine})")
    return np.loadtxt(path, delimiter="," if split != "test_label" else None)


def anomaly_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """scikit-learn's score_samples is 'higher = more normal'; negate once, here (HANDOFF section 1)."""
    return -model.score_samples(X)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default="machine-1-1", help="SMD machine whose train file fits the model")
    ap.add_argument("--deploy", default="machine-1-2", help="SMD machine whose test file is the deployment")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-cal", type=int, default=200, help="benign deployment points used to recalibrate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15, help="last fraction of source train held out")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "host_shift"), help="output stem (.png, .json)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)  # one explicit generator at the boundary; pass it down

    # 1. source: fit on the first 85 percent in time order, validate on the last 15 percent
    src_train = load("train", args.source)
    n_val = int(round(len(src_train) * args.val_frac))
    fit, val = src_train[:-n_val], src_train[-n_val:]
    scaler = StandardScaler().fit(fit)
    model = IsolationForest(
        n_estimators=200, random_state=int(rng.integers(2**31 - 1)), n_jobs=-1
    ).fit(scaler.transform(fit))
    val_scores = anomaly_scores(model, scaler.transform(val))

    # 2. threshold set at home
    thr_fixed = source_threshold(val_scores, args.alpha)

    # 3. deployment: same scaler, same model, nothing refitted
    dep_X = load("test", args.deploy)
    dep_y = load("test_label", args.deploy).astype(int)
    dep_scores = anomaly_scores(model, scaler.transform(dep_X))
    benign, anomalous = dep_scores[dep_y == 0], dep_scores[dep_y == 1]
    auroc = float(roc_auc_score(dep_y, dep_scores))
    if auroc < 0.5:
        print(f"WARNING: AUROC {auroc:.3f} < 0.5 on the deployment host. Score sign flipped? (HANDOFF section 1)")

    # 5. calibration slice: drawn once, removed from evaluation; both thresholds judged on the same rest
    if args.n_cal >= benign.size:
        sys.exit(f"--n-cal {args.n_cal} leaves nothing to evaluate on ({benign.size} benign points)")
    cal_mask = np.zeros(benign.size, dtype=bool)
    cal_mask[rng.choice(benign.size, size=args.n_cal, replace=False)] = True
    cal_slice, benign_eval = benign[cal_mask], benign[~cal_mask]
    thr_new = recalibrate(cal_slice, args.alpha)

    # 4 + 5. realised rates. realised_fpr on the anomalous scores is the true-positive rate.
    fpr_fixed = realised_fpr(benign_eval, thr_fixed)
    tpr_fixed = realised_fpr(anomalous, thr_fixed)
    fpr_new = realised_fpr(benign_eval, thr_new)
    tpr_new = realised_fpr(anomalous, thr_new)

    # 6. how much benign deployment data the correction needs
    curve = sample_size_curve(benign, args.alpha, sizes=CURVE_SIZES, draws=CURVE_DRAWS, seed=args.seed)

    # ---- report ---------------------------------------------------------------------------
    print(f"alertcal {__version__} host-shift demo   source={args.source}  deploy={args.deploy}  "
          f"alpha={args.alpha}  seed={args.seed}")
    print(f"fit {len(fit)} rows | validation {len(val)} rows | deployment benign {benign.size} rows, "
          f"anomalous {anomalous.size} rows")
    print()
    print(f"  AUROC on deployment host            {auroc:.4f}   (ranking; identical before and after)")
    print(f"  fixed threshold   thr={thr_fixed:8.4f}   FPR {fpr_fixed:.3f}   TPR {tpr_fixed:.3f}")
    print(f"  recalibrated      thr={thr_new:8.4f}   FPR {fpr_new:.3f}   TPR {tpr_new:.3f}"
          f"   n_calibration={args.n_cal}  n_eval={benign_eval.size}")
    if fpr_fixed < 2 * args.alpha:
        print(f"  note: weak shift on {args.deploy} (fixed FPR < 2 alpha). Try another --deploy machine.")
    print()
    print(f"sample-size curve: realised FPR of a recalibrated threshold, {CURVE_DRAWS} draws per n, "
          f"evaluated on the benign points not in the draw")
    print(f"  {'n':>6} {'mean':>7} {'p5':>7} {'p95':>7}")
    for n in sorted(curve):
        m, lo, hi = curve[n]
        print(f"  {n:>6} {m:7.3f} {lo:7.3f} {hi:7.3f}")

    # ---- outputs --------------------------------------------------------------------------
    out = Path(args.out)
    result = {
        "version": __version__, "source": args.source, "deploy": args.deploy,
        "alpha": args.alpha, "seed": args.seed, "val_frac": args.val_frac,
        "n_fit": int(len(fit)), "n_val": int(len(val)),
        "n_deploy_benign": int(benign.size), "n_deploy_anomalous": int(anomalous.size),
        "n_calibration": args.n_cal, "n_eval": int(benign_eval.size),
        "auroc": auroc,
        "threshold_fixed": thr_fixed, "fpr_fixed": fpr_fixed, "tpr_fixed": tpr_fixed,
        "fpr_fixed_all_benign": realised_fpr(benign, thr_fixed),
        "threshold_recalibrated": thr_new, "fpr_recalibrated": fpr_new, "tpr_recalibrated": tpr_new,
        "curve_draws": CURVE_DRAWS,
        "curve": {str(n): {"mean": m, "p5": lo, "p95": hi} for n, (m, lo, hi) in curve.items()},
    }
    out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    plot(args, val_scores, benign, thr_fixed, thr_new, fpr_fixed, fpr_new, curve, out.with_suffix(".png"))
    print(f"\nwrote {out.with_suffix('.png').relative_to(ROOT)}, {out.with_suffix('.json').relative_to(ROOT)}")


def plot(args, val_scores, benign, thr_fixed, thr_new, fpr_fixed, fpr_new, curve, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)

    # left: the two benign score distributions and the two thresholds
    pooled = np.concatenate([val_scores, benign])
    bins = np.linspace(pooled.min(), np.quantile(pooled, 0.995), 80)
    ax1.hist(val_scores, bins=bins, density=True, histtype="stepfilled", color=BLUE, alpha=0.22,
             edgecolor=BLUE, linewidth=1.2, label=f"source benign ({args.source}, validation)")
    ax1.hist(benign, bins=bins, density=True, histtype="stepfilled", color=ORANGE, alpha=0.22,
             edgecolor=ORANGE, linewidth=1.2, label=f"deployment benign ({args.deploy})")
    ax1.axvline(thr_fixed, color=BLUE, linestyle="--", linewidth=1.5)
    ax1.axvline(thr_new, color=ORANGE, linestyle="--", linewidth=1.5)
    ymax = ax1.get_ylim()[1]
    left_first = thr_fixed <= thr_new
    ax1.annotate(f"fixed threshold\nrealised FPR {fpr_fixed:.2f}", xy=(thr_fixed, ymax * 0.97),
                 xytext=(-6 if left_first else 6, 0), textcoords="offset points",
                 ha="right" if left_first else "left", va="top", fontsize=8.5, color=INK)
    ax1.annotate(f"recalibrated (n={args.n_cal})\nrealised FPR {fpr_new:.2f}", xy=(thr_new, ymax * 0.97),
                 xytext=(6 if left_first else -6, 0), textcoords="offset points",
                 ha="left" if left_first else "right", va="top", fontsize=8.5, color=INK)
    ax1.set_title("Same model, same ranking: benign scores shift, the fixed threshold does not",
                  fontsize=10.5, color=INK, loc="left")
    ax1.set_xlabel("anomaly score  (higher = more anomalous)", color=INK2, fontsize=9)
    ax1.set_ylabel("density", color=INK2, fontsize=9)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, 0.82))

    # right: realised FPR against calibration slice size
    ns = sorted(curve)
    means = [curve[n][0] for n in ns]
    p5s = [curve[n][1] for n in ns]
    p95s = [curve[n][2] for n in ns]
    ax2.axhline(fpr_fixed, color=BLUE, linestyle="--", linewidth=1.5,
                label=f"fixed threshold: realised FPR {fpr_fixed:.2f}")
    ax2.fill_between(ns, p5s, p95s, color=ORANGE, alpha=0.18, linewidth=0,
                     label=f"recalibrated: 5th to 95th percentile of {CURVE_DRAWS} draws")
    ax2.plot(ns, means, color=ORANGE, linewidth=2, marker="o", markersize=5.5,
             markeredgecolor="white", markeredgewidth=1, label="recalibrated: mean realised FPR")
    ax2.axhline(args.alpha, color=INK2, linestyle=":", linewidth=1.2, label=f"target alpha = {args.alpha}")
    ax2.set_xscale("log")
    ax2.set_xticks(ns)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.tick_params(axis="x", which="minor", length=0)
    ax2.set_ylim(0, max(fpr_fixed * 1.15, max(p95s) * 1.3, args.alpha * 3))
    ax2.set_title("How much benign deployment data the correction needs", fontsize=10.5, color=INK, loc="left")
    ax2.set_xlabel("calibration slice size n  (benign deployment points, drawn without replacement)",
                   color=INK2, fontsize=9)
    ax2.set_ylabel("realised false-positive rate", color=INK2, fontsize=9)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper right")

    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
