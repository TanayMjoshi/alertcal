"""Command-line face of alertcal.

    alertcal calibrate <csv> [--alpha 0.05] [--column score]
    alertcal evaluate  <csv> --threshold T [--column score]
    alertcal curve     <csv> [--alpha 0.05] [--sizes 50,100,...] [--draws 20] [--seed 0] [--column score]

The CSV needs a header row and one numeric column of anomaly scores (higher = more anomalous).
Each command prints one JSON object to stdout. Errors go to stderr with exit code 1 or 2.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys

import numpy as np

from . import __version__
from .core import realised_fpr, recalibrate, sample_size_curve


def read_scores(path: str, column: str) -> np.ndarray:
    """Read one numeric column from a CSV with a header row. Rejects NaN and inf."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise SystemExit(f"column {column!r} not found in {path}; columns: {reader.fieldnames}")
        values = []
        for line_no, row in enumerate(reader, start=2):
            try:
                values.append(float(row[column]))
            except (TypeError, ValueError):
                raise SystemExit(f"{path}:{line_no}: {row[column]!r} is not a number") from None
    if not values:
        raise SystemExit(f"{path}: no data rows")
    scores = np.asarray(values, dtype=float)
    if not np.isfinite(scores).all():
        raise SystemExit(f"{path}: column {column!r} contains NaN or inf")
    return scores


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alertcal",
        description="Recalibrate anomaly-alert thresholds after deployment shift. "
        "Scores in, thresholds and realised false-positive rates out.",
    )
    parser.add_argument("--version", action="version", version=f"alertcal {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("csv", help="CSV file with a header row")
        p.add_argument("--column", default="score", help="score column name (default: score)")

    p = sub.add_parser(
        "calibrate",
        help="threshold at the (1 - alpha) quantile of benign scores",
        description="Threshold at the (1 - alpha) quantile of the benign scores in the CSV. "
        "Same quantile as source_threshold; refuses fewer than ceil(1/alpha) scores.",
    )
    common(p)
    p.add_argument("--alpha", type=float, default=0.05, help="target false-positive rate in (0, 1)")

    p = sub.add_parser(
        "evaluate",
        help="realised false-positive rate of a threshold on benign scores",
        description="Fraction of the benign scores in the CSV at or above --threshold. "
        "Use scores that were NOT used to set the threshold.",
    )
    common(p)
    p.add_argument("--threshold", type=float, required=True)

    p = sub.add_parser(
        "curve",
        help="realised FPR of a recalibrated threshold versus calibration slice size",
        description="For each size n: draw n scores without replacement (--draws times), "
        "recalibrate on the draw, measure the realised FPR on the rest.",
    )
    common(p)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--sizes", default="50,100,200,500,1000,2000", help="comma-separated slice sizes")
    p.add_argument("--draws", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    scores = read_scores(args.csv, args.column)
    try:
        if args.command == "calibrate":
            out = {
                "threshold": recalibrate(scores, args.alpha),
                "alpha": args.alpha,
                "n": int(scores.size),
            }
        elif args.command == "evaluate":
            out = {
                "fpr": realised_fpr(scores, args.threshold),
                "threshold": args.threshold,
                "n": int(scores.size),
            }
        else:
            sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
            curve = sample_size_curve(scores, args.alpha, sizes=sizes, draws=args.draws, seed=args.seed)
            out = {
                "alpha": args.alpha,
                "draws": args.draws,
                "seed": args.seed,
                "n": int(scores.size),
                "curve": {str(n): {"mean": m, "p5": lo, "p95": hi} for n, (m, lo, hi) in curve.items()},
            }
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
