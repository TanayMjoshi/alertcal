"""alertcal: recalibrate anomaly-alert thresholds after deployment shift.

Scores in, thresholds out. Model-agnostic: nothing here ever sees raw data or a model.
"""
from .core import realised_fpr, recalibrate, sample_size_curve, source_threshold

__version__ = "0.1.0"
__all__ = ["source_threshold", "realised_fpr", "recalibrate", "sample_size_curve", "__version__"]
