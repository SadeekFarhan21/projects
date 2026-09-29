"""expope: an experimentation and off-policy evaluation engine.

Subpackages
-----------
metrics  DuckDB event-log schema and SQL metric definitions that emit sufficient statistics.
stats    From-scratch A/B statistics (Welch, delta method, CUPED, SRM, Holm, BH, mSPRT).
ope      From-scratch off-policy estimators (IPS, SNIPS, DM, DR, Switch-DR) and bootstrap CIs.
sim      Synthetic data generators with known ground truth.
"""

__version__ = "0.1.0"
