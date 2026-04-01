"""
Alpha result evaluation and filtering.

Applies a quality gate (硬过滤) to simulation results before
they're persisted or submitted.  All thresholds are configurable.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class QualityGate:
    """Configurable thresholds for accepting a simulated alpha."""
    min_sharpe: float = 1.25
    min_fitness: float = 1.0
    min_turnover: float = 0.01
    max_turnover: float = 0.70
    min_long_count: int = 50
    min_short_count: int = 50
    max_prod_correlation: float = 0.70   # used in submission pre-check


def passes_gate(metrics: dict, gate: QualityGate) -> tuple[bool, list[str]]:
    """
    Return (passed, reasons_failed).

    metrics should have the shape returned by BrainClient.extract_metrics().
    """
    reasons = []

    sharpe = metrics.get("sharpe")
    fitness = metrics.get("fitness")
    turnover = metrics.get("turnover")
    long_count = metrics.get("long_count", 0) or 0
    short_count = metrics.get("short_count", 0) or 0

    if sharpe is None:
        reasons.append("sharpe is None")
    elif abs(sharpe) < gate.min_sharpe:
        reasons.append(f"sharpe {sharpe:.3f} < {gate.min_sharpe}")

    if fitness is None:
        reasons.append("fitness is None")
    elif abs(fitness) < gate.min_fitness:
        reasons.append(f"fitness {fitness:.3f} < {gate.min_fitness}")

    if turnover is None:
        reasons.append("turnover is None")
    else:
        if turnover < gate.min_turnover:
            reasons.append(f"turnover {turnover:.3f} < {gate.min_turnover}")
        if turnover > gate.max_turnover:
            reasons.append(f"turnover {turnover:.3f} > {gate.max_turnover}")

    if long_count + short_count < gate.min_long_count + gate.min_short_count:
        reasons.append(
            f"instrument count {long_count + short_count} too low"
        )

    passed = len(reasons) == 0
    return passed, reasons


def recommend_decay(turnover: Optional[float], base_decay: int = 0) -> int:
    """
    Suggest a decay value based on observed turnover.
    High-turnover alphas need more decay to reduce transaction costs.
    """
    if turnover is None:
        return base_decay
    if turnover > 0.70:
        return max(base_decay, base_decay * 4)
    if turnover > 0.60:
        return max(base_decay, base_decay * 3 + 3)
    if turnover > 0.50:
        return max(base_decay, base_decay * 3)
    if turnover > 0.40:
        return max(base_decay, base_decay * 2)
    if turnover > 0.35:
        return max(base_decay, base_decay + 4)
    if turnover > 0.30:
        return max(base_decay, base_decay + 2)
    return base_decay


def log_metrics(expression: str, metrics: dict, passed: bool, reasons: list[str]):
    prefix = "PASS" if passed else "FAIL"
    sharpe = metrics.get("sharpe", "N/A")
    fitness = metrics.get("fitness", "N/A")
    turnover = metrics.get("turnover", "N/A")
    log.info(
        "[%s] sharpe=%.3f fitness=%.3f turnover=%.3f | %s",
        prefix,
        sharpe if isinstance(sharpe, float) else 0.0,
        fitness if isinstance(fitness, float) else 0.0,
        turnover if isinstance(turnover, float) else 0.0,
        expression[:60],
    )
    if reasons:
        log.debug("  Fail reasons: %s", "; ".join(reasons))
