"""
Alpha expression factories.

Each factory takes a field (or list of fields) and returns a list of
concrete FASTEXPR expression strings. No simulation logic here — pure
string generation.

Mirrors the day1/day2/day3 pipeline from WQ挖掘脚本/machine_lib.py,
refactored into composable, stateless functions.
"""

from itertools import product as iproduct
from typing import Optional


# ---------------------------------------------------------------------------
# Primitive factories
# ---------------------------------------------------------------------------

def ts_factory(op: str, field: str, windows: list[int] = None) -> list[str]:
    """op(field, window) for each window."""
    windows = windows or [5, 22, 66, 120, 240]
    return [f"{op}({field}, {w})" for w in windows]


def ts_comp_factory(
    op: str,
    field: str,
    factor: str,
    paras: list,
    windows: list[int] = None,
) -> list[str]:
    """op(field, window, factor=para) for each (window, para) combo."""
    windows = windows or [5, 22, 66, 240]
    out = []
    for w, p in iproduct(windows, paras):
        if isinstance(p, float):
            out.append(f"{op}({field}, {w}, {factor}={p:.1f})")
        else:
            out.append(f"{op}({field}, {w}, {factor}={p})")
    return out


def vector_factory(op: str, field: str, vectors: list[str] = None) -> list[str]:
    vectors = vectors or ["cap"]
    return [f"{op}({field}, {v})" for v in vectors]


def twin_field_factory(
    op: str,
    field: str,
    all_fields: list[str],
    windows: list[int] = None,
) -> list[str]:
    """op(field, other_field, window) for each (other, window) combo."""
    windows = windows or [5, 22, 66, 240]
    others = [f for f in all_fields if f != field]
    return [f"{op}({field}, {other}, {w})" for w, other in iproduct(windows, others)]


# ---------------------------------------------------------------------------
# First-order factory  (Day 1 equivalent)
# ---------------------------------------------------------------------------

_TWIN_OPS = {"ts_corr", "ts_covariance", "ts_co_kurtosis", "ts_co_skewness", "ts_theilsen"}

def first_order_factory(
    fields: list[str],
    ops: list[str],
    windows: list[int] = None,
) -> list[str]:
    """
    Expand every (field, op) pair into concrete expressions.
    Equivalent to machine_lib.first_order_factory.

    For ts_* ops: generates ts_op(field, window) for each window.
    For twin ops:  generates op(field, other, window) for each (other, window).
    For basic ops: generates op(field).
    """
    windows = windows or [5, 22, 66, 120, 240]
    out: list[str] = []

    for field in fields:
        out.append(field)   # raw field is itself a trivial 0-th order signal
        for op in ops:
            if op == "ts_percentage":
                out += ts_comp_factory(op, field, "percentage", [0.5], windows)
            elif op == "ts_decay_exp_window":
                out += ts_comp_factory(op, field, "factor", [0.5], windows)
            elif op == "ts_moment":
                out += ts_comp_factory(op, field, "k", [2, 3, 4], windows)
            elif op == "ts_entropy":
                out += ts_comp_factory(op, field, "buckets", [10], windows)
            elif op in _TWIN_OPS:
                out += twin_field_factory(op, field, fields, windows)
            elif op.startswith("ts_") or op == "inst_tvr":
                out += ts_factory(op, field, windows)
            elif op.startswith("vector"):
                out += vector_factory(op, field)
            elif op == "signed_power":
                out.append(f"{op}({field}, 2)")
            elif op == "normalize":
                out.append(f"{op}({field}, useStd=false, limit=0.0)")
            else:
                out.append(f"{op}({field})")

    return out


# ---------------------------------------------------------------------------
# Group factory  (used in Day 2)
# ---------------------------------------------------------------------------

# Group datasets by region — mirrors machine_lib.group_factory
_GROUPS_BY_REGION: dict[str, list[str]] = {
    "usa": [
        "market", "sector", "industry", "subindustry",
        "bucket(rank(cap), range='0.1, 1, 0.1')",
        "bucket(rank(assets), range='0.1, 1, 0.1')",
        "bucket(group_rank(cap, sector), range='0.1, 1, 0.1')",
        "bucket(group_rank(assets, sector), range='0.1, 1, 0.1')",
        "bucket(rank(ts_std_dev(returns, 20)), range='0.1, 1, 0.1')",
        "bucket(rank(close*volume), range='0.1, 1, 0.1')",
        # pv13 datasets
        "pv13_h_min2_3000_sector", "pv13_r2_min20_3000_sector",
        "pv13_r2_min2_3000_sector", "pv13_h_min2_focused_pureplay_3000_sector",
    ],
    "chn": [
        "market", "sector", "industry", "subindustry",
        "pv13_h_min2_sector", "pv13_parent", "pv13_level",
        "sta1_top3000c30", "sta1_top3000c20", "sta1_top3000c10",
    ],
    "eur": [
        "market", "sector", "industry", "subindustry",
        "pv13_5_sector", "pv13_2_sector",
        "sta1_allc10", "sta1_allc2", "sta1_top1200c2",
    ],
}

def group_factory(
    op: str,
    field: str,
    region: str = "usa",
    extra_groups: Optional[list[str]] = None,
) -> list[str]:
    """
    op(field, densify(group)) for every group in the region's group list.
    Equivalent to machine_lib.group_factory.
    """
    groups = list(_GROUPS_BY_REGION.get(region.lower(), _GROUPS_BY_REGION["usa"]))
    if extra_groups:
        groups += extra_groups

    out = []
    for group in groups:
        if op.startswith("group_vector"):
            out.append(f"{op}({field}, cap, densify({group}))")
        elif op.startswith("group_percentage"):
            out.append(f"{op}({field}, densify({group}), percentage=0.5)")
        else:
            out.append(f"{op}({field}, densify({group}))")
    return out


def group_second_order_factory(
    first_order_exprs: list[str],
    group_ops: list[str],
    region: str = "usa",
) -> list[str]:
    """
    For each first-order expression, apply every group op.
    Equivalent to machine_lib.get_group_second_order_factory.
    Used in Day 2.
    """
    out = []
    for expr in first_order_exprs:
        for op in group_ops:
            out += group_factory(op, expr, region)
    return out


# ---------------------------------------------------------------------------
# trade_when factory  (Day 3)
# ---------------------------------------------------------------------------

_OPEN_EVENTS = [
    "ts_arg_max(volume, 5) == 0",
    "ts_corr(close, volume, 20) < 0",
    "ts_corr(close, volume, 5) < 0",
    "ts_mean(volume, 10) > ts_mean(volume, 60)",
    "group_rank(ts_std_dev(returns, 60), sector) > 0.7",
    "ts_zscore(returns, 60) > 2",
    "ts_arg_min(volume, 5) > 3",
    "ts_std_dev(returns, 5) > ts_std_dev(returns, 20)",
    "ts_arg_max(close, 5) == 0",
    "ts_arg_max(close, 20) == 0",
    "ts_corr(close, volume, 5) > 0.3",
    "ts_corr(close, volume, 20) > 0.3",
    "ts_regression(returns, ts_step(20), 20, lag=0, rettype=2) > 0",
    "ts_regression(returns, ts_step(5), 5, lag=0, rettype=2) > 0",
    "pcr_oi_270 < 1",
]

_EXIT_EVENTS = [
    "abs(returns) > 0.1",
    "-1",
]

def trade_when_factory(
    field: str,
    open_events: Optional[list[str]] = None,
    exit_events: Optional[list[str]] = None,
) -> list[str]:
    """
    trade_when(open_event, field, exit_event) for each (open, exit) pair.
    Equivalent to machine_lib.trade_when_factory.
    Used in Day 3.
    """
    opens = open_events or _OPEN_EVENTS
    exits = exit_events or _EXIT_EVENTS

    # Include field-specific regression events
    field_opens = list(opens) + [
        f"ts_regression(returns, {field}, 5, lag=0, rettype=2) > 0",
        f"ts_regression(returns, {field}, 20, lag=0, rettype=2) > 0",
    ]

    return [
        f"trade_when({oe}, {field}, {ee})"
        for oe in field_opens
        for ee in exits
    ]


# ---------------------------------------------------------------------------
# prune  (inter-stage field deduplication)
# ---------------------------------------------------------------------------

def prune(
    alpha_decay_pairs: list[tuple[str, int]],
    field_prefix: str,
    keep_per_field: int = 5,
) -> list[tuple[str, int]]:
    """
    Given [(expression, decay), ...], keep only the top `keep_per_field`
    expressions per unique field suffix (identified by field_prefix).

    Equivalent to machine_lib.prune.
    """
    counts: dict[str, int] = {}
    out = []
    for expr, decay in alpha_decay_pairs:
        if field_prefix in expr:
            field_key = expr.split(field_prefix)[-1].split(",")[0].split(")")[0]
        else:
            field_key = expr
        counts.setdefault(field_key, 0)
        if counts[field_key] < keep_per_field:
            counts[field_key] += 1
            out.append((expr, decay))
    return out


# ---------------------------------------------------------------------------
# Default operator sets
# ---------------------------------------------------------------------------

BASIC_OPS = [
    "reverse", "inverse", "rank", "zscore", "normalize",
    "scale", "log", "sqrt", "abs",
]

TS_OPS = [
    "ts_delta", "ts_sum", "ts_product", "ts_std_dev", "ts_mean",
    "ts_arg_min", "ts_arg_max", "ts_scale", "ts_rank", "ts_zscore",
    "ts_returns", "ts_ir", "ts_skewness", "ts_kurtosis",
    "ts_min_diff", "ts_max_diff",
]

GROUP_OPS = [
    "group_rank", "group_neutralize", "group_zscore",
    "group_mean", "group_sum", "group_std_dev",
]

ALL_OPS = BASIC_OPS + TS_OPS
