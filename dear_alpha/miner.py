"""
Template-based systematic enumeration miner (模板枚举挖掘).

Four mining strategies, all returning list[MineResult]:

1. TemplateMiner  – Cartesian product enumeration with checkpoint resume.
2. LayeredMiner   – Two-pass pruning: coarse field screen → fine param sweep.
3. BayesianMiner  – Optuna TPE over numeric/categorical params.
4. PipelineMiner  – Multi-stage day1→day2→day3 recursive expansion pipeline.

Checkpoint / resume:
  Every miner that does sequential simulation writes progress to a JSON file
  after each completed simulation.  On restart it skips already-done work.
"""

import itertools
import json
import logging
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

class Checkpoint:
    """
    Minimal key-value store backed by a JSON file.
    Stores {stage_name: {"index": int, "results": [...]}} so a run can
    resume exactly where it left off after a crash or interruption.
    """

    def __init__(self, path: str = "results/progress.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, state: dict):
        self.path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    def get(self, stage: str) -> dict:
        return self._load().get(stage, {"index": 0, "results": []})

    def update(self, stage: str, index: int, result: Optional[dict] = None):
        state = self._load()
        entry = state.setdefault(stage, {"index": 0, "results": []})
        entry["index"] = index
        if result is not None:
            entry["results"].append(result)
        self._save(state)

    def results(self, stage: str) -> list[dict]:
        return self.get(stage)["results"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ParameterAxis:
    """Describes one dimension to vary in a template."""
    name: str           # human-readable label
    placeholder: str    # the token in the template to replace, e.g. "{window}"
    values: list[Any]   # candidate values


@dataclass
class MineResult:
    expression: str
    metrics: dict
    passed: bool


# ---------------------------------------------------------------------------
# Built-in parameter grids
# ---------------------------------------------------------------------------

WINDOW_AXIS = ParameterAxis(
    name="ts_window",
    placeholder="{window}",
    values=[5, 10, 22, 44, 66, 120, 240],
)

DECAY_AXIS = ParameterAxis(
    name="decay",
    placeholder="{decay}",
    values=[0, 2, 4, 6, 10, 15, 20],
)

NEUT_AXIS = ParameterAxis(
    name="neutralization",
    placeholder="{neut}",
    values=["NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"],
)


# ---------------------------------------------------------------------------
# Core miner
# ---------------------------------------------------------------------------

class TemplateMiner:
    """
    Mines a template expression by iterating over parameter combinations.

    Example usage:
        template = "group_rank(ts_mean({field}, {window}), densify(sector))"
        axes = [
            ParameterAxis("field", "{field}", ["close", "volume", "returns"]),
            WINDOW_AXIS,
        ]
        miner = TemplateMiner(brain_client, template, axes)
        results = miner.run(filter_fn=lambda m: m["sharpe"] and m["sharpe"] > 1.25)
    """

    def __init__(
        self,
        brain,                   # BrainClient instance
        template: str,
        axes: list[ParameterAxis],
        sim_settings: Optional[dict] = None,
        checkpoint: Optional[Checkpoint] = None,
        stage: str = "template",
    ):
        self.brain = brain
        self.template = template
        self.axes = axes
        self.sim_settings = sim_settings or {}
        self.checkpoint = checkpoint
        self.stage = stage

    def _expand(self) -> list[str]:
        """Produce all concrete expressions from the template."""
        if not self.axes:
            return [self.template]

        names = [ax.placeholder for ax in self.axes]
        value_lists = [ax.values for ax in self.axes]

        expressions = []
        for combo in itertools.product(*value_lists):
            expr = self.template
            for placeholder, value in zip(names, combo):
                expr = expr.replace(placeholder, str(value))
            expressions.append(expr)

        log.info(
            "Template expanded to %d combinations from %d axes",
            len(expressions),
            len(self.axes),
        )
        return expressions

    def run(self, filter_fn=None, max_expressions: int = 2000) -> list[MineResult]:
        """
        Test all combinations.  filter_fn(metrics: dict) -> bool selects passing ones.
        Resumes from checkpoint if one is provided and progress exists.
        """
        expressions = self._expand()[:max_expressions]
        results: list[MineResult] = []

        start = 0
        if self.checkpoint:
            saved = self.checkpoint.get(self.stage)
            start = saved["index"]
            # Reconstruct already-done results so callers get the full list
            for r in saved["results"]:
                results.append(MineResult(
                    expression=r["expression"],
                    metrics=r["metrics"],
                    passed=r["passed"],
                ))
            if start > 0:
                log.info("[%s] Resuming from index %d / %d", self.stage, start, len(expressions))

        for i, expr in enumerate(expressions):
            if i < start:
                continue
            log.info("[%d/%d] Testing: %s", i + 1, len(expressions), expr[:80])
            raw = self.brain.simulate(expr, **self.sim_settings)
            if raw is None:
                if self.checkpoint:
                    self.checkpoint.update(self.stage, i + 1)
                continue
            metrics = self.brain.extract_metrics(raw)
            passed = filter_fn(metrics) if filter_fn else True
            result = MineResult(expression=expr, metrics=metrics, passed=passed)
            results.append(result)

            if self.checkpoint:
                self.checkpoint.update(self.stage, i + 1, {
                    "expression": expr,
                    "metrics": metrics,
                    "passed": passed,
                })

        passed_count = sum(1 for r in results if r.passed)
        log.info(
            "Mining complete: %d/%d expressions passed filter",
            passed_count, len(results),
        )
        return results


# ---------------------------------------------------------------------------
# Numeric parameter sweep on an existing expression
# ---------------------------------------------------------------------------

def sweep_numeric_params(
    brain,
    expression: str,
    param_ranges: dict[str, tuple],   # {"252": (200, 300, 10)} → from 200 to 300 step 10
    sim_settings: Optional[dict] = None,
    filter_fn=None,
) -> list[MineResult]:
    """
    Given an existing expression, vary its numeric literals.

    param_ranges maps each literal string that appears in the expression to
    a (start, stop, step) tuple.

    Example:
        expression = "ts_mean(close, 252) / ts_std_dev(close, 252)"
        param_ranges = {"252": (100, 300, 20)}
    """
    # Build axes by finding occurrences of each param literal
    axes = []
    template = expression
    for literal, (start, stop, step) in param_ranges.items():
        placeholder = f"{{P_{literal}}}"
        template = template.replace(str(literal), placeholder, 1)  # replace first occurrence
        values = list(range(int(start), int(stop) + 1, int(step)))
        axes.append(ParameterAxis(name=f"param_{literal}", placeholder=placeholder, values=values))

    miner = TemplateMiner(brain, template, axes, sim_settings=sim_settings or {})
    return miner.run(filter_fn=filter_fn)


# ---------------------------------------------------------------------------
# Built-in template library
# ---------------------------------------------------------------------------

def get_template_library() -> list[tuple[str, list[ParameterAxis]]]:
    """
    A curated set of template expressions with their parameter axes.
    These templates encode recurring structural patterns in alpha research.
    """
    field_axis_price = ParameterAxis(
        "price_field", "{field}",
        ["close", "vwap", "high", "low", "open"],
    )
    field_axis_fundamental = ParameterAxis(
        "fundamental_field", "{field}",
        ["pe", "pb", "ps", "roe", "roa", "gm", "npm", "eps", "cashflow_op"],
    )
    field_axis_volume = ParameterAxis(
        "volume_field", "{field}",
        ["volume", "adv20", "turnover_ratio"],
    )

    return [
        # ── Momentum ──────────────────────────────────────────────────────
        (
            "group_rank(ts_mean({field}, {window}), densify(sector))",
            [field_axis_price, WINDOW_AXIS],
        ),
        (
            "-ts_returns({field}, {window})",
            [field_axis_price, WINDOW_AXIS],
        ),
        # ── Reversal ──────────────────────────────────────────────────────
        (
            "-rank(ts_returns(close, {window}))",
            [WINDOW_AXIS],
        ),
        (
            "rank(ts_delta({field}, 1)) - rank(ts_mean(ts_delta({field}, 1), {window}))",
            [field_axis_price, WINDOW_AXIS],
        ),
        # ── Volatility ────────────────────────────────────────────────────
        (
            "-rank(ts_std_dev(returns, {window}))",
            [WINDOW_AXIS],
        ),
        (
            "group_rank(-ts_std_dev({field}, {window}), densify(sector))",
            [field_axis_price, WINDOW_AXIS],
        ),
        # ── Value ─────────────────────────────────────────────────────────
        (
            "-rank(winsorize(ts_backfill({field}, 120), std=4))",
            [field_axis_fundamental],
        ),
        (
            "group_rank(-winsorize(ts_backfill({field}, 120), std=4), densify(industry))",
            [field_axis_fundamental],
        ),
        # ── Volume / Liquidity ────────────────────────────────────────────
        (
            "-rank(ts_mean({field}, {window}) / ts_std_dev({field}, {window}))",
            [field_axis_volume, WINDOW_AXIS],
        ),
        # ── Correlation ───────────────────────────────────────────────────
        (
            "rank(ts_corr(returns, {field}, {window}))",
            [field_axis_volume, WINDOW_AXIS],
        ),
    ]


# ---------------------------------------------------------------------------
# Strategy 2: Layered pruning miner  (分层剪枝)
# ---------------------------------------------------------------------------

class LayeredMiner:
    """
    Two-pass mining that avoids testing the full Cartesian product.

    Pass 1 – Coarse screen (field selection)
        Test each field with a small representative set of windows.
        Rank fields by mean |Sharpe| across all tested windows.
        Keep only the top `keep_fields` fields.

    Pass 2 – Fine sweep (parameter optimisation)
        For the surviving fields, test the full window grid.

    This reduces simulation count by roughly:
        (total_fields / keep_fields) × (full_windows / coarse_windows)

    Example: 20 fields × 7 windows = 140 → coarse (20×3) + fine (5×7) = 95
             Saving ~32% simulations, but more importantly the saving is front-
             loaded: bad fields are eliminated early.
    """

    def __init__(
        self,
        brain,
        template: str,
        field_axis: ParameterAxis,        # the {field} axis
        window_axis: ParameterAxis,       # the {window} axis
        coarse_windows: Optional[list] = None,
        keep_fields: int = 5,
        sim_settings: Optional[dict] = None,
    ):
        self.brain = brain
        self.template = template
        self.field_axis = field_axis
        self.window_axis = window_axis
        self.coarse_windows = coarse_windows or [5, 22, 120]
        self.keep_fields = keep_fields
        self.sim_settings = sim_settings or {}

    def _simulate(self, expr: str) -> Optional[dict]:
        raw = self.brain.simulate(expr, **self.sim_settings)
        if raw is None:
            return None
        return self.brain.extract_metrics(raw)

    def _fill(self, field: str, window) -> str:
        return (
            self.template
            .replace(self.field_axis.placeholder, str(field))
            .replace(self.window_axis.placeholder, str(window))
        )

    def run(self, filter_fn: Optional[Callable] = None) -> list[MineResult]:
        # ── Pass 1: coarse screen ────────────────────────────────────────
        field_scores: dict[str, list[float]] = {}

        total_fields = len(self.field_axis.values)
        log.info(
            "[Layered] Pass-1 coarse screen: %d fields × %d windows",
            total_fields, len(self.coarse_windows),
        )

        for field in self.field_axis.values:
            sharpes = []
            for w in self.coarse_windows:
                expr = self._fill(field, w)
                metrics = self._simulate(expr)
                if metrics and metrics.get("sharpe") is not None:
                    sharpes.append(abs(metrics["sharpe"]))
            field_scores[field] = sharpes
            mean_s = sum(sharpes) / len(sharpes) if sharpes else 0.0
            log.info("  field=%-20s  mean|sharpe|=%.3f", field, mean_s)

        # Rank by mean |Sharpe|
        ranked = sorted(
            field_scores.items(),
            key=lambda kv: sum(kv[1]) / len(kv[1]) if kv[1] else 0.0,
            reverse=True,
        )
        surviving_fields = [f for f, _ in ranked[: self.keep_fields]]
        log.info(
            "[Layered] Pass-1 done. Top-%d fields: %s",
            self.keep_fields, surviving_fields,
        )

        # ── Pass 2: fine sweep on survivors ─────────────────────────────
        log.info(
            "[Layered] Pass-2 fine sweep: %d fields × %d windows",
            len(surviving_fields), len(self.window_axis.values),
        )
        results: list[MineResult] = []

        for field in surviving_fields:
            for w in self.window_axis.values:
                expr = self._fill(field, w)
                log.info("  Testing: %s", expr[:80])
                metrics = self._simulate(expr)
                if metrics is None:
                    continue
                passed = filter_fn(metrics) if filter_fn else True
                results.append(MineResult(expression=expr, metrics=metrics, passed=passed))

        passed_count = sum(1 for r in results if r.passed)
        log.info(
            "[Layered] Done: %d/%d passed filter",
            passed_count, len(results),
        )
        return results


# ---------------------------------------------------------------------------
# Strategy 3: Bayesian optimisation miner  (贝叶斯优化)
# ---------------------------------------------------------------------------

class BayesianMiner:
    """
    Uses Optuna (TPE sampler) to search numeric parameter space efficiently.

    Instead of testing all combinations, Optuna builds a probabilistic model
    of which parameter regions produce high Sharpe and directs sampling there.

    Supports:
    - Integer parameters  (e.g. window: 5–240)
    - Float parameters    (e.g. truncation: 0.01–0.20)
    - Categorical params  (e.g. field: ["close", "volume", ...])

    The objective is abs(Sharpe) – you can override it via `objective_fn`.

    Usage:
        param_space = {
            "window": ("int",   5,   240),
            "field":  ("cat",   ["close", "volume", "returns"]),
        }
        template = "group_rank(ts_mean({field}, {window}), densify(sector))"
        miner = BayesianMiner(brain, template, param_space, n_trials=60)
        results = miner.run(filter_fn=lambda m: m["sharpe"] and abs(m["sharpe"]) > 1.25)
    """

    def __init__(
        self,
        brain,
        template: str,
        param_space: dict,
        n_trials: int = 50,
        sim_settings: Optional[dict] = None,
        objective_fn: Optional[Callable[[dict], float]] = None,
        direction: str = "maximize",
        optuna_verbosity: int = 0,   # 0=WARNING, 1=INFO, 2=DEBUG
    ):
        self.brain = brain
        self.template = template
        self.param_space = param_space   # {placeholder: (type, *args)}
        self.n_trials = n_trials
        self.sim_settings = sim_settings or {}
        self.objective_fn = objective_fn or _default_objective
        self.direction = direction
        self.optuna_verbosity = optuna_verbosity
        self._results: list[MineResult] = []

    def _build_expr(self, params: dict) -> str:
        expr = self.template
        for placeholder, value in params.items():
            expr = expr.replace(placeholder, str(value))
        return expr

    def _objective(self, trial) -> float:
        params = {}
        for placeholder, spec in self.param_space.items():
            kind = spec[0]
            name = placeholder.strip("{}")   # use as trial param name
            if kind == "int":
                _, lo, hi = spec
                params[placeholder] = trial.suggest_int(name, lo, hi)
            elif kind == "float":
                _, lo, hi = spec
                params[placeholder] = trial.suggest_float(name, lo, hi)
            elif kind == "cat":
                _, choices = spec
                params[placeholder] = trial.suggest_categorical(name, choices)
            else:
                raise ValueError(f"Unknown param kind: {kind}")

        expr = self._build_expr(params)
        log.info("[Bayes] Trial %d: %s", trial.number, expr[:80])

        raw = self.brain.simulate(expr, **self.sim_settings)
        if raw is None:
            # Penalise failed simulations
            return -999.0

        metrics = self.brain.extract_metrics(raw)
        score = self.objective_fn(metrics)

        # Store for later retrieval (filter applied in run())
        trial.set_user_attr("expression", expr)
        trial.set_user_attr("metrics", metrics)
        return score if score is not None else -999.0

    def run(self, filter_fn: Optional[Callable] = None) -> list[MineResult]:
        import optuna

        optuna.logging.set_verbosity(
            [optuna.logging.WARNING, optuna.logging.INFO, optuna.logging.DEBUG][
                min(self.optuna_verbosity, 2)
            ]
        )

        study = optuna.create_study(
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        log.info(
            "[Bayes] Starting optimisation: %d trials | template: %s",
            self.n_trials, self.template[:60],
        )
        study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=False)

        # Collect results from completed (non-penalised) trials
        results: list[MineResult] = []
        for trial in study.trials:
            expr = trial.user_attrs.get("expression")
            metrics = trial.user_attrs.get("metrics")
            if expr is None or metrics is None:
                continue
            passed = filter_fn(metrics) if filter_fn else True
            results.append(MineResult(expression=expr, metrics=metrics, passed=passed))

        passed_count = sum(1 for r in results if r.passed)
        best = study.best_trial
        log.info(
            "[Bayes] Done: %d/%d passed | best score=%.4f | best params=%s",
            passed_count, len(results),
            best.value if best else float("nan"),
            best.params if best else {},
        )
        return results


def _default_objective(metrics: dict) -> float:
    """Default: maximise abs(Sharpe). Returns -999 if unavailable."""
    sharpe = metrics.get("sharpe")
    if sharpe is None:
        return -999.0
    turnover = metrics.get("turnover") or 0.0
    if turnover > 0.70:
        return abs(sharpe) * 0.5
    return abs(sharpe)


# ---------------------------------------------------------------------------
# Strategy 4: Pipeline miner  (三阶递进流水线)
# ---------------------------------------------------------------------------

class PipelineMiner:
    """
    Multi-stage recursive expansion pipeline, mirroring the day1→day2→day3
    workflow in WQ挖掘脚本.

    Stage 1 (day1 equivalent):
        fields × ops → first_order_factory → simulate all → collect passing IDs
        Result IDs are stored on the WQ platform; we fetch them back with
        brain.get_user_alphas() filtered by the run's date window.

    Stage 2 (day2 equivalent):
        Pull passing stage-1 expressions → prune → group_second_order_factory
        → simulate all

    Stage 3 (day3 equivalent):
        Pull passing stage-2 expressions → prune → trade_when_factory
        → simulate all

    Each stage has its own checkpoint key so a crash at stage 2 resumes
    from stage 2, not from stage 1.

    Usage:
        from dear_alpha.factories import TS_OPS, GROUP_OPS
        pipeline = PipelineMiner(
            brain,
            fields=["ts_mean(close,22)", "ts_std_dev(volume,22)"],
            stage1_ops=TS_OPS,
            stage2_group_ops=GROUP_OPS,
            sim_settings={"region": "USA", "universe": "TOP3000", ...},
            checkpoint=Checkpoint("results/progress.json"),
            field_prefix="anl4",   # for prune() field-dedup
            prune_keep=5,
        )
        results = pipeline.run(filter_fns=[gate1, gate2, gate3])
    """

    def __init__(
        self,
        brain,
        fields: list[str],
        stage1_ops: list[str],
        stage2_group_ops: list[str],
        sim_settings: Optional[dict] = None,
        checkpoint: Optional[Checkpoint] = None,
        field_prefix: str = "",
        prune_keep: int = 5,
        init_decay: int = 6,
        windows: Optional[list[int]] = None,
    ):
        from .factories import first_order_factory, group_second_order_factory, trade_when_factory, prune

        self.brain = brain
        self.fields = fields
        self.stage1_ops = stage1_ops
        self.stage2_group_ops = stage2_group_ops
        self.sim_settings = sim_settings or {}
        self.checkpoint = checkpoint or Checkpoint()
        self.field_prefix = field_prefix
        self.prune_keep = prune_keep
        self.init_decay = init_decay
        self.windows = windows

        # Keep factory references
        self._first_order_factory = first_order_factory
        self._group_second_order_factory = group_second_order_factory
        self._trade_when_factory = trade_when_factory
        self._prune = prune

    def _simulate_batch(
        self,
        alpha_decay_pairs: list[tuple[str, int]],
        stage: str,
        filter_fn: Optional[Callable],
    ) -> list[MineResult]:
        """
        Simulate a list of (expression, decay) pairs sequentially with
        checkpoint resume.  Returns all MineResult objects (passed and failed).
        """
        results: list[MineResult] = []
        saved = self.checkpoint.get(stage)
        start = saved["index"]

        # Restore already-done results
        for r in saved["results"]:
            results.append(MineResult(
                expression=r["expression"],
                metrics=r["metrics"],
                passed=r["passed"],
            ))

        if start > 0:
            log.info("[%s] Resuming from %d / %d", stage, start, len(alpha_decay_pairs))

        region = self.sim_settings.get("region", "USA")

        for i, (expr, decay) in enumerate(alpha_decay_pairs):
            if i < start:
                continue

            sim_cfg = dict(self.sim_settings)
            sim_cfg["decay"] = decay

            log.info("[%s] %d/%d: %s", stage, i + 1, len(alpha_decay_pairs), expr[:80])
            raw = self.brain.simulate(expr, **sim_cfg)

            if raw is None:
                self.checkpoint.update(stage, i + 1)
                continue

            metrics = self.brain.extract_metrics(raw)
            passed = filter_fn(metrics) if filter_fn else True
            result = MineResult(expression=expr, metrics=metrics, passed=passed)
            results.append(result)

            self.checkpoint.update(stage, i + 1, {
                "expression": expr,
                "metrics": metrics,
                "passed": passed,
            })

        passed_count = sum(1 for r in results if r.passed)
        log.info("[%s] Done: %d/%d passed", stage, passed_count, len(results))
        return results

    def _passing_pairs(self, results: list[MineResult]) -> list[tuple[str, int]]:
        """Extract (expression, decay) for passing results."""
        out = []
        for r in results:
            if r.passed:
                decay = r.metrics.get("recommended_decay") or self.init_decay
                out.append((r.expression, int(decay)))
        return out

    def run(
        self,
        filter_fns: Optional[list[Optional[Callable]]] = None,
    ) -> dict[str, list[MineResult]]:
        """
        Run all three stages.

        filter_fns: list of up to 3 callables, one per stage.
                    None entries mean "accept all" for that stage.

        Returns {"stage1": [...], "stage2": [...], "stage3": [...]}
        """
        fns = (filter_fns or []) + [None, None, None]
        fn1, fn2, fn3 = fns[0], fns[1], fns[2]
        region = self.sim_settings.get("region", "usa").lower()

        # ── Stage 1: first-order expansion ──────────────────────────────
        log.info("=== Pipeline Stage 1: first-order expansion ===")
        fo_exprs = self._first_order_factory(self.fields, self.stage1_ops, self.windows)
        log.info("Stage 1 generated %d expressions", len(fo_exprs))

        fo_pairs = [(expr, self.init_decay) for expr in fo_exprs]
        random.shuffle(fo_pairs)

        s1_results = self._simulate_batch(fo_pairs, "pipeline_stage1", fn1)

        # ── Stage 2: group second-order expansion ────────────────────────
        log.info("=== Pipeline Stage 2: group second-order expansion ===")
        s1_passing = self._passing_pairs(s1_results)
        s1_pruned = self._prune(s1_passing, self.field_prefix, self.prune_keep)
        log.info("Stage 1 → %d passing → %d after prune", len(s1_passing), len(s1_pruned))

        s1_exprs = [expr for expr, _ in s1_pruned]
        so_exprs = self._group_second_order_factory(s1_exprs, self.stage2_group_ops, region)
        so_pairs = [(expr, decay) for (_, decay), expr in zip(s1_pruned, so_exprs) if expr]
        # Above zip is 1:1 per expr but group factory returns many — rebuild properly:
        so_pairs = []
        for (base_expr, decay) in s1_pruned:
            for grp_expr in self._group_second_order_factory([base_expr], self.stage2_group_ops, region):
                so_pairs.append((grp_expr, decay))

        random.shuffle(so_pairs)
        log.info("Stage 2 testing %d expressions", len(so_pairs))
        s2_results = self._simulate_batch(so_pairs, "pipeline_stage2", fn2)

        # ── Stage 3: trade_when expansion ───────────────────────────────
        log.info("=== Pipeline Stage 3: trade_when expansion ===")
        s2_passing = self._passing_pairs(s2_results)
        s2_pruned = self._prune(s2_passing, self.field_prefix, self.prune_keep)
        log.info("Stage 2 → %d passing → %d after prune", len(s2_passing), len(s2_pruned))

        th_pairs = []
        for (expr, decay) in s2_pruned:
            for tw_expr in self._trade_when_factory(expr):
                th_pairs.append((tw_expr, decay))

        random.shuffle(th_pairs)
        log.info("Stage 3 testing %d expressions", len(th_pairs))
        s3_results = self._simulate_batch(th_pairs, "pipeline_stage3", fn3)

        return {
            "stage1": s1_results,
            "stage2": s2_results,
            "stage3": s3_results,
        }
