#!/usr/bin/env python3
"""
DearAlpha – Alpha Mining CLI
=============================

Three mining modes, each reflecting a distinct research philosophy:

  ai       – LLM generates economically-motivated bare signals.
             Best for discovering novel ideas; high-variance output.

  template – Systematic enumeration of curated template expressions.
             Best for exhaustive coverage of a structural pattern.

  submit   – Check and submit passing alphas to WorldQuant Brain.

Usage:
  python mine.py ai       [options]
  python mine.py template [options]
  python mine.py submit   [--dry-run]

Quick start:
  1. Copy configs/default.yaml → config.yaml and fill in credentials/API key.
  2. python mine.py ai --rounds 3
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Logging setup (must come before any dear_alpha imports that log at module
# level, even though there are none currently)
# ---------------------------------------------------------------------------

def _setup_logging(log_path: str, verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler(sys.stdout)]
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    default_path = Path("configs/default.yaml")
    cfg: dict = {}

    # Load defaults first
    if default_path.exists():
        with default_path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    # Override with user config if it exists
    user_path = Path(path)
    if user_path.exists():
        with user_path.open(encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        _deep_merge(cfg, user)
    else:
        if path != "config.yaml":
            print(f"[warn] Config file not found: {path}")

    # Environment variable overrides
    if os.environ.get("OPENROUTER_API_KEY"):
        cfg.setdefault("llm", {})["api_key"] = os.environ["OPENROUTER_API_KEY"]
    if os.environ.get("ANTHROPIC_API_KEY"):
        cfg.setdefault("llm", {})["api_key"] = os.environ["ANTHROPIC_API_KEY"]

    return cfg


def _deep_merge(base: dict, override: dict):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ---------------------------------------------------------------------------
# Client / backend factories
# ---------------------------------------------------------------------------

def _build_brain(cfg: dict):
    from dear_alpha.brain import BrainClient

    brain_cfg = cfg.get("brain", {})
    cred_path = brain_cfg.get("credentials", "credential.txt")

    if not Path(cred_path).exists():
        sys.exit(
            f"[error] Credentials file not found: {cred_path}\n"
            "Create it as a JSON array: [\"email@example.com\", \"password\"]"
        )

    with open(cred_path, encoding="utf-8") as f:
        creds = json.load(f)

    return BrainClient(username=creds[0], password=creds[1])


def _build_generator(cfg: dict):
    from dear_alpha.generator import BareSignalGenerator, create_backend

    llm_cfg = cfg.get("llm", {})
    if not llm_cfg.get("api_key") and llm_cfg.get("provider") != "ollama":
        sys.exit(
            "[error] LLM API key not set.\n"
            "Set it in config.yaml under llm.api_key, or via environment variable\n"
            "(OPENROUTER_API_KEY or ANTHROPIC_API_KEY)."
        )

    backend = create_backend(llm_cfg)
    return BareSignalGenerator(backend)


def _sim_settings(cfg: dict) -> dict:
    s = cfg.get("simulation", {})
    return {
        "region": s.get("region", "USA"),
        "universe": s.get("universe", "TOP3000"),
        "neutralization": s.get("neutralization", "SUBINDUSTRY"),
        "decay": s.get("decay", 0),
        "delay": s.get("delay", 1),
        "truncation": s.get("truncation", 0.08),
    }


def _build_gate(cfg: dict):
    from dear_alpha.evaluator import QualityGate

    q = cfg.get("quality", {})
    return QualityGate(
        min_sharpe=q.get("min_sharpe", 1.25),
        min_fitness=q.get("min_fitness", 1.0),
        min_turnover=q.get("min_turnover", 0.01),
        max_turnover=q.get("max_turnover", 0.70),
        min_long_count=q.get("min_long_count", 50),
        min_short_count=q.get("min_short_count", 50),
    )


# ---------------------------------------------------------------------------
# Mining modes
# ---------------------------------------------------------------------------

def run_ai_mode(cfg: dict, args):
    """
    AI bare-signal generation loop.

    For each round:
      1. Ask LLM to generate N alpha expressions (one economic theme per round).
      2. Simulate each via WorldQuant Brain.
      3. Apply quality gate.
      4. Persist passing alphas.
    """
    from dear_alpha.evaluator import passes_gate, log_metrics, recommend_decay
    from dear_alpha.submitter import ResultStore

    log = logging.getLogger("ai_mode")
    brain = _build_brain(cfg)
    generator = _build_generator(cfg)
    gate = _build_gate(cfg)
    store = ResultStore(cfg.get("paths", {}).get("results", "results/passing_alphas.jsonl"))
    sim_cfg = _sim_settings(cfg)
    batch_size = cfg.get("mining", {}).get("ai_batch_size", 5)
    rounds = args.rounds

    log.info("=== AI Mode: %d rounds × %d signals each ===", rounds, batch_size)
    total_tested = 0
    total_passed = 0

    for rnd in range(1, rounds + 1):
        log.info("── Round %d/%d ──", rnd, rounds)
        expressions = generator.generate_batch(n=batch_size, theme=args.theme or None)

        if not expressions:
            log.warning("LLM returned no expressions this round, skipping")
            continue

        for expr in expressions:
            total_tested += 1
            log.info("Testing: %s", expr[:100])
            result = brain.simulate(expr, **sim_cfg)
            if result is None:
                log.warning("Simulation failed for: %s", expr[:80])
                continue

            metrics = brain.extract_metrics(result)
            passed, reasons = passes_gate(metrics, gate)
            log_metrics(expr, metrics, passed, reasons)

            if passed:
                total_passed += 1
                decay = recommend_decay(metrics.get("turnover"), sim_cfg["decay"])
                record = {
                    "expression": expr,
                    "alpha_id": metrics.get("alpha_id"),
                    "metrics": metrics,
                    "recommended_decay": decay,
                    "source": "ai",
                    "mode": "ai",
                }
                store.save(record)

    log.info(
        "=== AI Mode complete: %d/%d passed gate ===",
        total_passed, total_tested,
    )


def run_template_mode(cfg: dict, args):
    """
    Systematic template enumeration mode.

    Iterates through the built-in template library and mines each template
    against all parameter combinations.
    """
    from dear_alpha.miner import get_template_library, TemplateMiner
    from dear_alpha.evaluator import passes_gate, recommend_decay
    from dear_alpha.submitter import ResultStore

    log = logging.getLogger("template_mode")
    brain = _build_brain(cfg)
    gate = _build_gate(cfg)
    store = ResultStore(cfg.get("paths", {}).get("results", "results/passing_alphas.jsonl"))
    sim_cfg = _sim_settings(cfg)
    max_per_entry = cfg.get("mining", {}).get("template_max_per_entry", 500)

    library = get_template_library()
    log.info("=== Template Mode: %d templates ===", len(library))

    def filter_fn(metrics):
        passed, _ = passes_gate(metrics, gate)
        return passed

    total_passed = 0
    for i, (template, axes) in enumerate(library):
        log.info("[%d/%d] Template: %s", i + 1, len(library), template[:80])
        miner = TemplateMiner(brain, template, axes, sim_settings=sim_cfg)
        results = miner.run(filter_fn=filter_fn, max_expressions=max_per_entry)

        for r in results:
            if r.passed:
                total_passed += 1
                decay = recommend_decay(r.metrics.get("turnover"), sim_cfg["decay"])
                record = {
                    "expression": r.expression,
                    "alpha_id": r.metrics.get("alpha_id"),
                    "metrics": r.metrics,
                    "recommended_decay": decay,
                    "source": "template",
                    "template": template,
                }
                store.save(record)

    log.info("=== Template Mode complete: %d passed gate ===", total_passed)


def run_submit_mode(cfg: dict, args):
    """Submit passing, unsubmitted alphas to WorldQuant Brain."""
    from dear_alpha.submitter import ResultStore, Submitter

    log = logging.getLogger("submit_mode")
    brain = _build_brain(cfg)
    store = ResultStore(cfg.get("paths", {}).get("results", "results/passing_alphas.jsonl"))
    sub_cfg = cfg.get("submission", {})

    submitter = Submitter(
        brain=brain,
        store=store,
        max_per_day=sub_cfg.get("max_per_day", 10),
        max_prod_corr=sub_cfg.get("max_prod_correlation", 0.70),
    )

    log.info("=== Submit Mode (dry_run=%s) ===", args.dry_run)
    submitted = submitter.run(dry_run=args.dry_run)
    log.info("Submitted: %s", submitted)


def _save_results(results, store, sim_cfg, source: str, extra: dict = None):
    """Shared helper: persist passing MineResult objects to the store."""
    from dear_alpha.evaluator import recommend_decay

    total_passed = 0
    for r in results:
        if r.passed:
            total_passed += 1
            decay = recommend_decay(r.metrics.get("turnover"), sim_cfg["decay"])
            record = {
                "expression": r.expression,
                "alpha_id": r.metrics.get("alpha_id"),
                "metrics": r.metrics,
                "recommended_decay": decay,
                "source": source,
            }
            if extra:
                record.update(extra)
            store.save(record)
    return total_passed


def run_pipeline_mode(cfg: dict, args):
    """
    Three-stage recursive expansion pipeline (day1 → day2 → day3).

    Stage 1: fields × ops → first_order_factory → simulate
    Stage 2: passing stage-1 → prune → group expansion → simulate
    Stage 3: passing stage-2 → prune → trade_when expansion → simulate

    Each stage has its own checkpoint; interruptions resume mid-stage.
    """
    from dear_alpha.miner import PipelineMiner, Checkpoint
    from dear_alpha.evaluator import passes_gate, recommend_decay
    from dear_alpha.submitter import ResultStore
    from dear_alpha.factories import TS_OPS, GROUP_OPS

    log = logging.getLogger("pipeline_mode")
    brain = _build_brain(cfg)
    gate = _build_gate(cfg)
    store = ResultStore(cfg.get("paths", {}).get("results", "results/passing_alphas.jsonl"))
    sim_cfg = _sim_settings(cfg)
    p_cfg = cfg.get("mining", {}).get("pipeline", {})

    # Fields: from CLI or config or API
    if args.fields:
        fields = [f.strip() for f in args.fields.split(",")]
    elif p_cfg.get("fields"):
        fields = p_cfg["fields"]
    elif p_cfg.get("dataset_id"):
        # Fetch from API dynamically
        log.info("Fetching data fields for dataset %s …", p_cfg["dataset_id"])
        raw_fields = brain.get_datafields(
            region=sim_cfg["region"],
            universe=sim_cfg["universe"],
            delay=sim_cfg["delay"],
            dataset_id=p_cfg["dataset_id"],
        )
        import pandas as pd
        df = pd.DataFrame(raw_fields)
        from dear_alpha.factories import prune as _prune
        matrix = df[df["type"] == "MATRIX"]["id"].tolist() if "type" in df.columns else []
        fields = [f"winsorize(ts_backfill({f}, 120), std=4)" for f in matrix]
        log.info("Loaded %d fields from API", len(fields))
    else:
        fields = p_cfg.get("fields", ["close", "volume", "returns", "vwap"])

    stage1_ops = p_cfg.get("stage1_ops") or TS_OPS
    stage2_group_ops = p_cfg.get("stage2_group_ops") or GROUP_OPS
    field_prefix = args.field_prefix or p_cfg.get("field_prefix", "")
    prune_keep = args.prune_keep or p_cfg.get("prune_keep", 5)
    init_decay = p_cfg.get("init_decay", 6)
    checkpoint_path = cfg.get("paths", {}).get("checkpoint", "results/progress.json")

    def make_filter(stage_key):
        thresholds = p_cfg.get(stage_key, {})
        min_sharpe = thresholds.get("min_sharpe", gate.min_sharpe)
        min_fitness = thresholds.get("min_fitness", gate.min_fitness)
        def fn(metrics):
            s = metrics.get("sharpe")
            f = metrics.get("fitness")
            lc = (metrics.get("long_count") or 0) + (metrics.get("short_count") or 0)
            return (s is not None and abs(s) >= min_sharpe
                    and f is not None and abs(f) >= min_fitness
                    and lc >= 100)
        return fn

    log.info("=== Pipeline Mode | %d fields | ops: %d ===", len(fields), len(stage1_ops))

    pipeline = PipelineMiner(
        brain=brain,
        fields=fields,
        stage1_ops=stage1_ops,
        stage2_group_ops=stage2_group_ops,
        sim_settings=sim_cfg,
        checkpoint=Checkpoint(checkpoint_path),
        field_prefix=field_prefix,
        prune_keep=prune_keep,
        init_decay=init_decay,
    )

    all_results = pipeline.run(filter_fns=[
        make_filter("stage1_filter"),
        make_filter("stage2_filter"),
        make_filter("stage3_filter"),
    ])

    # Persist all passing results from all stages
    total_passed = 0
    for stage_name, results in all_results.items():
        for r in results:
            if r.passed:
                total_passed += 1
                decay = recommend_decay(r.metrics.get("turnover"), sim_cfg["decay"])
                store.save({
                    "expression": r.expression,
                    "alpha_id": r.metrics.get("alpha_id"),
                    "metrics": r.metrics,
                    "recommended_decay": decay,
                    "source": "pipeline",
                    "stage": stage_name,
                })

    log.info("=== Pipeline Mode complete: %d total passed ===", total_passed)


def run_layered_mode(cfg: dict, args):
    """
    Two-pass layered mining.

    Pass 1: test each field with a small coarse window set to rank fields.
    Pass 2: sweep the full window grid on the top-K surviving fields.

    Much faster than full Cartesian product when there are many fields.
    """
    from dear_alpha.miner import LayeredMiner, ParameterAxis, WINDOW_AXIS
    from dear_alpha.evaluator import passes_gate
    from dear_alpha.submitter import ResultStore

    log = logging.getLogger("layered_mode")
    brain = _build_brain(cfg)
    gate = _build_gate(cfg)
    store = ResultStore(cfg.get("paths", {}).get("results", "results/passing_alphas.jsonl"))
    sim_cfg = _sim_settings(cfg)
    m_cfg = cfg.get("mining", {}).get("layered", {})

    template = args.template or m_cfg.get(
        "template",
        "group_rank(ts_mean({field}, {window}), densify(sector))",
    )
    fields = args.fields.split(",") if args.fields else m_cfg.get("fields", [
        "close", "vwap", "open", "high", "low",
        "volume", "adv20",
        "returns", "pe", "pb", "ps", "roe", "roa", "gm", "npm",
        "cashflow_op", "eps", "revenue_growth",
    ])
    keep = args.keep_fields or m_cfg.get("keep_fields", 5)
    coarse = [int(x) for x in (args.coarse_windows or "5,22,120").split(",")]

    field_axis = ParameterAxis("field", "{field}", fields)

    def filter_fn(metrics):
        passed, _ = passes_gate(metrics, gate)
        return passed

    log.info("=== Layered Mode | template: %s ===", template[:60])
    log.info("Fields: %d  |  keep: %d  |  coarse windows: %s", len(fields), keep, coarse)

    miner = LayeredMiner(
        brain, template, field_axis, WINDOW_AXIS,
        coarse_windows=coarse,
        keep_fields=keep,
        sim_settings=sim_cfg,
    )
    results = miner.run(filter_fn=filter_fn)
    total_passed = _save_results(results, store, sim_cfg, "layered", {"template": template})
    log.info("=== Layered Mode complete: %d passed gate ===", total_passed)


def run_bayesian_mode(cfg: dict, args):
    """
    Bayesian optimisation over numeric + categorical parameters.

    The TPE sampler builds a model of which parameter regions produce
    high |Sharpe| and samples there, converging much faster than grid search.
    """
    from dear_alpha.miner import BayesianMiner
    from dear_alpha.evaluator import passes_gate
    from dear_alpha.submitter import ResultStore

    log = logging.getLogger("bayes_mode")
    brain = _build_brain(cfg)
    gate = _build_gate(cfg)
    store = ResultStore(cfg.get("paths", {}).get("results", "results/passing_alphas.jsonl"))
    sim_cfg = _sim_settings(cfg)
    m_cfg = cfg.get("mining", {}).get("bayesian", {})

    template = args.template or m_cfg.get(
        "template",
        "group_rank(ts_mean({field}, {window}), densify(sector))",
    )
    n_trials = args.n_trials or m_cfg.get("n_trials", 50)

    # Build param_space from CLI or config
    # Config format:
    #   bayesian:
    #     param_space:
    #       "{window}": ["int", 5, 240]
    #       "{field}":  ["cat", ["close", "volume", "returns"]]
    param_space = {}
    cfg_space = m_cfg.get("param_space", {})

    if cfg_space:
        for placeholder, spec in cfg_space.items():
            param_space[placeholder] = tuple(spec)
    else:
        # Sensible default matching the default template
        param_space = {
            "{window}": ("int", 5, 240),
            "{field}":  ("cat", ["close", "vwap", "volume", "returns", "adv20",
                                  "pe", "pb", "roe", "roa", "gm", "cashflow_op"]),
        }

    def filter_fn(metrics):
        passed, _ = passes_gate(metrics, gate)
        return passed

    log.info(
        "=== Bayesian Mode | %d trials | template: %s ===",
        n_trials, template[:60],
    )
    log.info("Param space: %s", {k: v[0] for k, v in param_space.items()})

    miner = BayesianMiner(
        brain, template, param_space,
        n_trials=n_trials,
        sim_settings=sim_cfg,
    )
    results = miner.run(filter_fn=filter_fn)
    total_passed = _save_results(results, store, sim_cfg, "bayesian", {"template": template})
    log.info("=== Bayesian Mode complete: %d passed gate ===", total_passed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mine.py",
        description="DearAlpha – WorldQuant Brain alpha miner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default="config.yaml", help="Path to config YAML (default: config.yaml)")
    p.add_argument("--verbose", "-v", action="store_true", help="Debug logging")

    sub = p.add_subparsers(dest="mode", required=True)

    # ai mode
    ai_p = sub.add_parser("ai", help="LLM bare-signal generation")
    ai_p.add_argument("--rounds", type=int, default=1, help="Number of generation rounds (default: 1)")
    ai_p.add_argument("--theme", type=str, default="", help="Economic theme override (optional)")

    # template mode (Cartesian product)
    sub.add_parser("template", help="Full Cartesian product enumeration over built-in library")

    # pipeline mode
    pipe_p = sub.add_parser("pipeline", help="Three-stage day1→day2→day3 recursive expansion")
    pipe_p.add_argument("--fields", type=str, default="", help="Comma-separated field names (overrides config)")
    pipe_p.add_argument("--field-prefix", type=str, default="", help="Field prefix for prune dedup (e.g. 'anl4')")
    pipe_p.add_argument("--prune-keep", type=int, default=0, help="Max expressions per field to carry forward (default: 5)")

    # layered mode
    lay_p = sub.add_parser("layered", help="Two-pass layered mining (coarse field screen → fine sweep)")
    lay_p.add_argument("--template", type=str, default="", help="Alpha template with {field} and {window}")
    lay_p.add_argument("--fields", type=str, default="", help="Comma-separated list of field names")
    lay_p.add_argument("--keep-fields", type=int, default=0, help="How many top fields to keep (default: 5)")
    lay_p.add_argument("--coarse-windows", type=str, default="", help="Comma-separated coarse windows, e.g. 5,22,120")

    # bayesian mode
    bay_p = sub.add_parser("bayesian", help="Bayesian optimisation over numeric/categorical params")
    bay_p.add_argument("--template", type=str, default="", help="Alpha template with {field} and/or {window}")
    bay_p.add_argument("--n-trials", type=int, default=0, help="Number of Optuna trials (default: 50)")

    # submit mode
    sub_p = sub.add_parser("submit", help="Submit passing alphas")
    sub_p.add_argument("--dry-run", action="store_true", help="Check eligibility but do not actually submit")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_path = cfg.get("paths", {}).get("logs", "results/dear_alpha.log")
    _setup_logging(log_path, verbose=args.verbose)

    log = logging.getLogger("main")
    log.info("DearAlpha starting | mode=%s", args.mode)

    dispatch = {
        "ai":       run_ai_mode,
        "template": run_template_mode,
        "pipeline": run_pipeline_mode,
        "layered":  run_layered_mode,
        "bayesian": run_bayesian_mode,
        "submit":   run_submit_mode,
    }
    dispatch[args.mode](cfg, args)


if __name__ == "__main__":
    main()
