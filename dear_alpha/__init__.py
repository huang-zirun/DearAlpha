"""
DearAlpha – WorldQuant Brain alpha mining system.
"""

from .brain import BrainClient
from .generator import BareSignalGenerator, create_backend
from .miner import (
    TemplateMiner, LayeredMiner, BayesianMiner, PipelineMiner,
    ParameterAxis, Checkpoint,
    sweep_numeric_params, get_template_library,
)
from .evaluator import QualityGate, passes_gate, recommend_decay
from .submitter import ResultStore, Submitter
from .factories import (
    first_order_factory, group_factory, group_second_order_factory,
    trade_when_factory, prune,
    BASIC_OPS, TS_OPS, GROUP_OPS, ALL_OPS,
)

__all__ = [
    "BrainClient",
    "BareSignalGenerator", "create_backend",
    "TemplateMiner", "LayeredMiner", "BayesianMiner", "PipelineMiner",
    "ParameterAxis", "Checkpoint",
    "sweep_numeric_params", "get_template_library",
    "QualityGate", "passes_gate", "recommend_decay",
    "ResultStore", "Submitter",
    "first_order_factory", "group_factory", "group_second_order_factory",
    "trade_when_factory", "prune",
    "BASIC_OPS", "TS_OPS", "GROUP_OPS", "ALL_OPS",
]
