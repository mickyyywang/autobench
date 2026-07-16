"""Automatic embodied task generation from JSONL view graphs."""

from .generator import GenerationConfig, TaskGenerator
from .graph_io import load_view_graphs_jsonl, write_tasks_jsonl

__all__ = [
    "GenerationConfig",
    "TaskGenerator",
    "load_view_graphs_jsonl",
    "TaskViewGraphSynthesisConfig",
    "synthesize_task_view_graph",
    "write_tasks_jsonl",
]


def __getattr__(name: str):
    if name in {"TaskViewGraphSynthesisConfig", "synthesize_task_view_graph"}:
        from .layout_synthesis import TaskViewGraphSynthesisConfig, synthesize_task_view_graph

        values = {
            "TaskViewGraphSynthesisConfig": TaskViewGraphSynthesisConfig,
            "synthesize_task_view_graph": synthesize_task_view_graph,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
