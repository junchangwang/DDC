#!/usr/bin/env python3
"""Render the same-literal JOB COMP result with the established style."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
PARENT_EXPERIMENT = HERE.parent / "cr_native_no_btv_20260820"
PLOT_MODULE = PARENT_EXPERIMENT / "plot_results.py"


def load_plot_module():
    spec = importlib.util.spec_from_file_location("native_plot_results", PLOT_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PLOT_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_plot_module()
    source = HERE / "results" / "job_literal_native_logical.csv"
    rows = module.read_logical(
        source,
        "job",
        module.JOB_CASES,
        ("COMP",),
    )
    output_dir = HERE / "figures"
    output_dir.mkdir(exist_ok=True)
    outputs = module.plot_relative_bars(
        rows,
        module.JOB_CASES,
        ("COMP",),
        module.JOB_LABELS,
        output_dir / "job_comp_literal_native_relative",
        240,
    )
    for path in outputs:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
